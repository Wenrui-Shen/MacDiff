"""Label-alignment, tie handling and paired sampling checks without GPUs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_stage1_text_geometry import (
    balanced_batches, cached_features, make_plan, normalize, read_captions,
    row_metrics, summarize,
)


class GeometryTests(unittest.TestCase):
    def test_pk_sampling_and_reproducibility(self):
        labels = np.repeat(np.arange(60), 12)
        indices = np.arange(len(labels)) + 1000
        batches = balanced_batches(indices, labels, count=5)
        self.assertEqual(batches, balanced_batches(indices, labels, count=5))
        self.assertNotEqual(batches, balanced_batches(indices, labels, count=5, seed=43))
        for batch in batches:
            self.assertEqual(len(set(batch)), 128)
            _, counts = np.unique(labels[np.array(batch) - 1000], return_counts=True)
            self.assertEqual(len(counts), 16)
            np.testing.assert_array_equal(counts, np.full(16, 8))

    def test_perfect_class_geometry(self):
        labels = np.repeat(np.arange(3), 8)
        features = np.eye(3)[labels]
        result = row_metrics(features @ features.T, labels)
        for key in ('auc', 'p1', 'p5', 'same_cosine', 'gap'):
            np.testing.assert_allclose(result[key], 1)
        np.testing.assert_allclose(result['different_cosine'], 0)

    def test_collapsed_embeddings_are_chance_not_perfect(self):
        labels = np.repeat(np.arange(3), 8)
        result = row_metrics(np.ones((24, 24)), labels)
        np.testing.assert_allclose(result['auc'], .5)
        np.testing.assert_allclose(result['p1'], 7 / 23)
        np.testing.assert_allclose(result['p5'], 7 / 23)
        np.testing.assert_allclose(result['gap'], 0)

    def test_auc_equals_explicit_triplets_and_is_permutation_invariant(self):
        rng = np.random.RandomState(9)
        labels = np.repeat(np.arange(3), 4)
        scores = rng.randint(0, 4, size=(12, 12)).astype(float)
        result = row_metrics(scores, labels)
        for i in range(12):
            values = [float(scores[i, p] > scores[i, n]) + .5 * (scores[i, p] == scores[i, n])
                      for p in range(12) for n in range(12)
                      if p != i and labels[p] == labels[i] and labels[n] != labels[i]]
            self.assertAlmostEqual(result['auc'][i], np.mean(values))
        order = rng.permutation(12)
        shuffled = row_metrics(scores[np.ix_(order, order)], labels[order])
        for key in result:
            np.testing.assert_allclose(shuffled[key], result[key][order])

    def test_diagonal_and_zero_vectors(self):
        scores = np.ones((12, 12))
        labels = np.repeat(np.arange(3), 4)
        expected = row_metrics(scores, labels)
        np.fill_diagonal(scores, -1e9)
        actual = row_metrics(scores, labels)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key])
        with self.assertRaises(ValueError):
            normalize(np.zeros((2, 3)))

    @staticmethod
    def caption(index, prompt='fixed'):
        return {'sample_index': index, 'source_split': 'train', 'model': '8b',
                'prompt': {'sha256': prompt}, 'status': 'accepted', 'actor_count': 1,
                'texts': [{'person_index': 0, 'text': 'The person raises an arm.'}]}

    def test_plan_keeps_raw_npz_indices_and_freezes_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.tile(np.arange(60), 10)
            np.savez(root / 'data.npz', y_train=np.eye(60)[labels])
            caption_path = root / 'captions.jsonl'
            caption_path.write_text('\n'.join(json.dumps(self.caption(i)) for i in range(600)), encoding='utf-8')
            args = argparse.Namespace(output_dir=str(root / 'out'), data_path=str(root / 'data.npz'),
                                      captions=[str(caption_path)], num_batches=2,
                                      classes_per_batch=16, samples_per_class=8, seed=42)
            make_plan(args)
            plan = json.loads((root / 'out/plan.json').read_text())
            for sample in plan['samples']:
                self.assertEqual(sample['label'], labels[sample['index']])
            features = np.eye(60)[[r['label'] for r in plan['samples']]]
            self.assertEqual(summarize(features, plan)['class_macro']['auc'], 1.)
            with self.assertRaises(FileExistsError):
                make_plan(args)

    def test_mixed_prompts_fail_and_invalid_attempt_does_not_replace_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'captions.jsonl'
            good = self.caption(0)
            invalid = {**good, 'status': 'invalid', 'texts': []}
            path.write_text('\n'.join(map(json.dumps, [good, invalid])), encoding='utf-8')
            rows, _, counts = read_captions([path], 10)
            self.assertEqual(rows[0], ['The person raises an arm.'])
            self.assertEqual(counts['other_records'], 1)
            path.write_text('\n'.join(map(json.dumps, [good, self.caption(1, 'changed')])), encoding='utf-8')
            with self.assertRaises(ValueError):
                read_captions([path], 10)

    def test_cache_refuses_changed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'model': 'first'}
            cached_features(root, 'cache', identity, lambda: np.eye(3))
            actual = cached_features(root, 'cache', identity, lambda: self.fail('Should reuse cache'))
            np.testing.assert_array_equal(actual, np.eye(3))
            with self.assertRaises(ValueError):
                cached_features(root, 'cache', {'model': 'second'}, lambda: np.eye(3))


if __name__ == '__main__':
    unittest.main()
