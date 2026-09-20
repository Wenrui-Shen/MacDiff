"""Discovery invariants and label-isolated end-to-end checks on synthetic caches."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_cache_v2_discovery as discovery


class DiscoveryTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        labels = np.repeat(np.arange(4), 8)
        x = np.eye(4, dtype=np.float32)[labels]
        cache = root / 'cache'
        cache.mkdir()
        # Deliberately different person1, to expose unintended person averaging.
        np.save(cache / 'person_features.npy', np.stack([x, -x], axis=1))
        np.save(cache / 'person_valid.npy', np.ones((len(x), 2), dtype=bool))
        data = root / 'data.npz'
        np.savez(data, y_train=np.eye(4)[labels])
        manifest = {'protocol': 'macdiff_clip_token_cache_v2', 'complete': True,
                    'sample_count': len(x), 'feature_dim': 4,
                    'identity': {'data': discovery.identity(data)},
                    'files': {name: discovery.identity(cache / name) for name in
                              ('person_features.npy', 'person_valid.npy')}}
        discovery.write_json(cache / 'manifest.json', manifest)
        return cache, data, labels, x

    def test_exact_block_neighbors_against_dense_reference(self):
        rng = np.random.RandomState(17)
        x = rng.randn(23, 7).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        ids, distances = discovery.exact_neighbors(x, [1, 5], 4)
        scores = x @ x.T
        np.fill_diagonal(scores, -np.inf)
        expected = np.argsort(-scores, axis=1)[:, :5]
        np.testing.assert_array_equal(ids, expected)
        d = np.sqrt(np.maximum(2 - 2 * np.take_along_axis(scores, expected, axis=1), 0))
        np.testing.assert_allclose(distances[5], d.mean(axis=1), atol=1e-6)
        self.assertFalse(np.any(ids == np.arange(len(x))[:, None]))

    def test_fps_covers_separated_classes_and_never_repeats(self):
        labels = np.repeat(np.arange(4), 8)
        x = np.eye(4, dtype=np.float32)[labels]
        for weights in (None, np.ones(len(x), dtype=np.float32)):
            ids, curves = discovery.select_sequence(x, len(x), 42, weights)
            self.assertEqual(len(np.unique(labels[ids[:4]])), 4)
            self.assertEqual(len(np.unique(ids)), len(x))
            self.assertTrue(np.all(np.diff(curves, axis=0) <= 1e-7))
            np.testing.assert_allclose(curves[3:], 0)
        random_ids, _ = discovery.select_sequence(x, 10, 42, random=True)
        np.testing.assert_array_equal(random_ids, np.random.RandomState(42).permutation(len(x))[:10])
        self.assertEqual(ids[0], random_ids[0])

    def test_density_ties_and_knee(self):
        weights = discovery.density_weights(np.array([0., 0., 1., 3.]), .1)
        self.assertEqual(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertGreater(weights[2], weights[3])
        self.assertGreater(weights.min(), 0)
        self.assertIsNone(discovery.elbow_candidate(np.ones(10))['candidate_k'])
        self.assertIsNone(discovery.elbow_candidate(np.linspace(1, 0, 10))['candidate_k'])
        self.assertEqual(discovery.elbow_candidate([1., .7, .4, .1, .1, .1, .1, .1])['candidate_k'], 4)

    def test_person0_provenance_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, data, labels, expected = self.fixture(root)
            x, manifest = discovery.load_features(cache)
            np.testing.assert_array_equal(x, expected)
            np.testing.assert_array_equal(discovery.read_labels(data, manifest), labels)
            other = root / 'other.npz'
            np.savez(other, y_train=labels[::-1])
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                discovery.read_labels(other, manifest)
            valid = np.ones((len(x), 2), dtype=bool)
            valid[3, 0] = False
            np.save(cache / 'person_valid.npy', valid)
            manifest['files']['person_valid.npy'] = discovery.identity(cache / 'person_valid.npy')
            discovery.write_json(cache / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'person0'):
                discovery.load_features(cache)

    def test_end_to_end_freezes_all_blind_results_before_label_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, data, labels, _ = self.fixture(root)
            output = root / 'result'
            real_reader = discovery.read_labels

            def audited_reader(path, manifest):
                blind = json.loads((output / 'label_blind_results.json').read_text())
                self.assertEqual(len(blind), 6)
                self.assertTrue(all('evaluation' not in r for r in blind))
                self.assertTrue(all((output / r['sequence_file']).exists() for r in blind))
                return real_reader(path, manifest)

            argv = ['test_cache_v2_discovery.py', '--cache', str(cache), '--data-path', str(data),
                    '--output-dir', str(output), '--device', 'cpu', '--max-centers', '12',
                    '--knn', '3', '--seeds', '42', '43', '--block-size', '7']
            with patch.object(sys, 'argv', argv), patch.object(discovery, 'read_labels', audited_reader):
                discovery.main()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['true_k_evaluation_only'], 4)
            self.assertEqual(summary['full_population_p1'], 1.)
            for run in summary['runs']:
                if run['method'] != 'random':
                    self.assertEqual(run['evaluation']['coverage_at_true_k'], 4)
            self.assertTrue((output / 'curves.csv').exists())
            self.assertTrue((output / 'report.md').exists())
            # Relabeling can change the audit, but never the persisted selections.
            ids = np.load(output / 'fps_seed42.npz')['selected_indices']
            original = discovery.evaluate_sequence(ids, labels, {})
            collapsed = discovery.evaluate_sequence(ids, np.zeros_like(labels), {})
            self.assertEqual(original['coverage_at_budget'], 4)
            self.assertEqual(collapsed['coverage_at_budget'], 1)

    def test_unlabelled_run_and_collapsed_features(self):
        x = np.ones((8, 3), dtype=np.float32) / np.sqrt(np.float32(3))
        selected, _ = discovery.select_sequence(x, 8, 42)
        self.assertEqual(len(np.unique(selected)), 8)
        with tempfile.TemporaryDirectory() as tmp:
            cache, _, _, _ = self.fixture(Path(tmp))
            output = Path(tmp) / 'unlabelled'
            argv = ['test_cache_v2_discovery.py', '--cache', str(cache), '--output-dir', str(output),
                    '--device', 'cpu', '--max-centers', '8', '--knn', '3', '--seeds', '42']
            with patch.object(sys, 'argv', argv), patch.object(discovery, 'read_labels') as reader:
                discovery.main()
                reader.assert_not_called()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertNotIn('true_k_evaluation_only', summary)
            self.assertTrue(all('evaluation' not in r for r in summary['runs']))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed locally')
    def test_torch_backend_matches_numpy(self):
        import torch
        rng = np.random.RandomState(91)
        x = rng.randn(30, 12).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        ids_np, density_np = discovery.exact_neighbors(x, [3], 7)
        weights = discovery.density_weights(density_np[3], .1)
        a, ac = discovery.select_sequence(x, 15, 42, weights)
        for device in (['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']):
            tensor = torch.as_tensor(x, device=device)
            ids_t, density_t = discovery.exact_neighbors(tensor, [3], 7)
            np.testing.assert_array_equal(ids_np, ids_t)
            np.testing.assert_allclose(density_np[3], density_t[3], atol=1e-6)
            b, bc = discovery.select_sequence(tensor, 15, 42, weights)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(ac, bc, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
