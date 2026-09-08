"""Numerical and leakage regressions. Run without torch or NTU data."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stage1_readout import (
    PROTOCOL, classification_metrics, compare_cache, fit_ridge, fit_transform,
    normalize, select_supports, subset_indices, transform,
)


class ReadoutTests(unittest.TestCase):
    def test_support_cache_and_seed_selection(self):
        labels = np.repeat([2, 5, 9], 8)
        classes, supports = select_supports(labels, [0, 1, 2])
        for ids in supports.values():
            np.testing.assert_array_equal(labels[ids], classes)
        self.assertEqual(supports, select_supports(labels, [0, 1, 2])[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'support.json'
            path.write_text(json.dumps({'indices': supports['0'], 'class_ids': classes.tolist(),
                                        'seed': 0, 'num_samples': len(labels)}))
            self.assertEqual(select_supports(labels, [0], [str(path)])[1]['0'], supports['0'])
            with self.assertRaises(ValueError):
                select_supports(labels, [1], [str(path)])

    def test_subsetting_keeps_original_support_ids(self):
        ids = subset_indices(100, 12, 17, [90, 2, 80])
        self.assertEqual(len(ids), 12)
        self.assertTrue(set([90, 2, 80]).issubset(ids))
        self.assertTrue(np.all(np.diff(ids) > 0))
        np.testing.assert_array_equal(ids, subset_indices(100, 12, 17, [90, 2, 80]))
        with self.assertRaises(ValueError):
            subset_indices(100, 3, 17, [90, 2, 80])

    def test_transforms_use_only_fit_rows(self):
        x = np.random.RandomState(7).normal(size=(40, 5)).astype(np.float32)
        indices = np.arange(5, 40)
        first = fit_transform(x, indices, pca_rank=3)
        x[:5] = 1e9  # excluded exemplars must not alter fitted statistics/PCA
        second = fit_transform(x, indices, pca_rank=3)
        for key in ('mean', 'std', 'components', 'eigenvalues', 'pca_fit_indices'):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertTrue(set(first['pca_fit_indices']).issubset(indices))

    def test_regularized_pca_whitening_matches_covariance(self):
        rng = np.random.RandomState(8)
        x = rng.normal(size=(90, 4)).astype(np.float32)
        x[:, 1] += 0.8 * x[:, 0]
        state = fit_transform(x, np.arange(len(x)), 4, 100, 0.1)
        components = state['components']
        np.testing.assert_allclose(components.T @ components, np.eye(4), atol=1e-5)
        z = ((x - state['mean']) / state['std'] - state['pca_mean']) @ components
        whitened = z / state['whiten_scale']
        expected_variance = state['eigenvalues'] / state['whiten_scale'] ** 2
        np.testing.assert_allclose(np.cov(whitened.T), np.diag(expected_variance), atol=2e-5)
        self.assertEqual(transform(x, state, 'pca').shape, transform(x, state, 'whitened').shape)

    def test_ridge_dual_equals_primal_with_unpenalized_bias(self):
        e = np.random.RandomState(9).normal(size=(5, 7))
        weights, bias, penalty = fit_ridge(e, 0.3)
        ec = e - e.mean(axis=0)
        y = np.eye(len(e)) - 1.0 / len(e)
        expected = np.linalg.solve(ec.T @ ec + penalty * np.eye(e.shape[1]), ec.T @ y)
        np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(e.mean(axis=0) @ weights + bias, np.full(len(e), 0.2), atol=1e-6)

    def test_linear_readout_retains_feature_magnitude(self):
        x = np.array([[1., 2.], [4., 8.], [-2., 1.]], dtype=np.float32)
        state = fit_transform(x, np.arange(3), pca_rank=0)
        linear = transform(x, state, 'linear_standardized')
        np.testing.assert_allclose(linear, (x - state['mean']) / state['std'])
        np.testing.assert_allclose(normalize(linear), transform(x, state, 'standardized'))

    def test_constant_dimensions_and_degenerate_supports_are_finite(self):
        x = np.ones((12, 5), dtype=np.float32)
        state = fit_transform(x, np.arange(12), 3, 12)
        for mode in ('raw', 'centered', 'standardized', 'pca', 'whitened', 'linear_standardized'):
            self.assertTrue(np.isfinite(transform(x, state, mode)).all())
        weights, bias, _ = fit_ridge(normalize(x[:3]), 1)
        np.testing.assert_allclose(x @ weights + bias, np.full((12, 3), 1 / 3), atol=1e-5)

    def test_metrics_class_mapping_and_extreme_logits(self):
        scores = np.array([[1000., -1000.], [-1000., 1000.], [-1000., 1000.]])
        result = classification_metrics(scores, np.array([3, 7, 3]), np.array([3, 7]), .04)
        self.assertAlmostEqual(result['top1'], 200 / 3)
        self.assertEqual(result['confusion'], [[1, 1], [0, 1]])
        self.assertTrue(np.isfinite(result['nll']))
        self.assertAlmostEqual(result['ece'], 1 / 3)
        with self.assertRaises(ValueError):
            classification_metrics(scores, np.array([3, 7, 9]), np.array([3, 7]), 1)

    def make_cache(self, root):
        rng = np.random.RandomState(11)
        labels = np.repeat(np.arange(3), 12)
        centers = np.eye(3, 6) * 4
        train = centers[labels] + rng.normal(scale=.1, size=(36, 6))
        eval_labels = np.repeat(np.arange(3), 5)
        evaluation = centers[eval_labels] + rng.normal(scale=.1, size=(15, 6))
        classes, supports = select_supports(labels, [0, 1, 2])
        root.mkdir()
        for split, x, y in [('train', train, labels), ('eval', evaluation, eval_labels)]:
            np.save(root / (split + '_I.npy'), x.astype(np.float32))
            np.save(root / (split + '_labels.npy'), y)
            np.save(root / (split + '_indices.npy'), np.arange(len(x)))
        manifest = {'protocol': PROTOCOL, 'project': 'synthetic', 'complete': True,
                    'supports': supports, 'branches': {'I': 6}, 'class_ids': classes.tolist()}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def compare_args(self, root, output):
        return argparse.Namespace(cache_dir=str(root), output_dir=str(output),
                                  pca_rank=3, pca_max_samples=20, whiten_shrinkage=.1,
                                  ridge_alphas=[.1, 1., 10.], cosine_temperature=.1,
                                  ridge_temperature=1., seed=17)

    def test_end_to_end_cache_scoring_and_eval_label_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / 'cache'
            manifest = self.make_cache(root)
            first_args = self.compare_args(root, parent / 'first')
            with contextlib.redirect_stdout(io.StringIO()):
                compare_cache(first_args, 'synthetic')
            report = json.loads((parent / 'first/comparison.json').read_text())
            self.assertEqual(len(report['results']), 33)  # 5 cosine + 6 ridge x 3 seeds
            union = set(sum(manifest['supports'].values(), []))
            fitted = np.load(parent / 'first/fit_original_indices.npy')
            self.assertFalse(union.intersection(fitted.tolist()))
            self.assertEqual(report['results'][0]['eval']['top1'], 100)
            # Relabel held-out samples: fitted transforms and train/support results stay equal.
            y = np.load(root / 'eval_labels.npy')
            np.save(root / 'eval_labels.npy', (y + 1) % 3)
            with contextlib.redirect_stdout(io.StringIO()):
                compare_cache(self.compare_args(root, parent / 'second'), 'synthetic')
            second = json.loads((parent / 'second/comparison.json').read_text())
            for left, right in zip(report['results'], second['results']):
                self.assertEqual(left['train_unlabeled'], right['train_unlabeled'])
                self.assertEqual(left['support'], right['support'])
            self.assertEqual(second['results'][0]['eval']['top1'], 0)
            with self.assertRaises(FileExistsError):
                compare_cache(first_args, 'synthetic')

    def test_incomplete_and_wrong_project_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cache'
            self.make_cache(root)
            with self.assertRaises(ValueError):
                compare_cache(self.compare_args(root, Path(directory) / 'out'), 'wrong_project')


if __name__ == '__main__':
    unittest.main()
