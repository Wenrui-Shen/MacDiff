"""SNN reference checks, real igraph fits, and label-isolated integration tests."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_cache_v2_graph as graph_test
from tests import test_cache_v2_discovery as discovery_tests


HAS_IGRAPH = importlib.util.find_spec('igraph') is not None


class GraphMathTests(unittest.TestCase):
    def test_snn_matches_set_intersections_on_union_edges(self):
        rng = np.random.RandomState(31)
        n, k = 19, 5
        neighbors = np.stack([rng.permutation([j for j in range(n) if i != j])[:k] for i in range(n)])
        actual_edges, weights, candidates = graph_test.snn_edges(neighbors, k, .1, 3)
        sets = [set(row) for row in neighbors]
        union = sorted({tuple(sorted((i, int(j)))) for i in range(n) for j in neighbors[i]})
        expected = {}
        for i, j in union:
            w = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            if w > .1:
                expected[(i, j)] = w
        self.assertEqual(candidates, len(union))
        self.assertEqual([tuple(e) for e in actual_edges], list(expected))
        np.testing.assert_allclose(weights, list(expected.values()))
        self.assertTrue(np.all(actual_edges[:, 0] < actual_edges[:, 1]))

    def test_metrics_permutation_invariant_and_penalize_splitting(self):
        truth = np.repeat([0, 1, 2], 4)
        remapped = np.repeat([55, -8, 900], 4)
        same = graph_test.partition_metrics(truth, remapped)
        self.assertAlmostEqual(same['ari'], 1.)
        self.assertAlmostEqual(same['nmi'], 1.)
        split = graph_test.partition_metrics(truth, np.arange(len(truth)))
        self.assertAlmostEqual(split['ari'], 0.)
        self.assertAlmostEqual(split['homogeneity'], 1.)
        self.assertLess(split['completeness'], 1.)
        collapsed = graph_test.partition_metrics(truth, np.zeros(len(truth)))
        self.assertAlmostEqual(collapsed['ari'], 0.)
        self.assertAlmostEqual(collapsed['nmi'], 0.)
        self.assertEqual(graph_test.partition_metrics([1], [2])['ari'], 1.)

    def test_evaluation_counts_and_winner_selection(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([0, 1, 2, 2, 2, 2])
        metrics, matrix, _, per_class, per_cluster = graph_test.evaluate(labels, predicted)
        self.assertEqual(int(matrix.sum()), 6)
        self.assertEqual(per_class[0]['clusters_touched'], 2)
        self.assertEqual(per_cluster[2]['classes_touched'], 2)
        self.assertAlmostEqual(metrics['weighted_purity'], 4 / 6)
        runs = [dict(setting_id='a', method='infomap', objective_value=score, run_id=str(i))
                for i, score in enumerate([3., 2., 4.])]
        runs += [dict(setting_id='b', method='leiden', objective_value=score, run_id='b' + str(i))
                 for i, score in enumerate([.2, .5, .3])]
        self.assertEqual(graph_test.choose_winners(runs), {'a': '1', 'b': 'b1'})
        runs = [dict(setting_id='a', method='infomap', objective_value=None, run_id='empty')]
        self.assertEqual(graph_test.choose_winners(runs), {'a': 'empty'})


@unittest.skipUnless(HAS_IGRAPH, 'Install igraph==0.11.8 for graph integration tests')
class GraphIntegrationTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        cache, data, labels, _ = discovery_tests.DiscoveryTests.fixture(root)
        neighbors = []
        for i in range(len(labels)):
            group = np.flatnonzero(labels == labels[i])
            own = int(np.flatnonzero(group == i)[0])
            neighbors.append([int(group[(own + j) % len(group)]) for j in range(1, len(group))])
        directory = root / 'neighbors'
        directory.mkdir()
        np.save(directory / 'neighbor_indices.npy', np.asarray(neighbors, dtype=np.int32))
        manifest = json.loads((cache / 'manifest.json').read_text())
        graph_test.write_json(directory / 'run_config.json', {
            'sample_count': len(labels), 'feature_dim': 4, 'cache_identity': manifest['identity'],
            'representation': 'person_features[:,0,:]; renormalized FP32; raw row = sample_index'})
        return cache, directory, data, labels

    def test_real_algorithms_find_cliques_and_preserve_isolates(self):
        edges = list(itertools.combinations(range(8), 2)) + list(itertools.combinations(range(8, 16), 2))
        graph = graph_test.make_graph(17, np.array(edges), np.ones(len(edges)))
        for method in ('infomap', 'leiden'):
            labels, stats = graph_test.fit_partition(graph, method, 42, 1., 1)
            self.assertEqual(stats['k_total'], 3)
            self.assertEqual(stats['k_nonisolated'], 2)
            self.assertEqual(stats['singleton_clusters'], 1)
            self.assertEqual(len(np.unique(labels[:8])), 1)
            self.assertNotEqual(labels[0], labels[8])
            again, _ = graph_test.fit_partition(graph, method, 42, 1., 1)
            np.testing.assert_array_equal(again, labels)
        empty = graph_test.make_graph(5, np.empty((0, 2), dtype=int), np.array([]))
        _, stats = graph_test.fit_partition(empty, 'infomap', 42)
        self.assertEqual(stats['k_total'], 5)
        self.assertEqual(stats['k_nonisolated'], 0)
        self.assertIsNone(stats['objective_value'])

    def test_metrics_match_igraph(self):
        import igraph as ig
        rng = np.random.RandomState(19)
        for size in (10, 200, 1000):
            a = rng.randint(0, 5, size).tolist()
            b = rng.randint(0, 7, size).tolist()
            ours = graph_test.partition_metrics(a, b)
            self.assertAlmostEqual(ours['ari'], ig.compare_communities(a, b, method='adjusted_rand'), places=10)
            self.assertAlmostEqual(ours['nmi'], ig.compare_communities(a, b, method='nmi'), places=10)

    def test_full_pipeline_label_separation_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, neighbor_dir, data, _ = self.fixture(root)
            output = root / 'out'
            argv = ['test_cache_v2_graph.py', '--cache', str(cache), '--neighbor-dir', str(neighbor_dir),
                    '--output-dir', str(output), '--knn', '3', '7', '--seeds', '42', '43',
                    '--resolutions', '1', '--edge-block-size', '11']
            # Initial run has no labels at all.
            with patch.object(sys, 'argv', argv), patch.object(graph_test, 'read_labels') as reader:
                graph_test.main()
                reader.assert_not_called()
            real_reader = graph_test.read_labels
            def audited_reader(path, manifest):
                blind = json.loads((output / 'label_blind_results.json').read_text())
                self.assertEqual(len(blind['runs']), 8)
                self.assertEqual(len(blind['winners']), 4)
                self.assertTrue(all('evaluation' not in r for r in blind['runs']))
                self.assertTrue(all((output / r['prediction_file']).exists() for r in blind['runs']))
                return real_reader(path, manifest)
            # Resume performs only label audit, no graph construction or model fit.
            with patch.object(sys, 'argv', argv + ['--resume', '--data-path', str(data)]), \
                 patch.object(graph_test, 'read_labels', audited_reader), \
                 patch.object(graph_test, 'fit_partition', side_effect=AssertionError('unneeded refit')), \
                 patch.object(graph_test, 'snn_edges', side_effect=AssertionError('unneeded rebuild')):
                graph_test.main()
            result = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result['true_k_evaluation_only'], 4)
            for r in result['runs']:
                if r['k'] == 7:
                    self.assertEqual(r['k_total'], 4)
                    self.assertAlmostEqual(r['evaluation']['ari'], 1.)
            self.assertTrue((output / 'per_cluster.csv').exists())
            self.assertTrue((output / 'per_class.csv').exists())
            self.assertTrue(any(r['comparison'].startswith('between_knn') for r in result['stability']))
            wrong = json.loads((neighbor_dir / 'run_config.json').read_text())
            wrong['representation'] = 'person1'
            graph_test.write_json(neighbor_dir / 'run_config.json', wrong)
            with self.assertRaisesRegex(ValueError, 'person0'):
                graph_test.load_neighbors(cache, neighbor_dir, 7)


if __name__ == '__main__':
    unittest.main()
