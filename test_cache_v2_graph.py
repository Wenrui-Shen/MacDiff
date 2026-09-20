"""Full-population SNN + Infomap/Leiden audit using saved cache-v2 neighbors.

Dependencies: NumPy, igraph==0.11.8 (Python >=3.8); no Torch/GPU/scikit-learn.
Keep test_cache_v2_discovery.py beside this script for cache/provenance helpers.
All partitions, per-setting winners and stability scores are frozen BEFORE labels.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np

from test_cache_v2_discovery import identity, load_features, read_labels, write_json


PROTOCOL = 'cache_v2_snn_graph_discovery_v1'


def load_neighbors(cache, neighbor_dir, max_k):
    features, manifest = load_features(cache)
    n, dim = features.shape
    del features
    root = Path(neighbor_dir)
    config = json.loads((root / 'run_config.json').read_text(encoding='utf-8'))
    if (config.get('cache_identity') != manifest['identity']
            or config.get('sample_count') != n or config.get('feature_dim') != dim
            or config.get('representation') !=
            'person_features[:,0,:]; renormalized FP32; raw row = sample_index'):
        raise ValueError('Neighbor run_config does not match this cache/person0/raw row order')
    neighbors = np.load(root / 'neighbor_indices.npy', allow_pickle=False)
    if (neighbors.ndim != 2 or neighbors.shape[0] != n or neighbors.shape[1] < max_k
            or neighbors.dtype.kind not in 'iu' or np.any(neighbors < 0) or np.any(neighbors >= n)):
        raise ValueError('Invalid neighbor indices or requested k exceeds saved neighbor count')
    neighbors = np.array(neighbors[:, :max_k], dtype=np.int64)
    if np.any(neighbors == np.arange(n)[:, None]):
        raise ValueError('Neighbors must exclude self')
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError('A neighbor row contains duplicate indices')
    return neighbors, manifest


def snn_edges(neighbors, k, min_jaccard=0., block_size=4096):
    """Jaccard on the undirected UNION of directed kNN candidate edges.

    Match encoded (row, neighbor) keys in blocks. Never compute A @ A.T, which
    can become dense around hubs. Memory is O(N*k + block_size*k), not O(N^2).
    Self is excluded from neighbor sets; zero-overlap edges are always dropped.
    """
    n = len(neighbors)
    near = np.asarray(neighbors[:, :k], dtype=np.int64)
    if near.shape[1] != k or k < 1 or not 0 <= min_jaccard < 1 or block_size < 1:
        raise ValueError('Invalid SNN parameters')
    row = np.repeat(np.arange(n, dtype=np.int64), k)
    other = near.ravel()
    candidate = np.unique(np.minimum(row, other) * n + np.maximum(row, other))
    sorted_keys = (np.arange(n, dtype=np.int64)[:, None] * n + np.sort(near, axis=1)).ravel()
    kept_codes, kept_weights = [], []
    for start in range(0, len(candidate), block_size):
        codes = candidate[start:start + block_size]
        u, v = codes // n, codes % n
        queries = v[:, None] * n + near[u]
        positions = np.searchsorted(sorted_keys, queries)
        common = (sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == queries).sum(axis=1)
        weights = common / (2. * k - common)
        keep = (u != v) & (weights > min_jaccard)
        kept_codes.append(codes[keep])
        kept_weights.append(weights[keep])
        if start == 0 or start + block_size >= len(candidate) or start // block_size % 100 == 0:
            print('SNN k={}: {}/{} candidate edges'.format(
                k, min(start + block_size, len(candidate)), len(candidate)), flush=True)
    codes = np.concatenate(kept_codes)
    weights = np.concatenate(kept_weights).astype(np.float64)
    edges = np.column_stack((codes // n, codes % n))
    return edges, weights, int(len(candidate))


def make_graph(n, edges, weights):
    import igraph as ig
    graph = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    graph.es['weight'] = weights.tolist()
    return graph


def graph_stats(graph, candidates):
    degrees = np.asarray(graph.degree())
    sizes = np.asarray(graph.connected_components().sizes())
    return {'nodes': graph.vcount(), 'edges': graph.ecount(), 'candidate_edges': candidates,
            'isolated_nodes': int((degrees == 0).sum()), 'components': int(len(sizes)),
            'largest_component_fraction': float(sizes.max() / graph.vcount()),
            'degree_quantiles_0_50_95_100': np.quantile(degrees, [0, .5, .95, 1]).tolist()}


def canonical_labels(labels):
    return np.unique(np.asarray(labels), return_inverse=True)[1].astype(np.int32)


def partition_stats(labels, active, small_size):
    sizes = np.bincount(labels)
    active_k = len(np.unique(labels[active])) if len(active) else 0
    return {'k_total': int(len(sizes)), 'k_nonisolated': int(active_k),
            'singleton_clusters': int((sizes == 1).sum()),
            'small_cluster_cutoff': small_size,
            'clusters_below_cutoff': int((sizes < small_size).sum()),
            'samples_below_cutoff': int(sizes[sizes < small_size].sum()),
            'largest_cluster_fraction': float(sizes.max() / len(labels)),
            'size_quantiles_0_25_50_75_100': np.quantile(sizes, [0, .25, .5, .75, 1]).tolist()}


def fit_partition(graph, method, seed, resolution=None, trials=1, small_size=10):
    import igraph as ig
    ig.set_random_number_generator(random.Random(seed))
    active = np.flatnonzero(np.asarray(graph.degree()) > 0)
    isolated = np.flatnonzero(np.asarray(graph.degree()) == 0)
    labels = np.empty(graph.vcount(), dtype=np.int32)
    score = None
    if len(active):
        subgraph = graph.induced_subgraph(active.tolist())
        if method == 'infomap':
            partition = subgraph.community_infomap(edge_weights='weight', trials=trials)
            score = float(partition.codelength)
        elif method == 'leiden':
            partition = subgraph.community_leiden(objective_function='modularity', weights='weight',
                                                 resolution=resolution, n_iterations=-1)
            score = float(subgraph.modularity(partition.membership, weights='weight', resolution=resolution))
        else:
            raise ValueError('Unknown method: ' + method)
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite clustering objective: ' + method)
        labels[active] = canonical_labels(partition.membership)
        next_label = int(labels[active].max()) + 1
    else:
        next_label = 0
    # No silent removal, nearest-neighbor repair, or arbitrary isolated-node merging.
    labels[isolated] = np.arange(next_label, next_label + len(isolated))
    labels = canonical_labels(labels)
    stats = partition_stats(labels, active, small_size)
    stats.update({'objective': 'codelength_min' if method == 'infomap' else 'modularity_max',
                  'objective_value': score,
                  'status': 'no_edges' if not len(active) else 'ok'})
    return labels, stats


def partition_metrics(a, b):
    """Exact ARI/NMI using observed contingency entries, without a dense KxK matrix."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Partitions must be nonempty, aligned one-dimensional arrays')
    _, ai, ac = np.unique(a, return_inverse=True, return_counts=True)
    _, bi, bc = np.unique(b, return_inverse=True, return_counts=True)
    keys, counts = np.unique(ai.astype(np.int64) * len(bc) + bi, return_counts=True)
    n = float(len(a))
    def pairs(c):
        c = c.astype(np.float64)
        return float(np.sum(c * (c - 1) / 2))
    total = n * (n - 1) / 2
    observed, pa, pb = pairs(counts), pairs(ac), pairs(bc)
    if not total:
        ari = 1.
    else:
        expected = pa * pb / total
        denominator = (pa + pb) / 2 - expected
        ari = (observed - expected) / denominator if abs(denominator) > 1e-12 else 1.
    ha = float(-np.sum(ac / n * np.log(ac / n)))
    hb = float(-np.sum(bc / n * np.log(bc / n)))
    mi = float(np.sum(counts / n * np.log(counts.astype(float) * n /
                   (ac[keys // len(bc)].astype(float) * bc[keys % len(bc)]))))
    mi = max(0., mi)
    return {'ari': float(ari), 'nmi': float(np.clip(2 * mi / (ha + hb), 0, 1)) if ha + hb else 1.,
            'homogeneity': float(np.clip(mi / ha, 0, 1)) if ha else 1.,
            'completeness': float(np.clip(mi / hb, 0, 1)) if hb else 1.}


def evaluate(labels, prediction):
    classes, target = np.unique(labels, return_inverse=True)
    cluster_ids, predicted = np.unique(prediction, return_inverse=True)
    counts = np.zeros((len(classes), len(cluster_ids)), dtype=np.int64)
    np.add.at(counts, (target, predicted), 1)
    metrics = partition_metrics(labels, prediction)
    metrics.update({'true_k': int(len(classes)), 'absolute_k_error': abs(len(cluster_ids) - len(classes)),
                    'weighted_purity': float(counts.max(axis=0).sum() / len(labels))})
    # Counts of any nonzero intersection expose even minor splitting/mixing.
    per_class = [{'class_id': int(c), 'samples': int(counts[i].sum()),
                  'clusters_touched': int((counts[i] > 0).sum()),
                  'largest_piece_fraction': float(counts[i].max() / counts[i].sum())}
                 for i, c in enumerate(classes)]
    per_cluster = [{'cluster_id': int(c), 'samples': int(counts[:, j].sum()),
                    'classes_touched': int((counts[:, j] > 0).sum()),
                    'dominant_class': int(classes[counts[:, j].argmax()]),
                    'purity': float(counts[:, j].max() / counts[:, j].sum())}
                   for j, c in enumerate(cluster_ids)]
    return metrics, counts, classes, per_class, per_cluster


def setting_id(k, method, resolution):
    return 'k{}_{}'.format(k, method) + (('_r' + format(resolution, '.8g')) if resolution is not None else '')


def choose_winners(runs):
    winners = {}
    for run in runs:
        key = run['setting_id']
        value = run['objective_value']
        candidate = value if run['method'] == 'infomap' else -value if value is not None else None
        old = winners.get(key)
        if old is None or (candidate is not None and candidate < old[0]):
            winners[key] = (candidate if candidate is not None else float('inf'), run['run_id'])
    return {k: v[1] for k, v in winners.items()}


def write_csv(path, records, fields=None):
    if not records and fields is None:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def compute_stability(runs, winners, output):
    predictions = {r['run_id']: np.load(output / r['prediction_file'], allow_pickle=False) for r in runs}
    records = []
    for a, b in itertools.combinations(runs, 2):
        if a['setting_id'] == b['setting_id']:
            kind = 'between_seeds_same_graph'
        elif (a['run_id'] in winners.values() and b['run_id'] in winners.values()
              and a['method'] == b['method'] and a['resolution'] == b['resolution'] and a['k'] != b['k']):
            kind = 'between_knn_same_method_and_resolution'
        else:
            continue
        metrics = partition_metrics(predictions[a['run_id']], predictions[b['run_id']])
        records.append({'comparison': kind, 'run_a': a['run_id'], 'run_b': b['run_id'],
                        'k_total_a': a['k_total'], 'k_total_b': b['k_total'],
                        'ari': metrics['ari'], 'nmi': metrics['nmi']})
    return records


def report(output, runs, graphs, winners, stability, evaluated):
    selected = [r for r in runs if r['run_id'] in winners.values()]
    lines = ['# Cache v2 SNN graph discovery', '',
             'Full-population person0. Predictions and winners were fixed before label evaluation.',
             'Infomap: flat two-level implementation in igraph. Leiden: generalized modularity.',
             'Winner selection only compares seeds on the SAME graph/method/resolution.',
             'There is no ground-truth-driven choice of k, resolution, or final class count.',
             'Isolates remain singleton clusters. Small clusters are reported, never removed.', '',
             '| kNN | Edges | Components | Isolates | Largest component |',
             '|---:|---:|---:|---:|---:|']
    for k, g in graphs.items():
        lines.append('| {} | {} | {} | {} | {:.1%} |'.format(
            k, g['edges'], g['components'], g['isolated_nodes'], g['largest_component_fraction']))
    lines += ['', '## Best seed within each setting', '',
              '| Setting | Seed | K including isolates | K on nonisolated nodes | ARI | NMI | Purity |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        e = r.get('evaluation', {})
        fields = [format(e[q], '.4f') if q in e else 'n.a.' for q in ('ari', 'nmi', 'weighted_purity')]
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            r['setting_id'], r['seed'], r['k_total'], r['k_nonisolated'], *fields))
    lines += ['', '## Stability across seeds', '',
              '| Setting | K range | Mean pairwise ARI | Minimum pairwise ARI |',
              '|---|---:|---:|---:|']
    for key in winners:
        current = [r for r in runs if r['setting_id'] == key]
        ids = {r['run_id'] for r in current}
        values = [s['ari'] for s in stability if s['run_a'] in ids and s['run_b'] in ids]
        lines.append('| {} | {}-{} | {} | {} |'.format(key, min(r['k_total'] for r in current),
            max(r['k_total'] for r in current), format(float(np.mean(values)), '.4f') if values else 'n.a.',
            format(min(values), '.4f') if values else 'n.a.'))
    lines += ['', 'Cross-neighborhood ARI/NMI are in stability.csv.',
              'High stability/purity or a count near the true K alone does not prove semantic recovery.',
              'Purity can increase through over-splitting; read ARI, homogeneity and completeness together.',
              'If component or isolate counts are large, inspect graph fragmentation before interpreting K.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    table = []
    for r in runs:
        row = {key: r[key] for key in ('run_id', 'method', 'k', 'resolution', 'seed', 'k_total',
                                      'k_nonisolated', 'singleton_clusters', 'clusters_below_cutoff',
                                      'samples_below_cutoff', 'largest_cluster_fraction', 'objective_value')}
        row['selected_within_setting'] = r['run_id'] in winners.values()
        if evaluated:
            row.update(r['evaluation'])
        table.append(row)
    write_csv(output / 'runs.csv', table)
    write_csv(output / 'stability.csv', stability, ['comparison', 'run_a', 'run_b', 'k_total_a', 'k_total_b', 'ari', 'nmi'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='vlm_pilot/ntu60_xsub_clip_cache_v2')
    parser.add_argument('--neighbor-dir', default='output_dir/cache_v2_discovery')
    parser.add_argument('--output-dir', default='output_dir/cache_v2_graph')
    parser.add_argument('--data-path', help='Optional original NPZ: read labels only after freezing all partitions')
    parser.add_argument('--knn', type=int, nargs='+', default=[10, 20, 30, 50])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--methods', nargs='+', choices=['infomap', 'leiden'], default=['infomap', 'leiden'])
    parser.add_argument('--resolutions', type=float, nargs='+', default=[.5, 1., 2.])
    parser.add_argument('--min-jaccard', type=float, default=0.)
    parser.add_argument('--edge-block-size', type=int, default=4096)
    parser.add_argument('--infomap-trials', type=int, default=1)
    parser.add_argument('--small-cluster-size', type=int, default=10, help='Reporting only; never used to remove/merge clusters')
    parser.add_argument('--resume', action='store_true', help='Reuse graph and prediction artifacts with identical provenance')
    args = parser.parse_args()
    if (min(args.knn) < 2 or args.edge_block_size < 1 or args.infomap_trials < 1
            or args.small_cluster_size < 2 or not 0 <= args.min_jaccard < 1
            or any(not np.isfinite(r) or r <= 0 for r in args.resolutions)
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('Invalid neighborhood, block, trials, reporting cutoff, resolution, pruning, or seed')
    for name in ('knn', 'seeds', 'resolutions', 'methods'):
        setattr(args, name, sorted(set(getattr(args, name))))
    try:
        import igraph as ig
    except ImportError as exc:
        raise SystemExit('Missing igraph. Install once: python -m pip install igraph==0.11.8') from exc
    neighbors, manifest = load_neighbors(args.cache, args.neighbor_dir, max(args.knn))
    output = Path(args.output_dir)
    config_args = {k: v for k, v in vars(args).items() if k not in ('resume', 'data_path', 'output_dir')}
    config = {'protocol': PROTOCOL, 'args': config_args, 'sample_count': len(neighbors),
              'cache_identity': manifest['identity'], 'script_identity': identity(__file__),
              'helper_identity': identity(Path(__file__).with_name('test_cache_v2_discovery.py')),
              'neighbor_file_identity': identity(Path(args.neighbor_dir) / 'neighbor_indices.npy'),
              'numpy_version': np.__version__, 'igraph_version': ig.__version__,
              'graph': 'undirected union kNN edges; weight=Jaccard of open neighbor sets; weight > min_jaccard',
              'isolate_policy': 'each isolated node is a singleton, included in K_total and evaluation',
              'objective_population': 'nonisolated induced graph; isolates have no flow and are not fitted',
              'seed_selection': 'minimum codelength / maximum modularity within identical graph and resolution only'}
    config_path = output / 'run_config.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not config_path.exists():
            raise FileExistsError('Use a new output directory, or --resume for the identical experiment')
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Resume configuration/provenance changed; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    runs, graphs = [], {}
    print('Reusing full-population neighbors: N={}, ks={}; CPU only, no CLIP/GPU'.format(len(neighbors), args.knn), flush=True)
    for k in args.knn:
        graph_file = output / 'graph_k{}.npz'.format(k)
        graph_meta = output / 'graph_k{}.json'.format(k)
        if args.resume and graph_file.exists() and graph_meta.exists():
            metadata = json.loads(graph_meta.read_text(encoding='utf-8'))
            if identity(graph_file) != metadata['file_identity']:
                raise ValueError('Saved graph checksum mismatch')
            with np.load(graph_file, allow_pickle=False) as saved:
                edges, weights = saved['edges'], saved['weights']
            candidates = metadata['stats']['candidate_edges']
            print('Reusing graph k={}'.format(k), flush=True)
        else:
            edges, weights, candidates = snn_edges(neighbors, k, args.min_jaccard, args.edge_block_size)
            np.savez(graph_file, edges=edges, weights=weights)
        graph = make_graph(len(neighbors), edges, weights)
        stats = graph_stats(graph, candidates)
        graphs[str(k)] = stats
        write_json(graph_meta, {'stats': stats, 'file_identity': identity(graph_file)})
        print('Graph k={}: edges={}, components={}, isolates={}'.format(k, stats['edges'], stats['components'], stats['isolated_nodes']), flush=True)
        for method in args.methods:
            resolutions = [None] if method == 'infomap' else args.resolutions
            for resolution in resolutions:
                setting = setting_id(k, method, resolution)
                for seed in args.seeds:
                    run_id = '{}_seed{}'.format(setting, seed)
                    pred_file, meta_file = output / (run_id + '.npy'), output / (run_id + '.json')
                    if args.resume and pred_file.exists() and meta_file.exists():
                        run = json.loads(meta_file.read_text(encoding='utf-8'))
                        if identity(pred_file) != run['prediction_identity']:
                            raise ValueError('Prediction checksum mismatch: ' + run_id)
                        print('Reusing ' + run_id, flush=True)
                    else:
                        print('Fitting ' + run_id, flush=True)
                        start = time.monotonic()
                        prediction, run_stats = fit_partition(graph, method, seed, resolution,
                                                              args.infomap_trials, args.small_cluster_size)
                        np.save(pred_file, prediction, allow_pickle=False)
                        run = dict(run_stats, run_id=run_id, setting_id=setting, method=method, k=k,
                                   resolution=resolution, seed=seed, prediction_file=pred_file.name,
                                   prediction_identity=identity(pred_file), seconds=time.monotonic() - start)
                        write_json(meta_file, run)
                        print('  K_total={}, K_nonisolated={}, score={}'.format(
                            run['k_total'], run['k_nonisolated'], run['objective_value']), flush=True)
                    runs.append(run)
        del graph, edges, weights
    winners = choose_winners(runs)
    stability = compute_stability(runs, winners, output)
    write_json(output / 'label_blind_results.json', {'graphs': graphs, 'runs': runs,
                                                   'winners': winners, 'stability': stability})
    # All model selection above is independent of labels and true class count.
    labels = read_labels(args.data_path, manifest) if args.data_path else None
    if labels is not None:
        class_rows, cluster_rows = [], []
        for run in runs:
            prediction = np.load(output / run['prediction_file'], allow_pickle=False)
            metrics, matrix, class_ids, per_class, per_cluster = evaluate(labels, prediction)
            run['evaluation'] = metrics
            if run['run_id'] in winners.values():
                np.savez(output / (run['run_id'] + '_contingency.npz'), counts=matrix, class_ids=class_ids)
                class_rows.extend([dict(run_id=run['run_id'], **r) for r in per_class])
                cluster_rows.extend([dict(run_id=run['run_id'], **r) for r in per_cluster])
        write_csv(output / 'per_class.csv', class_rows)
        write_csv(output / 'per_cluster.csv', cluster_rows)
    result = {'protocol': PROTOCOL, 'sample_count': len(neighbors), 'graphs': graphs,
              'runs': runs, 'winners_selected_without_labels': winners, 'stability': stability,
              'evaluation_dataset_identity': manifest['identity']['data'] if labels is not None else None,
              'true_k_evaluation_only': int(len(np.unique(labels))) if labels is not None else None}
    write_json(output / 'summary.json', result)
    report(output, runs, graphs, winners, stability, labels is not None)
    for run in runs:
        if run['run_id'] in winners.values():
            print('{}: K={}, eval={}'.format(run['run_id'], run['k_total'], run.get('evaluation', 'not requested')), flush=True)
    print('Done: {}'.format(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
