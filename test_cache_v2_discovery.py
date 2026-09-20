"""Full-cache, label-blind representative selection; labels are loaded only for audit.

Requires NumPy; CUDA acceleration uses PyTorch. Matplotlib is optional.
Does not import or modify cache extraction/training code, or read token_features.npy.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def identity(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def load_features(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'macdiff_clip_token_cache_v2' or manifest.get('complete') is not True:
        raise ValueError('Expected a complete cache v2')
    for name in ('person_features.npy', 'person_valid.npy'):
        if identity(root / name) != manifest['files'][name]:
            raise ValueError('Cache checksum mismatch: ' + name)
    values = np.load(root / 'person_features.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(root / 'person_valid.npy', allow_pickle=False)
    n, dim = manifest['sample_count'], manifest['feature_dim']
    if values.shape != (n, 2, dim) or values.dtype != np.float32:
        raise ValueError('Unexpected person_features shape/dtype')
    if valid.shape != (n, 2) or valid.dtype != np.bool_ or not valid[:, 0].all():
        raise ValueError('Every raw sample must have valid person0; refusing silent filtering')
    x = np.array(values[:, 0, :], dtype=np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not np.allclose(norms, 1., atol=1e-4):
        raise ValueError('Expected finite, unit-normalized person0 features')
    return x / norms, manifest


def exact_neighbors(x, ks, block_size):
    """Exact full-population neighbors, O(N^2 d) work, O(block_size*N) scores."""
    n = len(x)
    largest = max(ks)
    indices = np.empty((n, largest), dtype=np.int32)
    mean_distances = {k: np.empty(n, dtype=np.float32) for k in ks}
    if isinstance(x, np.ndarray):
        for start in range(0, n, block_size):
            end = min(n, start + block_size)
            scores = x[start:end] @ x.T
            scores[np.arange(end - start), np.arange(start, end)] = -np.inf
            neighbors = np.argpartition(-scores, largest - 1, axis=1)[:, :largest]
            similarities = np.take_along_axis(scores, neighbors, axis=1)
            order = np.argsort(-similarities, axis=1, kind='stable')
            indices[start:end] = np.take_along_axis(neighbors, order, axis=1)
            distances = np.sqrt(np.maximum(2 - 2 * np.take_along_axis(similarities, order, axis=1), 0))
            for k in ks:
                mean_distances[k][start:end] = distances[:, :k].mean(axis=1)
            if start == 0 or end == n or (start // block_size) % 20 == 0:
                print('Exact kNN: {}/{} samples'.format(end, n), flush=True)
        return indices, mean_distances
    import torch
    with torch.no_grad():
        for start in range(0, n, block_size):
            end = min(n, start + block_size)
            scores = torch.mm(x[start:end], x.t())
            scores[torch.arange(end - start, device=x.device),
                   torch.arange(start, end, device=x.device)] = -float('inf')
            similarities, neighbors = torch.topk(scores, largest, dim=1, sorted=True)
            indices[start:end] = neighbors.cpu().numpy()
            distances = torch.sqrt(torch.clamp(2 - 2 * similarities, min=0)).cpu().numpy()
            for k in ks:
                mean_distances[k][start:end] = distances[:, :k].mean(axis=1)
            if start == 0 or end == n or (start // block_size) % 20 == 0:
                print('Exact kNN: {}/{} samples'.format(end, n), flush=True)
    return indices, mean_distances


def density_weights(distances, floor):
    """Tie-aware empirical density ranks; smaller mean neighbor distance is denser.

    This is a heuristic weighting, not Density Peaks or a proven K estimator.
    All points remain eligible, including sparse groups.
    """
    ordered = np.sort(distances)
    left = np.searchsorted(ordered, distances, side='left')
    right = np.searchsorted(ordered, distances, side='right')
    rank = (left + right - 1) / 2.
    return (floor + (1 - floor) * (1 - rank / max(1, len(distances) - 1))).astype(np.float32)


def select_sequence(x, budget, seed, weights=None, random=False):
    n = len(x)
    permutation = np.random.RandomState(seed).permutation(n)
    selected = []
    curves = []
    if isinstance(x, np.ndarray):
        available = np.ones(n, dtype=bool)
        nearest_squared = np.full(n, np.inf, dtype=np.float32)
        next_index = int(permutation[0])
        for step in range(budget):
            selected.append(next_index)
            available[next_index] = False
            nearest_squared = np.minimum(nearest_squared, np.maximum(2 - 2 * (x @ x[next_index]), 0))
            nearest_squared[~available] = 0
            nearest = np.sqrt(nearest_squared)
            curves.append([float(nearest.max()), float(np.quantile(nearest, .99)),
                           float(np.quantile(nearest, .95)), float(nearest.mean())])
            score = nearest.copy() if weights is None else nearest * weights
            score[~available] = -1
            if step + 1 < budget:
                next_index = int(permutation[step + 1]) if random else int(np.argmax(score))
            if (step + 1) % 100 == 0:
                print('  selected {}/{}'.format(step + 1, budget), flush=True)
        return np.asarray(selected, dtype=np.int64), np.asarray(curves)
    import torch
    available = torch.ones(n, dtype=torch.bool, device=x.device)
    nearest_squared = torch.full((n,), float('inf'), device=x.device)
    weight = None if weights is None else torch.as_tensor(weights, device=x.device)
    next_index = int(permutation[0])  # paired initial point across methods
    with torch.no_grad():
        for step in range(budget):
            selected.append(next_index)
            available[next_index] = False
            distances = torch.clamp(2 - 2 * torch.mv(x, x[next_index]), min=0)
            nearest_squared = torch.minimum(nearest_squared, distances)
            nearest_squared[~available] = 0  # exactly zero at selected points
            nearest = torch.sqrt(nearest_squared)
            radii = nearest.cpu().numpy()
            curves.append([float(radii.max()), float(np.quantile(radii, .99)),
                           float(np.quantile(radii, .95)), float(radii.mean())])
            if step + 1 < budget:
                if random:
                    next_index = int(permutation[step + 1])
                else:
                    score = nearest if weight is None else nearest * weight
                    score = score.masked_fill(~available, -1)
                    next_index = int(torch.argmax(score).item())
            if (step + 1) % 100 == 0:
                print('  selected {}/{}'.format(step + 1, budget), flush=True)
    return np.asarray(selected, dtype=np.int64), np.asarray(curves)


def elbow_candidate(radius, min_strength=.05):
    """Maximum deviation below the endpoint chord; returns a candidate, not truth.

    Depends on the observation budget. A null result means no accepted knee.
    No class count or labels enter this function.
    """
    radius = np.asarray(radius, dtype=float)
    if len(radius) < 4 or radius[0] - radius[-1] <= 1e-8:
        return {'candidate_k': None, 'strength': 0., 'reason': 'flat_or_short_curve'}
    t = np.linspace(0, 1, len(radius))
    normalized = (radius - radius[-1]) / (radius[0] - radius[-1])
    deviation = 1 - t - normalized
    index = int(np.argmax(deviation))
    strength = float(deviation[index])
    accepted = 0 < index < len(radius) - 1 and strength >= min_strength
    return {'candidate_k': index + 1 if accepted else None,
            'strength': strength, 'reason': 'heuristic_knee' if accepted else 'weak_or_boundary_knee'}


def read_labels(data_path, manifest):
    # Deliberately invoked only after every sequence and blind knee is saved.
    print('Selection finished. Checking dataset identity before loading y_train...', flush=True)
    if identity(data_path) != manifest['identity']['data']:
        raise ValueError('Dataset SHA256 differs from cache provenance; refusing misaligned labels')
    with np.load(data_path, allow_pickle=False) as archive:
        y = archive['y_train']
    n = manifest['sample_count']
    if y.ndim == 2 and y.shape[0] == n and y.shape[1] > 1:
        if (not np.isfinite(y).all() or not np.all((y == 0) | (y == 1))
                or not np.all(y.sum(axis=1) == 1)):
            raise ValueError('Expected one-hot y_train')
        return y.argmax(axis=1)
    if y.shape == (n,) and y.dtype.kind in 'iu' and np.all(y >= 0):
        return y.astype(np.int64)
    raise ValueError('Unsupported y_train shape/dtype')


def evaluate_sequence(selected, labels, knees):
    classes = np.unique(labels)
    seen = set()
    coverage = []
    first = {}
    for step, index in enumerate(selected, 1):
        label = int(labels[index])
        seen.add(label)
        first.setdefault(str(label), step)
        coverage.append(len(seen))
    k = len(classes)
    budget = len(selected)
    result = {'coverage_curve': coverage,
              'coverage_at_true_k': coverage[k - 1] if k <= budget else None,
              'coverage_at_budget': coverage[-1],
              'first_hit_by_class': first,
              'missing_classes': [int(c) for c in classes if int(c) not in seen],
              'steps_to_90_percent': next((i + 1 for i, c in enumerate(coverage)
                                           if c >= math.ceil(.9 * k)), None),
              'steps_to_all_classes': next((i + 1 for i, c in enumerate(coverage) if c == k), None),
              'knee_audit': {}}
    for name, knee in knees.items():
        candidate = knee['candidate_k']
        result['knee_audit'][name] = {
            'candidate_k': candidate,
            'absolute_k_error': abs(candidate - k) if candidate is not None else None,
            'covered_classes': coverage[candidate - 1] if candidate is not None else None}
    return result


def write_outputs(output, runs, labels, neighbor_indices, ks):
    evaluated = labels is not None
    summary = {'protocol': 'cache_v2_person0_discovery_v1',
               'estimated_k_is_heuristic': True, 'runs': runs}
    if evaluated:
        classes, counts = np.unique(labels, return_counts=True)
        summary['true_k_evaluation_only'] = len(classes)
        summary['class_counts'] = {str(int(c)): int(n) for c, n in zip(classes, counts)}
        matches = labels[neighbor_indices] == labels[:, None]
        summary['full_population_knn_same_class_fraction'] = {
            str(k): float(matches[:, :k].mean()) for k in ks}
        summary['full_population_p1'] = float(matches[:, 0].mean())
        summary['knn_ties'] = 'topk tie order is backend dependent; self excluded; no tie averaging'
    with (output / 'curves.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'seed', 'm', 'radius_max', 'radius_q99',
                         'radius_q95', 'mean_distance', 'covered_classes'])
        for run in runs:
            arrays = np.load(output / run['sequence_file'], allow_pickle=False)
            coverage = run.get('evaluation', {}).get('coverage_curve', [])
            for i, row in enumerate(arrays['curves']):
                writer.writerow([run['method'], run['seed'], i + 1] + row.tolist()
                                + [coverage[i] if coverage else ''])
    write_json(output / 'summary.json', summary)
    lines = ['# Cache v2 person0 discovery', '',
             'Full population; frozen normalized sentence features; no token features or remap.',
             'All sequences and knee candidates were saved before labels were loaded.',
             'Density-FPS is a heuristic, not the published Density Peaks algorithm.',
             'Knee candidates depend on budget/geometry and do not guarantee semantic class counts.', '',
             '| Method | Seed | q99 knee candidate | Classes covered at true K | Final coverage | Steps to all |',
             '|---|---:|---:|---:|---:|---:|']
    for run in runs:
        audit = run.get('evaluation', {})
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            run['method'], run['seed'], run['knees']['q99']['candidate_k'],
            audit.get('coverage_at_true_k'), audit.get('coverage_at_budget'),
            audit.get('steps_to_all_classes')))
    lines += ['', 'Null means no accepted knee, unavailable evaluation, or target not reached.',
              'Inspect all seeds and the half-budget knees in summary.json; do not pick the run closest to ground truth.',
              'This test measures representative coverage and knee behavior, not final clustering accuracy.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable: CSV, JSON and report saved; skipping PNG.', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = sorted(set(run['method'] for run in runs))
    for method in methods:
        current = [run for run in runs if run['method'] == method]
        radii = np.asarray([np.load(output / r['sequence_file'])['curves'][:, 1] for r in current])
        m = np.arange(1, radii.shape[1] + 1)
        line, = axes[1].plot(m, radii.mean(axis=0), label=method)
        axes[1].fill_between(m, radii.min(axis=0), radii.max(axis=0), color=line.get_color(), alpha=.15)
        if evaluated:
            coverage = np.asarray([r['evaluation']['coverage_curve'] for r in current])
            line, = axes[0].plot(m, coverage.mean(axis=0), label=method)
            axes[0].fill_between(m, coverage.min(axis=0), coverage.max(axis=0), color=line.get_color(), alpha=.15)
    if evaluated:
        axes[0].axhline(len(np.unique(labels)), color='black', linestyle='--', label='True K (audit only)')
        axes[0].legend()
    else:
        axes[0].text(.5, .5, 'No labels supplied', ha='center', transform=axes[0].transAxes)
    axes[0].set(xlabel='Selected representatives', ylabel='Distinct ground-truth classes', title='Coverage: mean and seed range')
    axes[1].set(xlabel='Selected representatives', ylabel='99th percentile nearest-representative distance', title='Label-blind coverage radius')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='vlm_pilot/ntu60_xsub_clip_cache_v2')
    parser.add_argument('--data-path', default=None, help='Optional original NPZ, used ONLY for post-selection evaluation')
    parser.add_argument('--output-dir', default='output_dir/cache_v2_discovery')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-centers', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--knn', type=int, nargs='+', default=[50])
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--density-floor', type=float, default=.1)
    parser.add_argument('--min-knee-strength', type=float, default=.05)
    args = parser.parse_args()
    if (args.max_centers < 4 or args.block_size < 1 or min(args.knn) < 1
            or not 0 < args.density_floor <= 1 or not 0 <= args.min_knee_strength <= 1
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('Invalid budget, block size, kNN, weight floor, knee strength, or seed')
    args.seeds, args.knn = sorted(set(args.seeds)), sorted(set(args.knn))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory must be empty; choose a new --output-dir')
    x_numpy, manifest = load_features(args.cache)
    n = len(x_numpy)
    if args.max_centers > n or max(args.knn) >= n:
        parser.error('Need max-centers <= sample_count and kNN < sample_count')
    torch_version = None
    if args.device == 'cpu':
        x = x_numpy
    else:
        import torch
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use the server environment or explicitly --device cpu')
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch_version = torch.__version__
        x = torch.as_tensor(x_numpy, device=device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run_config.json', {
        'args': vars(args), 'sample_count': n, 'feature_dim': x_numpy.shape[1],
        'representation': 'person_features[:,0,:]; renormalized FP32; raw row = sample_index',
        'script': identity(__file__), 'numpy_version': np.__version__, 'torch_version': torch_version,
        'cache_identity': manifest['identity'],
        'density_score': 'nearest Euclidean distance * (floor + (1-floor)*tie-aware density percentile)',
        'density_proxy': 'negative mean distance to exact k nearest other samples',
        'knn_memory': 'score block is block_size * sample_count float32; no full NxN stored',
        'knee_rule': 'maximum normalized deviation below endpoint chord; threshold is heuristic'})
    print('Loaded ALL {} samples, {} dimensions, person0; device={}'.format(n, x_numpy.shape[1], args.device), flush=True)
    neighbors, local_distances = exact_neighbors(x, args.knn, args.block_size)
    np.save(output / 'neighbor_indices.npy', neighbors, allow_pickle=False)
    np.savez(output / 'local_density.npz', **{'mean_distance_k{}'.format(k): v for k, v in local_distances.items()})
    methods = [('random', None), ('fps', None)] + [
        ('density_fps_k{}'.format(k), density_weights(local_distances[k], args.density_floor)) for k in args.knn]
    runs = []
    for method, weights in methods:
        for seed in args.seeds:
            print('Running {}, seed={}'.format(method, seed), flush=True)
            selected, curves = select_sequence(x, args.max_centers, seed, weights, method == 'random')
            knees = {name: elbow_candidate(curves[:, j], args.min_knee_strength)
                     for j, name in enumerate(('max', 'q99', 'q95'))}
            half = args.max_centers // 2
            half_knees = {name: elbow_candidate(curves[:half, j], args.min_knee_strength)
                          for j, name in enumerate(('max', 'q99', 'q95'))}
            filename = '{}_seed{}.npz'.format(method, seed)
            np.savez(output / filename, selected_indices=selected, curves=curves)
            runs.append({'method': method, 'seed': seed, 'sequence_file': filename,
                         'knees': knees, 'half_budget': half, 'half_budget_knees': half_knees})
            write_json(output / 'label_blind_results.json', runs)
    labels = read_labels(args.data_path, manifest) if args.data_path else None
    if labels is not None:
        for run in runs:
            selected = np.load(output / run['sequence_file'])['selected_indices']
            run['evaluation'] = evaluate_sequence(selected, labels, run['knees'])
    write_outputs(output, runs, labels, neighbors, args.knn)
    if labels is not None:
        for run in runs:
            e = run['evaluation']
            print('{} seed={}: covered@trueK={}, final={}, steps_to_all={}, q99_knee={}'.format(
                run['method'], run['seed'], e['coverage_at_true_k'], e['coverage_at_budget'],
                e['steps_to_all_classes'], run['knees']['q99']['candidate_k']), flush=True)
    print('Done: {}'.format(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
