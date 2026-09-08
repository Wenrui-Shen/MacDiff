"""Frozen Stage1 readouts; identical portable NumPy core in both repositories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = 'frozen_stage1_readouts_v1'


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def select_supports(labels, seeds, caches=()):
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need >=2 classes and distinct exemplar seeds')
    if caches and len(caches) != len(seeds):
        raise ValueError('Supply one exemplar cache per seed, in seed order')
    supports = {}
    for position, seed in enumerate(seeds):
        if caches:
            path = Path(caches[position])
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    indices = np.asarray(data['indices'], dtype=np.int64)
                    cached_classes = np.asarray(data['class_ids'])
                    saved_seed = int(data['seed'])
                    saved_count = int(data['num_samples'])
            else:
                data = json.loads(path.read_text(encoding='utf-8'))
                indices = np.asarray(data['indices'], dtype=np.int64)
                cached_classes = np.asarray(data['class_ids'])
                saved_seed = int(data['seed'])
                saved_count = int(data['num_samples'])
            if (saved_seed != seed or saved_count != len(labels)
                    or not np.array_equal(cached_classes, classes)):
                raise ValueError('Exemplar cache seed/classes/dataset size mismatch')
        else:
            rng = np.random.RandomState(seed)
            indices = np.array([rng.choice(np.flatnonzero(labels == c))
                                for c in classes], dtype=np.int64)
        if (indices.shape != classes.shape or np.any(indices < 0)
                or np.any(indices >= len(labels))
                or not np.array_equal(labels[indices], classes)):
            raise ValueError('Need exactly one valid exemplar per class')
        supports[str(seed)] = indices.tolist()
    return classes, supports


def subset_indices(size, maximum, seed, required=()):
    required = np.unique(np.asarray(required, dtype=np.int64))
    if maximum < 0 or np.any(required < 0) or np.any(required >= size):
        raise ValueError('Invalid sample limit or required indices')
    if maximum == 0 or maximum >= size:
        return np.arange(size, dtype=np.int64)
    if maximum <= len(required):
        raise ValueError('Sample limit must leave non-exemplar training samples')
    candidates = np.setdiff1d(np.arange(size), required)
    extra = np.random.RandomState(seed).choice(
        candidates, maximum - len(required), replace=False)
    return np.sort(np.concatenate((required, extra)))


def fit_transform(features, indices, pca_rank=256, pca_max_samples=4096,
                  shrinkage=0.1, seed=17):
    """Only unlabeled TRAIN rows enter. Query labels/features are not arguments."""
    if len(indices) < 2 or pca_rank < 0 or pca_max_samples < 2:
        raise ValueError('Need >=2 fit samples and valid PCA settings')
    if not 0 < shrinkage <= 1:
        raise ValueError('Whitening shrinkage must be in (0,1]')
    dim = features.shape[1]
    total = np.zeros(dim, dtype=np.float64)
    squares = np.zeros(dim, dtype=np.float64)
    for start in range(0, len(indices), 1024):
        block = np.asarray(features[indices[start:start + 1024]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Non-finite training features')
        total += block.sum(axis=0)
        squares += (block * block).sum(axis=0)
    mean = total / len(indices)
    variance = np.maximum(squares / len(indices) - mean * mean, 0)
    floor = max(float(variance.mean()) * 1e-6, 1e-12)
    state = {'mean': mean.astype(np.float32),
             'std': np.sqrt(np.maximum(variance, floor)).astype(np.float32),
             'variance_floor': floor, 'fit_count': len(indices),
             'shrinkage': shrinkage}
    if pca_rank:
        rng = np.random.RandomState(seed)
        chosen = np.sort(rng.choice(indices, min(len(indices), pca_max_samples),
                                    replace=False))
        x = (np.asarray(features[chosen], dtype=np.float32) - state['mean']) / state['std']
        state['pca_mean'] = x.mean(axis=0)
        x -= state['pca_mean']
        rank = min(pca_rank, len(x) - 1, dim)
        width = min(rank + 16, len(x) - 1, dim)
        omega = rng.normal(size=(dim, width)).astype(np.float32)
        q, _ = np.linalg.qr(x @ omega, mode='reduced')
        # Stabilized randomized PCA: avoids a 6400 x 6400 covariance eigensolve.
        z, _ = np.linalg.qr(x.T @ q, mode='reduced')
        q, _ = np.linalg.qr(x @ z, mode='reduced')
        _, singular, vt = np.linalg.svd(q.T @ x, full_matrices=False)
        state['components'] = vt[:rank].T.astype(np.float32)
        eigenvalues = singular[:rank] ** 2 / max(len(x) - 1, 1)
        state['eigenvalues'] = eigenvalues.astype(np.float32)
        state['whiten_scale'] = np.sqrt(np.maximum(
            (1 - shrinkage) * eigenvalues + shrinkage * eigenvalues.mean(),
            1e-12)).astype(np.float32)
        state['pca_fit_indices'] = chosen
    return state


def transform(features, state, mode):
    x = np.asarray(features, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite input features')
    if mode == 'raw':
        return normalize(x)
    x = x - state['mean']
    if mode == 'centered':
        return normalize(x)
    x = x / state['std']
    if mode in ('pca', 'whitened'):
        x = (x - state['pca_mean']) @ state['components']
        if mode == 'whitened':
            x = x / state['whiten_scale']
    elif mode == 'linear_standardized':
        return x
    elif mode != 'standardized':
        raise ValueError('Unknown transform: ' + mode)
    return normalize(x)


def fit_ridge(support_features, alpha):
    """One exemplar/class, ordered by class. Dual ridge with free intercept.

    min ||EW + b - I||_F^2 + lambda ||W||_F^2,
    lambda = alpha * trace(E_centered E_centered.T) / C.
    """
    e = np.asarray(support_features, dtype=np.float64)
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Ridge alpha must be finite and positive')
    mean = e.mean(axis=0)
    ec = e - mean
    classes = len(e)
    target = np.eye(classes) - 1.0 / classes
    gram = ec @ ec.T
    penalty = float(alpha) * max(float(np.trace(gram) / classes), 1e-12)
    weights = ec.T @ np.linalg.solve(gram + penalty * np.eye(classes), target)
    bias = np.full(classes, 1.0 / classes) - mean @ weights
    return weights.astype(np.float32), bias.astype(np.float32), penalty


def probabilities(logits, temperature):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Temperature must be finite and positive')
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def classification_metrics(scores, labels, class_ids, temperature):
    """Labels enter only after scoring; all results cover the entire split."""
    labels = np.asarray(labels)
    positions = np.searchsorted(class_ids, labels)
    if (np.any(positions >= len(class_ids))
            or not np.array_equal(class_ids[positions], labels)):
        raise ValueError('Query label outside training class set')
    if len(scores) != len(labels) or not len(labels) or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction matrix')
    p = probabilities(scores, temperature)
    order = np.argsort(-scores, axis=1, kind='stable')
    pred = order[:, 0]
    correct = pred == positions
    confidence = p.max(axis=1)
    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    np.add.at(confusion, (positions, pred), 1)
    counts = confusion.sum(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    true_p = p[np.arange(len(p)), positions]
    other = p.copy()
    other[np.arange(len(p)), positions] = -np.inf
    return {
        'n': len(labels), 'top1': float(correct.mean() * 100),
        'top5': float((order[:, :min(5, len(class_ids))] == positions[:, None]).any(axis=1).mean() * 100),
        'macro_top1': float((np.diag(confusion)[counts > 0] / counts[counts > 0]).mean() * 100),
        'confidence': float(confidence.mean()),
        'entropy': float(-(p * np.log(np.maximum(p, 1e-300))).sum(axis=1).mean()),
        'nll': float(-np.log(np.maximum(true_p, 1e-300)).mean()),
        'ece': ece, 'true_class_margin': float((true_p - other.max(axis=1)).mean()),
        'active_classes': int(np.count_nonzero(confusion.sum(axis=0))),
        'predicted_counts': confusion.sum(axis=0).tolist(),
        'confusion': confusion.tolist(),
    }


def build_parser(project, checkpoint, batch_size):
    parser = argparse.ArgumentParser(description='Frozen Stage1 readouts: ' + project)
    sub = parser.add_subparsers(dest='command', required=True)
    extract = sub.add_parser('extract', help='GPU: cache raw LP-input features once')
    extract.add_argument('--checkpoint', default=checkpoint)
    extract.add_argument('--cache-dir', required=True)
    extract.add_argument('--device', default='cuda:0')
    extract.add_argument('--batch-size', type=int, default=batch_size)
    extract.add_argument('--workers', type=int, default=4)
    extract.add_argument('--exemplar-seeds', type=int, nargs='+', default=[0, 1, 2])
    extract.add_argument('--exemplar-caches', nargs='*', default=[])
    extract.add_argument('--seed', type=int, default=17)
    extract.add_argument('--max-train', type=int, default=0)
    extract.add_argument('--max-eval', type=int, default=0)
    compare = sub.add_parser('compare', help='CPU/NumPy: compare a complete cache')
    compare.add_argument('--cache-dir', required=True)
    compare.add_argument('--output-dir', required=True)
    compare.add_argument('--pca-rank', type=int, default=256)
    compare.add_argument('--pca-max-samples', type=int, default=4096)
    compare.add_argument('--whiten-shrinkage', type=float, default=0.1)
    compare.add_argument('--ridge-alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0])
    compare.add_argument('--cosine-temperature', type=float, default=0.1)
    compare.add_argument('--ridge-temperature', type=float, default=1.0)
    compare.add_argument('--seed', type=int, default=17)
    return parser, extract


def extract_cache(args, project, model, datasets, forward, provenance):
    """Sequential frozen forward; new directory only; completion manifest last."""
    import torch
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError('Invalid batch size/workers')
    train, evaluation = datasets
    train_labels = np.asarray(train.label, dtype=np.int64)
    eval_labels = np.asarray(evaluation.label, dtype=np.int64)
    classes, supports = select_supports(train_labels, args.exemplar_seeds, args.exemplar_caches)
    union = np.unique(np.concatenate([np.asarray(x) for x in supports.values()]))
    indices = {
        'train': subset_indices(len(train), args.max_train, args.seed, union),
        'eval': subset_indices(len(evaluation), args.max_eval, args.seed + 1),
    }
    if len(np.setdiff1d(indices['train'], union)) < 2:
        raise ValueError('Need >=2 non-exemplar training samples')
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False).to(args.device)
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    branch_dims = None
    for split, dataset, labels in (('train', train, train_labels), ('eval', evaluation, eval_labels)):
        selected = indices[split]
        np.save(root / (split + '_indices.npy'), selected)
        np.save(root / (split + '_labels.npy'), labels[selected])
        names = getattr(dataset, 'sample_name', [str(i) for i in range(len(dataset))])
        np.save(root / (split + '_names.npy'), np.asarray(names, dtype=str)[selected])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, selected.tolist()),
            batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.workers, pin_memory=str(args.device).startswith('cuda'))
        arrays = {}
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                values = forward(model, batch[0].to(args.device).float())
                dimensions = {name: int(value.shape[1]) for name, value in values.items()}
                if branch_dims is None:
                    branch_dims = dimensions
                if dimensions != branch_dims:
                    raise ValueError('Feature dimensions changed between batches/splits')
                for name, value in values.items():
                    value = value.detach().float().cpu().numpy()
                    if value.ndim != 2 or not np.isfinite(value).all():
                        raise ValueError('Invalid frozen features: ' + name)
                    if name not in arrays:
                        arrays[name] = np.lib.format.open_memmap(
                            root / (split + '_' + name + '.npy'), mode='w+', dtype=np.float32,
                            shape=(len(selected), value.shape[1]))
                    arrays[name][cursor:cursor + len(value)] = value
                cursor += len(batch[0])
                if cursor == len(selected) or cursor % (args.batch_size * 100) == 0:
                    print('{}: {}/{}'.format(split, cursor, len(selected)), flush=True)
        if cursor != len(selected):
            raise RuntimeError('Incomplete extraction')
        for array in arrays.values():
            array.flush()
        del arrays
    for name, value in model.named_buffers():
        if not torch.equal(value.detach().cpu(), buffers[name]):
            raise RuntimeError('Frozen forward modified buffer ' + name)
    manifest = {'protocol': PROTOCOL, 'project': project, 'class_ids': classes.tolist(),
                'supports': supports, 'branches': branch_dims, 'extraction': vars(args),
                'provenance': provenance, 'checkpoint': fingerprint(args.checkpoint),
                'sample_selection': 'uniform without query labels; support union always included',
                'feature_dtype': 'float32, raw unnormalized backbone output',
                'evaluation': 'native held-out test split; deterministic LP evaluation input',
                'complete': True}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Complete feature cache: ' + str(root), flush=True)


def compare_cache(args, expected_project):
    root = Path(args.cache_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('protocol') != PROTOCOL or not manifest.get('complete')
            or manifest.get('project') != expected_project):
        raise ValueError('Incomplete, incompatible, or wrong-project cache')
    if not args.ridge_alphas or any(a <= 0 or not np.isfinite(a) for a in args.ridge_alphas):
        raise ValueError('Ridge alphas must be finite and positive')
    if min(args.cosine_temperature, args.ridge_temperature) <= 0:
        raise ValueError('Temperatures must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    labels = {split: np.load(root / (split + '_labels.npy'), allow_pickle=False)
              for split in ('train', 'eval')}
    original = np.load(root / 'train_indices.npy', allow_pickle=False)
    classes = np.asarray(manifest['class_ids'])
    supports = {}
    for seed, ids in manifest['supports'].items():
        positions = np.searchsorted(original, ids)
        if (np.any(positions >= len(original)) or not np.array_equal(original[positions], ids)
                or not np.array_equal(labels['train'][positions], classes)):
            raise ValueError('Cached support mapping does not match training rows')
        supports[seed] = positions
    union = np.unique(np.concatenate(list(supports.values())))
    fit_indices = np.setdiff1d(np.arange(len(original)), union)
    np.save(output / 'fit_original_indices.npy', original[fit_indices])
    rows = []
    modes = ['raw', 'centered', 'standardized', 'linear_standardized']
    if args.pca_rank:
        modes += ['pca', 'whitened']
    for branch, dim in manifest['branches'].items():
        print('Fitting train-only transforms for ' + branch, flush=True)
        train = np.load(root / ('train_' + branch + '.npy'), mmap_mode='r')
        evaluation = np.load(root / ('eval_' + branch + '.npy'), mmap_mode='r')
        if train.shape != (len(original), dim) or evaluation.shape != (len(labels['eval']), dim):
            raise ValueError('Feature cache shape mismatch')
        state = fit_transform(train, fit_indices, args.pca_rank, args.pca_max_samples,
                              args.whiten_shrinkage, args.seed)
        np.savez(output / (branch + '_transform.npz'), **state)
        for mode in modes:
            mapped = {}
            for split, features in (('train', train), ('eval', evaluation)):
                mapped[split] = np.concatenate([
                    transform(features[start:start + 1024], state, mode)
                    for start in range(0, len(features), 1024)])
            for seed, positions in supports.items():
                exemplars = mapped['train'][positions]
                heads = []
                if mode != 'linear_standardized':
                    heads.append(('cosine', exemplars.T, np.zeros(len(classes), dtype=np.float32),
                                  args.cosine_temperature, None))
                if mode in ('standardized', 'linear_standardized'):
                    for alpha in args.ridge_alphas:
                        weights, bias, penalty = fit_ridge(exemplars, alpha)
                        heads.append(('ridge_alpha_' + str(alpha), weights, bias,
                                      args.ridge_temperature, penalty))
                for head, weights, bias, temperature, penalty in heads:
                    predictions = {split: values @ weights + bias for split, values in mapped.items()}
                    row = {'branch': branch, 'mode': mode, 'head': head, 'exemplar_seed': int(seed),
                           'temperature': temperature, 'ridge_penalty': penalty,
                           'train_unlabeled': classification_metrics(
                               predictions['train'][fit_indices], labels['train'][fit_indices], classes, temperature),
                           'eval': classification_metrics(predictions['eval'], labels['eval'], classes, temperature),
                           'support': classification_metrics(
                               predictions['train'][positions], classes, classes, temperature)}
                    rows.append(row)
                    print('{} {} {} seed{}: train={:.2f} eval={:.2f}'.format(
                        branch, mode, head, seed, row['train_unlabeled']['top1'], row['eval']['top1']), flush=True)
            del mapped
    summary = []
    for key in sorted(set((r['branch'], r['mode'], r['head']) for r in rows)):
        group = [r for r in rows if (r['branch'], r['mode'], r['head']) == key]
        values = np.asarray([r['eval']['top1'] for r in group])
        deltas = []
        for row in group:
            base = next(r for r in rows if r['branch'] == row['branch'] and r['mode'] == 'raw'
                        and r['head'] == 'cosine' and r['exemplar_seed'] == row['exemplar_seed'])
            deltas.append(row['eval']['top1'] - base['eval']['top1'])
        summary.append({'branch': key[0], 'mode': key[1], 'head': key[2],
                        'eval_top1_mean': float(values.mean()),
                        'eval_top1_sample_std': float(values.std(ddof=1)) if len(values) > 1 else None,
                        'paired_delta_vs_raw': deltas, 'paired_delta_mean': float(np.mean(deltas))})
    report = {'protocol': PROTOCOL, 'project': expected_project, 'settings': vars(args),
              'source_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
              'source_manifest': manifest, 'fit_count': len(fit_indices), 'results': rows, 'summary': summary,
              'interpretation': 'No test-based method/alpha selection. Ridge is one-shot least squares, not full-label LP. '
              'Train query metrics are transductive diagnostics. Compare PCA vs whitened at equal rank. '
              'Confidence/ECE depend on the declared score temperature and are not LP accuracy.'}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Frozen Stage1 readouts: ' + expected_project, '', report['interpretation'], '',
             '| Branch | Transform | Head | Eval Top1 mean | Exemplar std | Paired delta vs raw (pp) |',
             '|---|---|---|---:|---:|---:|']
    for row in summary:
        std = row['eval_top1_sample_std']
        lines.append('| {} | {} | {} | {:.2f} | {} | {:+.2f} |'.format(
            row['branch'], row['mode'], row['head'], row['eval_top1_mean'],
            '{:.2f}'.format(std) if std is not None else 'n/a', row['paired_delta_mean']))
    lines += ['', 'Statistics are fit on training rows excluding the union of all exemplar seeds.',
              'Only exemplar labels enter ridge fitting. Held-out labels enter scoring only.',
              'Feature extraction is clean/deterministic; augmentation stability is a separate experiment.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Report: ' + str(output / 'comparison.md'), flush=True)
