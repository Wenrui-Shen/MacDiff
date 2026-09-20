"""Read-only diagnostics for the current person-0 Stage1 text checkpoints.

Uses saved training arguments, strict model/cache identity checks, fixed crops,
and equal-length derangements. No training, loss changes, or cache regeneration.
Compatible with the project's Python 3.8 / old PyTorch environment.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def geometry(x, sample_ids, labels):
    """Raw scale plus scale-independent spread and centered spectral rank."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('Geometry requires >=3 finite vectors')
    centered = x - x.mean(axis=0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(x)).clip(0)
    total = eigen.sum()
    rank = 0.
    if total > 1e-14:
        p = eigen[eigen > total * 1e-12] / total
        rank = float(np.exp(-(p * np.log(p)).sum()))
    normalized = unit(x)
    similarity = normalized @ normalized.T
    allowed = np.asarray(sample_ids)[:, None] != np.asarray(sample_ids)[None, :]
    if not allowed.any(axis=1).all():
        raise ValueError('Need vectors from distinct samples')
    candidates = np.where(allowed, similarity, -np.inf)
    nn = candidates.argmax(axis=1)
    labels = np.asarray(labels)
    return {
        'vectors': len(x), 'dimensions': x.shape[1],
        'mean_channel_variance': float((centered ** 2).mean()),
        'mean_squared_energy': float((x ** 2).mean()),
        'variance_fraction_of_energy': float((centered ** 2).sum() / max((x ** 2).sum(), 1e-30)),
        'centered_effective_rank': rank,
        'cross_sample_mean_cosine': float(similarity[allowed].mean()),
        'diagnostic_label_nn_accuracy': float((labels[nn] == labels).mean()),
        'diagnostic_label_random_neighbor_accuracy': float(
            (((labels[:, None] == labels[None, :]) & allowed).sum(axis=1) / allowed.sum(axis=1)).mean()),
    }


def relation_retention(before, after, sample_ids, k=10):
    """Compare corresponding token identities; never align different words by slot."""
    allowed = np.asarray(sample_ids)[:, None] != np.asarray(sample_ids)[None, :]
    k = min(k, int(allowed.sum(axis=1).min()))
    result = {}
    for center in (False, True):
        a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
        if center:
            a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        a, b = unit(a) @ unit(a).T, unit(b) @ unit(b).T
        av, bv = a[allowed], b[allowed]
        corr = None if min(av.std(), bv.std()) < 1e-12 else float(np.corrcoef(av, bv)[0, 1])
        an = np.argsort(np.where(allowed, a, -np.inf), axis=1)[:, -k:]
        bn = np.argsort(np.where(allowed, b, -np.inf), axis=1)[:, -k:]
        overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(an, bn)])
        result['centered' if center else 'raw'] = {
            'cosine_pearson': corr, 'neighbor_overlap': float(overlap), 'k': k}
    return result


def paired_batches(indices, masks, batch_size):
    """Every batch shares EXACT original valid positions, including EOS position."""
    buckets = defaultdict(list)
    for i in indices:
        buckets[tuple(np.flatnonzero(masks[int(i), 0]))].append(int(i))
    batches, excluded = [], []
    for signature in sorted(buckets):
        rows = buckets[signature]
        chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        # Avoid losing a singleton tail when a preceding batch can give up a row.
        if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-2]) > 2:
            chunks[-1].insert(0, chunks[-2].pop())
        for chunk in chunks:
            (batches if len(chunk) >= 2 else excluded).append(chunk)
    return batches, [i for row in excluded for i in row]


def derangement(size, rng):
    if size < 2:
        raise ValueError('Cannot shuffle a singleton')
    order = rng.permutation(size)
    perm = np.empty(size, dtype=np.int64)
    perm[order] = np.roll(order, 1)
    return perm


def text_batch(model, cache, indices, device):
    data = {k: torch.from_numpy(v).to(device) for k, v in cache.get_batch(
        np.asarray(indices, dtype=np.int64), one_person=True).items()}
    r = model.text_remap(data['text_features'])
    local = model.remap_tokens(data['text_tokens'], data['text_token_mask'],
                               data['text_person_ids'], data['text_positions'])
    valid = torch.cat([torch.ones_like(data['text_token_mask'][:, :1]), data['text_token_mask']], 1)
    return data, r, local, torch.cat([r[:, None], local], 1), valid


@torch.no_grad()
def collect_geometry(model, cache, indices, labels, device, batch_size, seed):
    saved = defaultdict(list)
    local_ids, local_labels = [], []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        data, r, local, _, _ = text_batch(model, cache, ids, device)
        # The extra structural embeddings must not manufacture content diversity.
        content = model.text_remap(data['text_tokens'])
        saved['global_clip'].append(data['text_features'].cpu().numpy())
        saved['global_remap'].append(r.cpu().numpy())
        for row, idx in enumerate(ids):
            length = int(data['text_token_mask'][row].sum())
            # Exclude the real EOS (already represented in the global view).
            count = min(4, length - 1)
            if count == 0:
                continue
            positions = np.sort(np.random.RandomState(seed + int(idx)).choice(length - 1, count, replace=False))
            take = torch.as_tensor(positions, device=device)
            for name, tensor in [('local_clip', data['text_tokens']),
                                 ('local_content_remap', content), ('local_memory', local)]:
                saved[name].append(tensor[row, take].cpu().numpy())
            local_ids.extend([int(idx)] * count)
            local_labels.extend([int(labels[int(idx)])] * count)
    arrays = {k: np.concatenate(v) for k, v in saved.items()}
    result = {}
    for name, x in arrays.items():
        global_view = name.startswith('global')
        result[name] = geometry(x, indices if global_view else local_ids,
                                labels[indices] if global_view else local_labels)
    result['global_relation_retention'] = relation_retention(arrays['global_clip'], arrays['global_remap'], indices)
    if 'local_clip' in arrays:
        result['local_content_relation_retention'] = relation_retention(
            arrays['local_clip'], arrays['local_content_remap'], local_ids)
    return result


def skeleton_losses(model, prediction, noise, mask, t):
    mse = (prediction.float() - model.patchify(noise).float()).square().mean(-1)
    return (mse * mask).sum(-1) / mask.sum(-1) * model.loss_reweight[t].to(mse.device)


def text_losses(prediction, noise, valid):
    mse = (prediction.float() - noise.float()).square().mean(-1)
    return {
        's2t_all': ((mse * valid).sum(1) / valid.sum(1), valid.sum(1)),
        's2t_global': (mse[:, 0], torch.ones_like(mse[:, 0])),
        's2t_local': ((mse[:, 1:] * valid[:, 1:]).sum(1) / valid[:, 1:].sum(1), valid[:, 1:].sum(1)),
    }


@torch.no_grad()
def condition_trial(model, raw, aug, data, r, local, memory, valid,
                    timestep, mask_ratio, motion_tau, permutation):
    """Correct and perturbed conditions share noisy inputs, targets, masks and t."""
    center = raw.mean(dim=(1, 2), keepdim=True) if model.self_shift else None
    clean = model._normalize_sequence(raw, center)
    aug = model._normalize_sequence(aug, center)
    latent, pooled, mask, _ = model.forward_encoder(
        aug, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_tau)
    h = torch.cat([pooled, latent], 1)
    hmask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
    t = torch.full((len(raw),), timestep, dtype=torch.long, device=raw.device)
    noise = torch.randn_like(clean)
    noisy = model.diffusion.q_sample(clean, t, noise=noise)
    perm = torch.as_tensor(permutation, dtype=torch.long, device=raw.device)
    for key in ('text_token_mask', 'text_person_ids', 'text_positions'):
        if not torch.equal(data[key], data[key][perm]):
            raise ValueError('Permutation changes text structure; refuse confounded comparison')
    results = {}
    correct = None
    for name, global_r, local_r in [('correct', r, local), ('global_shuffled', r[perm], local),
                                    ('local_shuffled', r, local[perm]), ('all_shuffled', r[perm], local[perm])]:
        pred = model.text_skeleton_decoder(noisy, t, global_r, local_r, data['text_token_mask'],
                                          source=model if model.share_skeleton_decoder else None)
        loss = skeleton_losses(model, pred, noise, mask, t)
        if correct is None:
            correct, baseline = pred, loss
        else:
            change = ((pred - correct).square().mean(-1) * mask).sum(1) / mask.sum(1)
            results['t2s_' + name] = (baseline, loss, torch.ones_like(loss), change)
    text_noise = torch.randn_like(memory).masked_fill(~valid[..., None], 0)
    noisy_text = model.diffusion.q_sample(memory, t, noise=text_noise)
    predictions = [model.text_noise_decoder(noisy_text, t, cond, hmask, valid,
                   data['text_person_ids'], data['text_positions']) for cond in (h, h[perm])]
    original, shuffled = [text_losses(p, text_noise, valid) for p in predictions]
    changes = text_losses(predictions[0], predictions[1], valid)
    for key in original:
        results[key] = (original[key][0], shuffled[key][0], original[key][1], changes[key][0])
    return {key: [v.cpu().numpy() for v in values] for key, values in results.items()}


def summarize_pairs(records, seed):
    """Token-weighted estimates; cluster bootstrap keeps repeats/pairs together."""
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['test'], row['t'])].append(row)
    result = []
    for (test, t), rows in sorted(grouped.items()):
        weights = np.asarray([r['weight'] for r in rows], dtype=float)
        good = np.asarray([r['correct'] for r in rows])
        bad = np.asarray([r['shuffled'] for r in rows])
        baseline, perturbed = np.average(good, weights=weights), np.average(bad, weights=weights)
        clusters = defaultdict(lambda: [0., 0.])
        for r in rows:
            clusters[r['batch']][0] += r['weight'] * (r['shuffled'] - r['correct'])
            clusters[r['batch']][1] += r['weight']
        aggregates = np.asarray(list(clusters.values()))
        rng = np.random.RandomState(seed)
        ci = None
        if len(aggregates) >= 2:
            draws = aggregates[rng.randint(len(aggregates), size=(1000, len(aggregates)))].sum(axis=1)
            ci = np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5]).tolist()
        result.append({'test': test, 't': t, 'samples': len(set(r['sample'] for r in rows)),
                       'matched_batch_clusters': len(clusters), 'correct_mse': float(baseline),
                       'shuffled_mse': float(perturbed), 'delta': float(perturbed - baseline),
                       'relative_increase_percent': float(100 * (perturbed - baseline) / max(baseline, 1e-12)),
                       'delta_cluster_bootstrap_95ci': ci,
                       'prediction_change_mse': float(np.average([r['prediction_change'] for r in rows], weights=weights)),
                       'same_label_donor_fraction': float(np.mean([r['same_label'] for r in rows]))})
    return result


def load_checkpoint(path):
    # Saved argparse Namespace is required. This CLI is for the user's own checkpoint.
    import inspect
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    checkpoint = torch.load(path, **options)
    saved = checkpoint.get('args')
    args = dict(saved) if isinstance(saved, dict) else vars(saved) if saved is not None else {}
    if args.get('model') != 'model.transformer_macdiff_text.Transformer':
        raise ValueError('Need a complete text Stage1 checkpoint with saved training args')
    if tuple(args.get('text_person_alignment', ())) != ('per_person_v1', True):
        raise ValueError('This diagnostic supports the current person-0 protocol only')
    return checkpoint, args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text-cache', default=None)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[100, 500, 900])
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    if args.samples < 3 or args.batch_size < 2 or args.repeats < 1:
        parser.error('Need samples >=3, batch-size >=2, repeats >=1')
    out = Path(args.output_dir or str(Path(args.checkpoint).with_suffix('')) + '_text_diagnostic')
    if out.exists():
        raise FileExistsError('Choose a new --output-dir; existing diagnostics are not overwritten: ' + str(out))
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint, saved = load_checkpoint(args.checkpoint)
    from model.transformer_macdiff_text import Transformer
    from util.person_text_cache import load_person_token_cache
    from feeder.feeder_ntu import Feeder
    options = dict(saved['model_args'])
    options['share_skeleton_decoder'] = saved['text_share_skeleton_decoder']
    for name, value in zip(('lambda_text_to_skeleton', 'lambda_skeleton_to_text'), saved['text_training_weights']):
        options[name] = value
    model = Transformer(**options)
    model.load_state_dict(checkpoint['model'], strict=True)
    if not all(0 <= t < model.diffusion.num_timesteps for t in args.timesteps):
        parser.error('Timesteps are outside this checkpoint diffusion schedule')
    model.eval().requires_grad_(False).to(args.device)
    epoch = checkpoint['epoch']
    del checkpoint
    feeder_args = dict(saved['train_feeder_args'])
    if args.data_path:
        feeder_args['data_path'] = args.data_path
    if saved.get('feeder') != 'feeder.feeder_ntu.Feeder' or feeder_args.get('split') != 'train':
        raise ValueError('Only the native training feeder is supported')
    dataset = Feeder(**feeder_args)
    print('Validating existing cache and dataset checksums (can take several minutes)...', flush=True)
    cache = load_person_token_cache(args.text_cache or saved['text_cache'],
                                   dataset.data_path, expected_count=len(dataset))
    identity = {k: cache.manifest[k] for k in ('protocol', 'identity', 'files')}
    if identity != saved['text_cache_identity']:
        raise ValueError('Checkpoint/cache provenance mismatch')
    eligible = np.flatnonzero(cache.mask[:, 0].any(axis=1))
    if len(eligible) < args.samples:
        raise ValueError('Not enough person-0 descriptions for requested sample count')
    indices = np.random.RandomState(args.seed).permutation(eligible)[:args.samples]
    out.mkdir(parents=True)
    report = {'protocol': 'person0_text_condition_diagnostic_v1', 'checkpoint': str(Path(args.checkpoint).resolve()),
              'checkpoint_epoch': epoch, 'arguments': vars(args), 'training_feeder_args': feeder_args,
              'model_args': options, 'sample_indices': indices.tolist(),
              'cache_identity_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              'geometry_scope': 'Fixed train descriptions; local: up to 4 body tokens/sample, EOS excluded; same-sample neighbors excluded.',
              'interpretation': 'Descriptive diagnostics, not a causal attribution of LP accuracy. Small delta does not prove no semantic information.',
              'torch_version': torch.__version__}
    print('Computing fixed-description geometry...', flush=True)
    report['geometry'] = collect_geometry(model, cache, indices, np.asarray(dataset.label),
                                           args.device, args.batch_size, args.seed)
    (out / 'geometry.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    records, empty, singleton = [], [], []
    if not args.geometry_only:
        batches, singleton = paired_batches(indices, cache.mask, args.batch_size)
        for number, ids in enumerate(batches):
            examples = []
            active_ids = []
            for idx in ids:
                seed_all((args.seed + idx) % (2 ** 32))
                source, augmented, _, _ = dataset[idx]
                if not np.any(source[..., 0] != 0):
                    empty.append(idx)
                    continue
                examples.append((source[..., 0].transpose(1, 2, 0), augmented[..., 0].transpose(1, 2, 0)))
                active_ids.append(idx)
            if len(active_ids) < 2:
                singleton.extend(active_ids)
                continue
            raw = torch.as_tensor(np.stack([x[0] for x in examples]), dtype=torch.float32, device=args.device)
            aug = torch.as_tensor(np.stack([x[1] for x in examples]), dtype=torch.float32, device=args.device)
            with torch.no_grad():
                data, r, local, memory, valid = text_batch(model, cache, active_ids, args.device)
            for repeat in range(args.repeats):
                trial_seed = (args.seed + number * 1009 + repeat * 9176) % (2 ** 32)
                permutation = derangement(len(active_ids), np.random.RandomState(trial_seed))
                for t in args.timesteps:
                    seed_all(trial_seed)  # Same encoder mask and base noise across t/checkpoints.
                    values = condition_trial(model, raw, aug, data, r, local, memory, valid, t,
                                             saved['mask_ratio'], saved.get('motion_aware_tau', -1), permutation)
                    for test, (correct, shuffled, weights, change) in values.items():
                        for j, idx in enumerate(active_ids):
                            donor = active_ids[permutation[j]]
                            records.append({'test': test, 't': t, 'batch': number, 'repeat': repeat,
                                            'sample': idx, 'donor': donor, 'tokens': int(valid[j].sum()),
                                            'same_label': bool(dataset.label[idx] == dataset.label[donor]),
                                            'correct': float(correct[j]), 'shuffled': float(shuffled[j]),
                                            'weight': float(weights[j]), 'prediction_change': float(change[j])})
            print('Condition batch {}/{} finished ({} samples).'.format(number + 1, len(batches), len(active_ids)), flush=True)
        if not records:
            raise ValueError('No valid equal-length pairs; increase --samples. Geometry was saved.')
        report['conditioning'] = summarize_pairs(records, args.seed)
        report['timestep_alpha_cumprod'] = {str(t): float(model.diffusion.alphas_cumprod[t]) for t in args.timesteps}
        report['conditioning_excluded_empty_person0'] = empty
        report['conditioning_excluded_unpaired'] = singleton
        report['conditioning_sample_count'] = len(set(r['sample'] for r in records))
        report['conditioning_scope'] = 'Equal token-length matched subset, fixed training crop per sample. FP32/eval; no optimizer. CI clusters by matched batch across all repeats.'
        with (out / 'paired_records.jsonl').open('w', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        for row in report['conditioning']:
            print('{test:22s} t={t:3d} correct={correct_mse:.6g} shuffled={shuffled_mse:.6g} change={relative_increase_percent:+.2f}% CI={delta_cluster_bootstrap_95ci}'.format(**row))
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved ' + str(out / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
