"""Read-only checks for the share and independent Stage1 EMA-remap runs.

The same train samples/crops, decoder masks, timesteps and noise seeds are used
for both checkpoints. Within a trial, T->S swaps only the online text condition;
S->T swaps only the encoded skeleton condition. CLIP cache and checkpoints are
never edited.

Run from the repository root in the server's macdiff environment. Python 3.8
and the project's older PyTorch release are supported.
"""
import argparse
from collections import defaultdict
import hashlib
import inspect
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


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def geometry(vectors, sample_ids, labels):
    """Measure spread and label-neighbor agreement on a fixed train subset."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) < 3 or not np.isfinite(vectors).all():
        raise ValueError('Geometry needs at least three finite vectors')
    centered = vectors - vectors.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(vectors)).clip(0)
    total = eigenvalues.sum()
    rank = 0.
    if total > 1e-14:
        p = eigenvalues[eigenvalues > total * 1e-12] / total
        rank = float(np.exp(-(p * np.log(p)).sum()))
    similarities = unit(vectors) @ unit(vectors).T
    sample_ids = np.asarray(sample_ids)
    allowed = sample_ids[:, None] != sample_ids[None, :]
    if not allowed.any(axis=1).all():
        raise ValueError('Geometry needs distinct samples for nearest neighbors')
    nearest = np.where(allowed, similarities, -np.inf).argmax(axis=1)
    labels = np.asarray(labels)
    return {
        'vectors': len(vectors), 'dimensions': vectors.shape[1],
        'mean_channel_variance': float(np.square(centered).mean()),
        'mean_squared_energy': float(np.square(vectors).mean()),
        'variance_fraction_of_energy': float(np.square(centered).sum() /
                                             max(np.square(vectors).sum(), 1e-30)),
        'centered_effective_rank': rank,
        'cross_sample_mean_cosine': float(similarities[allowed].mean()),
        'diagnostic_label_nn_accuracy': float((labels[nearest] == labels).mean()),
        'diagnostic_label_random_neighbor_accuracy': float(
            (((labels[:, None] == labels[None, :]) & allowed).sum(axis=1)
             / allowed.sum(axis=1)).mean()),
    }


def relation_retention(before, after, sample_ids, neighbors=10):
    """Compare cosine relations for the same samples/tokens, raw and centered."""
    ids = np.asarray(sample_ids)
    allowed = ids[:, None] != ids[None, :]
    neighbors = min(neighbors, int(allowed.sum(axis=1).min()))
    result = {}
    for center in (False, True):
        first = np.asarray(before, dtype=np.float64)
        second = np.asarray(after, dtype=np.float64)
        if center:
            first, second = first - first.mean(axis=0), second - second.mean(axis=0)
        first, second = unit(first) @ unit(first).T, unit(second) @ unit(second).T
        a, b = first[allowed], second[allowed]
        correlation = None if min(a.std(), b.std()) < 1e-12 else float(np.corrcoef(a, b)[0, 1])
        first_neighbors = np.argsort(np.where(allowed, first, -np.inf), axis=1)[:, -neighbors:]
        second_neighbors = np.argsort(np.where(allowed, second, -np.inf), axis=1)[:, -neighbors:]
        overlap = np.mean([
            len(set(a_ids) & set(b_ids)) / neighbors
            for a_ids, b_ids in zip(first_neighbors, second_neighbors)])
        result['centered' if center else 'raw'] = {
            'cosine_pearson': correlation, 'neighbor_overlap': float(overlap),
            'neighbors': neighbors,
        }
    return result


def paired_batches(indices, masks, batch_size):
    """Text donors in one batch must have identical valid token positions."""
    buckets = defaultdict(list)
    for index in indices:
        buckets[tuple(np.flatnonzero(masks[int(index), 0]))].append(int(index))
    batches, excluded = [], []
    for signature in sorted(buckets):
        rows = buckets[signature]
        chunks = [rows[start:start + batch_size] for start in range(0, len(rows), batch_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-2]) > 2:
            chunks[-1].insert(0, chunks[-2].pop())
        for chunk in chunks:
            (batches if len(chunk) >= 2 else excluded).append(chunk)
    return batches, [index for chunk in excluded for index in chunk]


def derangement(size, rng):
    if size < 2:
        raise ValueError('Cannot shuffle a singleton')
    order = rng.permutation(size)
    permutation = np.empty(size, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    return permutation


def skeleton_losses(model, prediction, noise, mask, timestep):
    mse = (prediction.float() - model.patchify(noise).float()).square().mean(-1)
    return (mse * mask).sum(-1) / mask.sum(-1) * model.loss_reweight[timestep].to(mse.device)


def text_losses(prediction, noise, valid):
    mse = (prediction.float() - noise.float()).square().mean(-1)
    return {
        's2t_all': ((mse * valid).sum(-1) / valid.sum(-1), valid.sum(-1)),
        's2t_global': (mse[:, 0], torch.ones_like(mse[:, 0])),
        's2t_local': ((mse[:, 1:] * valid[:, 1:]).sum(-1)
                      / valid[:, 1:].sum(-1), valid[:, 1:].sum(-1)),
    }


def summarize_pairs(records, seed):
    """Summarize matched MSE differences, resampling whole mini-batches for CI."""
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['test'], row['t'])].append(row)
    result = []
    for (test, timestep), rows in sorted(grouped.items()):
        weights = np.asarray([row['weight'] for row in rows], dtype=float)
        correct = np.asarray([row['correct'] for row in rows])
        shuffled = np.asarray([row['shuffled'] for row in rows])
        baseline = np.average(correct, weights=weights)
        perturbed = np.average(shuffled, weights=weights)
        clusters = defaultdict(lambda: [0., 0.])
        for row in rows:
            clusters[row['batch']][0] += row['weight'] * (row['shuffled'] - row['correct'])
            clusters[row['batch']][1] += row['weight']
        aggregates = np.asarray(list(clusters.values()))
        interval = None
        if len(aggregates) >= 2:
            rng = np.random.RandomState(seed)
            draws = aggregates[rng.randint(
                len(aggregates), size=(1000, len(aggregates)))].sum(axis=1)
            interval = np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5]).tolist()
        result.append({
            'test': test, 't': timestep,
            'samples': len(set(row['sample'] for row in rows)),
            'matched_batch_clusters': len(clusters),
            'correct_mse': float(baseline), 'shuffled_mse': float(perturbed),
            'delta': float(perturbed - baseline),
            'relative_increase_percent': float(
                100 * (perturbed - baseline) / max(baseline, 1e-12)),
            'delta_cluster_bootstrap_95ci': interval,
            'prediction_change_mse': float(np.average(
                [row['prediction_change'] for row in rows], weights=weights)),
            'same_label_donor_fraction': float(np.mean(
                [row['same_label'] for row in rows])),
        })
    return result


def load_checked_checkpoint(path, expected_share):
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    checkpoint = torch.load(str(path), **options)
    saved_args = checkpoint.get('args')
    saved = (dict(saved_args) if isinstance(saved_args, dict)
             else vars(saved_args) if saved_args is not None else {})
    model_args = saved.get('model_args')
    if (saved.get('model') != 'model.transformer_macdiff_text.Transformer'
            or not isinstance(model_args, dict)
            or model_args.get('text_target_mode') != 'ema_remap'
            or model_args.get('text_target_norm') != 'rms'
            or model_args.get('one_person') is not True
            or tuple(saved.get('text_person_alignment', ())) != ('per_person_v1', True)
            or saved.get('text_target_mode') != 'ema_remap'
            or saved.get('text_target_norm') != 'rms'
            or saved.get('text_share_skeleton_decoder') is not expected_share
            or model_args.get('share_skeleton_decoder') is not expected_share
            or saved.get('feeder') != 'feeder.feeder_ntu.Feeder'
            or 'model' not in checkpoint or 'epoch' not in checkpoint):
        raise ValueError('{} is not the expected complete EMA-remap {} checkpoint'.format(
            path, 'share' if expected_share else 'independent'))
    weights = saved.get('text_training_weights')
    if (weights is None or len(weights) != 2 or
            not all(np.isfinite(float(value)) and float(value) > 0 for value in weights)):
        raise ValueError('{} has missing or disabled bidirectional losses'.format(path))
    if not np.isclose(float(saved.get('text_target_momentum', -1)),
                      float(model_args.get('text_target_momentum', -2))):
        raise ValueError('{} has inconsistent EMA momentum'.format(path))
    if not np.isclose(float(saved.get('text_uniformity_weight', -1)),
                      float(model_args.get('lambda_text_uniformity', -2))):
        raise ValueError('{} has inconsistent text uniformity weight'.format(path))
    return checkpoint, saved


def comparable_setup(saved):
    model_args = dict(saved['model_args'])
    model_args.pop('share_skeleton_decoder')
    return {
        'model_args': model_args,
        'weights': tuple(saved['text_training_weights']),
        'momentum': saved['text_target_momentum'],
        'uniformity_weight': saved['text_uniformity_weight'],
        'train_feeder_args': saved['train_feeder_args'],
        'mask_ratio': saved['mask_ratio'],
        'motion_aware_tau': saved.get('motion_aware_tau', -1),
        'cache_identity': saved['text_cache_identity'],
    }


def construct_model(checkpoint, saved, device):
    from model.transformer_macdiff_text import Transformer

    options = dict(saved['model_args'])
    options['share_skeleton_decoder'] = saved['text_share_skeleton_decoder']
    options['lambda_text_to_skeleton'], options['lambda_skeleton_to_text'] = (
        saved['text_training_weights'])
    model = Transformer(**options)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False).to(device)


def prepare_examples(dataset, cache, count, seed):
    eligible = np.flatnonzero(cache.mask[:, 0].any(axis=1))
    if len(eligible) < count:
        raise ValueError('Not enough person-0 descriptions')
    examples, empty = {}, []
    for index in np.random.RandomState(seed).permutation(eligible):
        index = int(index)
        seed_all((seed + index) % (2 ** 32))
        source, augmented, _, _ = dataset[index]
        if not np.any(source[..., 0] != 0):
            empty.append(index)
            continue
        examples[index] = (
            source[..., 0].transpose(1, 2, 0).copy(),
            augmented[..., 0].transpose(1, 2, 0).copy())
        if len(examples) == count:
            break
    if len(examples) < count:
        raise ValueError('Not enough nonempty person-0 training crops')
    return examples, empty


@torch.no_grad()
def text_views(model, cache, indices, device):
    data = {name: torch.from_numpy(array).to(device) for name, array in
            cache.get_batch(np.asarray(indices, dtype=np.int64), one_person=True).items()}
    mask = data['text_token_mask']
    model.validate_text_tokens(data['text_tokens'], mask,
                               data['text_person_ids'], data['text_positions'])
    fixed_global = model.fixed_clip_target(data['text_features'])
    fixed_local = model.fixed_clip_target(data['text_tokens'], mask)
    online_global = model.text_remap(fixed_global)
    online_local = model.text_remap(fixed_local).masked_fill(~mask[..., None], 0)
    teacher_global = model.text_target_remap(fixed_global)
    teacher_local = model.text_target_remap(fixed_local).masked_fill(~mask[..., None], 0)
    person_ids = data['text_person_ids'].masked_fill(~mask, 0)
    positions = data['text_positions'].masked_fill(~mask, 0)
    condition_local = (online_local + model.text_person_embedding(person_ids)
                       + model.text_position_embedding(positions)).masked_fill(
                           ~mask[..., None], 0)
    return data, {
        'global_clip': fixed_global, 'global_online': online_global,
        'global_teacher': teacher_global, 'local_clip': fixed_local,
        'local_online': online_local, 'local_teacher': teacher_local,
        'local_condition': condition_local,
    }


def paired_distance(before, after):
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    cosine = (before * after).sum(axis=-1) / np.maximum(
        np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1), 1e-12)
    return {'mse': float(np.square(before - after).mean()),
            'mean_cosine': float(cosine.mean())}


@torch.no_grad()
def collect_geometry(model, cache, indices, labels, device, batch_size, seed):
    chunks = {name: [] for name in (
        'global_clip', 'global_online', 'global_teacher',
        'local_clip', 'local_online', 'local_teacher')}
    local_sample_ids = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        data, views = text_views(model, cache, ids, device)
        for name in ('global_clip', 'global_online', 'global_teacher'):
            chunks[name].append(views[name].cpu().numpy())
        for row, index in enumerate(ids):
            valid = np.flatnonzero(data['text_token_mask'][row].cpu().numpy())
            body = valid[:-1]  # the final valid token is EOS; global already represents it
            if not len(body):
                continue
            chosen = np.sort(np.random.RandomState((seed + int(index)) % (2 ** 32)).choice(
                body, min(4, len(body)), replace=False))
            for name in ('local_clip', 'local_online', 'local_teacher'):
                chunks[name].append(views[name][row, chosen].cpu().numpy())
            local_sample_ids.extend([int(index)] * len(chosen))
    arrays = {name: np.concatenate(parts) for name, parts in chunks.items() if parts}
    if 'local_clip' not in arrays:
        raise ValueError('Selected samples have no body tokens for local geometry')
    result = {}
    for name, array in arrays.items():
        ids = indices if name.startswith('global') else local_sample_ids
        result[name] = geometry(array, ids, labels[np.asarray(ids, dtype=np.int64)])
    for suffix in ('online', 'teacher'):
        for scope, ids in (('global', indices), ('local', local_sample_ids)):
            fixed, transformed = arrays[scope + '_clip'], arrays[scope + '_' + suffix]
            result[scope + '_' + suffix + '_versus_clip'] = {
                'paired': paired_distance(fixed, transformed),
                'relations': relation_retention(fixed, transformed, ids),
            }
    return result


@torch.no_grad()
def t2s_trial(model, raw, aug, data, views, timestep, mask_ratio,
              motion_tau, permutation, mask_seed, noise_seed):
    """Swap one condition at a time; noisy targets and masks stay fixed."""
    order = torch.as_tensor(permutation, dtype=torch.long, device=raw.device)
    if (sorted(order.cpu().tolist()) != list(range(len(raw))) or
            (len(raw) > 1 and torch.any(order == torch.arange(len(raw), device=raw.device)))):
        raise ValueError('Text permutation must be a derangement')
    for name in ('text_token_mask', 'text_person_ids', 'text_positions'):
        if not torch.equal(data[name], data[name][order]):
            raise ValueError('Text permutation changes valid positions or person metadata')
    center = raw.mean(dim=(1, 2), keepdim=True) if model.self_shift else None
    clean = model._normalize_sequence(raw, center)
    augmented = model._normalize_sequence(aug, center)
    seed_all(mask_seed)
    latent, pooled, mask, _ = model.forward_encoder(
        augmented, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_tau)
    skeleton_memory = torch.cat([pooled, latent], dim=1)
    skeleton_mask = torch.ones(skeleton_memory.shape[:2], dtype=torch.bool,
                               device=raw.device)
    t = torch.full((len(raw),), timestep, dtype=torch.long, device=raw.device)
    seed_all(noise_seed)
    noise = torch.randn_like(clean)
    noisy = model.diffusion.q_sample(clean, t, noise=noise)
    global_text, local_text = views['global_online'], views['local_condition']
    predictions = {}
    for name, global_condition, local_condition in (
            ('correct', global_text, local_text),
            ('global_shuffled', global_text[order], local_text),
            ('local_shuffled', global_text, local_text[order]),
            ('all_shuffled', global_text[order], local_text[order])):
        predictions[name] = model.text_skeleton_decoder(
            noisy, t, global_condition, local_condition, data['text_token_mask'],
            source=model if model.share_skeleton_decoder else None)
    baseline = skeleton_losses(model, predictions['correct'], noise, mask, t)
    result = {}
    for name in ('global_shuffled', 'local_shuffled', 'all_shuffled'):
        shuffled = skeleton_losses(model, predictions[name], noise, mask, t)
        change = ((predictions[name] - predictions['correct']).float().square().mean(-1)
                  * mask).sum(-1) / mask.sum(-1)
        result['t2s_' + name] = (
            baseline.cpu().numpy(), shuffled.cpu().numpy(),
            np.ones(len(raw), dtype=np.float32), change.cpu().numpy())
    teacher_memory = torch.cat([
        views['global_teacher'][:, None], views['local_teacher']], dim=1)
    valid = torch.cat([torch.ones_like(data['text_token_mask'][:, :1]),
                       data['text_token_mask']], dim=1)
    if not valid[:, 1:].any(dim=1).all():
        raise ValueError('Every paired sample needs a local text token')
    seed_all((noise_seed + 1) % (2 ** 32))
    text_noise = torch.randn_like(teacher_memory).masked_fill(~valid[..., None], 0)
    noisy_text = model.diffusion.q_sample(teacher_memory, t, noise=text_noise)
    text_predictions = [model.text_noise_decoder(
        noisy_text, t, condition, skeleton_mask, valid,
        data['text_person_ids'], data['text_positions'])
        for condition in (skeleton_memory, skeleton_memory[order])]
    original = text_losses(text_predictions[0], text_noise, valid)
    shuffled = text_losses(text_predictions[1], text_noise, valid)
    changes = text_losses(text_predictions[0], text_predictions[1], valid)
    for name in original:
        result[name] = (original[name][0].cpu().numpy(),
                        shuffled[name][0].cpu().numpy(),
                        original[name][1].cpu().numpy(),
                        changes[name][0].cpu().numpy())
    return result


@torch.no_grad()
def evaluate_conditions(model, saved, cache, dataset, examples, batches,
                        timesteps, repeats, seed, device):
    records = []
    for batch_number, ids in enumerate(batches):
        raw = torch.as_tensor(np.stack([examples[index][0] for index in ids]),
                              dtype=torch.float32, device=device)
        aug = torch.as_tensor(np.stack([examples[index][1] for index in ids]),
                              dtype=torch.float32, device=device)
        data, views = text_views(model, cache, ids, device)
        for repeat in range(repeats):
            trial_seed = (seed + batch_number * 1009 + repeat * 9176) % (2 ** 32)
            permutation = derangement(len(ids), np.random.RandomState(trial_seed))
            for timestep in timesteps:
                values = t2s_trial(
                    model, raw, aug, data, views, timestep, saved['mask_ratio'],
                    saved.get('motion_aware_tau', -1), permutation,
                    trial_seed, (trial_seed + timestep * 7919) % (2 ** 32))
                for test, (correct, shuffled, weights, change) in values.items():
                    for row, index in enumerate(ids):
                        donor = ids[permutation[row]]
                        records.append({
                            'test': test, 't': timestep, 'batch': batch_number,
                            'repeat': repeat, 'sample': index, 'donor': donor,
                            'same_label': bool(dataset.label[index] == dataset.label[donor]),
                            'correct': float(correct[row]), 'shuffled': float(shuffled[row]),
                            'weight': float(weights[row]),
                            'prediction_change': float(change[row]),
                        })
        print('  condition batch {}/{} ({} samples)'.format(
            batch_number + 1, len(batches), len(ids)), flush=True)
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--share-checkpoint', required=True)
    parser.add_argument('--independent-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-cache', default=None)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[100, 500, 900])
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args(argv)
    if (args.samples < 3 or args.batch_size < 2 or args.repeats < 1 or args.seed < 0
            or not args.timesteps or len(set(args.timesteps)) != len(args.timesteps)):
        parser.error('Invalid samples, batch size, repeats, seed or timesteps')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for a small smoke test')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Output directory already exists: ' + str(output))
    paths = {'share': Path(args.share_checkpoint),
             'independent': Path(args.independent_checkpoint)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint, saved = load_checked_checkpoint(paths['share'], True)
    reference = comparable_setup(saved)
    epoch = int(checkpoint['epoch'])
    del checkpoint
    other, independent_saved = load_checked_checkpoint(paths['independent'], False)
    if int(other['epoch']) != epoch or comparable_setup(independent_saved) != reference:
        raise ValueError('Checkpoints have different epochs or training/data settings')
    del other, independent_saved

    from feeder.feeder_ntu import Feeder
    from util.person_text_cache import load_person_token_cache

    feeder_args = dict(saved['train_feeder_args'])
    if saved['feeder'] != 'feeder.feeder_ntu.Feeder' or feeder_args.get('split') != 'train':
        raise ValueError('Expected the NTU training feeder')
    if args.data_path:
        feeder_args['data_path'] = args.data_path
    dataset = Feeder(**feeder_args)
    print('Loading trusted v2 CLIP cache (header/size checks only)...', flush=True)
    cache = load_person_token_cache(args.text_cache or saved['text_cache'],
                                    dataset.data_path, expected_count=len(dataset),
                                    skip_full_validation=True)
    identity = {key: cache.manifest[key] for key in ('protocol', 'identity', 'files')}
    if identity != saved['text_cache_identity']:
        raise ValueError('Checkpoint/cache provenance mismatch')
    examples, empty = prepare_examples(dataset, cache, args.samples, args.seed)
    indices = list(examples)
    batches, unpaired = paired_batches(indices, cache.mask, args.batch_size)
    if not args.geometry_only and not batches:
        raise ValueError('No equal-position text pairs; increase --samples')

    output.mkdir(parents=True)
    report = {
        'protocol': 'ema_bidirectional_targeted_diagnostic_v1',
        'arguments': vars(args), 'checkpoint_epoch': epoch,
        'checkpoint_paths': {key: str(path.resolve()) for key, path in paths.items()},
        'sample_indices': indices, 'empty_person0_crops_skipped': empty,
        'condition_unpaired_indices': unpaired,
        'condition_paired_sample_count': sum(map(len, batches)),
        'cache_identity_sha256': hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest(),
        'scope': ('Fixed train crops. FP32/eval, same seeds, samples, masks, noises and '
                  'timesteps across checkpoints. T->S swaps paired online text; S->T '
                  'swaps paired skeleton features. Text donors have identical token '
                  'positions/person metadata. This is not LP or evidence of LP causality.'),
        'geometry': {}, 'conditioning': {}, 'torch_version': torch.__version__,
    }
    all_records = []
    for group, share in (('share', True), ('independent', False)):
        print('Checking {} checkpoint...'.format(group), flush=True)
        checkpoint, group_saved = load_checked_checkpoint(paths[group], share)
        seed_all(args.seed)
        model = construct_model(checkpoint, group_saved, args.device)
        del checkpoint
        if not all(0 <= t < model.diffusion.num_timesteps for t in args.timesteps):
            raise ValueError('Timesteps outside checkpoint diffusion schedule')
        report['geometry'][group] = collect_geometry(
            model, cache, indices, np.asarray(dataset.label),
            args.device, args.batch_size, args.seed)
        geometry_result = report['geometry'][group]
        print('  global variance: CLIP={:.6g}, online={:.6g}, teacher={:.6g}; '
              'teacher/CLIP cosine={:.4f}'.format(
                  geometry_result['global_clip']['mean_channel_variance'],
                  geometry_result['global_online']['mean_channel_variance'],
                  geometry_result['global_teacher']['mean_channel_variance'],
                  geometry_result['global_teacher_versus_clip']['paired']['mean_cosine']),
              flush=True)
        (output / 'geometry.json').write_text(
            json.dumps(report['geometry'], indent=2, allow_nan=False), encoding='utf-8')
        if not args.geometry_only:
            records = evaluate_conditions(
                model, group_saved, cache, dataset, examples, batches,
                args.timesteps, args.repeats, args.seed, args.device)
            if not records:
                raise ValueError('No paired records; increase --samples')
            for row in records:
                row['group'] = group
            all_records.extend(records)
            report['conditioning'][group] = summarize_pairs(records, args.seed)
            for row in report['conditioning'][group]:
                print('  {test:20s} t={t:3d} correct={correct_mse:.6g} '
                      'shuffled={shuffled_mse:.6g} delta={delta:+.6g} '
                      '({relative_increase_percent:+.2f}%) CI={delta_cluster_bootstrap_95ci}'.format(
                          **row), flush=True)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if all_records:
        with (output / 'paired_records.jsonl').open('w', encoding='utf-8') as stream:
            for row in all_records:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    (output / 'summary.json').write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved ' + str(output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
