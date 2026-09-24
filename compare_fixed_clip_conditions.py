"""Compare skeleton conditioning across fixed-CLIP Stage1 checkpoints.

For every model, use the same training samples/crops, encoder-mask seeds, text
noise, timesteps and skeleton derangements. Only the skeleton memory is swapped
within each correct/shuffled pair. This is a read-only training-set diagnostic,
not a linear-probe measurement or a test of causal LP improvement.

Compatible with the server's Python 3.8 and older PyTorch releases.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import random

import numpy as np
import torch

GROUPS = (
    ('fixed', 'none', 1.0),
    ('st01', 'none', 0.1),
    ('rms', 'rms', 1.0),
)
OPTIONAL_GROUP = ('rms_st01', 'rms', 0.1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derangement(size, rng):
    if size < 2:
        raise ValueError('Cannot shuffle a singleton batch')
    order = rng.permutation(size)
    permutation = np.empty(size, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    return permutation


def text_losses(prediction, target, valid):
    mse = (prediction.float() - target.float()).square().mean(-1)
    return {
        's2t_all': ((mse * valid).sum(1) / valid.sum(1), valid.sum(1)),
        's2t_global': (mse[:, 0], torch.ones_like(mse[:, 0])),
        's2t_local': ((mse[:, 1:] * valid[:, 1:]).sum(1) /
                      valid[:, 1:].sum(1), valid[:, 1:].sum(1)),
    }


def summarize_pairs(records, seed):
    """Token-weighted MSE; resample batches so repeated trials stay together."""
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
        ci = None
        if len(aggregates) >= 2:
            rng = np.random.RandomState(seed)
            draws = aggregates[rng.randint(
                len(aggregates), size=(1000, len(aggregates)))].sum(axis=1)
            ci = np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5]).tolist()
        result.append({
            'test': test, 't': timestep,
            'samples': len(set(row['sample'] for row in rows)),
            'matched_batch_clusters': len(clusters),
            'correct_mse': float(baseline), 'shuffled_mse': float(perturbed),
            'delta': float(perturbed - baseline),
            'relative_increase_percent': float(
                100 * (perturbed - baseline) / max(baseline, 1e-12)),
            'delta_cluster_bootstrap_95ci': ci,
            'prediction_change_mse': float(np.average(
                [row['prediction_change'] for row in rows], weights=weights)),
            'same_label_donor_fraction': float(np.mean(
                [row['same_label'] for row in rows])),
        })
    return result


def checkpoint_args(path, group):
    """Load a trusted local training checkpoint and reject wrong experiments."""
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    checkpoint = torch.load(str(path), **options)
    saved = checkpoint.get('args')
    saved = dict(saved) if isinstance(saved, dict) else vars(saved) if saved is not None else {}
    if saved.get('model') != 'model.transformer_macdiff_text.Transformer':
        raise ValueError('{}: not a complete text Stage1 checkpoint'.format(path))
    if tuple(saved.get('text_person_alignment', ())) != ('per_person_v1', True):
        raise ValueError('{}: expected strict person-0 alignment'.format(path))
    if saved.get('feeder') != 'feeder.feeder_ntu.Feeder':
        raise ValueError('{}: expected the NTU training feeder'.format(path))
    model_args = saved.get('model_args')
    if not isinstance(model_args, dict) or model_args.get('one_person') is not True:
        raise ValueError('{}: missing person-0 model arguments'.format(path))
    expected = {name: (norm, weight) for name, norm, weight in
                GROUPS + (OPTIONAL_GROUP,)}
    norm, weight = expected[group]
    if (model_args.get('text_target_mode') != 'fixed_clip'
            or saved.get('text_target_mode') != 'fixed_clip'
            or model_args.get('text_target_norm', 'none') != norm
            or saved.get('text_target_norm', norm) != norm):
        raise ValueError('{}: wrong fixed-CLIP target mode for {}'.format(path, group))
    weights = saved.get('text_training_weights')
    if (weights is None or len(weights) != 2 or float(weights[0]) != 0.
            or not np.isclose(float(weights[1]), weight, rtol=0, atol=1e-8)):
        raise ValueError('{}: wrong S->T weight for {}'.format(path, group))
    if saved.get('text_share_skeleton_decoder') is not False:
        raise ValueError('{}: unexpected decoder sharing'.format(path))
    if checkpoint.get('epoch') is None or 'model' not in checkpoint:
        raise ValueError('{}: missing model or epoch'.format(path))
    return checkpoint, saved


def common_setup(saved):
    """Configuration that must agree across the three groups and all epochs."""
    model_args = dict(saved['model_args'])
    model_args.pop('text_target_norm', None)
    model_args.pop('lambda_text_to_skeleton', None)
    model_args.pop('lambda_skeleton_to_text', None)
    return {
        'model_args': model_args,
        'train_feeder_args': saved['train_feeder_args'],
        'text_cache_identity': saved['text_cache_identity'],
        'mask_ratio': saved['mask_ratio'],
        'motion_aware_tau': saved.get('motion_aware_tau', -1),
    }


def construct_model(checkpoint, saved, device):
    from model.transformer_macdiff_text import Transformer

    options = dict(saved['model_args'])
    options['text_target_norm'] = saved.get(
        'text_target_norm', options.get('text_target_norm', 'none'))
    options['share_skeleton_decoder'] = saved['text_share_skeleton_decoder']
    options['lambda_text_to_skeleton'], options['lambda_skeleton_to_text'] = (
        saved['text_training_weights'])
    model = Transformer(**options)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False).to(device)


def prepare_batches(dataset, cache, samples, batch_size, seed):
    """Draw one fixed crop/augmentation per eligible person-0 sample."""
    eligible = np.flatnonzero(cache.mask[:, 0].any(axis=1))
    if len(eligible) < samples:
        raise ValueError('Not enough person-0 descriptions')
    chosen, empty = [], []
    for idx in np.random.RandomState(seed).permutation(eligible):
        idx = int(idx)
        seed_all((seed + idx) % (2 ** 32))
        source, aug, _, _ = dataset[idx]
        if not np.any(source[..., 0] != 0):
            empty.append(idx)
            continue
        chosen.append((idx, source[..., 0].transpose(1, 2, 0).copy(),
                       aug[..., 0].transpose(1, 2, 0).copy()))
        if len(chosen) == samples:
            break
    if len(chosen) < samples:
        raise ValueError('Not enough nonempty person-0 crops')
    batches = [chosen[i:i + batch_size] for i in range(0, len(chosen), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        if len(batches[-2]) > 2:
            batches[-1].insert(0, batches[-2].pop())
        else:
            batches[-2].extend(batches.pop())
    return batches, empty


def fixed_text_batch(model, cache, indices, device):
    data = {name: torch.from_numpy(value).to(device) for name, value in
            cache.get_batch(np.asarray(indices, dtype=np.int64), one_person=True).items()}
    model.validate_text_tokens(data['text_tokens'], data['text_token_mask'],
                               data['text_person_ids'], data['text_positions'])
    global_text = model.fixed_clip_target(data['text_features'])
    local_text = model.fixed_clip_target(data['text_tokens'], data['text_token_mask'])
    memory = torch.cat([global_text[:, None], local_text], dim=1)
    valid = torch.cat([torch.ones_like(data['text_token_mask'][:, :1]),
                       data['text_token_mask']], dim=1)
    return data, memory, valid


@torch.no_grad()
def skeleton_condition(model, raw, aug, mask_ratio, motion_tau, mask_seed):
    """Encode a fixed crop once per repeat, then reuse it across timesteps."""
    seed_all(mask_seed)
    center = raw.mean(dim=(1, 2), keepdim=True) if model.self_shift else None
    aug = model._normalize_sequence(aug, center)
    latent, pooled, _, _ = model.forward_encoder(
        aug, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_tau)
    skeleton = torch.cat([pooled, latent], dim=1)
    skeleton_mask = torch.ones(skeleton.shape[:2], dtype=torch.bool, device=raw.device)
    return skeleton, skeleton_mask


@torch.no_grad()
def paired_trial(model, raw, aug, data, memory, valid, timestep, mask_ratio,
                 motion_tau, permutation, mask_seed, noise_seed,
                 skeleton=None, skeleton_mask=None):
    """Return per-sample losses with identical noisy text in both conditions."""
    if skeleton is None:
        skeleton, skeleton_mask = skeleton_condition(
            model, raw, aug, mask_ratio, motion_tau, mask_seed)
    permutation = torch.as_tensor(permutation, dtype=torch.long, device=raw.device)
    if sorted(permutation.cpu().tolist()) != list(range(len(raw))):
        raise ValueError('Skeleton permutation is not a bijection')
    t = torch.full((len(raw),), timestep, dtype=torch.long, device=raw.device)
    seed_all(noise_seed)
    noise = torch.randn_like(memory).masked_fill(~valid[..., None], 0)
    noisy = model.diffusion.q_sample(memory, t, noise=noise)
    predictions = [model.text_noise_decoder(
        noisy, t, condition, skeleton_mask, valid,
        data['text_person_ids'], data['text_positions'])
        for condition in (skeleton, skeleton[permutation])]
    correct = text_losses(predictions[0], noise, valid)
    shuffled = text_losses(predictions[1], noise, valid)
    change = text_losses(predictions[0], predictions[1], valid)
    return {name: (correct[name][0].cpu().numpy(), shuffled[name][0].cpu().numpy(),
                   correct[name][1].cpu().numpy(), change[name][0].cpu().numpy())
            for name in correct}


def evaluate_checkpoint(model, saved, cache, dataset, batches, timesteps,
                        repeats, seed, device):
    if not all(0 <= t < model.diffusion.num_timesteps for t in timesteps):
        raise ValueError('Timesteps outside checkpoint diffusion schedule')
    records = []
    for batch_number, batch in enumerate(batches):
        indices = [item[0] for item in batch]
        raw = torch.as_tensor(np.stack([item[1] for item in batch]),
                              dtype=torch.float32, device=device)
        aug = torch.as_tensor(np.stack([item[2] for item in batch]),
                              dtype=torch.float32, device=device)
        data, memory, valid = fixed_text_batch(model, cache, indices, device)
        for repeat in range(repeats):
            trial_seed = (seed + batch_number * 1009 + repeat * 9176) % (2 ** 32)
            permutation = derangement(len(batch), np.random.RandomState(trial_seed))
            skeleton, skeleton_mask = skeleton_condition(
                model, raw, aug, saved['mask_ratio'],
                saved.get('motion_aware_tau', -1), trial_seed)
            for timestep in timesteps:
                values = paired_trial(
                    model, raw, aug, data, memory, valid, timestep,
                    saved['mask_ratio'], saved.get('motion_aware_tau', -1),
                    permutation, trial_seed, (trial_seed + timestep * 7919) % (2 ** 32),
                    skeleton=skeleton, skeleton_mask=skeleton_mask)
                for name, (correct, shuffled, weights, change) in values.items():
                    for j, idx in enumerate(indices):
                        donor = indices[permutation[j]]
                        records.append({
                            'test': name, 't': timestep, 'batch': batch_number,
                            'repeat': repeat, 'sample': idx, 'donor': donor,
                            'same_label': bool(dataset.label[idx] == dataset.label[donor]),
                            'tokens': int(valid[j].sum().item()),
                            'correct': float(correct[j]), 'shuffled': float(shuffled[j]),
                            'weight': float(weights[j]), 'prediction_change': float(change[j]),
                        })
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-dir', required=True)
    parser.add_argument('--st01-dir', required=True)
    parser.add_argument('--rms-dir', required=True)
    parser.add_argument('--rms-st01-dir', default=None)
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 100, 200, 300, 399])
    parser.add_argument('--text-cache', default=None)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[100, 500, 900])
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-text-cache-validation', action='store_true')
    args = parser.parse_args(argv)
    if (args.samples < 3 or args.batch_size < 2 or args.repeats < 1
            or args.seed < 0 or not args.epochs or len(set(args.epochs)) != len(args.epochs)
            or min(args.epochs) < 0 or not args.timesteps
            or len(set(args.timesteps)) != len(args.timesteps)):
        parser.error('Invalid samples, batch size, repeats, seed, epochs or timesteps')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for a small smoke test')
    return args


def main(argv=None):
    args = parse_args(argv)
    directories = {'fixed': Path(args.fixed_dir), 'st01': Path(args.st01_dir),
                   'rms': Path(args.rms_dir)}
    groups = GROUPS
    if args.rms_st01_dir:
        directories['rms_st01'] = Path(args.rms_st01_dir)
        groups += (OPTIONAL_GROUP,)
    paths = {(group, epoch): directory / ('checkpoint-{}.pth'.format(epoch))
             for group, directory in directories.items() for epoch in args.epochs}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoints:\n' + '\n'.join(missing))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Output directory already exists: ' + str(output))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # One checkpoint determines the shared data protocol. Each later checkpoint
    # is strictly checked before use, including saved mode/weight and state keys.
    first_path = paths[('fixed', args.epochs[0])]
    checkpoint, saved = checkpoint_args(first_path, 'fixed')
    reference = common_setup(saved)
    del checkpoint
    from feeder.feeder_ntu import Feeder
    from util.person_text_cache import load_person_token_cache

    feeder_args = dict(saved['train_feeder_args'])
    if feeder_args.get('split') != 'train':
        raise ValueError('Expected the training split')
    if args.data_path:
        feeder_args['data_path'] = args.data_path
    dataset = Feeder(**feeder_args)
    cache_path = args.text_cache or saved['text_cache']
    print('Loading existing CLIP cache...', flush=True)
    cache = load_person_token_cache(
        cache_path, dataset.data_path, expected_count=len(dataset),
        skip_full_validation=args.skip_text_cache_validation)
    identity = {key: cache.manifest[key] for key in ('protocol', 'identity', 'files')}
    if identity != reference['text_cache_identity']:
        raise ValueError('Checkpoint/cache identity mismatch')
    batches, empty = prepare_batches(dataset, cache, args.samples, args.batch_size, args.seed)
    output.mkdir(parents=True)
    report = {
        'protocol': 'fixed_clip_skeleton_condition_comparison_v1',
        'arguments': vars(args),
        'sample_indices': [int(item[0]) for batch in batches for item in batch],
        'empty_person0_crops_skipped': empty,
        'cache_identity_sha256': hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest(),
        'cache_validation': 'headers_and_saved_manifest' if args.skip_text_cache_validation else 'full',
        'scope': 'Fixed train crops; FP32/eval; skeleton-only derangement within each batch; identical text target/noise/t/positions within each pair. Mask and noise seeds shared across checkpoints.',
        'interpretation': 'Positive shuffled-minus-correct MSE means this checkpoint uses paired skeleton information for text denoising. It does not establish that the information helps linear-probe accuracy.',
        'torch_version': torch.__version__,
        'results': [],
    }
    all_records = []
    for group, _, _ in groups:
        for epoch in args.epochs:
            path = paths[(group, epoch)]
            checkpoint, saved = checkpoint_args(path, group)
            if int(checkpoint['epoch']) != epoch:
                raise ValueError('{}: saved epoch does not match filename'.format(path))
            if common_setup(saved) != reference:
                raise ValueError('{}: data/model protocol differs from first checkpoint'.format(path))
            seed_all(args.seed)
            model = construct_model(checkpoint, saved, args.device)
            del checkpoint
            if 'timestep_alpha_cumprod' not in report:
                report['timestep_alpha_cumprod'] = {
                    str(t): float(model.diffusion.alphas_cumprod[t]) for t in args.timesteps}
            print('Evaluating {} epoch {}...'.format(group, epoch), flush=True)
            records = evaluate_checkpoint(model, saved, cache, dataset, batches,
                                          args.timesteps, args.repeats, args.seed, args.device)
            for row in records:
                row['group'] = group
                row['epoch'] = epoch
            all_records.extend(records)
            rows = summarize_pairs(records, args.seed)
            for row in rows:
                row['group'] = group
                row['epoch'] = epoch
                row['checkpoint'] = str(path)
                row['target_norm'] = model.text_target_norm
                row['s2t_weight'] = model.lambda_skeleton_to_text
                report['results'].append(row)
                print('{} e={:3d} {:10s} t={:3d} correct={:.6f} shuffled={:.6f} delta={:+.6f} ({:+.2f}%) CI={}'.format(
                    group, epoch, row['test'], row['t'], row['correct_mse'],
                    row['shuffled_mse'], row['delta'], row['relative_increase_percent'],
                    row['delta_cluster_bootstrap_95ci']), flush=True)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    with (output / 'paired_records.jsonl').open('w', encoding='utf-8') as stream:
        for row in all_records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    columns = ('epoch', 't', 'test', 'group', 'target_norm', 's2t_weight',
               'samples', 'matched_batch_clusters', 'correct_mse',
               'shuffled_mse', 'delta', 'relative_increase_percent',
               'ci_low', 'ci_high', 'prediction_change_mse',
               'same_label_donor_fraction', 'checkpoint')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(report['results'], key=lambda item:
                          (item['epoch'], item['t'], item['test'], item['group'])):
            flat = dict(row)
            ci = flat.pop('delta_cluster_bootstrap_95ci')
            flat['ci_low'], flat['ci_high'] = (ci if ci is not None else (None, None))
            writer.writerow({key: flat[key] for key in columns})
    (output / 'summary.json').write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved ' + str(output / 'summary.json'), flush=True)

if __name__ == '__main__':
    main()
