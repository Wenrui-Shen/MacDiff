"""Diagnose whether frozen MacDiff Stage1 features fit the Stage2 geometry.

The script never updates model weights.  It compares full-token and masked
features, measures classwise cosine separation, and evaluates the exact
Stage2 exemplar selection as nearest class prototypes in the frozen encoder
space.  These are diagnostics only; labels are never used by Stage2 training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from main_pretrain_stage2 import (
    import_class,
    load_or_create_exemplars,
)
from model.transformer_stage2 import (
    MacDiffStage2,
    transfer_macdiff_stage1,
)


class _DiagnosticDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        sample = torch.from_numpy(self.dataset.get_base_sample(index))
        return sample, int(self.dataset.label[index]), index


def _mean_std(total, total_squared, count):
    if count <= 0:
        return {'mean': float('nan'), 'std': float('nan'), 'count': 0}
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    return {
        'mean': float(mean),
        'std': float(variance ** 0.5),
        'count': int(count),
    }


def _select_balanced_indices(labels, excluded, maximum, seed):
    excluded = {int(index) for index in excluded}
    buckets = {}
    for index, label in enumerate(labels):
        if index not in excluded:
            buckets.setdefault(int(label), []).append(int(index))
    rng = np.random.RandomState(seed)
    for values in buckets.values():
        rng.shuffle(values)
    available = sum(len(values) for values in buckets.values())
    target = available if maximum <= 0 else min(int(maximum), available)
    selected = []
    positions = {label: 0 for label in buckets}
    classes = sorted(buckets)
    while len(selected) < target:
        added = False
        for label in classes:
            position = positions[label]
            if position < len(buckets[label]):
                selected.append(buckets[label][position])
                positions[label] += 1
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
    rng.shuffle(selected)
    return selected


@torch.no_grad()
def _forward_full(encoder, samples):
    mask_ratio = encoder.mask_ratio
    encoder.mask_ratio = 0.0
    try:
        return encoder.forward_features(samples)
    finally:
        encoder.mask_ratio = mask_ratio


@torch.no_grad()
def _forward_masked(encoder, samples):
    mask_indices = encoder.sample_mask_indices(samples)
    features = encoder.forward_features(
        samples, mask_indices=mask_indices)
    return features, mask_indices


def _accumulate_values(values, state):
    values = values.float()
    state['sum'] += float(values.sum().item())
    state['sum_squared'] += float(values.square().sum().item())
    state['count'] += int(values.numel())


@torch.no_grad()
def _extract_query_features(encoder, loader, device, mask_repeats):
    full_features = []
    masked_features = []
    labels = []
    full_masked = {'sum': 0.0, 'sum_squared': 0.0, 'count': 0}
    masked_masked = {'sum': 0.0, 'sum_squared': 0.0, 'count': 0}
    missing_joints = 0
    samples_with_missing = 0
    mask_samples = 0
    joint_count = int(encoder.num_joint_patches)

    for samples, batch_labels, _ in loader:
        samples = samples.to(device, non_blocking=True).float()
        full = F.normalize(_forward_full(encoder, samples), dim=1)
        first_masked = None
        for repeat in range(mask_repeats):
            masked, mask_indices = _forward_masked(encoder, samples)
            masked = F.normalize(masked, dim=1)
            _accumulate_values((full * masked).sum(dim=1), full_masked)
            if first_masked is None:
                first_masked = masked
            else:
                _accumulate_values(
                    (first_masked * masked).sum(dim=1), masked_masked)

            joint_ids = mask_indices.remainder(joint_count)
            counts = torch.zeros(
                mask_indices.shape[0], joint_count,
                device=mask_indices.device, dtype=torch.long)
            counts.scatter_add_(
                1, joint_ids, torch.ones_like(joint_ids))
            missing = counts.eq(0)
            missing_joints += int(missing.sum().item())
            samples_with_missing += int(missing.any(dim=1).sum().item())
            mask_samples += int(mask_indices.shape[0])

        full_features.append(full.cpu())
        masked_features.append(first_masked.cpu())
        labels.append(batch_labels.long().cpu())

    return {
        'full': torch.cat(full_features),
        'masked': torch.cat(masked_features),
        'labels': torch.cat(labels),
        'full_masked_cosine': _mean_std(
            full_masked['sum'], full_masked['sum_squared'],
            full_masked['count']),
        'masked_masked_cosine': _mean_std(
            masked_masked['sum'], masked_masked['sum_squared'],
            masked_masked['count']),
        'mean_missing_joints': (
            float(missing_joints) / max(mask_samples, 1)),
        'fraction_with_missing_joint': (
            float(samples_with_missing) / max(mask_samples, 1)),
    }


def _draw_pairs(labels, count, same_class, rng):
    labels = np.asarray(labels, dtype=np.int64)
    buckets = {
        int(label): np.flatnonzero(labels == label)
        for label in np.unique(labels)
    }
    classes = sorted(buckets)
    if same_class:
        classes = [label for label in classes if len(buckets[label]) >= 2]
        if not classes:
            raise ValueError('No class has two diagnostic samples')
    if not same_class and len(classes) < 2:
        raise ValueError('Need at least two classes for different-class pairs')
    left = np.empty(count, dtype=np.int64)
    right = np.empty(count, dtype=np.int64)
    for position in range(count):
        first_class = classes[rng.randint(len(classes))]
        if same_class:
            pair = rng.choice(buckets[first_class], size=2, replace=False)
            left[position], right[position] = pair
        else:
            second_offset = rng.randint(1, len(classes))
            first_position = classes.index(first_class)
            second_class = classes[
                (first_position + second_offset) % len(classes)]
            left[position] = rng.choice(buckets[first_class])
            right[position] = rng.choice(buckets[second_class])
    return torch.from_numpy(left), torch.from_numpy(right)


@torch.no_grad()
def _pair_cosine(features, labels, pair_count, seed, device):
    rng = np.random.RandomState(seed)
    same_left, same_right = _draw_pairs(
        labels.numpy(), pair_count, True, rng)
    diff_left, diff_right = _draw_pairs(
        labels.numpy(), pair_count, False, rng)

    def calculate(left, right):
        state = {'sum': 0.0, 'sum_squared': 0.0, 'count': 0}
        chunk_size = 1024
        for start in range(0, left.numel(), chunk_size):
            end = min(start + chunk_size, left.numel())
            left_features = features[left[start:end]].to(device)
            right_features = features[right[start:end]].to(device)
            _accumulate_values(
                (left_features * right_features).sum(dim=1), state)
        return _mean_std(
            state['sum'], state['sum_squared'], state['count'])

    same = calculate(same_left, same_right)
    different = calculate(diff_left, diff_right)
    return {
        'same_class': same,
        'different_class': different,
        'mean_gap': float(same['mean'] - different['mean']),
    }


@torch.no_grad()
def _encode_exemplars(encoder, dataset, indices, batch_size, device):
    loader = DataLoader(
        _DiagnosticDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    full_views = []
    masked_views = [[], []]
    for samples, _, _ in loader:
        samples = samples.to(device, non_blocking=True).float()
        full_views.append(F.normalize(
            _forward_full(encoder, samples), dim=1).cpu())
        for repeat in range(2):
            masked, _ = _forward_masked(encoder, samples)
            masked_views[repeat].append(F.normalize(masked, dim=1).cpu())
    full_prototypes = torch.cat(full_views)
    stacked_masked = torch.stack([
        torch.cat(masked_views[0]),
        torch.cat(masked_views[1]),
    ], dim=1)
    masked_prototypes = F.normalize(stacked_masked.mean(dim=1), dim=1)
    return full_prototypes, masked_prototypes


@torch.no_grad()
def _prototype_metrics(features, labels, prototypes, class_ids,
                       temperatures, device):
    prototypes = prototypes.to(device)
    class_ids = torch.as_tensor(class_ids, device=device, dtype=torch.long)
    correct = 0
    total = 0
    temperature_stats = {
        str(temperature): {'confidence': 0.0, 'entropy': 0.0}
        for temperature in temperatures
    }
    chunk_size = 512
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        batch_features = features[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        scores = torch.matmul(batch_features, prototypes.t())
        predictions = class_ids[scores.argmax(dim=1)]
        correct += int(predictions.eq(batch_labels).sum().item())
        total += int(batch_labels.numel())
        for temperature in temperatures:
            probabilities = torch.softmax(scores / temperature, dim=1)
            confidence = probabilities.max(dim=1).values
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=1)
            key = str(temperature)
            temperature_stats[key]['confidence'] += float(
                confidence.sum().item())
            temperature_stats[key]['entropy'] += float(entropy.sum().item())
    for values in temperature_stats.values():
        values['mean_top1_confidence'] = values.pop('confidence') / total
        values['mean_entropy'] = values.pop('entropy') / total
    return {
        'top1_accuracy': 100.0 * correct / total,
        'samples': total,
        'temperature_stats': temperature_stats,
    }


def get_parser():
    parser = argparse.ArgumentParser('MacDiff Stage1 geometry diagnostics')
    parser.add_argument(
        '--checkpoint',
        default='./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth')
    parser.add_argument(
        '--config',
        default='./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--exemplar_index_path', default='')
    parser.add_argument('--output', default='./output_dir/stage1_geometry.json')
    parser.add_argument('--max_samples', default=4096, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--mask_repeats', default=2, type=int)
    parser.add_argument('--pair_count', default=10000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    return parser


def main(args):
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Stage1 checkpoint does not exist: {}'.format(checkpoint_path))
    if not config_path.is_file():
        raise FileNotFoundError(
            'Stage2 config does not exist: {}'.format(config_path))
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('Invalid batch_size or num_workers')
    if args.mask_repeats < 2:
        raise ValueError('mask_repeats must be at least 2')
    if args.pair_count <= 0:
        raise ValueError('pair_count must be positive')

    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    feeder_args = dict(config['train_feeder_args'])
    if args.data_path:
        feeder_args['data_path'] = args.data_path
    data_path = Path(feeder_args['data_path'])
    if not data_path.is_file():
        raise FileNotFoundError(
            'NTU data does not exist: {}'.format(data_path))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = import_class(config['feeder'])(**feeder_args)
    exemplar_path = (
        args.exemplar_index_path or config['exemplar_index_path'])
    class_ids, exemplar_indices = load_or_create_exemplars(
        dataset,
        exemplar_path,
        seed=int(config['exemplar_seed']),
        num_classes=int(config['num_classes']),
    )
    selected = _select_balanced_indices(
        dataset.label,
        exemplar_indices,
        args.max_samples,
        args.seed,
    )
    loader = DataLoader(
        _DiagnosticDataset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Construct the transfer container on CPU, then retain only encoder_q.
    # Moving the four fresh 6400-D Stage2 projector branches to the diagnostic
    # GPU would waste substantial memory without affecting any metric here.
    model = MacDiffStage2(**config['model_args'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    transfer_report = transfer_macdiff_stage1(model, checkpoint)
    encoder = model.encoder_q
    del model
    del checkpoint
    encoder = encoder.to(device).eval()
    print('Loaded frozen Stage1 encoder: {}'.format(transfer_report))
    print('Diagnostic samples: {}'.format(len(selected)))

    queries = _extract_query_features(
        encoder, loader, device, args.mask_repeats)
    full_prototypes, masked_prototypes = _encode_exemplars(
        encoder,
        dataset,
        exemplar_indices,
        args.batch_size,
        device,
    )

    results = {
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
        'samples': len(selected),
        'feature_dim': int(queries['full'].shape[1]),
        'mask_ratio': float(encoder.mask_ratio),
        'mask_repeats': int(args.mask_repeats),
        'prototype_space': (
            'normalized frozen encoder outputs; no OSE projector'),
        'full_masked_cosine': queries['full_masked_cosine'],
        'masked_masked_cosine': queries['masked_masked_cosine'],
        'mask_visibility': {
            'mean_missing_joints': queries['mean_missing_joints'],
            'fraction_with_missing_joint': (
                queries['fraction_with_missing_joint']),
        },
        'full_feature_cosine': _pair_cosine(
            queries['full'], queries['labels'], args.pair_count,
            args.seed + 11, device),
        'masked_feature_cosine': _pair_cosine(
            queries['masked'], queries['labels'], args.pair_count,
            args.seed + 17, device),
        'full_exemplar_prototypes': _prototype_metrics(
            queries['full'], queries['labels'], full_prototypes,
            class_ids, (0.04, 0.06, 0.1), device),
        'masked_k2_exemplar_prototypes': _prototype_metrics(
            queries['masked'], queries['labels'], masked_prototypes,
            class_ids, (0.04, 0.06, 0.1), device),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print('Saved diagnostics to {}'.format(output_path))


if __name__ == '__main__':
    main(get_parser().parse_args())
