"""Compare frozen Stage1 and Stage2 geometry on identical NTU samples/masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from diagnose_stage1_geometry import (
    _DiagnosticDataset,
    _encode_exemplars,
    _extract_query_features,
    _mean_std,
    _pair_cosine,
    _prototype_metrics,
    _select_balanced_indices,
)
from main_pretrain_stage2 import import_class, load_or_create_exemplars
from model.transformer_stage2 import MacDiffStage2, transfer_macdiff_stage1


def _reset_mask_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))


def _unwrap_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a dictionary')
    state = checkpoint.get('model', checkpoint)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint model state must be a dictionary')
    if state and all(str(name).startswith('module.') for name in state):
        state = {
            str(name)[len('module.'):]: value
            for name, value in state.items()
        }
    return state


def _load_stage1_encoder(model_args, path):
    container = MacDiffStage2(**model_args)
    checkpoint = torch.load(path, map_location='cpu')
    report = transfer_macdiff_stage1(container, checkpoint)
    encoder = container.encoder_q
    del container
    del checkpoint
    return encoder, report


def _load_stage2_modules(model_args, path):
    container = MacDiffStage2(**model_args)
    checkpoint = torch.load(path, map_location='cpu')
    state = _unwrap_state(checkpoint)
    encoder_keys = set(container.encoder_q.state_dict())
    state_keys = set(state)
    complete = any(name.startswith('encoder_q.') for name in state_keys)
    if complete:
        container.load_state_dict(state, strict=True)
    elif state_keys == encoder_keys:
        container.encoder_q.load_state_dict(state, strict=True)
    else:
        missing = sorted(encoder_keys - state_keys)[:5]
        unexpected = sorted(state_keys - encoder_keys)[:5]
        raise ValueError(
            'Unrecognized Stage2 checkpoint state; missing encoder keys {}, '
            'unexpected keys {}'.format(missing, unexpected))

    metadata = checkpoint.get('args', {})
    modules = {
        'encoder_q': container.encoder_q,
        'encoder_k': container.encoder_k if complete else None,
        'ose_projector_q': (
            container.ose_online_projector if complete else None),
        'ose_projector_k': (
            container.ose_teacher_projector if complete else None),
    }
    del container
    del checkpoint
    return modules, complete, metadata


def _parameter_group(name):
    components = name.split('.')
    if components[0] == 'blocks' and len(components) > 1:
        return 'blocks.{}'.format(components[1])
    return components[0]


@torch.no_grad()
def _parameter_drift(stage1, stage2):
    first = dict(stage1.named_parameters())
    second = dict(stage2.named_parameters())
    if set(first) != set(second):
        raise ValueError('Stage1 and Stage2 encoder parameter keys differ')
    accumulators = {
        'all': {'difference': 0.0, 'reference': 0.0,
                'second': 0.0, 'dot': 0.0}
    }
    for name in sorted(first):
        group = _parameter_group(name)
        accumulators.setdefault(
            group, {'difference': 0.0, 'reference': 0.0,
                    'second': 0.0, 'dot': 0.0})
        reference = first[name].detach().double().view(-1)
        updated = second[name].detach().double().view(-1)
        values = {
            'difference': float((updated - reference).square().sum().item()),
            'reference': float(reference.square().sum().item()),
            'second': float(updated.square().sum().item()),
            'dot': float(torch.dot(reference, updated).item()),
        }
        for target in (accumulators['all'], accumulators[group]):
            for key, value in values.items():
                target[key] += value

    result = {}
    for group, values in accumulators.items():
        denominator = max(values['reference'], 1e-24)
        cosine_denominator = max(
            (values['reference'] * values['second']) ** 0.5, 1e-24)
        result[group] = {
            'relative_l2': float(
                (values['difference'] / denominator) ** 0.5),
            'parameter_cosine': float(
                values['dot'] / cosine_denominator),
        }
    return result


def _geometry_metrics(queries, full_prototypes, masked_prototypes,
                      class_ids, pair_count, seed, device):
    return {
        'full_masked_cosine': queries['full_masked_cosine'],
        'masked_masked_cosine': queries['masked_masked_cosine'],
        'mask_visibility': {
            'mean_missing_joints': queries['mean_missing_joints'],
            'fraction_with_missing_joint': (
                queries['fraction_with_missing_joint']),
        },
        'full_feature_cosine': _pair_cosine(
            queries['full'], queries['labels'], pair_count,
            seed + 11, device),
        'masked_feature_cosine': _pair_cosine(
            queries['masked'], queries['labels'], pair_count,
            seed + 17, device),
        'full_exemplar_prototypes': _prototype_metrics(
            queries['full'], queries['labels'], full_prototypes,
            class_ids, (0.04, 0.06, 0.1), device),
        'masked_k2_exemplar_prototypes': _prototype_metrics(
            queries['masked'], queries['labels'], masked_prototypes,
            class_ids, (0.04, 0.06, 0.1), device),
    }


@torch.no_grad()
def _paired_cosine(first, second):
    values = (first.float() * second.float()).sum(dim=1)
    return _mean_std(
        float(values.sum().item()),
        float(values.square().sum().item()),
        int(values.numel()),
    )


@torch.no_grad()
def _linear_cka(first, second, maximum, seed, device):
    sample_count = min(int(maximum), first.shape[0])
    rng = np.random.RandomState(seed)
    indices = rng.choice(first.shape[0], sample_count, replace=False)
    indices = torch.from_numpy(indices).long()
    first = first[indices].to(device).float()
    second = second[indices].to(device).float()
    first_gram = torch.matmul(first, first.t())
    second_gram = torch.matmul(second, second.t())
    first_gram = (
        first_gram
        - first_gram.mean(dim=0, keepdim=True)
        - first_gram.mean(dim=1, keepdim=True)
        + first_gram.mean())
    second_gram = (
        second_gram
        - second_gram.mean(dim=0, keepdim=True)
        - second_gram.mean(dim=1, keepdim=True)
        + second_gram.mean())
    numerator = (first_gram * second_gram).sum()
    denominator = (
        first_gram.square().sum().sqrt()
        * second_gram.square().sum().sqrt()).clamp_min(1e-12)
    return {
        'linear_cka': float((numerator / denominator).item()),
        'samples': int(sample_count),
    }


@torch.no_grad()
def _encode_projected_prototypes(encoder, projector, dataset, indices,
                                 batch_size, device):
    loader = DataLoader(
        _DiagnosticDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    views = [[], []]
    for samples, _, _ in loader:
        samples = samples.to(device, non_blocking=True).float()
        for repeat in range(2):
            mask_indices = encoder.sample_mask_indices(samples)
            raw = encoder.forward_features(
                samples, mask_indices=mask_indices)
            views[repeat].append(F.normalize(projector(raw), dim=1).cpu())
    stacked = torch.stack([
        torch.cat(views[0]),
        torch.cat(views[1]),
    ], dim=1)
    return F.normalize(stacked.mean(dim=1), dim=1)


@torch.no_grad()
def _extract_projected_queries(encoder, projector, loader, device,
                               mask_repeats):
    features = []
    labels = []
    for samples, batch_labels, _ in loader:
        samples = samples.to(device, non_blocking=True).float()
        mask_indices = None
        for repeat in range(mask_repeats):
            candidate = encoder.sample_mask_indices(samples)
            if repeat == 0:
                mask_indices = candidate
        raw = encoder.forward_features(
            samples, mask_indices=mask_indices)
        features.append(F.normalize(projector(raw), dim=1).cpu())
        labels.append(batch_labels.long().cpu())
    return torch.cat(features), torch.cat(labels)


def _selected_checkpoint_args(metadata):
    if not isinstance(metadata, dict):
        return {}
    names = (
        'lr', 'head_lr', 'resa_weight', 'ose_lambda',
        'ose_mix_proto_weight', 'ose_mix_ins_weight',
        'ose_tau_s', 'ose_tau_t', 'mask_protocol',
    )
    return {name: metadata[name] for name in names if name in metadata}


def get_parser():
    parser = argparse.ArgumentParser('Compare MacDiff Stage1/Stage2 geometry')
    parser.add_argument(
        '--stage1_checkpoint',
        default='./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth')
    parser.add_argument(
        '--stage2_checkpoint',
        default=(
            './output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_'
            'syncbn_lr1e3/checkpoint-100.pth'))
    parser.add_argument(
        '--config',
        default='./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--exemplar_index_path', default='')
    parser.add_argument(
        '--output', default='./output_dir/stage1_vs_stage2_geometry.json')
    parser.add_argument('--max_samples', default=4096, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--mask_repeats', default=2, type=int)
    parser.add_argument('--pair_count', default=10000, type=int)
    parser.add_argument('--cka_samples', default=512, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    return parser


def main(args):
    stage1_path = Path(args.stage1_checkpoint)
    stage2_path = Path(args.stage2_checkpoint)
    config_path = Path(args.config)
    for description, path in (
            ('Stage1 checkpoint', stage1_path),
            ('Stage2 checkpoint', stage2_path),
            ('Stage2 config', config_path)):
        if not path.is_file():
            raise FileNotFoundError('{} does not exist: {}'.format(
                description, path))
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('Invalid batch_size or num_workers')
    if args.mask_repeats < 2 or args.pair_count <= 0:
        raise ValueError('mask_repeats must be >=2 and pair_count positive')
    if args.cka_samples <= 1:
        raise ValueError('cka_samples must be greater than one')

    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    feeder_args = dict(config['train_feeder_args'])
    if args.data_path:
        feeder_args['data_path'] = args.data_path
    data_path = Path(feeder_args['data_path'])
    if not data_path.is_file():
        raise FileNotFoundError('NTU data does not exist: {}'.format(
            data_path))

    np.random.seed(args.seed)
    _reset_mask_seed(args.seed)
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
        dataset.label, exemplar_indices, args.max_samples, args.seed)
    loader = DataLoader(
        _DiagnosticDataset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    stage1_encoder, stage1_report = _load_stage1_encoder(
        config['model_args'], stage1_path)
    stage2_modules, complete_stage2, stage2_metadata = _load_stage2_modules(
        config['model_args'], stage2_path)
    stage2_encoder = stage2_modules['encoder_q']
    drift = _parameter_drift(stage1_encoder, stage2_encoder)
    print('Loaded Stage1 encoder: {}'.format(stage1_report))
    print('Loaded {} Stage2 checkpoint'.format(
        'complete' if complete_stage2 else 'backbone-only'))
    print('Comparison samples: {}'.format(len(selected)))

    query_seed = args.seed + 1000
    exemplar_seed = args.seed + 2000

    stage1_encoder = stage1_encoder.to(device).eval()
    _reset_mask_seed(query_seed)
    stage1_queries = _extract_query_features(
        stage1_encoder, loader, device, args.mask_repeats)
    _reset_mask_seed(exemplar_seed)
    stage1_full_prototypes, stage1_masked_prototypes = _encode_exemplars(
        stage1_encoder, dataset, exemplar_indices,
        args.batch_size, device)
    stage1_encoder = stage1_encoder.cpu()

    stage2_encoder = stage2_encoder.to(device).eval()
    _reset_mask_seed(query_seed)
    stage2_queries = _extract_query_features(
        stage2_encoder, loader, device, args.mask_repeats)
    _reset_mask_seed(exemplar_seed)
    stage2_full_prototypes, stage2_masked_prototypes = _encode_exemplars(
        stage2_encoder, dataset, exemplar_indices,
        args.batch_size, device)
    stage2_encoder = stage2_encoder.cpu()

    if not torch.equal(stage1_queries['labels'], stage2_queries['labels']):
        raise RuntimeError('Stage1/Stage2 diagnostic sample order differs')

    stage1_metrics = _geometry_metrics(
        stage1_queries, stage1_full_prototypes,
        stage1_masked_prototypes, class_ids,
        args.pair_count, args.seed, device)
    stage2_metrics = _geometry_metrics(
        stage2_queries, stage2_full_prototypes,
        stage2_masked_prototypes, class_ids,
        args.pair_count, args.seed, device)
    results = {
        'stage1_checkpoint': str(stage1_path),
        'stage2_checkpoint': str(stage2_path),
        'stage2_checkpoint_type': (
            'complete' if complete_stage2 else 'backbone-only'),
        'stage2_checkpoint_args': _selected_checkpoint_args(stage2_metadata),
        'config': str(config_path),
        'samples': len(selected),
        'feature_dim': int(stage1_queries['full'].shape[1]),
        'same_samples_and_masks': True,
        'prototype_space': (
            'normalized frozen encoder outputs; no projector'),
        'stage1': stage1_metrics,
        'stage2': stage2_metrics,
        'stage2_minus_stage1': {
            'full_masked_cosine_mean': float(
                stage2_metrics['full_masked_cosine']['mean']
                - stage1_metrics['full_masked_cosine']['mean']),
            'masked_masked_cosine_mean': float(
                stage2_metrics['masked_masked_cosine']['mean']
                - stage1_metrics['masked_masked_cosine']['mean']),
            'full_same_different_gap': float(
                stage2_metrics['full_feature_cosine']['mean_gap']
                - stage1_metrics['full_feature_cosine']['mean_gap']),
            'masked_same_different_gap': float(
                stage2_metrics['masked_feature_cosine']['mean_gap']
                - stage1_metrics['masked_feature_cosine']['mean_gap']),
            'full_exemplar_top1_accuracy': float(
                stage2_metrics['full_exemplar_prototypes']['top1_accuracy']
                - stage1_metrics['full_exemplar_prototypes'][
                    'top1_accuracy']),
            'masked_k2_exemplar_top1_accuracy': float(
                stage2_metrics['masked_k2_exemplar_prototypes'][
                    'top1_accuracy']
                - stage1_metrics['masked_k2_exemplar_prototypes'][
                    'top1_accuracy']),
        },
        'stage1_stage2_alignment': {
            'full_feature_cosine': _paired_cosine(
                stage1_queries['full'], stage2_queries['full']),
            'masked_feature_cosine': _paired_cosine(
                stage1_queries['masked'], stage2_queries['masked']),
            'full_feature_cka': _linear_cka(
                stage1_queries['full'], stage2_queries['full'],
                args.cka_samples, args.seed + 31, device),
            'masked_feature_cka': _linear_cka(
                stage1_queries['masked'], stage2_queries['masked'],
                args.cka_samples, args.seed + 37, device),
        },
        'encoder_parameter_drift': drift,
    }

    if complete_stage2:
        online_encoder = stage2_modules['encoder_q'].to(device).eval()
        online_projector = (
            stage2_modules['ose_projector_q'].to(device).eval())
        _reset_mask_seed(exemplar_seed)
        projected_prototypes = _encode_projected_prototypes(
            online_encoder, online_projector, dataset, exemplar_indices,
            args.batch_size, device)
        online_encoder = online_encoder.cpu()
        online_projector = online_projector.cpu()

        teacher_encoder = stage2_modules['encoder_k'].to(device).eval()
        teacher_projector = (
            stage2_modules['ose_projector_k'].to(device).eval())
        _reset_mask_seed(query_seed)
        teacher_features, teacher_labels = _extract_projected_queries(
            teacher_encoder, teacher_projector, loader, device,
            args.mask_repeats)
        teacher_encoder = teacher_encoder.cpu()
        teacher_projector = teacher_projector.cpu()
        results['stage2_ose_teacher_eval_bn'] = _prototype_metrics(
            teacher_features, teacher_labels, projected_prototypes,
            class_ids, (0.04, 0.06, 0.1), device)
        results['stage2_ose_teacher_eval_bn']['note'] = (
            'EMA encoder/projector queries versus online K2 prototypes; '
            'projector BatchNorm uses checkpoint running statistics')
    else:
        results['stage2_ose_teacher_eval_bn'] = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print('Saved comparison to {}'.format(output_path))


if __name__ == '__main__':
    main(get_parser().parse_args())
