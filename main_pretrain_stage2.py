"""Independent MacDiff RSDG Stage2 training entry point.

This is intentionally separate from ``main_pretrain.py``.  A fresh run loads
only the Stage1 MacDiff skeleton encoder and creates new ReSA/OSE heads, EMA
branches, optimizer and schedule state.
"""

import argparse
import datetime
import json
import math
import os
import random
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from model.transformer_stage2 import (
    MacDiffDenseOSE,
    MacDiffStage2,
    transfer_macdiff_stage1,
)
from util.dense_ose_diagnostics import (
    DenseOSEJsonlLogger,
    assignment_distribution,
    dense_epoch_geometry,
    extract_dense_features,
    extract_stage1_reference,
    prototype_geometry,
    reference_geometry,
    select_balanced_indices,
)


TRAIN_METRICS = (
    'loss', 'cluster', 'cluster_entropy', 'cluster_kl', 'proto', 'align',
    'disp', 'mix_proto', 'mix_ins', 'ose_target_entropy',
    'ose_target_confidence', 'ose_prototype_usage', 'ema_momentum',
)

DENSE_DIAGNOSTIC_METRICS = (
    'ose_target_entropy_p10', 'ose_target_entropy_p50',
    'ose_target_entropy_p90', 'ose_target_confidence_p10',
    'ose_target_confidence_p50', 'ose_target_confidence_p90',
    'ose_teacher_accuracy', 'ose_student_accuracy',
    'ose_teacher_student_agreement',
    'ose_teacher_true_class_probability', 'ose_teacher_margin',
)

RESUME_CONTRACT_FIELDS = (
    'feeder', 'train_feeder_args', 'model_args', 'epochs', 'batch_size',
    'accum_iter', 'num_workers', 'max_train_steps', 'seed', 'enable_amp',
    'lr', 'head_lr',
    'final_lr', 'head_final_lr', 'weight_decay', 'optimizer_momentum',
    'nesterov', 'ema_momentum', 'exemplar_index_path', 'exemplar_seed',
    'num_classes', 'ose_exemplar_views', 'ose_exemplar_batch_size',
    'resa_weight', 'ose_lambda',
    'ose_mix_proto_weight', 'ose_mix_ins_weight', 'ose_mix_alpha',
    'ose_tau_s', 'ose_tau_t', 'mask_protocol', 'sync_batchnorm',
    'world_size',
)

MASK_PROTOCOL_STRATEGIES = {
    'shared_qk_joint_v1': 'global_random',
    'shared_qk_per_joint_v1': 'per_joint_random',
    'dense_ose_proto_ema_v1': 'global_random',
}

DENSE_OSE_PROTOCOL = 'dense_ose_proto_ema_v1'
DENSE_REFERENCE_CACHE = 'dense_ose_diagnostic_reference.pth'


def is_dense_ose_protocol(protocol):
    return str(protocol) == DENSE_OSE_PROTOCOL


def mask_strategy_from_protocol(protocol):
    try:
        return MASK_PROTOCOL_STRATEGIES[str(protocol)]
    except KeyError as error:
        raise ValueError(
            'Unsupported Stage2 mask_protocol: {}'.format(protocol)
        ) from error


def import_class(name):
    components = name.split('.')
    module = __import__(components[0])
    for component in components[1:]:
        module = getattr(module, component)
    return module


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in ('1', 'true', 'yes', 'y'):
        return True
    if lowered in ('0', 'false', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value')


def get_args_parser():
    parser = argparse.ArgumentParser('MacDiff RSDG Stage2')
    parser.add_argument(
        '--config',
        default='./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml')
    parser.add_argument('--stage1_weights', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--output_dir', default='./output_dir/stage2')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--feeder', default='feeder.feeder_stage2.FeederStage2')
    parser.add_argument('--train_feeder_args', default=dict())
    parser.add_argument('--model_args', default=dict())

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)
    parser.add_argument('--max_train_steps', default=0, type=int)
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--enable_amp', type=str2bool, default=False)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--sync_batchnorm', type=str2bool, default=True)

    parser.add_argument('--lr', default=0.25, type=float)
    parser.add_argument('--head_lr', default=0.25, type=float)
    parser.add_argument('--final_lr', default=0.0, type=float)
    parser.add_argument('--head_final_lr', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--optimizer_momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', type=str2bool, default=False)
    parser.add_argument('--ema_momentum', default=0.996, type=float)

    parser.add_argument('--exemplar_index_path', default='', type=str)
    parser.add_argument('--exemplar_seed', default=0, type=int)
    parser.add_argument('--num_classes', default=60, type=int)
    parser.add_argument('--ose_exemplar_views', default=2, type=int)
    parser.add_argument('--ose_exemplar_batch_size', default=0, type=int)
    parser.add_argument('--resa_weight', default=1.0, type=float)
    parser.add_argument('--ose_lambda', default=1.0, type=float)
    parser.add_argument('--ose_mix_proto_weight', default=1.0, type=float)
    parser.add_argument('--ose_mix_ins_weight', default=1.0, type=float)
    parser.add_argument('--ose_mix_alpha', default=1.0, type=float)
    parser.add_argument('--ose_tau_s', default=0.1, type=float)
    parser.add_argument('--ose_tau_t', default=0.04, type=float)
    parser.add_argument(
        '--mask_protocol', default='shared_qk_joint_v1', type=str)
    parser.add_argument(
        '--diagnostic_log_name', default='dense_ose_diagnostics.jsonl',
        type=str)
    parser.add_argument('--diagnostic_step_interval', default=20, type=int)
    parser.add_argument('--diagnostic_epoch_interval', default=5, type=int)
    parser.add_argument('--diagnostic_max_samples', default=600, type=int)
    parser.add_argument('--diagnostic_batch_size', default=4, type=int)
    parser.add_argument('--diagnostic_seed', default=17, type=int)

    return parser


def parse_args():
    initial = get_args_parser().parse_known_args()[0]
    parser = get_args_parser()
    if initial.config:
        with open(initial.config, 'r', encoding='utf-8') as handle:
            defaults = yaml.load(handle, Loader=yaml.FullLoader)
        unknown = sorted(set(defaults) - set(vars(initial)))
        if unknown:
            raise ValueError(
                'Unknown Stage2 config keys: {}'.format(unknown))
        parser.set_defaults(**defaults)
    return parser.parse_args()


def validate_args(args):
    if not args.stage1_weights and not args.resume:
        raise ValueError('A fresh Stage2 run requires --stage1_weights')
    if args.stage1_weights and not args.resume and not os.path.isfile(
            args.stage1_weights):
        raise FileNotFoundError(
            'Stage1 checkpoint does not exist: {}'.format(
                args.stage1_weights))
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(
            'Stage2 resume checkpoint does not exist: {}'.format(
                args.resume))
    if args.epochs <= 0 or args.batch_size <= 1:
        raise ValueError('Stage2 requires positive epochs and batch_size > 1')
    if args.accum_iter <= 0:
        raise ValueError('Stage2 accum_iter must be positive')
    if args.num_workers < 0 or args.max_train_steps < 0:
        raise ValueError('Worker and step limits must be non-negative')
    if args.world_size <= 0:
        raise ValueError('world_size must be positive')
    if args.save_interval <= 0:
        raise ValueError('save_interval must be positive')
    if args.lr <= 0 or args.head_lr <= 0:
        raise ValueError('Stage2 initial learning rates must be positive')
    if args.final_lr < 0 or args.head_final_lr < 0:
        raise ValueError('Stage2 final learning rates must be non-negative')
    if not 0.0 <= args.ema_momentum < 1.0:
        raise ValueError('ema_momentum must be in [0, 1)')
    if args.ose_mix_alpha <= 0:
        raise ValueError('ose_mix_alpha must be positive')
    if min(
            args.resa_weight, args.ose_lambda,
            args.ose_mix_proto_weight,
            args.ose_mix_ins_weight) < 0:
        raise ValueError('Stage2 loss weights must be non-negative')
    if max(
            args.resa_weight, args.ose_lambda,
            args.ose_mix_proto_weight,
            args.ose_mix_ins_weight) == 0:
        raise ValueError('Stage2 requires at least one positive loss weight')
    if args.num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if args.ose_exemplar_views < 1:
        raise ValueError('ose_exemplar_views must be at least 1')
    if args.ose_exemplar_batch_size < 0:
        raise ValueError('ose_exemplar_batch_size must be non-negative')
    mask_strategy = mask_strategy_from_protocol(args.mask_protocol)
    dense_ose = is_dense_ose_protocol(args.mask_protocol)
    expected_mask_ratio = 0.0 if dense_ose else 0.9
    if float(args.model_args.get('mask_ratio', -1.0)) != expected_mask_ratio:
        raise ValueError(
            '{} requires model_args.mask_ratio={}'.format(
                args.mask_protocol, expected_mask_ratio))
    if dense_ose:
        if not (
                args.resa_weight == 0.0
                and args.ose_lambda > 0.0
                and args.ose_mix_proto_weight == 0.0
                and args.ose_mix_ins_weight == 0.0):
            raise ValueError(
                'dense_ose_proto_ema_v1 requires prototype-only OSE weights')
        augmentation_probability = float(
            args.train_feeder_args.get('augmentation_probability', 0.0))
        if augmentation_probability <= 0.0:
            raise ValueError(
                'Dense OSE requires independently augmented online/EMA views')
        if args.ose_exemplar_batch_size <= 0:
            raise ValueError(
                'Dense OSE requires a positive EMA exemplar micro-batch')
        if not args.diagnostic_log_name:
            raise ValueError('Dense OSE diagnostic_log_name must be set')
        if Path(args.diagnostic_log_name).name != args.diagnostic_log_name:
            raise ValueError(
                'diagnostic_log_name must be a file name, not a path')
        if min(
                args.diagnostic_step_interval,
                args.diagnostic_epoch_interval,
                args.diagnostic_max_samples,
                args.diagnostic_batch_size) <= 0:
            raise ValueError('Dense OSE diagnostic settings must be positive')
    elif args.accum_iter != 1:
        raise ValueError(
            'Gradient accumulation is currently reserved for dense OSE')
    if mask_strategy == 'per_joint_random':
        num_frames = int(args.model_args.get('num_frames', 0))
        num_joints = int(args.model_args.get('num_joints', 0))
        patch_size = int(args.model_args.get('patch_size', 0))
        temporal_patch = int(args.model_args.get('t_patch_size', 0))
        if min(num_frames, num_joints, patch_size, temporal_patch) <= 0:
            raise ValueError(
                'Per-joint masking requires valid model patch dimensions')
        if num_frames % temporal_patch or num_joints % patch_size:
            raise ValueError(
                'Per-joint masking requires divisible temporal/joint grids')
        joint_patches = num_joints // patch_size
        tokens = (num_frames // temporal_patch) * joint_patches
        keep = round(tokens * (1.0 - 0.9))
        if keep != joint_patches * 3:
            raise ValueError(
                'shared_qk_per_joint_v1 requires exactly three visible '
                'temporal tokens per joint')
    if not args.exemplar_index_path:
        raise ValueError('exemplar_index_path must be set')
    if not args.output_dir:
        raise ValueError('output_dir must be set')
    if not isinstance(args.model_args, dict):
        raise ValueError('model_args must be a dictionary')
    if not isinstance(args.train_feeder_args, dict):
        raise ValueError('train_feeder_args must be a dictionary')
    if not bool(args.model_args.get('ose_separate_projector', False)):
        raise ValueError(
            'Stage2 requires model_args.ose_separate_projector=True')
    if not args.log_dir:
        args.log_dir = os.path.join(args.output_dir, 'tensorboard')


def stage2_training_mode(args):
    """Return a stable label for the active Stage2 objective branches."""
    if is_dense_ose_protocol(getattr(args, 'mask_protocol', '')):
        return DENSE_OSE_PROTOCOL
    resa_enabled = args.resa_weight > 0
    ose_enabled = any(weight > 0 for weight in (
        args.ose_lambda,
        args.ose_mix_proto_weight,
        args.ose_mix_ins_weight,
    ))
    if resa_enabled and ose_enabled:
        return 'resa_ose_separate_projector'
    if ose_enabled:
        return 'ose_only_separate_projector'
    return 'resa_only'


def load_or_create_exemplars(dataset, path, seed, num_classes):
    labels = np.asarray(dataset.label, dtype=np.int64)
    class_ids = sorted(np.unique(labels).tolist())
    if len(class_ids) != num_classes:
        raise ValueError(
            'Stage2 dataset has {} classes, expected {}'.format(
                len(class_ids), num_classes))
    path = Path(path)
    if path.is_file():
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        cached_classes = [int(value) for value in payload['class_ids']]
        indices = [int(value) for value in payload['indices']]
        errors = []
        if int(payload.get('seed', -1)) != int(seed):
            errors.append('seed mismatch')
        if int(payload.get('num_samples', -1)) != len(labels):
            errors.append('dataset size mismatch')
        if cached_classes != class_ids:
            errors.append('class IDs mismatch')
        if len(indices) != len(class_ids):
            errors.append('exemplar count mismatch')
        for class_id, index in zip(cached_classes, indices):
            if index < 0 or index >= len(labels):
                errors.append('index {} is out of range'.format(index))
                break
            if int(labels[index]) != int(class_id):
                errors.append(
                    'index {} has label {}, expected {}'.format(
                        index, int(labels[index]), class_id))
                break
        if errors:
            raise ValueError(
                'Invalid Stage2 exemplar cache {}: {}'.format(
                    path, '; '.join(errors)))
        return class_ids, indices

    rng = np.random.RandomState(seed)
    indices = []
    for class_id in class_ids:
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size == 0:
            raise ValueError('No exemplar candidate for class {}'.format(
                class_id))
        indices.append(int(rng.choice(candidates)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump({
            'seed': int(seed),
            'num_samples': len(labels),
            'class_ids': class_ids,
            'indices': indices,
        }, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return class_ids, indices


class ExemplarProvider:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.base_samples = [
            dataset.get_base_sample(index) for index in indices]

    def joint_views(self, device, num_views):
        views = []
        for _ in range(int(num_views)):
            augmented = np.stack([
                self.dataset.augment(sample) for sample in self.base_samples])
            joint = torch.from_numpy(augmented).float().to(
                device, non_blocking=True)
            views.append(joint)
        return views


@torch.no_grad()
def concat_all_gather(tensor):
    if not misc.is_dist_avail_and_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(misc.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def reduce_epoch_totals(totals, count, device):
    if not misc.is_dist_avail_and_initialized():
        return totals, count
    names = tuple(totals)
    packed = torch.tensor(
        [totals[name] for name in names] + [float(count)],
        dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = {
        name: float(packed[index].item())
        for index, name in enumerate(names)
    }
    return reduced, int(packed[-1].item())


def reduce_scalar_means(values, device):
    """Return rank-averaged scalar diagnostics in deterministic key order."""
    names = sorted(values)
    packed = torch.tensor(
        [float(values[name]) for name in names],
        dtype=torch.float64, device=device)
    if misc.is_dist_avail_and_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= float(misc.get_world_size())
    return {
        name: float(packed[index].item())
        for index, name in enumerate(names)
    }


def optimizer_group_diagnostics(optimizer):
    """Measure gradients and a momentum-free relative step estimate."""
    result = {}
    for group in optimizer.param_groups:
        role = str(group.get('stage2_role'))
        parameter_square = None
        gradient_square = None
        for parameter in group['params']:
            value = parameter.detach().float().square().sum()
            parameter_square = (
                value if parameter_square is None
                else parameter_square + value)
            if parameter.grad is not None:
                value = parameter.grad.detach().float().square().sum()
                gradient_square = (
                    value if gradient_square is None
                    else gradient_square + value)
        parameter_norm = math.sqrt(float(parameter_square.item()))
        gradient_norm = math.sqrt(float(
            gradient_square.item() if gradient_square is not None else 0.0))
        result[role + '_parameter_norm'] = parameter_norm
        result[role + '_gradient_norm'] = gradient_norm
        result[role + '_gradient_to_parameter_ratio'] = (
            gradient_norm / max(parameter_norm, 1e-12))
        result[role + '_gradient_step_ratio_estimate'] = (
            float(group['lr']) * gradient_norm
            / max(parameter_norm, 1e-12))
    return result


def cosine_value(start, end, progress):
    progress = min(max(float(progress), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end) + (float(start) - float(end)) * cosine


def ema_momentum(base_momentum, progress):
    return 1.0 - (1.0 - float(base_momentum)) * (
        math.cos(math.pi * min(max(progress, 0.0), 1.0)) + 1.0
    ) / 2.0


def set_learning_rates(optimizer, args, progress):
    backbone_lr = cosine_value(args.lr, args.final_lr, progress)
    head_lr = cosine_value(args.head_lr, args.head_final_lr, progress)
    for group in optimizer.param_groups:
        role = group.get('stage2_role')
        if role == 'backbone':
            group['lr'] = backbone_lr
        elif role == 'head':
            group['lr'] = head_lr
        else:
            raise ValueError('Unknown Stage2 optimizer group {}'.format(role))
    return backbone_lr, head_lr


def build_optimizer(model, args):
    backbone = [
        parameter for parameter in model.encoder_q.parameters()
        if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone}
    heads = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids]
    if not backbone or not heads:
        raise ValueError('Stage2 optimizer requires backbone and head params')
    return torch.optim.SGD([
        {'params': backbone, 'lr': args.lr, 'stage2_role': 'backbone'},
        {'params': heads, 'lr': args.head_lr, 'stage2_role': 'head'},
    ], momentum=args.optimizer_momentum, nesterov=args.nesterov,
        weight_decay=args.weight_decay)


def _capture_rng_state():
    state = {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state


@contextmanager
def preserve_rng_state():
    """Prevent rank-zero-only diagnostics from perturbing training RNGs."""
    state = _capture_rng_state()
    try:
        yield
    finally:
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in state:
            torch.cuda.set_rng_state(state['cuda_rng_state'])


def save_checkpoint(model, optimizer, scaler, args, completed_epochs):
    output = Path(args.output_dir)
    checkpoint_path = output / 'checkpoint-{:03d}.pth'.format(
        completed_epochs)
    rng_state = _capture_rng_state()
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': int(completed_epochs),
        'args': vars(args),
    }
    if misc.is_dist_avail_and_initialized():
        sidecars = [
            'checkpoint-{:03d}-rng-rank-{:03d}.pth'.format(
                completed_epochs, rank)
            for rank in range(misc.get_world_size())
        ]
        torch.save(rng_state, output / sidecars[misc.get_rank()])
        dist.barrier()
        state['rng_world_size'] = misc.get_world_size()
        state['rng_sidecars'] = sidecars
    else:
        state.update(rng_state)
    if misc.is_main_process():
        torch.save(state, checkpoint_path)
    backbone_path = output / 'checkpoint-{:03d}-backbone.pth'.format(
        completed_epochs)
    if misc.is_main_process():
        torch.save({
            'model': model.encoder_q.state_dict(),
            'epoch': int(completed_epochs),
            'source': 'MacDiff RSDG Stage2 online encoder',
        }, backbone_path)
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    return checkpoint_path, backbone_path


def restore_rng(checkpoint, args):
    state = checkpoint
    if misc.is_dist_avail_and_initialized():
        sidecar = checkpoint['rng_sidecars'][misc.get_rank()]
        state = torch.load(
            Path(args.resume).resolve().parent / sidecar,
            map_location='cpu')
    random.setstate(state['python_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    torch.set_rng_state(state['torch_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state' in state:
        torch.cuda.set_rng_state(state['cuda_rng_state'])


def validate_resume_checkpoint(checkpoint, args):
    if not isinstance(checkpoint, dict):
        raise ValueError('Stage2 resume checkpoint must be a dictionary')
    required = {'model', 'optimizer', 'scaler', 'epoch', 'args'}
    if misc.is_dist_avail_and_initialized():
        required.update({'rng_world_size', 'rng_sidecars'})
    else:
        required.update({
            'python_rng_state', 'numpy_rng_state', 'torch_rng_state'})
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            'Stage2 resume checkpoint is incomplete: missing {}'.format(
                missing))
    stored_args = checkpoint['args']
    if not isinstance(stored_args, dict):
        raise ValueError('Stage2 resume checkpoint args must be a dictionary')
    if misc.is_dist_avail_and_initialized():
        if int(checkpoint['rng_world_size']) != misc.get_world_size():
            raise ValueError('Stage2 resume world size does not match')
        sidecars = checkpoint['rng_sidecars']
        if len(sidecars) != misc.get_world_size():
            raise ValueError('Stage2 resume RNG sidecar count does not match')
        resume_dir = Path(args.resume).resolve().parent
        missing_sidecars = [
            name for name in sidecars
            if not (resume_dir / name).is_file()]
        if missing_sidecars:
            raise ValueError(
                'Stage2 resume RNG sidecars are missing: {}'.format(
                    missing_sidecars))
    mismatches = []
    for name in RESUME_CONTRACT_FIELDS:
        if name not in stored_args:
            # Checkpoints created before dense OSE support implicitly used one
            # optimizer step per loader batch.
            if name == 'accum_iter' and int(args.accum_iter) == 1:
                continue
            if (name == 'ose_exemplar_batch_size'
                    and int(args.ose_exemplar_batch_size) == 0):
                continue
            mismatches.append('{} missing'.format(name))
        elif stored_args[name] != getattr(args, name):
            mismatches.append('{} changed'.format(name))
    if mismatches:
        raise ValueError(
            'Stage2 resume contract mismatch: {}'.format(
                ', '.join(mismatches)))


def prepare_output(args):
    output = Path(args.output_dir)
    if not args.resume and output.exists():
        if not output.is_dir():
            raise RuntimeError(
                'Stage2 output_dir exists but is not a directory: {}'.format(
                    output))
        resolved_output = output.resolve()
        safe_root = (Path.cwd() / 'output_dir').resolve()
        if (resolved_output == safe_root or
                safe_root not in resolved_output.parents):
            raise RuntimeError(
                'Automatic Stage2 output replacement is only allowed for '
                'a child directory of {}: {}'.format(
                    safe_root, resolved_output))
        shutil.rmtree(str(resolved_output))
        print('Removed previous Stage2 output: {}'.format(resolved_output))
    output.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, loader, exemplar_provider, optimizer, scaler,
                    device, epoch, args, log_writer):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    totals = {name: 0.0 for name in TRAIN_METRICS}
    count = 0
    steps = len(loader)
    if args.max_train_steps > 0:
        steps = min(steps, args.max_train_steps)

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        view_a, view_b, _, _ = batch
        view_a = view_a.float().to(device, non_blocking=True)
        view_b = view_b.float().to(device, non_blocking=True)
        global_view_a = concat_all_gather(view_a)
        global_batch = global_view_a.shape[0]
        expected_global_batch = view_a.shape[0] * misc.get_world_size()
        if global_batch != expected_global_batch:
            raise RuntimeError(
                'Every DDP rank must contribute the same Stage2 batch size')
        if misc.is_main_process():
            global_mix_index = torch.randperm(global_batch, device=device)
        else:
            global_mix_index = torch.empty(
                global_batch, dtype=torch.long, device=device)
        if misc.is_dist_avail_and_initialized():
            dist.broadcast(global_mix_index, src=0)
        row_start = misc.get_rank() * view_a.shape[0]
        mix_index = global_mix_index[
            row_start:row_start + view_a.shape[0]]
        mix_beta = float(np.random.beta(
            args.ose_mix_alpha, args.ose_mix_alpha))
        if misc.is_dist_avail_and_initialized():
            beta_tensor = torch.tensor(mix_beta, device=device)
            dist.broadcast(beta_tensor, src=0)
            mix_beta = float(beta_tensor.item())
        mixed_view = (
            mix_beta * view_b
            + (1.0 - mix_beta) * global_view_a[mix_index])
        exemplar_views = exemplar_provider.joint_views(
            device, args.ose_exemplar_views)

        progress = (
            epoch + float(step + 1) / max(len(loader), 1)
        ) / max(args.epochs, 1)
        backbone_lr, head_lr = set_learning_rates(
            optimizer, args, progress)
        momentum = ema_momentum(args.ema_momentum, progress)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            losses = model(
                view_a,
                view_b,
                exemplar_views,
                momentum=momentum,
                ose_tau_s=args.ose_tau_s,
                ose_tau_t=args.ose_tau_t,
                mixed_view=mixed_view,
                mix_index=mix_index,
                mix_beta=mix_beta,
                exemplar_mask_seed=(
                    int(args.seed) * 1000003
                    + (epoch * max(len(loader), 1) + step) * 2),
            )
            resa_objective = args.resa_weight * losses['cluster']
            ose_objective = (
                args.ose_lambda * losses['proto']
                + args.ose_mix_proto_weight * losses['mix_proto']
                + args.ose_mix_ins_weight * losses['mix_ins'])
            loss = resa_objective + ose_objective

        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(
                'Non-finite Stage2 loss at epoch {}, step {}'.format(
                    epoch + 1, step))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        values = {
            'loss': float(loss.item()),
            'cluster': float(losses['cluster'].item()),
            'cluster_entropy': float(losses['cluster_entropy'].item()),
            'cluster_kl': float(losses['cluster_kl'].item()),
            'proto': float(losses['proto'].item()),
            'align': float(losses['align'].item()),
            'disp': float(losses['disp'].item()),
            'mix_proto': float(losses['mix_proto'].item()),
            'mix_ins': float(losses['mix_ins'].item()),
            'ose_target_entropy': float(
                losses['ose_target_entropy'].item()),
            'ose_target_confidence': float(
                losses['ose_target_confidence'].item()),
            'ose_prototype_usage': float(losses['ose_prototype_usage']),
            'ema_momentum': float(momentum),
        }
        for name, value in values.items():
            totals[name] += value
        count += 1

        global_step = epoch * len(loader) + step
        if log_writer is not None:
            for name, value in values.items():
                log_writer.add_scalar('train/' + name, value, global_step)
            log_writer.add_scalar('train/backbone_lr', backbone_lr, global_step)
            log_writer.add_scalar('train/head_lr', head_lr, global_step)
        if step % 20 == 0 or step + 1 == steps:
            print(
                'Epoch {:03d} [{:04d}/{:04d}] loss {:.4f} '
                'ReSA {:.4f} (H {:.4f}, KL {:.4f}) proto {:.4f} '
                'mix-p {:.4f} mix-i {:.4f} OSE-H {:.4f} '
                'conf {:.4f} usage {:.3f} '
                'lr {:.6f} head_lr {:.6f} m {:.6f}'.format(
                    epoch + 1, step + 1, steps, values['loss'],
                    values['cluster'], values['cluster_entropy'],
                    values['cluster_kl'], values['proto'],
                    values['mix_proto'], values['mix_ins'],
                    values['ose_target_entropy'],
                    values['ose_target_confidence'],
                    values['ose_prototype_usage'],
                    backbone_lr, head_lr, momentum), flush=True)

    if count == 0:
        raise RuntimeError('Stage2 epoch produced no optimizer step')
    optimizer_steps = count
    totals, count = reduce_epoch_totals(totals, count, device)
    means = {name: totals[name] / count for name in totals}
    return means, backbone_lr, head_lr, optimizer_steps


@torch.no_grad()
def refresh_dense_ose_prototypes(
        model, exemplar_provider, device, args, previous_prototypes=None):
    """Build one epoch-frozen prototype table entirely with the EMA pair."""
    exemplar_views = exemplar_provider.joint_views(
        device, args.ose_exemplar_views)
    with torch.cuda.amp.autocast(enabled=args.enable_amp):
        prototypes = model.refresh_ema_prototypes(
            exemplar_views, batch_size=args.ose_exemplar_batch_size)
    prototypes = prototypes.detach().float()
    if misc.is_dist_avail_and_initialized():
        # Every rank uses exactly the rank-zero cache even if low-level CUDA
        # kernels or augmentation RNGs differ.
        dist.broadcast(prototypes, src=0)
    if prototypes.shape != (args.num_classes, model.feature_dim):
        raise RuntimeError(
            'Dense OSE prototype cache has shape {}, expected [{},{}]'.format(
                tuple(prototypes.shape), args.num_classes,
                model.feature_dim))
    geometry = prototype_geometry(prototypes, previous_prototypes)
    print(
        'Refreshed epoch EMA prototypes: shape {} mean off-diagonal cosine '
        '{:.4f}'.format(
            tuple(prototypes.shape),
            geometry['offdiagonal_cosine_mean']),
        flush=True)
    return prototypes, geometry


def initialize_dense_diagnostics(
        model, dataset, exemplar_indices, device, args, is_resume):
    """Create/load the fixed Stage1 feature reference on rank zero."""
    if not misc.is_main_process():
        return None, None, None, None

    log_path = Path(args.output_dir) / args.diagnostic_log_name
    cache_path = Path(args.output_dir) / DENSE_REFERENCE_CACHE
    logger = DenseOSEJsonlLogger(log_path)
    if is_resume:
        if not cache_path.is_file():
            raise FileNotFoundError(
                'Dense OSE resume needs diagnostic reference cache: {}'.format(
                    cache_path))
        cached = torch.load(cache_path, map_location='cpu')
        diagnostic_indices = [
            int(value) for value in cached['indices']]
        diagnostic_labels = cached['labels'].long()
        reference_features = cached['reference_features'].float()
    else:
        diagnostic_indices = select_balanced_indices(
            dataset.label,
            exemplar_indices,
            args.diagnostic_max_samples,
            args.diagnostic_seed,
            args.num_classes,
        )
        diagnostic_labels = torch.as_tensor(
            np.asarray(dataset.label)[diagnostic_indices], dtype=torch.long)
        with preserve_rng_state():
            reference_features = extract_stage1_reference(
                model,
                dataset,
                diagnostic_indices,
                device,
                args.diagnostic_batch_size,
                args.enable_amp,
            )
        torch.save({
            'indices': diagnostic_indices,
            'labels': diagnostic_labels,
            'reference_features': reference_features,
        }, cache_path)

    if reference_features.shape[0] != len(diagnostic_indices):
        raise RuntimeError('Dense OSE diagnostic reference row mismatch')
    if diagnostic_labels.shape[0] != len(diagnostic_indices):
        raise RuntimeError('Dense OSE diagnostic label row mismatch')
    class_counts = torch.bincount(
        diagnostic_labels, minlength=args.num_classes)
    baseline_geometry = reference_geometry(
        reference_features, diagnostic_labels, device)
    logger.write(
        'run_start',
        resume=bool(is_resume),
        start_checkpoint=(args.resume or args.stage1_weights),
        output_dir=str(args.output_dir),
        protocol=str(args.mask_protocol),
        num_classes=int(args.num_classes),
        maximum_target_entropy=float(math.log(args.num_classes)),
        teacher_temperature=float(args.ose_tau_t),
        student_temperature=float(args.ose_tau_s),
        backbone_lr=float(args.lr),
        head_lr=float(args.head_lr),
        ema_momentum=float(args.ema_momentum),
        batch_size_per_rank=int(args.batch_size),
        world_size=int(misc.get_world_size()),
        accumulation_steps=int(args.accum_iter),
        effective_optimizer_batch=int(
            args.batch_size * misc.get_world_size() * args.accum_iter),
        augmentation_methods=list(args.train_feeder_args.get(
            'augmentation_methods', [])),
        augmentation_probability=float(args.train_feeder_args.get(
            'augmentation_probability', 0.0)),
        diagnostic_sample_count=int(len(diagnostic_indices)),
        diagnostic_class_counts=[
            int(value) for value in class_counts.tolist()],
        diagnostic_index_sum=int(sum(diagnostic_indices)),
        diagnostic_epoch_interval=int(args.diagnostic_epoch_interval),
        diagnostic_step_interval=int(args.diagnostic_step_interval),
        stage1_reference_geometry=baseline_geometry,
    )
    print('Dense OSE diagnostics: {} fixed samples -> {}'.format(
        len(diagnostic_indices), log_path), flush=True)
    return (
        logger, diagnostic_indices, diagnostic_labels, reference_features)


def train_one_epoch_dense_ose(
        model, loader, prototypes, optimizer, scaler,
        device, epoch, args, log_writer, diagnostic_logger=None):
    """Train full-token prototype-only OSE with one online encoder graph."""
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    totals = {name: 0.0 for name in TRAIN_METRICS}
    diagnostic_totals = {
        name: 0.0 for name in DENSE_DIAGNOSTIC_METRICS}
    window_diagnostic_totals = {
        name: 0.0 for name in DENSE_DIAGNOSTIC_METRICS}
    window_diagnostic_count = 0
    window_loss_total = 0.0
    count = 0
    optimizer_steps = 0
    optimizer_attempts = 0
    steps = len(loader)
    if args.max_train_steps > 0:
        steps = min(steps, args.max_train_steps)
    teacher_assignment_counts = torch.zeros(
        args.num_classes, dtype=torch.float64, device=device)
    student_assignment_counts = torch.zeros_like(
        teacher_assignment_counts)
    teacher_probability_sum = torch.zeros_like(
        teacher_assignment_counts)
    assignment_observations = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        # Ground-truth labels are consumed only by detached diagnostics.  They
        # are never passed into the OSE objective or prototype construction.
        teacher_view, online_view, labels, _ = batch
        teacher_view = teacher_view.float().to(
            device, non_blocking=True)
        online_view = online_view.float().to(
            device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.long)

        progress = (
            epoch + float(step + 1) / max(len(loader), 1)
        ) / max(args.epochs, 1)
        backbone_lr, head_lr = set_learning_rates(
            optimizer, args, progress)
        momentum = ema_momentum(args.ema_momentum, progress)

        window_start = (step // args.accum_iter) * args.accum_iter
        window_size = min(args.accum_iter, steps - window_start)
        accumulation_boundary = (
            (step + 1) % args.accum_iter == 0 or step + 1 == steps)
        sync_context = (
            model.no_sync()
            if hasattr(model, 'no_sync') and not accumulation_boundary
            else nullcontext())
        with sync_context:
            with torch.cuda.amp.autocast(enabled=args.enable_amp):
                losses = model(
                    online_view,
                    teacher_view,
                    prototypes,
                    ose_tau_s=args.ose_tau_s,
                    ose_tau_t=args.ose_tau_t,
                    labels=labels,
                )
                loss = args.ose_lambda * losses['proto']
            if not math.isfinite(float(loss.item())):
                raise FloatingPointError(
                    'Non-finite dense OSE loss at epoch {}, step {}'.format(
                        epoch + 1, step))
            scaler.scale(loss / window_size).backward()

        dense_values = {
            name: float(losses[name].item())
            for name in DENSE_DIAGNOSTIC_METRICS
        }
        for name, value in dense_values.items():
            diagnostic_totals[name] += value
            window_diagnostic_totals[name] += value
        window_diagnostic_count += 1
        window_loss_total += float(loss.item())
        teacher_assignment_counts += torch.bincount(
            losses['_ose_teacher_assignment'],
            minlength=args.num_classes).to(torch.float64)
        student_assignment_counts += torch.bincount(
            losses['_ose_student_assignment'],
            minlength=args.num_classes).to(torch.float64)
        teacher_probability_sum += losses[
            '_ose_teacher_probability_sum'].to(torch.float64)
        assignment_observations += int(labels.shape[0])

        if accumulation_boundary:
            optimizer_attempts += 1
            should_log_step = (
                optimizer_attempts == 1
                or optimizer_attempts % args.diagnostic_step_interval == 0
                or step + 1 == steps)
            optimizer_diagnostics = {}
            if should_log_step:
                # Unscale exactly once before inspecting AMP gradients.
                scaler.unscale_(optimizer)
                optimizer_diagnostics = optimizer_group_diagnostics(
                    optimizer)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
            if optimizer_stepped:
                model_without_ddp.momentum_update(momentum)
                optimizer_steps += 1
            if should_log_step:
                step_diagnostics = {
                    name: value / max(window_diagnostic_count, 1)
                    for name, value in window_diagnostic_totals.items()
                }
                step_diagnostics.update(optimizer_diagnostics)
                step_diagnostics.update({
                    'loss': window_loss_total / max(
                        window_diagnostic_count, 1),
                    'backbone_lr': float(backbone_lr),
                    'head_lr': float(head_lr),
                    'ema_momentum': float(momentum),
                })
                step_diagnostics = reduce_scalar_means(
                    step_diagnostics, device)
                if diagnostic_logger is not None:
                    diagnostic_logger.write(
                        'optimizer_step',
                        epoch=int(epoch + 1),
                        micro_step=int(step + 1),
                        micro_steps_in_epoch=int(steps),
                        optimizer_attempt_in_epoch=int(optimizer_attempts),
                        optimizer_steps_in_epoch=int(optimizer_steps),
                        optimizer_stepped=bool(optimizer_stepped),
                        accumulation_window=int(window_diagnostic_count),
                        metrics=step_diagnostics,
                    )
            optimizer.zero_grad()
            window_diagnostic_totals = {
                name: 0.0 for name in DENSE_DIAGNOSTIC_METRICS}
            window_diagnostic_count = 0
            window_loss_total = 0.0

        values = {
            'loss': float(loss.item()),
            'cluster': 0.0,
            'cluster_entropy': 0.0,
            'cluster_kl': 0.0,
            'proto': float(losses['proto'].item()),
            'align': float(losses['align'].item()),
            'disp': float(losses['disp'].item()),
            'mix_proto': 0.0,
            'mix_ins': 0.0,
            'ose_target_entropy': float(
                losses['ose_target_entropy'].item()),
            'ose_target_confidence': float(
                losses['ose_target_confidence'].item()),
            'ose_prototype_usage': float(losses['ose_prototype_usage']),
            'ema_momentum': float(momentum),
        }
        for name, value in values.items():
            totals[name] += value
        count += 1

        global_step = epoch * len(loader) + step
        if log_writer is not None:
            for name, value in values.items():
                log_writer.add_scalar('train/' + name, value, global_step)
            log_writer.add_scalar(
                'train/backbone_lr', backbone_lr, global_step)
            log_writer.add_scalar('train/head_lr', head_lr, global_step)
        if step % 20 == 0 or step + 1 == steps:
            print(
                'Epoch {:03d} [{:04d}/{:04d}] dense-OSE {:.4f} '
                'H {:.4f} conf {:.4f} usage {:.3f} disp {:.4f} '
                'lr {:.6f} head_lr {:.6f} m {:.6f} accum {:d}'.format(
                    epoch + 1, step + 1, steps, values['proto'],
                    values['ose_target_entropy'],
                    values['ose_target_confidence'],
                    values['ose_prototype_usage'], values['disp'],
                    backbone_lr, head_lr, momentum, args.accum_iter),
                flush=True)

    if count == 0 or optimizer_steps == 0:
        raise RuntimeError('Dense OSE epoch produced no optimizer step')
    local_count = count
    totals, reduced_count = reduce_epoch_totals(totals, local_count, device)
    means = {name: totals[name] / reduced_count for name in totals}
    diagnostic_totals, diagnostic_count = reduce_epoch_totals(
        diagnostic_totals, local_count, device)
    diagnostic_means = {
        name: diagnostic_totals[name] / diagnostic_count
        for name in diagnostic_totals
    }
    if misc.is_dist_avail_and_initialized():
        for tensor in (
                teacher_assignment_counts,
                student_assignment_counts,
                teacher_probability_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        observation_tensor = torch.tensor(
            assignment_observations, dtype=torch.float64, device=device)
        dist.all_reduce(observation_tensor, op=dist.ReduceOp.SUM)
        assignment_observations = int(observation_tensor.item())
    soft_probability = teacher_probability_sum / max(
        assignment_observations, 1)
    nonzero_probability = soft_probability[soft_probability > 0]
    soft_entropy = -(
        nonzero_probability * nonzero_probability.log()).sum()
    epoch_diagnostics = {
        'means': diagnostic_means,
        'teacher_top1_distribution': assignment_distribution(
            teacher_assignment_counts),
        'student_top1_distribution': assignment_distribution(
            student_assignment_counts),
        'teacher_soft_class_distribution': {
            'probability_mass': [
                float(value) for value in soft_probability.cpu().tolist()],
            'perplexity': float(torch.exp(soft_entropy).item()),
            'kl_to_uniform': float(
                math.log(args.num_classes) - soft_entropy.item()),
        },
        'assignment_observations': int(assignment_observations),
        'optimizer_attempts': int(optimizer_attempts),
        'successful_optimizer_steps': int(optimizer_steps),
    }
    return (
        means, backbone_lr, head_lr, optimizer_steps, epoch_diagnostics)


def main(args):
    misc.init_distributed_mode(args)
    validate_args(args)
    if misc.is_main_process():
        prepare_output(args)
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    device = (
        torch.device('cuda', args.gpu)
        if args.distributed else torch.device(args.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    rank = misc.get_rank()
    # Exemplar augmentation and Beta mixing stay identical across ranks;
    # torch randomness is rank-specific for permutations and workers.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed + rank)
    cudnn.benchmark = True

    feeder_class = import_class(args.feeder)
    dataset = feeder_class(**args.train_feeder_args)
    class_ids = None
    exemplar_indices = None
    if misc.is_main_process():
        class_ids, exemplar_indices = load_or_create_exemplars(
            dataset, args.exemplar_index_path, args.exemplar_seed,
            args.num_classes)
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    if not misc.is_main_process():
        class_ids, exemplar_indices = load_or_create_exemplars(
            dataset, args.exemplar_index_path, args.exemplar_seed,
            args.num_classes)
    excluded = set(exemplar_indices)
    unlabeled_indices = [
        index for index in range(len(dataset)) if index not in excluded]
    if len(unlabeled_indices) < args.batch_size * misc.get_world_size():
        raise ValueError('Not enough Stage2 unlabeled samples after exclusion')
    unlabeled_dataset = torch.utils.data.Subset(dataset, unlabeled_indices)
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            unlabeled_dataset,
            num_replicas=misc.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = torch.utils.data.RandomSampler(unlabeled_dataset)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    exemplar_provider = ExemplarProvider(dataset, exemplar_indices)
    print('Stage2 dataset: {} unlabeled + {} exemplars, classes {}'.format(
        len(unlabeled_indices), len(exemplar_indices), len(class_ids)))

    dense_ose = is_dense_ose_protocol(args.mask_protocol)
    model_class = MacDiffDenseOSE if dense_ose else MacDiffStage2
    model = model_class(
        mask_strategy=mask_strategy_from_protocol(args.mask_protocol),
        **args.model_args).to(device)
    if args.distributed and args.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Enabled SyncBatchNorm for Stage2 DDP heads')
    model_without_ddp = model
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        validate_resume_checkpoint(resume_checkpoint, args)
        model.load_state_dict(resume_checkpoint['model'], strict=True)
        start_epoch = int(resume_checkpoint['epoch'])
        print('Strictly resumed Stage2 model from epoch {}'.format(
            start_epoch))
    else:
        stage1_checkpoint = torch.load(
            args.stage1_weights, map_location='cpu')
        report = transfer_macdiff_stage1(model, stage1_checkpoint)
        print(
            'Stage2 transfer: {} encoder tensors, {} Stage1 tensors ignored '
            '({})'.format(
                report['encoder_tensors'], report['ignored_tensors'],
                report['source']))

    diagnostic_logger = None
    diagnostic_indices = None
    diagnostic_labels = None
    reference_features = None
    if dense_ose:
        (
            diagnostic_logger,
            diagnostic_indices,
            diagnostic_labels,
            reference_features,
        ) = initialize_dense_diagnostics(
            model_without_ddp,
            dataset,
            exemplar_indices,
            device,
            args,
            is_resume=resume_checkpoint is not None,
        )
        if misc.is_dist_avail_and_initialized():
            dist.barrier()

    optimizer = build_optimizer(model_without_ddp, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        scaler.load_state_dict(resume_checkpoint['scaler'])
    if args.distributed:
        model = DistributedDataParallel(
            model_without_ddp,
            device_ids=[args.gpu],
            output_device=args.gpu,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
    if resume_checkpoint is not None:
        restore_rng(resume_checkpoint, args)
    if start_epoch >= args.epochs:
        raise ValueError('Stage2 checkpoint already reached requested epochs')

    log_writer = (
        SummaryWriter(log_dir=args.log_dir)
        if misc.is_main_process() else None)
    start_time = time.time()
    training_mode = stage2_training_mode(args)
    print('Stage2 training mode: {}'.format(training_mode))
    previous_prototypes = None
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        if dense_ose:
            prototypes, prototype_diagnostics = (
                refresh_dense_ose_prototypes(
                    model_without_ddp,
                    exemplar_provider,
                    device,
                    args,
                    previous_prototypes=previous_prototypes,
                ))
            if diagnostic_logger is not None:
                diagnostic_logger.write(
                    'prototype_refresh',
                    epoch=int(epoch + 1),
                    geometry=prototype_diagnostics,
                )
            (
                means,
                backbone_lr,
                head_lr,
                batches,
                dense_epoch_diagnostics,
            ) = train_one_epoch_dense_ose(
                model, loader, prototypes, optimizer, scaler,
                device, epoch, args, log_writer, diagnostic_logger)
            previous_prototypes = prototypes.detach().clone()
        else:
            means, backbone_lr, head_lr, batches = train_one_epoch(
                model, loader, exemplar_provider, optimizer, scaler,
                device, epoch, args, log_writer)
        row = {
            'epoch': epoch + 1,
            'training_mode': training_mode,
            'batches': batches,
            'backbone_lr': backbone_lr,
            'head_lr': head_lr,
        }
        row.update({'mean_' + name: means[name]
                    for name in TRAIN_METRICS})
        if misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a',
                      encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            if dense_ose:
                diagnostic_logger.write(
                    'epoch_summary',
                    epoch=int(epoch + 1),
                    training_mode=training_mode,
                    batches=int(batches),
                    backbone_lr=float(backbone_lr),
                    head_lr=float(head_lr),
                    train_means={
                        name: float(value)
                        for name, value in means.items()
                    },
                    ose_diagnostics=dense_epoch_diagnostics,
                )
                print(
                    'Epoch {:03d} mean dense-OSE {:.4f} | H {:.4f} | '
                    'confidence {:.4f} | usage {:.3f}'.format(
                        epoch + 1, means['proto'],
                        means['ose_target_entropy'],
                        means['ose_target_confidence'],
                        means['ose_prototype_usage']))
            else:
                print(
                    'Epoch {:03d} mean loss {:.4f} | ReSA {:.4f} | '
                    'H {:.4f} | KL {:.4f} | prototype {:.4f}'.format(
                        epoch + 1, means['loss'], means['cluster'],
                        means['cluster_entropy'], means['cluster_kl'],
                        means['proto']))
        completed = epoch + 1
        run_geometry = dense_ose and (
            completed == 1
            or completed % args.diagnostic_epoch_interval == 0
            or completed == args.epochs)
        if run_geometry:
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            if misc.is_main_process():
                with preserve_rng_state():
                    extracted = extract_dense_features(
                        model_without_ddp,
                        dataset,
                        diagnostic_indices,
                        device,
                        args.diagnostic_batch_size,
                        args.enable_amp,
                    )
                    geometry = dense_epoch_geometry(
                        model_without_ddp,
                        extracted,
                        reference_features,
                        diagnostic_labels,
                        prototypes,
                        device,
                        args.ose_tau_s,
                        args.ose_tau_t,
                        args.batch_size * misc.get_world_size(),
                    )
                diagnostic_logger.write(
                    'representation_geometry',
                    epoch=int(completed),
                    sample_count=int(len(diagnostic_indices)),
                    metrics=geometry,
                )
                print(
                    'Dense OSE geometry epoch {:03d}: raw gap {:.4f}, '
                    'projected gap {:.4f}, Stage1 CKA {:.4f}'.format(
                        completed,
                        geometry['online_raw_geometry']['class_gap'],
                        geometry['online_projected_geometry']['class_gap'],
                        geometry['stage1_reference']['linear_cka']),
                    flush=True)
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
        if completed % args.save_interval == 0 or completed == args.epochs:
            checkpoint_path, backbone_path = save_checkpoint(
                model_without_ddp, optimizer, scaler, args, completed)
            if misc.is_main_process():
                print('Saved Stage2 checkpoint: {}'.format(checkpoint_path))
                print('Saved LP backbone: {}'.format(backbone_path))
        if log_writer is not None:
            log_writer.flush()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Stage2 training time {}'.format(elapsed))
    if log_writer is not None:
        log_writer.close()
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
