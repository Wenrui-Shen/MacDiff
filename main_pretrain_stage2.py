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
    MacDiffStage2,
    transfer_macdiff_stage1,
)


NTU_BONE_PAIRS = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
    (21, 21), (22, 23), (23, 8), (24, 25), (25, 12),
)

TRAIN_METRICS = (
    'loss', 'cluster', 'cluster_entropy', 'cluster_kl', 'proto', 'align',
    'disp', 'mix_proto', 'mix_ins', 'ema_momentum',
)

RESUME_CONTRACT_FIELDS = (
    'feeder', 'train_feeder_args', 'model_args', 'epochs', 'batch_size',
    'num_workers', 'max_train_steps', 'seed', 'enable_amp', 'lr', 'head_lr',
    'final_lr', 'head_final_lr', 'weight_decay', 'optimizer_momentum',
    'nesterov', 'ema_momentum', 'exemplar_index_path', 'exemplar_seed',
    'num_classes', 'ose_exemplar_views', 'resa_weight', 'ose_lambda',
    'ose_mix_proto_weight', 'ose_mix_ins_weight', 'ose_mix_alpha',
    'ose_tau_s', 'ose_tau_t', 'mask_protocol', 'world_size',
)


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
    parser.add_argument('--resa_weight', default=1.0, type=float)
    parser.add_argument('--ose_lambda', default=1.0, type=float)
    parser.add_argument('--ose_mix_proto_weight', default=1.0, type=float)
    parser.add_argument('--ose_mix_ins_weight', default=1.0, type=float)
    parser.add_argument('--ose_mix_alpha', default=1.0, type=float)
    parser.add_argument('--ose_tau_s', default=0.1, type=float)
    parser.add_argument('--ose_tau_t', default=0.04, type=float)
    parser.add_argument(
        '--mask_protocol', default='shared_qk_jmb_v1', type=str)

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
    if args.num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if args.ose_exemplar_views < 1:
        raise ValueError('ose_exemplar_views must be at least 1')
    if args.mask_protocol != 'shared_qk_jmb_v1':
        raise ValueError(
            'Stage2 requires mask_protocol=shared_qk_jmb_v1')
    if float(args.model_args.get('mask_ratio', -1.0)) != 0.9:
        raise ValueError(
            'shared_qk_jmb_v1 requires model_args.mask_ratio=0.9')
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
            'This Stage2 entry implements the Dual-space protocol; '
            'model_args.ose_separate_projector must be True')
    if not args.log_dir:
        args.log_dir = os.path.join(args.output_dir, 'tensorboard')


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


def derive_motion(skeleton):
    motion = torch.zeros_like(skeleton)
    motion[:, :, :-1] = skeleton[:, :, 1:] - skeleton[:, :, :-1]
    return motion


def derive_bone(skeleton):
    if skeleton.shape[3] != 25:
        raise ValueError('NTU Stage2 bone stream requires 25 joints')
    bone = torch.zeros_like(skeleton)
    for first, second in NTU_BONE_PAIRS:
        bone[:, :, :, first - 1] = (
            skeleton[:, :, :, first - 1]
            - skeleton[:, :, :, second - 1])
    return bone


class ExemplarProvider:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.base_samples = [
            dataset.get_base_sample(index) for index in indices]

    def jmb_groups(self, device, num_views):
        groups = []
        for _ in range(int(num_views)):
            augmented = np.stack([
                self.dataset.augment(sample) for sample in self.base_samples])
            joint = torch.from_numpy(augmented).float().to(
                device, non_blocking=True)
            groups.append((joint, derive_motion(joint), derive_bone(joint)))
        return groups


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
    packed = torch.tensor(
        [totals[name] for name in TRAIN_METRICS] + [float(count)],
        dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = {
        name: float(packed[index].item())
        for index, name in enumerate(TRAIN_METRICS)
    }
    return reduced, int(packed[-1].item())


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
        exemplar_groups = exemplar_provider.jmb_groups(
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
                exemplar_groups,
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
                'mix-p {:.4f} mix-i {:.4f} '
                'lr {:.6f} head_lr {:.6f} m {:.6f}'.format(
                    epoch + 1, step + 1, steps, values['loss'],
                    values['cluster'], values['cluster_entropy'],
                    values['cluster_kl'], values['proto'],
                    values['mix_proto'], values['mix_ins'],
                    backbone_lr, head_lr, momentum), flush=True)

    if count == 0:
        raise RuntimeError('Stage2 epoch produced no optimizer step')
    optimizer_steps = count
    totals, count = reduce_epoch_totals(totals, count, device)
    means = {name: totals[name] / count for name in totals}
    return means, backbone_lr, head_lr, optimizer_steps


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

    model = MacDiffStage2(**args.model_args).to(device)
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
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        means, backbone_lr, head_lr, batches = train_one_epoch(
            model, loader, exemplar_provider, optimizer, scaler,
            device, epoch, args, log_writer)
        row = {
            'epoch': epoch + 1,
            'training_mode': 'resa_ose_separate_projector',
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
            print(
                'Epoch {:03d} mean loss {:.4f} | ReSA {:.4f} | '
                'H {:.4f} | KL {:.4f} | prototype {:.4f}'.format(
                    epoch + 1, means['loss'], means['cluster'],
                    means['cluster_entropy'], means['cluster_kl'],
                    means['proto']))
        completed = epoch + 1
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
