# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import yaml
import numpy as np
import os
import time
from pathlib import Path

import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--config', default='./config/ntu60_xsub_joint_pretrain_debug.yaml', help='path to the configuration file')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument(
        '--max_train_steps', default=0, type=int,
        help='Limit iterations per epoch for diagnostics; 0 runs the full epoch')

    ###
    parser.add_argument('--task', default='recognition', type=str, help='recognition / 2dto3d')
    parser.add_argument('--args_2dto3d', default=dict(), help='the arguments of evaluation for 2dto3d task')
    
    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')

    parser.add_argument('--mask_ratio', default=0.90, type=float, nargs='+', ###
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--motion_stride', default=1, type=int,
                        help='stride of motion to be predicted.')
    parser.add_argument('--motion_aware_tau', default=0.75, type=float,
                        help='temperature of motion aware masking.')      
                        
    parser.add_argument('--mask_ratio_inter', default=0.75, type=float,
                        help='Masking ratio inter (percentage of removed patches).')
    parser.add_argument('--mask_ratio_intra', default=0.80, type=float,
                        help='Masking ratio intra (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=True, # False
                        help='Enabling automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--min_lr_epochs', type=int, default=20, metavar='N', help='epochs to keep min LR at the end of training')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')

    # OSE-guided cross-instance diffusion
    parser.add_argument(
        '--enable_ose', action='store_true', default=False,
        help='Enable OSE prototypes and cross-instance reconstruction')
    parser.add_argument('--ose_exemplar_indices', default='', type=str)
    parser.add_argument('--ose_momentum', default=0.999, type=float)
    parser.add_argument('--ose_queue_size', default=32768, type=int)
    parser.add_argument(
        '--ose_start_epoch', default=100, type=int,
        help='First epoch that enables OSE loss and peer reconstruction')
    parser.add_argument(
        '--ose_exemplar_checkpoint', action='store_true', default=False,
        help='Checkpoint online exemplar encoder blocks to trade compute for memory')
    parser.add_argument('--ose_refresh_interval', default=1, type=int)
    parser.add_argument('--ose_topk', default=4, type=int)
    parser.add_argument('--ose_alpha', default=0.75, type=float)
    parser.add_argument('--ose_tau_s', default=0.1, type=float)
    parser.add_argument('--ose_tau_t', default=0.04, type=float)
    parser.add_argument('--ose_assignment_confidence', default=0.8, type=float)
    parser.add_argument('--lambda_ose', default=1.0, type=float)
    parser.add_argument('--self_prob_start', default=0.9, type=float)
    parser.add_argument('--self_prob_end', default=0.1, type=float)
    parser.add_argument('--peer_prob_start', default=0.1, type=float)
    parser.add_argument('--peer_prob_end', default=0.9, type=float)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def load_exemplar_mapping(path, dataset):
    if not path:
        raise ValueError('ose_exemplar_indices must point to a one-exemplar-per-class JSON file')
    with open(path, 'r', encoding='utf-8') as file:
        raw_mapping = json.load(file)
    if not isinstance(raw_mapping, dict):
        raise ValueError('OSE exemplar mapping must be a JSON object')
    mapping = {str(int(class_id)): int(dataset_id) for class_id, dataset_id in raw_mapping.items()}
    if len(mapping) != len(raw_mapping):
        raise ValueError('OSE class IDs must remain unique after integer normalization')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Every OSE class must use a distinct exemplar dataset index')
    if any(dataset_id < 0 or dataset_id >= len(dataset.label) for dataset_id in mapping.values()):
        raise ValueError('OSE exemplar dataset index is outside the training split')
    class_ids = sorted(int(class_id) for class_id in mapping)
    dataset_class_ids = sorted(int(class_id) for class_id in np.unique(dataset.label))
    if class_ids != dataset_class_ids:
        missing = sorted(set(dataset_class_ids) - set(class_ids))
        extra = sorted(set(class_ids) - set(dataset_class_ids))
        raise ValueError(
            'OSE mapping must contain exactly one exemplar for every dataset '
            'class; missing={}, extra={}'.format(missing, extra))
    for class_id, dataset_id in mapping.items():
        if int(dataset.label[dataset_id]) != int(class_id):
            raise ValueError(
                'OSE exemplar {} has label {}, expected class {}'.format(
                    dataset_id, int(dataset.label[dataset_id]), class_id))
    return mapping


@torch.no_grad()
def refresh_ose_state(model, dataset, args, device):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = misc.get_rank()
    exemplar_ids = model.ose_memory.exemplar_ids.cpu().tolist()
    exemplar_samples = torch.from_numpy(
        dataset.get_ose_samples(exemplar_ids)).float().to(device)
    model.set_ose_exemplars(exemplar_samples)

    neighbor_map = None
    if int(model.ose_memory.queue_count.item()) > 0 and rank == 0:
        neighbor_map = model.refresh_ose(exemplar_samples, args.mask_ratio)

    if distributed:
        for state in (
            model.ose_memory.prototypes,
            model.ose_memory.prototype_neighbor_ids,
            model.ose_memory.prototype_valid,
            model.ose_memory.snapshot_version,
            model.ose_memory.neighbor_map_entries,
        ):
            torch.distributed.broadcast(state, src=0)
        # Queue state is kept identical by all-gather before every enqueue, so
        # the Python routing map can be rebuilt locally from broadcast tensors.
        neighbor_map = model.ose_memory.build_neighbor_map()

    if neighbor_map is None:
        neighbor_map = {}
    dataset.set_neighbor_map(neighbor_map)
    neighbor_edges = 0
    correct_neighbor_edges = 0
    for source_id, peer_ids in neighbor_map.items():
        source_label = int(dataset.label[int(source_id)])
        for peer_id in peer_ids:
            neighbor_edges += 1
            correct_neighbor_edges += int(
                source_label == int(dataset.label[int(peer_id)]))
    neighbor_accuracy = (
        correct_neighbor_edges / neighbor_edges if neighbor_edges > 0 else 0.0
    )
    if rank == 0:
        print('OSE snapshot {}: {} source samples have valid peers'.format(
            int(model.ose_memory.snapshot_version.item()), len(neighbor_map)
        ))
        print(
            'Offline neighbor label accuracy: {:.4f} ({}/{})'.format(
                neighbor_accuracy, correct_neighbor_edges, neighbor_edges))
    return {
        'offline_neighbor_label_accuracy': neighbor_accuracy,
        'offline_neighbor_edge_count': neighbor_edges,
    }


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.max_train_steps < 0:
        raise ValueError('max_train_steps must be non-negative')
    if args.enable_ose:
        if isinstance(args.mask_ratio, list):
            if len(args.mask_ratio) != 1:
                raise ValueError('OSE peer diffusion requires one fixed mask_ratio')
            args.mask_ratio = args.mask_ratio[0]
        if not 0.0 < args.mask_ratio < 1.0:
            raise ValueError('OSE peer diffusion requires mask_ratio in (0, 1)')
        if args.motion_aware_tau > 0:
            raise ValueError('OSE peer diffusion requires motion_aware_tau <= 0')
        if args.ose_refresh_interval <= 0:
            raise ValueError('ose_refresh_interval must be positive')
        if args.ose_start_epoch < 0 or args.ose_start_epoch >= args.epochs:
            raise ValueError('ose_start_epoch must be in [0, epochs)')
        if not 0.0 <= args.ose_momentum < 1.0:
            raise ValueError('ose_momentum must be in [0, 1)')
        if args.lambda_ose < 0:
            raise ValueError('lambda_ose must be non-negative')
        routing_probabilities = (
            args.self_prob_start, args.self_prob_end,
            args.peer_prob_start, args.peer_prob_end)
        if any(probability < 0.0 or probability > 1.0
               for probability in routing_probabilities):
            raise ValueError('Routing probabilities must be in [0, 1]')
        if not np.isclose(args.self_prob_start + args.peer_prob_start, 1.0):
            raise ValueError('Initial self/peer routing probabilities must sum to one')
        if not np.isclose(args.self_prob_end + args.peer_prob_end, 1.0):
            raise ValueError('Final self/peer routing probabilities must sum to one')

    # Load dataset
    Feeder = import_class(args.feeder)
    dataset_train = Feeder(**args.train_feeder_args)
    exemplar_mapping = None
    if args.enable_ose:
        if not hasattr(dataset_train, 'set_neighbor_map'):
            raise TypeError('OSE pretraining requires feeder.feeder_ntu.FeederOSE')
        exemplar_mapping = load_exemplar_mapping(
            args.ose_exemplar_indices, dataset_train)
        dataset_train.exclude_ose_exemplars(exemplar_mapping.values())
    print(dataset_train)

    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    def worker_init_fn(worker_id):                                                          
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    Model = import_class(args.model)
    model = Model(**args.model_args)
    if args.enable_ose:
        model.initialize_ose(
            exemplar_mapping=exemplar_mapping,
            ose_momentum=args.ose_momentum,
            ose_queue_size=args.ose_queue_size,
            ose_start_epoch=args.ose_start_epoch,
            ose_exemplar_checkpoint=args.ose_exemplar_checkpoint,
            ose_topk=args.ose_topk,
            ose_alpha=args.ose_alpha,
            ose_tau_s=args.ose_tau_s,
            ose_tau_t=args.ose_tau_t,
            ose_assignment_confidence=args.ose_assignment_confidence,
            lambda_ose=args.lambda_ose,
            self_prob_start=args.self_prob_start,
            self_prob_end=args.self_prob_end,
            peer_prob_start=args.peer_prob_start,
            peer_prob_end=args.peer_prob_end,
        )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False,
            broadcast_buffers=False)
        model_without_ddp = model.module
    if args.enable_ose:
        model_without_ddp.reset_momentum_encoder()
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    offline_neighbor_stats = {
        'offline_neighbor_label_accuracy': 0.0,
        'offline_neighbor_edge_count': 0,
    }

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        ose_refresh_due = (
            args.enable_ose
            and epoch >= args.ose_start_epoch
            and (
                epoch == args.start_epoch
                or (epoch - args.ose_start_epoch) % args.ose_refresh_interval == 0
            )
        )
        if ose_refresh_due:
            offline_neighbor_stats = refresh_ose_state(
                model_without_ddp, dataset_train, args, device)
    
        if args.task == 'recognition':
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        else:
            assert 0
        if args.enable_ose:
            train_stats.update(offline_neighbor_stats)
        if log_writer is not None:
            scalar_names = [
                'cuda_peak_allocated_mb',
                'cuda_peak_reserved_mb',
            ]
            if args.enable_ose:
                scalar_names.extend([
                    'offline_neighbor_label_accuracy',
                    'offline_neighbor_edge_count',
                    'offline_cross_reconstruction_label_accuracy',
                    'offline_cross_reconstruction_count',
                ])
            for name in scalar_names:
                log_writer.add_scalar(name, train_stats[name], epoch)
        ###
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
