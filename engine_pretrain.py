# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


@torch.no_grad()
def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    model_without_ddp = model.module if hasattr(model, 'module') else model
    model_without_ddp.update_diffusion_sampler(epoch, args.epochs)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40 #20

    accum_iter = args.accum_iter
    steps_this_epoch = len(data_loader)
    if args.max_train_steps > 0:
        steps_this_epoch = min(steps_this_epoch, args.max_train_steps)

    optimizer.zero_grad()
    pending_teacher_features = []
    pending_source_ids = []
    offline_cross_correct = torch.zeros((), dtype=torch.long, device=device)
    offline_cross_total = torch.zeros((), dtype=torch.long, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step >= steps_this_epoch:
            break
        (
            samples, samples_aug, peers, source_ids, _, has_peer,
            source_labels, peer_labels,
        ) = batch

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        samples = samples.float().to(device, non_blocking=True)
        samples_aug = samples_aug.float().to(device, non_blocking=True)
        peers = peers.float().to(device, non_blocking=True)
        source_ids = source_ids.long().to(device, non_blocking=True)
        has_peer = has_peer.bool().to(device, non_blocking=True)
        source_labels = source_labels.long().to(device, non_blocking=True)
        peer_labels = peer_labels.long().to(device, non_blocking=True)

        # mask ratio
        mask_ratio = args.mask_ratio
        if isinstance(mask_ratio, list):
            if len(mask_ratio) == 1: mask_ratio = mask_ratio[0]
            elif len(mask_ratio) == 2: mask_ratio = np.random.uniform(mask_ratio[0], mask_ratio[1])
            
        
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            loss, _, _, aux, teacher_features, use_peer = model(
                samples,
                samples_aug,
                peers,
                source_ids,
                has_peer,
                epoch,
                args.epochs,
                mask_ratio=mask_ratio,
                motion_aware_tau=args.motion_aware_tau,
            )

        # Offline diagnostic only: labels never enter the model, routing
        # decision, loss, or backward graph. Device-side counters avoid a
        # per-iteration GPU synchronization.
        with torch.no_grad():
            offline_cross_total.add_(use_peer.sum())
            offline_cross_correct.add_((
                (source_labels == peer_labels) & use_peer
            ).sum())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(11)

        window_start = (data_iter_step // accum_iter) * accum_iter
        window_size = min(accum_iter, steps_this_epoch - window_start)
        loss /= window_size
        accumulation_boundary = (
            (data_iter_step + 1) % accum_iter == 0
            or data_iter_step + 1 == steps_this_epoch)
        pending_teacher_features.append(teacher_features.detach())
        pending_source_ids.append(source_ids.detach())
        scale_before = loss_scaler._scaler.get_scale()
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=accumulation_boundary)
        optimizer_step = accumulation_boundary and loss_scaler._scaler.get_scale() >= scale_before
        if optimizer_step:
            model_without_ddp.update_momentum_encoder()
            model_without_ddp.enqueue_ose(
                concat_all_gather(torch.cat(pending_teacher_features, dim=0)),
                concat_all_gather(torch.cat(pending_source_ids, dim=0)),
            )
        if accumulation_boundary:
            optimizer.zero_grad()
            pending_teacher_features.clear()
            pending_source_ids.clear()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        for name, value in aux.items():
            metric_logger.update(**{name: float(value.item())})
        for name, value in model_without_ddp.ose_metrics().items():
            metric_logger.update(**{name: value})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        aux_reduced = None
        if optimizer_step:
            aux_reduced = {
                name: misc.all_reduce_mean(float(value.item()))
                for name, value in aux.items()
            }
        if log_writer is not None and optimizer_step:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            for name, value in aux_reduced.items():
                log_writer.add_scalar(name, value, epoch_1000x)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    offline_counts = torch.stack([
        offline_cross_correct, offline_cross_total,
    ]).to(dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(offline_counts)
    offline_correct, offline_total = offline_counts.tolist()
    offline_cross_accuracy = (
        offline_correct / offline_total if offline_total > 0 else 0.0
    )
    if device.type == 'cuda':
        memory_peak = torch.tensor([
            torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        ], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(memory_peak, op=dist.ReduceOp.MAX)
        peak_allocated_mb, peak_reserved_mb = memory_peak.tolist()
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
    print("Averaged stats:", metric_logger)
    print(
        'Offline cross-reconstruction label accuracy: '
        '{:.4f} ({}/{})'.format(
            offline_cross_accuracy, int(offline_correct), int(offline_total)))
    print(
        'CUDA peak memory: allocated={:.1f} MiB, reserved={:.1f} MiB'.format(
            peak_allocated_mb, peak_reserved_mb))
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({
        'offline_cross_reconstruction_label_accuracy': offline_cross_accuracy,
        'offline_cross_reconstruction_count': int(offline_total),
        'cuda_peak_allocated_mb': peak_allocated_mb,
        'cuda_peak_reserved_mb': peak_reserved_mb,
    })
    return stats
