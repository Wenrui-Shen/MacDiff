"""Native MacDiff with a full-context EMA teacher and positional layer alignment.

The online encoder/decoder keep their native state-dict names for linear probing.
This borrows global InfoNCE and adaptive cross-layer mixing; it is not a literal
ACA2Net reproduction (no motion-input teacher or asymmetric temporal patching).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from .transformer_macdiff import (
    MomentumSkeletonEncoder, Transformer as NativeTransformer,
)


METHOD_VERSION = 'raw_full_position_student_adapter_aa1_v3'


def validate_teacher_options(ema_momentum, contrast_temperature, contrast_queue_size,
                             alignment_teacher_layers, depth,
                             lambda_contrastive, lambda_alignment):
    for name, value in (('ema_momentum', ema_momentum),
                        ('contrast_temperature', contrast_temperature),
                        ('lambda_contrastive', lambda_contrastive),
                        ('lambda_alignment', lambda_alignment)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite number')
    if not 0. <= ema_momentum < 1.:
        raise ValueError('ema_momentum must be in [0, 1)')
    if contrast_temperature <= 0.:
        raise ValueError('contrast_temperature must be positive')
    if lambda_contrastive < 0. or lambda_alignment < 0.:
        raise ValueError('Auxiliary loss weights must be non-negative')
    for name, value in (('contrast_queue_size', contrast_queue_size),
                        ('alignment_teacher_layers', alignment_teacher_layers)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(name + ' must be a positive integer')
    if alignment_teacher_layers > depth:
        raise ValueError('alignment_teacher_layers must not exceed encoder depth')


def gather_token_positions(features, ids_keep):
    """features are in original temporal-major/joint-minor patch order."""
    return torch.gather(features, 1, ids_keep.unsqueeze(-1).expand(-1, -1, features.shape[-1]))


class FullContextTeacher(MomentumSkeletonEncoder):
    @torch.no_grad()
    def forward(self, x, ids_keep, target_layers):
        x = self.joints_embed(x)
        batch, temporal, joints, dim = x.shape
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        # Bypass random_masking even at mask_ratio=0: it otherwise permutes
        # all tokens, invalidating positional gathers without ids_restore.
        x = x.reshape(batch, temporal * joints, dim)
        targets = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in target_layers:
                targets.append(gather_token_positions(x, ids_keep))
        pooled = self.norm(x).float().mean(dim=1)
        return F.normalize(pooled, dim=-1, eps=1.e-6), targets


class Transformer(NativeTransformer):
    supports_skeleton_teacher = True

    def __init__(self, ema_momentum=.999, contrast_temperature=.07,
                 contrast_queue_size=65536, alignment_teacher_layers=5,
                 lambda_contrastive=1.e-4, lambda_alignment=1.,
                 teacher_method_version=METHOD_VERSION, **kwargs):
        if teacher_method_version != METHOD_VERSION:
            raise ValueError('Unsupported teacher_method_version')
        if kwargs.get('one_person', True) is not True:
            raise ValueError('Skeleton teacher requires one_person=True')
        if kwargs.get('diff_prediction', 'noise') != 'noise':
            raise ValueError('Skeleton teacher preserves native epsilon prediction')
        validate_teacher_options(
            ema_momentum, contrast_temperature, contrast_queue_size,
            alignment_teacher_layers, kwargs.get('depth', 5),
            lambda_contrastive, lambda_alignment)
        super().__init__(**kwargs)
        self.ema_momentum = ema_momentum
        self.contrast_temperature = contrast_temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_alignment = lambda_alignment
        self.target_layers = tuple(range(len(self.blocks) - alignment_teacher_layers,
                                         len(self.blocks)))
        self.ema_teacher = FullContextTeacher(self)
        self.ema_teacher.eval()
        # ACA2Net Eq. (9): a direct linear combination, not a softmax mixture.
        # These are the Top-K target columns of the full S-by-T matrix.
        self.alignment_weights = nn.Parameter(torch.empty(len(self.blocks), alignment_teacher_layers))
        self.alignment_adapters = nn.ModuleList([
            nn.Linear(self.dim_feat, self.dim_feat) for _ in self.target_layers])
        # Align to detached, unprojected teacher features. Only the student
        # has adapters, so two trainable projections cannot shrink together.
        # The paper does not specify initialization. Use zero-mean small W
        # and native Xavier/zero-bias Linear initialization, with no layer prior.
        nn.init.normal_(self.alignment_weights, std=.02)
        self.alignment_adapters.apply(self._init_weights)
        self.register_buffer('contrast_queue', torch.zeros(contrast_queue_size, self.dim_feat))
        self.register_buffer('contrast_queue_ids', torch.full((contrast_queue_size,), -1, dtype=torch.long))
        self.register_buffer('contrast_queue_ptr', torch.zeros((), dtype=torch.long))
        self.register_buffer('contrast_queue_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.long))
        self._student_alignment = None
        self._native_sample_ids = None
        self._native_loss_diffusion = None

    def train(self, mode=True):
        super().train(mode)
        self.ema_teacher.eval()
        return self

    @torch.no_grad()
    def initialize_pretraining_state(self):
        """Called after DDP sync and before resume loads the saved EMA/queue."""
        self.ema_teacher.update_from(self, 0.)

    def initialize_ose(self, *args, **kwargs):
        raise ValueError('Skeleton teacher pretraining does not use OSE')

    @torch.no_grad()
    def _raise_native_failure(self, stage, **tensors):
        details = {}
        for name, value in tensors.items():
            if value is None:
                continue
            finite = torch.isfinite(value)
            valid = value.detach()[finite].double()
            details[name] = dict(
                shape=tuple(value.shape), dtype=str(value.dtype),
                nonfinite=int((~finite).sum().item()),
                absmax=valid.abs().max().item() if valid.numel() else None,
                rms=valid.pow(2).mean().sqrt().item() if valid.numel() else None)
        sample_ids = getattr(self, '_native_sample_ids', None)
        sample_ids = sample_ids.detach().cpu().tolist() if sample_ids is not None else None
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        raise FloatingPointError(
            'Skeleton teacher native {} is non-finite: rank={}, sample_ids={}, details={}'.format(
                stage, rank, sample_ids, details))

    def forward_decoder(self, x, *, z, t):
        # The native decoder's modulation and QK products can overflow FP16
        # before softmax/loss promotion. Keep this teacher-only path in FP32;
        # casting only its final prediction would be too late.
        with torch.cuda.amp.autocast(enabled=False):
            prediction = super().forward_decoder(x.float(), z=z.float(), t=t)
        if not torch.isfinite(prediction).all().item():
            self._raise_native_failure(
                'decoder_prediction', prediction=prediction, decoder_x=x, condition=z, timesteps=t)
        return prediction

    def forward_loss(self, imgs, pred, mask, t):
        with torch.cuda.amp.autocast(enabled=False):
            loss = super().forward_loss(imgs.float(), pred.float(), mask.float(), t)
        self._native_loss_diffusion = loss.detach()
        if not torch.isfinite(loss).all().item():
            self._raise_native_failure(
                'diffusion_loss', loss=loss, target=imgs, prediction=pred, mask=mask, timesteps=t)
        return loss

    def forward_encoder(self, x, *, x_orig, mask_ratio, motion_aware_tau,
                        checkpoint_blocks=False):
        if checkpoint_blocks:
            raise ValueError('Skeleton teacher does not use encoder checkpointing')
        x = self.joints_embed(x)
        batch, temporal, joints, dim = x.shape
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        x = x.reshape(batch, temporal * joints, dim)
        if motion_aware_tau <= 0.:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            original = self.patchify(x_orig).reshape(batch, temporal, joints, -1)
            x, mask, ids_restore, ids_keep = self.motion_aware_random_masking(
                x, x_orig=original, mask_ratio=mask_ratio, tau=motion_aware_tau)
        layers = []
        for block in self.blocks:
            x = block(x)
            layers.append(x)
        latent = self.norm(x)
        self._student_alignment = (layers, latent, ids_keep)
        return latent, latent.mean(dim=1, keepdim=True), mask, ids_restore

    def contrastive_loss(self, latent, teacher_keys, sample_ids):
        with torch.cuda.amp.autocast(enabled=False):
            queries = F.normalize(latent.float().mean(dim=1), dim=-1, eps=1.e-6)
            positive = (queries * teacher_keys.detach().float()).sum(dim=-1, keepdim=True)
            count = int(self.contrast_queue_count.item())
            negatives = queries @ self.contrast_queue[:count].detach().float().T
            # A previous view of this same training sequence is not a negative.
            same_sample = sample_ids[:, None] == self.contrast_queue_ids[None, :count]
            negatives = negatives.masked_fill(same_sample, float('-inf'))
            logits = torch.cat((positive, negatives), dim=1) / self.contrast_temperature
            labels = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
            # The empty queue has only the positive: a finite differentiable
            # zero on the first optimizer window, not random negative targets.
            return F.cross_entropy(logits, labels)

    def alignment_loss(self, student_layers, teacher_targets):
        with torch.cuda.amp.autocast(enabled=False):
            mixed = torch.einsum('ij,ibnc->jbnc', self.alignment_weights.float(),
                                 torch.stack(student_layers, dim=0).float())
            losses = []
            for index, (adapter, target) in enumerate(zip(
                    self.alignment_adapters, teacher_targets)):
                prediction = adapter(mixed[index])
                target = target.detach().float()
                losses.append(F.smooth_l1_loss(prediction, target, reduction='mean'))
            return torch.stack(losses).mean()

    def forward(self, source, source_aug, *, sample_indices=None, mask_ratio=.9,
                motion_stride=1, motion_aware_tau=-1, enable_ose=False, **kwargs):
        if enable_ose or kwargs:
            raise ValueError('Use the native skeleton-only teacher entry point')
        if sample_indices is None:
            raise ValueError('Raw training sample_indices are required for the contrast queue')
        if not 0. < mask_ratio < 1.:
            raise ValueError('Skeleton teacher requires mask_ratio in (0, 1)')
        keep = round(self.joints_embed.num_patches * (1. - mask_ratio))
        if keep <= 0 or keep >= self.joints_embed.num_patches:
            raise ValueError('mask_ratio must retain at least one visible and one masked token')
        sample_ids = sample_indices.to(device=source.device, dtype=torch.long).reshape(-1)
        if sample_ids.shape[0] != source.shape[0] or (sample_ids < 0).any():
            raise ValueError('One non-negative sample index is required per sequence')
        # Prepare before native forward, whose normalization can change views
        # of its inputs. This is the clean, SAME-crop target, not x_aug or x_t.
        with torch.no_grad():
            teacher_input = self._prepare_target(source)
        self._native_sample_ids = sample_ids
        self._native_loss_diffusion = None
        try:
            native_loss, prediction, mask = super().forward_macdiff(
                source, source_aug, mask_ratio=mask_ratio,
                motion_stride=motion_stride, motion_aware_tau=motion_aware_tau)
            student_layers, latent, ids_keep = self._student_alignment
            if not torch.isfinite(native_loss).all().item():
                self._raise_native_failure(
                    'total_loss', native=native_loss, diffusion=self._native_loss_diffusion,
                    latent=latent, prediction=prediction, mask=mask)
            with torch.no_grad():
                teacher_keys, teacher_targets = self.ema_teacher(
                    teacher_input, ids_keep, self.target_layers)
            loss_contrastive = self.contrastive_loss(latent, teacher_keys, sample_ids)
            loss_alignment = self.alignment_loss(student_layers, teacher_targets)
            loss = (native_loss + self.lambda_contrastive * loss_contrastive
                    + self.lambda_alignment * loss_alignment)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    'Skeleton teacher loss is non-finite: native={}, contrastive={}, alignment={}'.format(
                        native_loss.detach().item(), loss_contrastive.detach().item(),
                        loss_alignment.detach().item()))
            metrics = dict(loss_native=native_loss.detach(),
                           loss_contrastive=loss_contrastive.detach(),
                           loss_alignment=loss_alignment.detach())
            payload = dict(teacher_keys=teacher_keys.detach(), teacher_ids=sample_ids.detach())
            return loss, prediction, mask, metrics, payload
        finally:
            self._student_alignment = None
            self._native_sample_ids = None
            self._native_loss_diffusion = None

    @torch.no_grad()
    def complete_pretraining_step(self, pending_batches, successful):
        """Commit EMA/queue only after a successful, complete optimizer step."""
        if not successful:
            return
        keys = torch.cat([batch['teacher_keys'] for batch in pending_batches], dim=0).float()
        ids = torch.cat([batch['teacher_ids'] for batch in pending_batches], dim=0)
        if dist.is_available() and dist.is_initialized():
            all_keys = [torch.empty_like(keys) for _ in range(dist.get_world_size())]
            all_ids = [torch.empty_like(ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_keys, keys.contiguous())
            dist.all_gather(all_ids, ids.contiguous())
            keys, ids = torch.cat(all_keys), torch.cat(all_ids)
        self.ema_teacher.update_from(self, self.ema_momentum)
        self.teacher_updates.add_(1)
        capacity = self.contrast_queue.shape[0]
        if keys.shape[0] >= capacity:
            self.contrast_queue.copy_(keys[-capacity:])
            self.contrast_queue_ids.copy_(ids[-capacity:])
            self.contrast_queue_ptr.zero_()
            self.contrast_queue_count.fill_(capacity)
        else:
            positions = (torch.arange(keys.shape[0], device=keys.device)
                         + self.contrast_queue_ptr) % capacity
            self.contrast_queue.index_copy_(0, positions, keys)
            self.contrast_queue_ids.index_copy_(0, positions, ids)
            self.contrast_queue_ptr.copy_((self.contrast_queue_ptr + keys.shape[0]) % capacity)
            self.contrast_queue_count.fill_(min(capacity, int(self.contrast_queue_count.item()) + keys.shape[0]))
