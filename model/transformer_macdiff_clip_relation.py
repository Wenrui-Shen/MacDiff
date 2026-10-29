"""Native MacDiff + full EMA InfoNCE + frozen-CLIP global relation KL.

There is no text remap, text decoder, or per-sample moving text target.
"""
import math

import torch
import torch.nn.functional as F

from .transformer_macdiff import Transformer as NativeTransformer
from .transformer_macdiff_text_relation import paired_relation_kl, gather_relation_pairs
from .transformer_macdiff_text_relation_teacher import (
    FullContextRelationKeyEncoder, shared_queue_info_nce,
)
from .util import token_uniformity_loss


CLIP_RELATION_VERSION = 'fixed_clip_full_ema_gap_shared_queue_kl_v1'


def validate_clip_relation_options(clip_relation_version, relation_bank_size, lambda_relation,
                                   relation_temperature_text, relation_temperature_skeleton,
                                   ema_momentum, lambda_contrastive, contrast_temperature,
                                   relation_feature_mode, one_person, text_target_mode,
                                   text_target_norm, lambda_text_to_skeleton,
                                   lambda_skeleton_to_text):
    if clip_relation_version != CLIP_RELATION_VERSION:
        raise ValueError('Unsupported clip_relation_version')
    if type(relation_bank_size) is not int or relation_bank_size < 2:
        raise ValueError('relation_bank_size must be an integer >= 2')
    if (one_person is not True or relation_feature_mode != 'gap'
            or text_target_mode != 'fixed_clip' or text_target_norm != 'rms'):
        raise ValueError('Use person0, GAP skeleton queries and fixed RMS CLIP text')
    if lambda_text_to_skeleton != 0 or lambda_skeleton_to_text != 0:
        raise ValueError('Fixed-CLIP relation PT has no T->S or S->T diffusion')
    for name, value in (('lambda_relation', lambda_relation),
                        ('relation_temperature_text', relation_temperature_text),
                        ('relation_temperature_skeleton', relation_temperature_skeleton),
                        ('ema_momentum', ema_momentum), ('lambda_contrastive', lambda_contrastive),
                        ('contrast_temperature', contrast_temperature)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(name + ' must be finite and non-negative')
        if 'temperature' in name and value == 0:
            raise ValueError(name + ' must be positive')
    if ema_momentum >= 1:
        raise ValueError('ema_momentum must be in [0,1)')


class Transformer(NativeTransformer):
    supports_text_cache = True
    supports_relation_memory = True
    text_global_only = True

    def __init__(self, clip_relation_version=CLIP_RELATION_VERSION, relation_bank_size=65536,
                 lambda_relation=.1, relation_temperature_text=.1, relation_temperature_skeleton=.1,
                 ema_momentum=.999, lambda_contrastive=1.e-4, contrast_temperature=.07,
                 relation_feature_mode='gap', text_input_dim=512, text_context_length=7,
                 text_target_mode='fixed_clip', text_target_norm='rms',
                 lambda_text_to_skeleton=0., lambda_skeleton_to_text=0., **kwargs):
        validate_clip_relation_options(clip_relation_version, relation_bank_size, lambda_relation,
            relation_temperature_text, relation_temperature_skeleton, ema_momentum,
            lambda_contrastive, contrast_temperature, relation_feature_mode,
            kwargs.get('one_person', True), text_target_mode, text_target_norm,
            lambda_text_to_skeleton, lambda_skeleton_to_text)
        if type(text_input_dim) is not int or text_input_dim < 1 or text_context_length != 7:
            raise ValueError('Use the existing seven-sentence CLIP cache and positive text dimension')
        super().__init__(**kwargs)
        if self.diff_prediction != 'noise' or self.dim_t_embed != 64:
            raise ValueError('Retain native epsilon diffusion and 64-dim time embedding')
        self.clip_relation_version = clip_relation_version
        self.relation_bank_size = relation_bank_size
        self.lambda_relation = float(lambda_relation)
        self.relation_temperature_text = float(relation_temperature_text)
        self.relation_temperature_skeleton = float(relation_temperature_skeleton)
        self.ema_momentum = float(ema_momentum)
        self.lambda_contrastive = float(lambda_contrastive)
        self.contrast_temperature = float(contrast_temperature)
        self.relation_feature_mode = relation_feature_mode
        self.text_input_dim, self.text_context_length = text_input_dim, text_context_length
        # Metadata expected by the common text-cache/checkpoint entry; no modules use these flags.
        self.text_target_mode, self.text_target_norm = text_target_mode, text_target_norm
        self.lambda_text_to_skeleton = self.lambda_skeleton_to_text = 0.
        self.lambda_text_uniformity = 0.
        self.share_skeleton_decoder = False
        self.text_conditioning_mode = 'global_relation_only'
        self.ema_teacher = FullContextRelationKeyEncoder(self)
        self.ema_teacher.eval()
        self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('relation_skeleton_keys', torch.zeros(relation_bank_size, self.dim_feat))
        self.register_buffer('relation_text_keys', torch.zeros(relation_bank_size, text_input_dim))
        self.register_buffer('relation_sample_ids', torch.full((relation_bank_size,), -1, dtype=torch.long))
        self.register_buffer('relation_pointer', torch.zeros((), dtype=torch.long))
        print('Fixed CLIP relation: %s; K=%d; temperatures(T/S)=%g/%g; CL=%g; KL=%g; '
              'no remap, text diffusion, or moving text target'
              % (clip_relation_version, relation_bank_size, relation_temperature_text,
                 relation_temperature_skeleton, lambda_contrastive, lambda_relation))

    def train(self, mode=True):
        super().train(mode)
        self.ema_teacher.eval()
        return self

    @torch.no_grad()
    def initialize_pretraining_state(self):
        self.ema_teacher.update_from(self, 0.)

    @torch.no_grad()
    def update_text_target(self):
        # Common engine hook. Fixed cached CLIP vectors never change.
        pass

    @torch.no_grad()
    def complete_relation_step(self, batches, successful):
        if not successful or not batches:
            return
        ids, skeleton, text = [torch.cat([batch[name] for batch in batches], dim=0)
                               for name in ('sample_ids', 'skeleton', 'text')]
        ids, skeleton, text = gather_relation_pairs(ids, skeleton, text)
        if not ids.numel():
            return
        if not torch.isfinite(skeleton).all() or not torch.isfinite(text).all() or torch.any(ids < 0):
            raise FloatingPointError('Invalid paired fixed-CLIP relation-memory update')
        total = ids.shape[0]
        start = max(0, total - self.relation_bank_size)
        positions = (torch.arange(start, total, device=ids.device)
                     + self.relation_pointer) % self.relation_bank_size
        self.relation_skeleton_keys[positions] = F.normalize(skeleton[start:].float(), dim=-1)
        self.relation_text_keys[positions] = F.normalize(text[start:].float(), dim=-1)
        self.relation_sample_ids[positions] = ids[start:]
        self.relation_pointer.copy_((self.relation_pointer + total) % self.relation_bank_size)
        # Commit keys produced before this optimizer step, then update the EMA once.
        self.ema_teacher.update_from(self, self.ema_momentum)
        self.teacher_updates.add_(1)

    def forward(self, source, source_aug, text_features=None, mask_ratio=.9,
                motion_stride=1, motion_aware_tau=-1, enable_ose=False,
                sample_indices=None, **unused):
        if enable_ose or not 0 < mask_ratio < 1:
            raise ValueError('Fixed-CLIP relation PT requires masked native diffusion without OSE')
        batch_size = source.shape[0]
        if (sample_indices is None or sample_indices.shape != (batch_size,)
                or sample_indices.dtype != torch.long or torch.any(sample_indices < 0)):
            raise ValueError('Stable dataset sample IDs are required')
        if text_features is None or text_features.shape != (batch_size, self.text_input_dim):
            raise ValueError('Use the aligned person0 global CLIP sentence features')
        with torch.no_grad():
            source, source_aug = source[..., :1], source_aug[..., :1]
            active = source.abs().sum(dim=(1, 2, 3, 4)) > 0
            if not active.any():
                raise ValueError('The entire batch has no active person0')
            teacher = text_features[active].detach().float()
            if not torch.isfinite(teacher).all() or torch.any(teacher.square().sum(dim=-1) <= 1.e-12):
                raise ValueError('Active samples require finite nonzero fixed CLIP vectors')
            teacher = F.normalize(teacher, dim=-1)
            raw = self._sequence_layout(source).float()
            augmented = self._sequence_layout(source_aug).float()
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center)
            augmented = self._normalize_sequence(augmented, center)
            time, _ = self.schedule_sampler.sample(batch_size, clean.device)
            noise = torch.randn_like(clean)
            noisy = self.diffusion.q_sample(clean, time, noise=noise)
        latent, pooled, mask, restore = self.forward_encoder(
            augmented, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        uniformity = token_uniformity_loss(latent[active])
        condition = self.build_global_local_condition(latent, pooled, restore)
        prediction = self.forward_decoder(noisy, z=condition, t=time)
        native = self.forward_loss(noise[active], prediction[active], mask[active], time[active])
        query = pooled[active, 0].float()
        with torch.no_grad():
            keys = self.ema_teacher(clean[active])
        contrastive, positive = shared_queue_info_nce(query, keys, sample_indices[active],
            self.relation_skeleton_keys, self.relation_sample_ids, self.contrast_temperature)
        relation, relation_metrics = paired_relation_kl(query, teacher, sample_indices[active],
            self.relation_skeleton_keys, self.relation_text_keys, self.relation_sample_ids,
            self.relation_temperature_skeleton, self.relation_temperature_text)
        loss = (native + self.lambda_loss_uni * uniformity
                + self.lambda_contrastive * contrastive + self.lambda_relation * relation)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native/InfoNCE/fixed-CLIP relation loss')
        metrics = dict(loss_diff=native.detach(), loss_uniformity=uniformity.detach(),
            loss_contrastive=contrastive.detach(),
            loss_contrastive_weighted=(self.lambda_contrastive * contrastive).detach(),
            contrast_positive_cosine=positive,
            loss_relation=relation.detach(), loss_relation_weighted=(self.lambda_relation * relation).detach(),
            relation_bank_fill=(self.relation_sample_ids >= 0).float().sum(), **relation_metrics)
        pending = dict(sample_ids=sample_indices[active].detach(), skeleton=keys.detach(), text=teacher)
        return loss, prediction, mask, metrics, pending
