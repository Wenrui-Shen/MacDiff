"""Coordinate-space OT conditional Flow Matching with native MacDiff modules."""
import math

import numpy as np
import torch

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from .transformer_macdiff import Transformer as NativeTransformer
from .util import token_uniformity_loss


FLOW_MATCHING_VERSION = 'conditional_ot_noise_to_data_v2'
FLOW_TIME_SAMPLERS = ('uniform', 'inverse_cosine_snr_v1')


def build_inverse_cosine_snr_times(sigma_min):
    """Map the native 1000-step inverse_cosine(1) SNRs onto physical OT time.

    A uniform index preserves the native discrete SNR distribution, including
    beta clipping. This changes sampling density, not the OT path derivative.
    """
    betas = get_named_beta_schedule(['inverse_cosine', 1], 1000)
    alpha_bar = np.cumprod(1 - betas, dtype=np.float64)
    signal, noise = np.sqrt(alpha_bar), np.sqrt(1 - alpha_bar)
    time = signal / (noise + (1 - sigma_min) * signal)
    if not np.isfinite(time).all() or np.any((time < 0) | (time > 1)):
        raise ValueError('flow_sigma_min cannot reach the native inverse-cosine SNR range')
    return time


def validate_flow_options(flow_time_scale, flow_matching_version, one_person, lambda_loss_uni,
                          flow_sigma_min=1.e-4, flow_time_sampling='uniform'):
    if flow_matching_version != FLOW_MATCHING_VERSION:
        raise ValueError('Unsupported flow_matching_version')
    if (isinstance(flow_time_scale, bool) or not isinstance(flow_time_scale, (int, float))
            or not math.isfinite(flow_time_scale) or flow_time_scale <= 0):
        raise ValueError('flow_time_scale must be finite and positive')
    if one_person is not True:
        raise ValueError('This native FM baseline requires one_person=True')
    if not math.isfinite(lambda_loss_uni) or lambda_loss_uni < 0:
        raise ValueError('lambda_loss_uni must be finite and non-negative')
    if (isinstance(flow_sigma_min, bool) or not isinstance(flow_sigma_min, (int, float))
            or not math.isfinite(flow_sigma_min) or not 0 <= flow_sigma_min < 1):
        raise ValueError('flow_sigma_min must be finite and in [0, 1)')
    if flow_time_sampling not in FLOW_TIME_SAMPLERS:
        raise ValueError('Unsupported flow_time_sampling: %s' % flow_time_sampling)
    if flow_time_sampling == 'inverse_cosine_snr_v1':
        build_inverse_cosine_snr_times(flow_sigma_min)


def conditional_ot_flow_path(clean, noise, time, sigma_min):
    """Lipman et al., Eqs. 22-23: t=0 noise, t=1 data + sigma_min*noise.

    This is OT between the per-target Gaussian endpoints, without minibatch
    transport assignment. The derivative form avoids division near t=1.
    """
    if clean.shape != noise.shape or time.shape != (clean.shape[0],):
        raise ValueError('Flow endpoints and per-person time shapes must match')
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError('Flow time must be finite and in [0, 1]')
    weight = time.reshape((-1,) + (1,) * (clean.ndim - 1))
    sigma_t = 1 - (1 - sigma_min) * weight
    return sigma_t * noise + weight * clean, clean - (1 - sigma_min) * noise


class Transformer(NativeTransformer):
    # Only enables scalar logging. No teacher, queue, or optimizer-step hooks.
    supports_pretrain_metrics = True
    pretrain_metric_prefix = 'flow_matching/'

    def __init__(self, flow_time_scale=999., flow_sigma_min=1.e-4,
                 flow_matching_version=FLOW_MATCHING_VERSION, flow_time_sampling='uniform', **kwargs):
        forbidden = {'diff_prediction', 'diff_steps', 'diff_noise_schedule', 'loss_reweight'} & set(kwargs)
        if forbidden:
            raise ValueError('Remove diffusion-only model arguments for FM: %s' % sorted(forbidden))
        validate_flow_options(flow_time_scale, flow_matching_version,
                              kwargs.get('one_person', True), kwargs.get('lambda_loss_uni', 0.),
                              flow_sigma_min=flow_sigma_min, flow_time_sampling=flow_time_sampling)
        # Native construction preserves every learned tensor's name, shape and
        # initialization. Its parameter-free diffusion helpers are discarded;
        # none of them is used in FM training or included in the state dict.
        super().__init__(**kwargs)
        for name in ('diffusion', 'schedule_sampler', 'loss_reweight',
                     'diff_steps', 'diff_noise_schedule'):
            delattr(self, name)
        self.diff_prediction = 'flow_velocity'
        self.flow_time_scale = float(flow_time_scale)
        self.flow_sigma_min = float(flow_sigma_min)
        self.flow_matching_version = flow_matching_version
        self.flow_time_sampling = flow_time_sampling
        if flow_time_sampling == 'inverse_cosine_snr_v1':
            self.register_buffer('_flow_snr_time_lookup', torch.tensor(
                build_inverse_cosine_snr_times(self.flow_sigma_min), dtype=torch.float32),
                persistent=False)
        print('OT conditional Flow Matching: %s; sigma_min=%g; time_sampling=%s; '
              'embedding time=t*%g; target=data-(1-sigma_min)*noise'
              % (self.flow_matching_version, self.flow_sigma_min, self.flow_time_sampling,
                 self.flow_time_scale))

    def sample_flow_time(self, batch_size, device):
        if self.flow_time_sampling == 'uniform':
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        indices = torch.randint(self._flow_snr_time_lookup.numel(), (batch_size,), device=device)
        return self._flow_snr_time_lookup[indices]

    def update_diffusion_sampler(self, epoch, total_epoch):
        """Compatibility hook for the native engine; FM has no discrete sampler."""
        return None

    def forward_loss(self, velocity, prediction, mask, t=None):
        """Mean over coordinates, masked patches, then retained people; no t weights."""
        target = self.patchify(velocity.float())
        if prediction.shape != target.shape or mask.shape != target.shape[:2]:
            raise ValueError('FM prediction, target and mask shapes must match')
        weights = mask.float()
        count = weights.sum(dim=-1)
        if torch.any(count <= 0):
            raise ValueError('FM requires at least one masked patch per person')
        error = (prediction.float() - target).square().mean(dim=-1)
        return ((error * weights).sum(dim=-1) / count).mean()

    def forward_macdiff(self, source, source_aug, mask_ratio=.9,
                        motion_stride=1, motion_aware_tau=-1):
        if source.ndim != 5 or source.shape != source_aug.shape:
            raise ValueError('FM requires matching [B,C,T,V,M] skeleton inputs')
        if not 0 < mask_ratio < 1:
            raise ValueError('FM mask_ratio must be in (0, 1)')
        with torch.no_grad():
            # Keep native person0, augmentation, normalization, and crop pairing.
            raw = self._sequence_layout(source).float()
            augmented = self._sequence_layout(source_aug).float()
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center=center)
            augmented = self._normalize_sequence(augmented, center=center)
            if not torch.isfinite(clean).all() or not torch.isfinite(augmented).all():
                raise FloatingPointError('FM nonfinite normalized skeleton input')
            time = self.sample_flow_time(clean.shape[0], clean.device)
            noise = torch.randn_like(clean)
            mixed, velocity = conditional_ot_flow_path(clean, noise, time, self.flow_sigma_min)
            coefficient_snr = (time / (1 - (1 - self.flow_sigma_min) * time)).square()
            middle_snr = ((coefficient_snr >= .1) & (coefficient_snr <= 10)).float().mean()

        latent, pooled, mask, ids_restore = self.forward_encoder(
            augmented, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        uniformity = token_uniformity_loss(latent)
        condition = self.build_global_local_condition(latent, pooled, ids_restore)
        prediction = self.forward_decoder(mixed, z=condition, t=time * self.flow_time_scale)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('FM nonfinite decoder velocity prediction')
        flow_loss = self.forward_loss(velocity, prediction, mask)
        total = flow_loss + self.lambda_loss_uni * uniformity
        if not torch.isfinite(total):
            raise FloatingPointError('FM nonfinite loss (velocity MSE or skeleton uniformity)')
        metrics = {'loss_flow_matching': flow_loss.detach(),
                   'loss_uniformity': uniformity.detach(),
                   'time_mean': time.mean().detach(),
                   'snr_middle_fraction': middle_snr.detach()}
        return total, prediction, mask, metrics

    def forward(self, source, source_aug, mask_ratio=.9, motion_stride=1,
                motion_aware_tau=-1, enable_ose=False):
        if enable_ose:
            raise ValueError('This native FM baseline does not use OSE')
        return self.forward_macdiff(source, source_aug, mask_ratio=mask_ratio,
                                    motion_stride=motion_stride, motion_aware_tau=motion_aware_tau)
