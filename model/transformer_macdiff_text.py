"""Stage1 MacDiff with task-adapted global text and bidirectional noise prediction.

Native encoder/decoder names are unchanged for downstream encoder transfer.
The two auxiliary decoders have independent parameters; only S->T sends an
auxiliary gradient to the skeleton encoder. T->S trains the text remap.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer_macdiff import Transformer as MacDiff, MLP
from .util import timestep_embedding, token_uniformity_loss


class TextConditionedSkeletonDecoder(nn.Module):
    """Independent native-shaped decoder, conditioned globally on remapped text."""
    def __init__(self, source):
        super().__init__()
        for name in ('decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred'):
            setattr(self, name, copy.deepcopy(getattr(source, name)))
        self.pos_embed = nn.Parameter(source.decoder_pos_embed.detach().clone())
        self.temp_embed = nn.Parameter(source.decoder_temp_embed.detach().clone())
        self.dim_t_embed = source.dim_t_embed

    def forward(self, noisy_skeleton, t, condition):
        x = self.decoder_embed(noisy_skeleton)
        n, tp, vp, dim = x.shape
        x = (x + self.pos_embed[:, :, :vp] + self.temp_embed[:, :tp]).reshape(n, tp * vp, dim)
        z = condition[:, None, :].expand(-1, tp * vp, -1)
        time = timestep_embedding(t, self.dim_t_embed)
        for block in self.decoder_blocks:
            x = block(x, t=time, z=z)
        return self.decoder_pred(self.decoder_norm(x))


class TextNoiseDecoder(nn.Module):
    """Small conditional residual MLP for vector-valued text diffusion."""
    def __init__(self, dim, time_dim, hidden_dim, depth):
        super().__init__()
        self.time_dim = time_dim
        self.input = nn.Linear(dim, hidden_dim)
        self.condition = nn.Sequential(nn.Linear(dim + time_dim, hidden_dim), nn.SiLU())
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim, elementwise_affine=False)
                                   for _ in range(depth)])
        self.modulations = nn.ModuleList([nn.Linear(hidden_dim, 2 * hidden_dim)
                                         for _ in range(depth)])
        self.blocks = nn.ModuleList([MLP(hidden_dim, hidden_dim * 2, hidden_dim)
                                    for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))

    def forward(self, noisy_text, t, skeleton_feature):
        x = self.input(noisy_text)
        condition = self.condition(torch.cat([
            skeleton_feature, timestep_embedding(t, self.time_dim)], dim=-1))
        for norm, modulation, block in zip(self.norms, self.modulations, self.blocks):
            scale, shift = modulation(condition).chunk(2, dim=-1)
            x = x + block(norm(x) * (1 + scale) + shift)
        return self.output(x)


class Transformer(MacDiff):
    supports_text_cache = True

    def __init__(self, text_input_dim=512, text_hidden_dim=512,
                 text_decoder_hidden_dim=256, text_decoder_depth=3,
                 lambda_text_to_skeleton=1., lambda_skeleton_to_text=1., **kwargs):
        super().__init__(**kwargs)
        if self.diff_prediction != 'noise':
            raise ValueError('All three text-Stage1 tasks require diff_prediction=noise')
        if self.dim_t_embed != 64:
            raise ValueError('The native MacDiff decoder requires dim_t_embed=64')
        if min(text_input_dim, text_hidden_dim, text_decoder_hidden_dim, text_decoder_depth) < 1:
            raise ValueError('Text dimensions and decoder depth must be positive')
        if any(not math.isfinite(w) or w < 0 for w in (
                lambda_text_to_skeleton, lambda_skeleton_to_text)):
            raise ValueError('Text loss weights must be finite and non-negative')
        self.text_input_dim = text_input_dim
        self.lambda_text_to_skeleton = float(lambda_text_to_skeleton)
        self.lambda_skeleton_to_text = float(lambda_skeleton_to_text)
        self.text_remap = nn.Sequential(
            nn.Linear(text_input_dim, text_hidden_dim), nn.GELU(),
            nn.Linear(text_hidden_dim, self.dim_feat),
            nn.LayerNorm(self.dim_feat, elementwise_affine=False))
        self.text_skeleton_decoder = TextConditionedSkeletonDecoder(self)
        self.text_noise_decoder = TextNoiseDecoder(
            self.dim_feat, self.dim_t_embed, text_decoder_hidden_dim, text_decoder_depth)
        self.text_remap.apply(self._init_weights)
        self.text_noise_decoder.apply(self._init_weights)
        # Disabled branches have no optimizer/DDP parameters awaiting gradients.
        # With T->S disabled the remap stays at its fixed initialization; it cannot
        # learn from S->T because that branch deliberately detaches its target.
        if not self.lambda_text_to_skeleton:
            self.text_remap.requires_grad_(False)
            self.text_skeleton_decoder.requires_grad_(False)
        if not self.lambda_skeleton_to_text:
            self.text_noise_decoder.requires_grad_(False)

    def text_to_skeleton_loss(self, noisy, noise, t, mask, r, active, people):
        condition = r[:, None, :].expand(-1, people, -1).reshape(-1, r.shape[-1])
        prediction = self.text_skeleton_decoder(noisy, t, condition)
        return self.forward_loss(noise[active], prediction[active], mask[active], t[active])

    def skeleton_to_text_loss(self, r, h):
        # Detach BEFORE corruption; no path from this loss into the remap.
        target = r.detach().float()
        t, _ = self.schedule_sampler.sample(target.shape[0], target.device)
        noise = torch.randn_like(target)
        noisy = self.diffusion.q_sample(target, t, noise=noise)
        predicted_noise = self.text_noise_decoder(noisy, t, h)
        return F.mse_loss(predicted_noise.float(), noise)

    def forward(self, source, source_aug, text_features=None, mask_ratio=.9,
                motion_stride=1, motion_aware_tau=-1, enable_ose=False, **unused):
        if enable_ose:
            raise ValueError('Text Stage1 does not support OSE routing')
        if not 0 < mask_ratio < 1:
            raise ValueError('Text Stage1 requires 0 < mask_ratio < 1')
        if text_features is None or text_features.shape != (source.shape[0], self.text_input_dim):
            raise ValueError('Expected one cached global text vector per sample')
        if not self.lambda_text_to_skeleton and not self.lambda_skeleton_to_text:
            loss, prediction, mask = self.forward_macdiff(
                source, source_aug, mask_ratio, motion_stride, motion_aware_tau)
            return loss, prediction, mask, {'loss_native_total': loss.detach()}

        with torch.no_grad():
            if self.one_person:
                source, source_aug = source[..., :1], source_aug[..., :1]
            batch, channels, frames, joints, people = source.shape
            # Detect padded people before centering/standardization makes zeros nonzero.
            active = source.abs().sum(dim=(1, 2, 3)) > 0
            if not active.any(dim=1).all():
                raise ValueError('A training sample contains no active skeleton person')
            raw = source.permute(0, 4, 2, 3, 1).reshape(batch * people, frames, joints, channels)
            aug = source_aug.permute(0, 4, 2, 3, 1).reshape_as(raw)
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center)
            aug = self._normalize_sequence(aug, center)
            t, _ = self.schedule_sampler.sample(clean.shape[0], clean.device)
            noise = torch.randn_like(clean)
            noisy = self.diffusion.q_sample(clean, t, noise=noise)

        latent, pooled, mask, ids_restore = self.forward_encoder(
            aug, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        uniformity = token_uniformity_loss(latent.reshape(batch, people, -1, latent.shape[-1])[:, 0])
        condition = self.build_global_local_condition(latent, pooled, ids_restore)
        prediction = self.forward_decoder(noisy, z=condition, t=t)
        native = self.forward_loss(noise, prediction, mask, t)
        # Keep normalization in FP32 so its unit-energy convention also holds under AMP.
        r = self.text_remap[:-1](text_features)
        r = self.text_remap[-1](r.float())
        h_person = pooled.reshape(batch, people, -1)
        h = (h_person * active[..., None]).sum(dim=1) / active.sum(dim=1, keepdim=True)
        zero = native.new_zeros(())
        t2s = self.text_to_skeleton_loss(
            noisy, noise, t, mask, r, active.reshape(-1), people
        ) if self.lambda_text_to_skeleton else zero
        s2t = self.skeleton_to_text_loss(r, h) if self.lambda_skeleton_to_text else zero
        loss = (native + self.lambda_loss_uni * uniformity
                + self.lambda_text_to_skeleton * t2s + self.lambda_skeleton_to_text * s2t)
        metrics = {'loss_diff': native.detach(), 'loss_uniformity': uniformity.detach(),
                   'loss_text_to_skeleton': t2s.detach(), 'loss_skeleton_to_text': s2t.detach(),
                   'text_energy': r.detach().square().mean(),
                   'text_batch_variance': r.detach().var(dim=0, unbiased=False).mean()}
        return loss, prediction, mask, metrics
