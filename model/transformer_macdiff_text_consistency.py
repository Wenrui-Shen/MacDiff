"""Native + T->S with independent-noise, bidirectional x0 consistency.

There is no text denoising decoder, target bank, EMA teacher, or stop-gradient
between the two reconstructed skeletons. Existing Stage1 models are unchanged.
"""
import math

import torch
from torch import nn

from .transformer_macdiff import Transformer as MacDiff
from .transformer_macdiff_text import (
    ResidualTextRemap, TextConditionedSkeletonDecoder, Transformer as TextStage1)
from .util import timestep_embedding, token_uniformity_loss


TARGET_MODE = 'reconstruction_consistency'


def validate_consistency_definition(lambda_consistency, lambda_text_to_skeleton,
                                    one_person, share_skeleton_decoder,
                                    text_target_mode, text_target_norm,
                                    lambda_skeleton_to_text, lambda_text_uniformity):
    if lambda_consistency is None:
        raise ValueError('lambda_consistency must be finite and non-negative')
    for name, value in (('lambda_consistency', lambda_consistency),
                        ('lambda_text_to_skeleton', lambda_text_to_skeleton)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be finite and non-negative')
    if lambda_text_to_skeleton == 0:
        raise ValueError('T->S ground-truth supervision must remain enabled')
    if one_person is not True or share_skeleton_decoder is not False:
        raise ValueError('This method requires one_person=True and no-share decoders')
    if text_target_mode != TARGET_MODE or text_target_norm != 'rms':
        raise ValueError('Use reconstruction_consistency with RMS sentence features')
    if lambda_skeleton_to_text != 0 or lambda_text_uniformity != 0:
        raise ValueError('S->T denoising and text uniformity are not part of this method')


def masked_reconstruction_consistency(diffusion, noisy_native, prediction_native,
                                      t_native, noisy_text, prediction_text,
                                      t_text, mask, condition_kept):
    """One undetached MSE trains both branches; every diffusion t is eligible.

    Arguments use native patch layout [N,L,patch_coordinates]. Only retained
    native conditions and encoder-masked positions contribute. No clipping,
    SNR weighting, timestep gating, or coordinate de-normalization is applied.
    """
    fraction = condition_kept.float().mean()
    if not condition_kept.any():
        # Keep a differentiable zero for both predictions, including CPU tests
        # and a DDP rank whose whole mini-batch lost its native condition.
        return (prediction_native.float().sum() + prediction_text.float().sum()) * 0., fraction
    with torch.cuda.amp.autocast(enabled=False):
        keep = condition_kept
        native_x0 = diffusion._predict_xstart_from_eps(
            noisy_native[keep].float(), t_native[keep], prediction_native[keep].float())
        text_x0 = diffusion._predict_xstart_from_eps(
            noisy_text[keep].float(), t_text[keep], prediction_text[keep].float())
        squared_error = (native_x0 - text_x0).square().mean(dim=-1)
        weights = mask[keep].float()
        loss = ((squared_error * weights).sum(dim=-1) / weights.sum(dim=-1)).mean()
    return loss, fraction


class Transformer(MacDiff):
    supports_text_cache = True
    # Reuse the established cache validation and RMS conversion, without
    # constructing TextStage1 or its S->T decoder/target-bank machinery.
    validate_text_tokens = TextStage1.validate_text_tokens
    fixed_clip_target = TextStage1.fixed_clip_target

    def __init__(self, text_input_dim=512, text_hidden_dim=2048,
                 text_context_length=7, text_target_mode=TARGET_MODE,
                 text_target_norm='rms', share_skeleton_decoder=False,
                 lambda_text_to_skeleton=1., lambda_consistency=1.,
                 lambda_skeleton_to_text=0., lambda_text_uniformity=0., **kwargs):
        validate_consistency_definition(
            lambda_consistency, lambda_text_to_skeleton,
            kwargs.get('one_person', True), share_skeleton_decoder,
            text_target_mode, text_target_norm,
            lambda_skeleton_to_text, lambda_text_uniformity)
        if min(text_input_dim, text_hidden_dim) < 1 or text_context_length < 2:
            raise ValueError('Text dimensions must be positive; context includes global and local')
        super().__init__(**kwargs)
        if self.diff_prediction != 'noise' or self.dim_t_embed != 64:
            raise ValueError('Use native epsilon prediction and dim_t_embed=64')
        if not 0. <= self.uncond_ratio <= 1.:
            raise ValueError('uncond_ratio must be in [0,1]')
        self.text_input_dim = text_input_dim
        self.text_context_length = text_context_length
        self.text_target_mode = text_target_mode
        self.text_target_norm = text_target_norm
        self.share_skeleton_decoder = False
        self.lambda_text_to_skeleton = float(lambda_text_to_skeleton)
        self.lambda_consistency = float(lambda_consistency)
        # Required by the existing cache/resume metadata contract, not losses
        # or modules left behind from the removed S->T method.
        self.lambda_skeleton_to_text = 0.
        self.lambda_text_uniformity = 0.
        self.text_remap = ResidualTextRemap(text_input_dim, text_hidden_dim)
        self.text_remap.apply(self._init_weights)
        self.text_person_embedding = nn.Embedding(2, text_input_dim)
        self.text_position_embedding = nn.Embedding(text_context_length, text_input_dim)
        nn.init.normal_(self.text_person_embedding.weight, std=.02)
        nn.init.normal_(self.text_position_embedding.weight, std=.02)
        self.text_skeleton_decoder = TextConditionedSkeletonDecoder(
            self, shared=False, text_dim=text_input_dim)

    def update_text_target(self):
        """Engine compatibility: this method has no EMA or persistent targets."""
        return

    def _native_noise_prediction(self, noisy, condition, t):
        # The native forward keeps its 0.1 condition dropout. Expose the keep
        # mask so consistency never asks an unconditional sample to train E.
        kept = torch.rand(noisy.shape[0], device=noisy.device) > self.uncond_ratio
        condition = condition * kept[:, None, None].to(condition.dtype)
        x = self.decoder_embed(noisy)
        n, tp, vp, dim = x.shape
        x = (x + self.decoder_pos_embed[:, :, :vp]
             + self.decoder_temp_embed[:, :tp]).reshape(n, tp * vp, dim)
        time = timestep_embedding(t, self.dim_t_embed)
        for block in self.decoder_blocks:
            x = block(x, z=condition, t=time)
        return self.decoder_pred(self.decoder_norm(x)), kept

    def forward(self, source, source_aug, text_features=None,
                text_tokens=None, text_token_mask=None, text_person_ids=None,
                text_positions=None, mask_ratio=.9, motion_stride=1,
                motion_aware_tau=-1, enable_ose=False):
        if enable_ose:
            raise ValueError('Text reconstruction consistency does not support OSE')
        length = self.joints_embed.t_grid_size * self.joints_embed.grid_size
        if (not 0. < mask_ratio < 1.
                or not 0 < round(length * (1. - mask_ratio)) < length):
            raise ValueError('Masking must leave visible and masked skeleton patches')
        if source.ndim != 5 or source_aug.shape != source.shape or source.shape[1] != 3:
            raise ValueError('Expected matching source/source_aug [N,3,T,V,M]')
        batch = source.shape[0]
        if text_features is None or text_features.shape != (batch, self.text_input_dim):
            raise ValueError('Expected one cached global sentence vector per person0')
        if any(value is None for value in (
                text_tokens, text_token_mask, text_person_ids, text_positions)):
            raise ValueError('This method requires the global + local sentence cache')

        with torch.no_grad():
            raw = source[..., 0].permute(0, 2, 3, 1).contiguous()
            aug = source_aug[..., 0].permute(0, 2, 3, 1).contiguous()
            active = raw.abs().sum(dim=(1, 2, 3)) > 0
            if not active.any():
                raise ValueError('The entire batch has an empty retained person0')
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center)
            aug = self._normalize_sequence(aug, center)
            # Independent draws from the entire existing native schedule.
            # Independently drawn integer timesteps may coincide by chance.
            t_native, _ = self.schedule_sampler.sample(batch, clean.device)
            noise_native = torch.randn_like(clean)
            noisy_native = self.diffusion.q_sample(clean, t_native, noise=noise_native)
            t_text, _ = self.schedule_sampler.sample(batch, clean.device)
            noise_text = torch.randn_like(clean)
            noisy_text = self.diffusion.q_sample(clean, t_text, noise=noise_text)

        latent, pooled, mask, ids_restore = self.forward_encoder(
            aug, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        condition = self.build_global_local_condition(latent, pooled, ids_restore)
        prediction_native, kept = self._native_noise_prediction(
            noisy_native, condition, t_native)
        native = self.forward_loss(
            noise_native[active], prediction_native[active].float(), mask[active], t_native[active])
        uniformity = token_uniformity_loss(latent[active])

        global_clip = text_features[active]
        tokens, valid = text_tokens[active], text_token_mask[active]
        people, positions = text_person_ids[active], text_positions[active]
        self.validate_text_tokens(tokens, valid, people, positions)
        r = self.text_remap(self.fixed_clip_target(global_clip))
        local = self.text_remap(self.fixed_clip_target(tokens, valid))
        people = people.masked_fill(~valid, 0)
        positions = positions.masked_fill(~valid, 0)
        local = (local + self.text_person_embedding(people)
                 + self.text_position_embedding(positions)).masked_fill(~valid[..., None], 0)
        prediction_text = self.text_skeleton_decoder(
            noisy_text[active], t_text[active], r, local, valid)
        t2s = self.forward_loss(
            noise_text[active], prediction_text.float(), mask[active], t_text[active])

        if self.lambda_consistency:
            consistency, fraction = masked_reconstruction_consistency(
                self.diffusion, self.patchify(noisy_native[active]), prediction_native[active],
                t_native[active], self.patchify(noisy_text[active]), prediction_text,
                t_text[active], mask[active], kept[active])
        else:
            consistency = prediction_native.float().sum() * 0. + prediction_text.float().sum() * 0.
            fraction = consistency.detach()
        loss = (native + self.lambda_loss_uni * uniformity
                + self.lambda_text_to_skeleton * t2s
                + self.lambda_consistency * consistency)
        metrics = {'loss_diff': native.detach(), 'loss_uniformity': uniformity.detach(),
                   'loss_text_to_skeleton': t2s.detach(),
                   'loss_consistency': consistency.detach(),
                   'consistency_fraction': fraction.detach()}
        return loss, prediction_native, mask, metrics
