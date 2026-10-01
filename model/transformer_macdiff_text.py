"""Stage1 with fixed CLIP, parameter-EMA or per-sample blended text targets.

Native encoder/decoder names are unchanged for downstream encoder transfer.
The skeleton decoder body can be shared; only S->T sends a direct
auxiliary gradient to the skeleton encoder. T->S trains the text remap.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer_macdiff import Transformer as MacDiff, MLP, FeatureModulation
from .util import timestep_embedding, token_uniformity_loss


class TextConditionedSkeletonDecoder(nn.Module):
    """Read token memory at each noisy skeleton position, then apply native modulation."""
    def __init__(self, source, shared=False, text_dim=None):
        super().__init__()
        # Shared modules remain registered only under the native model names.
        # Pass their owner at forward time, avoiding duplicate checkpoint keys.
        if not shared:
            for name in ('decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred'):
                setattr(self, name, copy.deepcopy(getattr(source, name)))
            self.pos_embed = nn.Parameter(source.decoder_pos_embed.detach().clone())
            self.temp_embed = nn.Parameter(source.decoder_temp_embed.detach().clone())
        self.dim_t_embed = source.dim_t_embed
        dim = source.dim_feat
        self.text_input = (nn.Identity() if text_dim is None or text_dim == dim
                           else nn.Linear(text_dim, dim))
        self.text_readers = nn.ModuleList([
            nn.MultiheadAttention(dim, block.attn.num_heads, dropout=0.)
            for block in source.decoder_blocks])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(dim, elementwise_affine=False) for _ in source.decoder_blocks])

    def forward(self, noisy_skeleton, t, condition, token_condition, token_mask, source=None):
        body = self if source is None else source
        pos = self.pos_embed if source is None else source.decoder_pos_embed
        temp = self.temp_embed if source is None else source.decoder_temp_embed
        x = body.decoder_embed(noisy_skeleton)
        n, tp, vp, dim = x.shape
        x = (x + pos[:, :, :vp] + temp[:, :tp]).reshape(n, tp * vp, dim)
        time = timestep_embedding(t, self.dim_t_embed)
        memory = self.text_input(
            torch.cat([condition[:, None, :], token_condition], dim=1)).transpose(0, 1)
        token_mask = torch.cat([torch.ones_like(token_mask[:, :1]), token_mask], dim=1)
        for block, reader, norm in zip(body.decoder_blocks, self.text_readers, self.query_norms):
            # Q is the current noisy decoder state, never the clean skeleton encoder.
            local, _ = reader(norm(x).transpose(0, 1), memory, memory,
                              key_padding_mask=~token_mask, need_weights=False)
            z = local.transpose(0, 1)
            x = block(x, t=time, z=z)
        return body.decoder_pred(body.decoder_norm(x))


class TextNoiseBlock(nn.Module):
    """MacDiff modulation order with masked text self-attention."""
    def __init__(self, dim, time_dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.reader = nn.MultiheadAttention(dim, heads, dropout=0.)
        self.norm1 = FeatureModulation(dim, time_dim, 0.)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.)
        self.norm2 = FeatureModulation(dim, time_dim, 0.)
        self.mlp = MLP(dim, dim * 4, dim)

    def forward(self, x, time, skeleton, skeleton_mask, valid):
        z, _ = self.reader(self.query_norm(x).transpose(0, 1), skeleton, skeleton,
                           key_padding_mask=~skeleton_mask, need_weights=False)
        z = z.transpose(0, 1)
        q = self.norm1(x, z=z, t=time).transpose(0, 1)
        update, _ = self.attn(q, q, q, key_padding_mask=~valid, need_weights=False)
        x = x + update.transpose(0, 1)
        x = x + self.mlp(self.norm2(x, z=z, t=time))
        return x.masked_fill(~valid[..., None], 0)


class TextNoiseDecoder(nn.Module):
    """Denoise global + local text tokens conditioned on visible skeleton tokens."""
    def __init__(self, dim, time_dim, hidden_dim, depth, heads, context_length,
                 skeleton_dim=None):
        super().__init__()
        self.time_dim = time_dim
        self.input = nn.Linear(dim, hidden_dim)
        self.skeleton_input = nn.Linear(dim if skeleton_dim is None else skeleton_dim, hidden_dim)
        # Decoder-owned structural hints; no clean text feature enters this branch.
        self.person_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(context_length, hidden_dim)
        self.global_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.global_embedding, std=.02)
        self.blocks = nn.ModuleList([
            TextNoiseBlock(hidden_dim, time_dim, heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))

    def forward(self, noisy_text, t, skeleton, skeleton_mask, valid, person_ids, positions):
        person_ids = person_ids.masked_fill(~valid[:, 1:], 0)
        positions = positions.masked_fill(~valid[:, 1:], 0)
        structure = torch.cat([self.global_embedding.expand(noisy_text.shape[0], -1, -1),
            self.person_embedding(person_ids) + self.position_embedding(positions)], dim=1)
        x = self.input(noisy_text.masked_fill(~valid[..., None], 0)) + structure
        x = x.masked_fill(~valid[..., None], 0)
        skeleton = self.skeleton_input(skeleton.masked_fill(~skeleton_mask[..., None], 0)).transpose(0, 1)
        time = timestep_embedding(t, self.time_dim)
        for block in self.blocks:
            x = block(x, time, skeleton, skeleton_mask, valid)
        return self.output(x).masked_fill(~valid[..., None], 0)


class ResidualTextRemap(nn.Module):
    """Learn a residual content remap while preserving RMS CLIP scale."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def reset_identity(self):
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features):
        # Target-value changes can be smaller than FP16 resolution.
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()
            result = features + self.body(features)
            return result * torch.rsqrt(
                result.square().mean(dim=-1, keepdim=True).clamp_min(1e-12))


def masked_text_uniformity_loss(features, valid):
    """Skeleton token-uniformity formula, excluding padded text token pairs."""
    normalized = F.normalize(
        features.float().masked_fill(~valid[..., None], 0), dim=-1)
    similarities = normalized @ normalized.transpose(1, 2)
    pairs = valid[:, :, None] & valid[:, None, :]
    return ((similarities.square() * pairs).sum(dim=(1, 2)) /
            pairs.sum(dim=(1, 2)).clamp_min(1)).mean()


def text_content_batch_variance(global_features, local_features, valid):
    """Channel variance over online global + valid local content vectors."""
    vectors = torch.cat([global_features.detach().float(),
                         local_features.detach().float()[valid]], dim=0)
    return vectors.var(dim=0, unbiased=False).mean()


class Transformer(MacDiff):
    supports_text_cache = True

    def __init__(self, text_input_dim=512, text_hidden_dim=512,
                 text_context_length=77,
                 text_decoder_hidden_dim=256, text_decoder_depth=5,
                 share_skeleton_decoder=False,
                 text_target_mode='remap',
                 text_target_norm='none',
                 text_target_momentum=0.999, text_target_update_ratio=0.1,
                 lambda_text_uniformity=0.,
                 lambda_text_to_skeleton=1., lambda_skeleton_to_text=1., **kwargs):
        super().__init__(**kwargs)
        if not isinstance(share_skeleton_decoder, bool):
            raise ValueError('share_skeleton_decoder must be a boolean')
        self.share_skeleton_decoder = share_skeleton_decoder
        if self.diff_prediction != 'noise':
            raise ValueError('Text Stage1 requires diff_prediction=noise')
        if self.dim_t_embed != 64:
            raise ValueError('The native MacDiff decoder requires dim_t_embed=64')
        if min(text_input_dim, text_hidden_dim, text_decoder_hidden_dim, text_decoder_depth) < 1:
            raise ValueError('Text dimensions and decoder depth must be positive')
        if any(not math.isfinite(w) or w < 0 for w in (
                lambda_text_to_skeleton, lambda_skeleton_to_text, lambda_text_uniformity)):
            raise ValueError('Text loss weights must be finite and non-negative')
        self.text_input_dim = text_input_dim
        if text_context_length < 2:
            raise ValueError('text_context_length must include BOS and EOS positions')
        self.text_context_length = text_context_length
        self.lambda_text_to_skeleton = float(lambda_text_to_skeleton)
        self.lambda_skeleton_to_text = float(lambda_skeleton_to_text)
        if text_target_mode not in ('remap', 'fixed_clip', 'ema_remap', 'sample_target_blend'):
            raise ValueError('Unknown text_target_mode')
        if text_target_mode == 'fixed_clip' and self.lambda_text_to_skeleton:
            raise ValueError('fixed_clip requires lambda_text_to_skeleton=0')
        if text_target_norm not in ('none', 'rms'):
            raise ValueError('text_target_norm must be none or rms')
        if text_target_mode == 'remap' and text_target_norm != 'none':
            raise ValueError('text_target_norm is only supported with fixed_clip or remapped target modes')
        if text_target_mode in ('ema_remap', 'sample_target_blend') and text_target_norm != 'rms':
            raise ValueError('Remapped text targets require text_target_norm=rms')
        if not math.isfinite(text_target_momentum) or not 0 <= text_target_momentum < 1:
            raise ValueError('text_target_momentum must be in [0, 1)')
        if (not math.isfinite(text_target_update_ratio)
                or not 0 < text_target_update_ratio <= 1):
            raise ValueError('text_target_update_ratio must be in (0, 1]')
        if text_target_mode not in ('ema_remap', 'sample_target_blend') and lambda_text_uniformity:
            raise ValueError('lambda_text_uniformity requires a remapped text target mode')
        self.text_target_mode = text_target_mode
        self.text_target_norm = text_target_norm
        self.text_target_momentum = float(text_target_momentum)
        self.text_target_update_ratio = float(text_target_update_ratio)
        self.lambda_text_uniformity = float(lambda_text_uniformity)
        if text_target_mode == 'remap':
            self.text_remap = nn.Sequential(
                nn.Linear(text_input_dim, text_hidden_dim), nn.GELU(),
                nn.Linear(text_hidden_dim, self.dim_feat),
                nn.LayerNorm(self.dim_feat, elementwise_affine=False))
            self.text_person_embedding = nn.Embedding(2, self.dim_feat)
            self.text_position_embedding = nn.Embedding(text_context_length, self.dim_feat)
            nn.init.normal_(self.text_person_embedding.weight, std=.02)
            nn.init.normal_(self.text_position_embedding.weight, std=.02)
            self.text_skeleton_decoder = TextConditionedSkeletonDecoder(self, share_skeleton_decoder)
        elif text_target_mode in ('ema_remap', 'sample_target_blend'):
            self.text_remap = ResidualTextRemap(text_input_dim, text_hidden_dim)
            self.text_remap.apply(self._init_weights)
            if text_target_mode == 'ema_remap':
                # Legacy experiment: average remap parameters, not target values.
                self.text_target_remap = copy.deepcopy(self.text_remap)
                self.text_target_remap.reset_identity()
                self.text_target_remap.requires_grad_(False)
            self.text_person_embedding = nn.Embedding(2, text_input_dim)
            self.text_position_embedding = nn.Embedding(text_context_length, text_input_dim)
            nn.init.normal_(self.text_person_embedding.weight, std=.02)
            nn.init.normal_(self.text_position_embedding.weight, std=.02)
            self.text_skeleton_decoder = TextConditionedSkeletonDecoder(
                self, share_skeleton_decoder, text_dim=text_input_dim)
        target_dim = text_input_dim if text_target_mode != 'remap' else self.dim_feat
        self.text_noise_decoder = TextNoiseDecoder(
            target_dim, self.dim_t_embed, text_decoder_hidden_dim, text_decoder_depth,
            self.decoder_blocks[0].attn.num_heads, text_context_length,
            skeleton_dim=self.dim_feat)
        if text_target_mode == 'remap':
            self.text_remap.apply(self._init_weights)
        self.text_noise_decoder.apply(self._init_weights)
        # Disabled branches have no optimizer/DDP parameters awaiting gradients.
        # With T->S disabled the remap stays at its fixed initialization; it cannot
        # learn from S->T because that branch deliberately detaches its target.
        if text_target_mode in ('remap', 'ema_remap', 'sample_target_blend') and not self.lambda_text_to_skeleton:
            self.text_person_embedding.requires_grad_(False)
            self.text_position_embedding.requires_grad_(False)
            self.text_skeleton_decoder.requires_grad_(False)
            if text_target_mode == 'remap' or not self.lambda_text_uniformity:
                self.text_remap.requires_grad_(False)
        if not self.lambda_skeleton_to_text:
            self.text_noise_decoder.requires_grad_(False)

    def validate_text_tokens(self, features, valid, person_ids, positions):
        if (features.ndim != 3 or features.shape[-1] != self.text_input_dim
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or person_ids.shape != valid.shape or positions.shape != valid.shape
                or person_ids.dtype != torch.long or positions.dtype != torch.long
                or not valid.any(dim=1).all()):
            raise ValueError('Expected token features, nonempty masks, person IDs and original positions')
        if (torch.any(person_ids[valid] < 0) or torch.any(person_ids[valid] >= 2)
                or torch.any(positions[valid] < 1)
                or torch.any(positions[valid] >= self.text_context_length)):
            raise ValueError('Invalid text person/position IDs')

    def remap_tokens(self, features, valid, person_ids, positions, return_content=False):
        self.validate_text_tokens(features, valid, person_ids, positions)
        # Padding must be irrelevant even if its caller-supplied values are nonzero.
        features = features.masked_fill(~valid[..., None], 0)
        person_ids = person_ids.masked_fill(~valid, 0)
        positions = positions.masked_fill(~valid, 0)
        if self.text_target_mode in ('ema_remap', 'sample_target_blend'):
            content = self.text_remap(self.fixed_clip_target(features, valid))
            result = (content + self.text_person_embedding(person_ids)
                      + self.text_position_embedding(positions)).masked_fill(
                          ~valid[..., None], 0)
            return (result, content.masked_fill(~valid[..., None], 0)) if return_content else result
        result = self.text_remap[:-1](features).float()
        # Reuse the MLP result for logging content before structural embeddings.
        content = self.text_remap[-1](result).masked_fill(~valid[..., None], 0) if return_content else None
        result = result + self.text_person_embedding(person_ids) + self.text_position_embedding(positions)
        result = self.text_remap[-1](result.float()).masked_fill(~valid[..., None], 0)
        return (result, content) if return_content else result

    def fixed_clip_target(self, features, valid=None):
        """Detach fixed CLIP targets and optionally give each token unit RMS."""
        result = features.detach().float()
        if valid is not None:
            result = result.masked_fill(~valid[..., None], 0)
        if self.text_target_norm == 'rms':
            inverse_rms = torch.rsqrt(
                result.square().mean(dim=-1, keepdim=True).clamp_min(1e-12))
            result = result * inverse_rms
        if valid is not None:
            result = result.masked_fill(~valid[..., None], 0)
        return result

    @torch.no_grad()
    def update_text_target(self):
        """Call once after each successful optimizer step, never each forward."""
        if self.text_target_mode != 'ema_remap':
            return
        momentum = self.text_target_momentum
        for target, online in zip(self.text_target_remap.parameters(), self.text_remap.parameters()):
            target.mul_(momentum).add_(online.detach(), alpha=1. - momentum)

    def text_to_skeleton_loss(self, noisy, noise, t, mask, r, active, people,
                              token_condition, token_mask):
        condition = r[:, None, :].expand(-1, people, -1).reshape(-1, r.shape[-1])
        token_condition = token_condition.repeat_interleave(people, dim=0)
        token_mask = token_mask.repeat_interleave(people, dim=0)
        prediction = self.text_skeleton_decoder(
            noisy, t, condition, token_condition, token_mask,
            source=self if self.share_skeleton_decoder else None)
        return self.forward_loss(noise[active], prediction[active], mask[active], t[active])

    def skeleton_to_text_loss(self, memory, skeleton, valid, skeleton_mask, person_ids, positions):
        # Detach BEFORE corruption; neither global nor local targets train the remap.
        target = memory.detach().float()
        t, _ = self.schedule_sampler.sample(target.shape[0], target.device)
        noise = torch.randn_like(target).masked_fill(~valid[..., None], 0)
        noisy = self.diffusion.q_sample(target, t, noise=noise)
        predicted_noise = self.text_noise_decoder(
            noisy, t, skeleton, skeleton_mask, valid, person_ids, positions)
        # One mean over all valid global/local tokens and channels, no separate weights.
        return F.mse_loss(predicted_noise.float()[valid], noise[valid])

    def forward(self, source, source_aug, text_features=None, mask_ratio=.9,
                motion_stride=1, motion_aware_tau=-1, enable_ose=False,
                text_tokens=None, text_token_mask=None, text_person_ids=None,
                text_positions=None, text_target_global=None,
                text_target_tokens=None, **unused):
        if enable_ose:
            raise ValueError('Text Stage1 does not support OSE routing')
        if not 0 < mask_ratio < 1:
            raise ValueError('Text Stage1 requires 0 < mask_ratio < 1')
        text_rows = source.shape[0] * (1 if self.one_person else source.shape[-1])
        if text_features is None or text_features.shape != (text_rows, self.text_input_dim):
            raise ValueError('Expected one cached sentence vector per retained skeleton person')
        if (self.lambda_text_to_skeleton or self.lambda_skeleton_to_text) and (any(value is None for value in (
                text_tokens, text_token_mask, text_person_ids, text_positions))
                or text_tokens.shape[0] != text_rows):
            raise ValueError('Multi-token training requires the v2 token cache and metadata')
        if self.text_target_mode == 'sample_target_blend' and (
                text_tokens is None or text_target_global is None or text_target_tokens is None
                or text_target_global.shape != text_features.shape
                or text_target_tokens.shape != text_tokens.shape):
            raise ValueError('sample_target_blend requires matching persistent target vectors')
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
            # A crop can have an empty retained person. Existing active-row
            # masks exclude it from all losses; do not reject the whole batch.
            if not active.any():
                raise ValueError('The entire batch contains no active retained skeleton person')
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
        active_rows = active.reshape(-1)
        if not text_token_mask[active_rows].any(dim=1).all():
            raise ValueError("An active skeleton person has no matching cached description")
        uniformity = token_uniformity_loss(latent[active_rows])
        condition = self.build_global_local_condition(latent, pooled, ids_restore)
        prediction = self.forward_decoder(noisy, z=condition, t=t)
        native = self.forward_loss(noise[active_rows], prediction[active_rows], mask[active_rows], t[active_rows])
        text_features = text_features[active_rows]
        text_tokens = text_tokens[active_rows]
        text_token_mask = text_token_mask[active_rows]
        text_person_ids = text_person_ids[active_rows]
        text_positions = text_positions[active_rows]
        if self.text_target_mode == 'sample_target_blend':
            text_target_global = text_target_global[active_rows]
            text_target_tokens = text_target_tokens[active_rows]
        zero = native.new_zeros(())
        text_uniformity = zero
        if self.text_target_mode == 'fixed_clip':
            self.validate_text_tokens(text_tokens, text_token_mask, text_person_ids, text_positions)
            # Structure belongs only to the decoder, never to the clean target.
            # RMS normalization preserves every CLIP direction and only changes scale.
            r = self.fixed_clip_target(text_features)
            token_condition = self.fixed_clip_target(text_tokens, text_token_mask)
            remapped_local = token_condition  # fixed_clip has no trainable remap
            memory = torch.cat([r[:, None], token_condition], dim=1)
        elif self.text_target_mode in ('ema_remap', 'sample_target_blend'):
            self.validate_text_tokens(text_tokens, text_token_mask, text_person_ids, text_positions)
            fixed_global = self.fixed_clip_target(text_features)
            fixed_local = self.fixed_clip_target(text_tokens, text_token_mask)
            r = self.text_remap(fixed_global)
            remapped_local = self.text_remap(fixed_local).masked_fill(
                ~text_token_mask[..., None], 0)
            text_uniformity = masked_text_uniformity_loss(remapped_local, text_token_mask)
            person_ids = text_person_ids.masked_fill(~text_token_mask, 0)
            positions = text_positions.masked_fill(~text_token_mask, 0)
            token_condition = (remapped_local + self.text_person_embedding(person_ids)
                + self.text_position_embedding(positions)).masked_fill(
                    ~text_token_mask[..., None], 0)
            with torch.no_grad():
                if self.text_target_mode == 'ema_remap':
                    target_global = self.text_target_remap(fixed_global)
                    target_local = self.text_target_remap(fixed_local)
                else:
                    target_global = text_target_global.detach().float()
                    target_local = text_target_tokens.detach().float()
                target_local = target_local.masked_fill(~text_token_mask[..., None], 0)
                memory = torch.cat([target_global[:, None], target_local], dim=1)
        else:
            # Keep legacy normalization in FP32 under AMP.
            r = self.text_remap[:-1](text_features)
            r = self.text_remap[-1](r.float())
            token_condition, remapped_local = self.remap_tokens(
                text_tokens, text_token_mask, text_person_ids, text_positions,
                return_content=True)
            memory = torch.cat([r[:, None], token_condition], dim=1)
        memory_mask = torch.cat([torch.ones_like(text_token_mask[:, :1]), text_token_mask], dim=1)
        h = torch.cat([pooled[active_rows], latent[active_rows]], dim=1)
        h_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        t2s = self.text_to_skeleton_loss(
            noisy[active_rows], noise[active_rows], t[active_rows], mask[active_rows], r,
            torch.ones(r.shape[0], dtype=torch.bool, device=r.device), 1,
            token_condition, text_token_mask
        ) if self.lambda_text_to_skeleton else zero
        s2t = self.skeleton_to_text_loss(
            memory, h, memory_mask, h_mask, text_person_ids, text_positions
        ) if self.lambda_skeleton_to_text else zero
        loss = (native + self.lambda_loss_uni * uniformity
                + self.lambda_text_uniformity * text_uniformity
                + self.lambda_text_to_skeleton * t2s + self.lambda_skeleton_to_text * s2t)
        metrics = {'loss_diff': native.detach(), 'loss_uniformity': uniformity.detach(),
                   'loss_text_uniformity': text_uniformity.detach(),
                   'loss_text_to_skeleton': t2s.detach(), 'loss_skeleton_to_text': s2t.detach(),
                   'text_energy': memory[:, 0].detach().square().mean(),
                   'text_batch_variance': text_content_batch_variance(
                       r, remapped_local, text_token_mask)}
        if self.lambda_text_to_skeleton or self.lambda_skeleton_to_text:
            metrics['text_valid_tokens'] = text_token_mask.sum(dim=1).float().mean()
        return loss, prediction, mask, metrics
