"""The native final norm and decoder jointly supervise the last three layers."""
import math

import torch

from .transformer_macdiff import Transformer as NativeTransformer
from .util import timestep_embedding, token_uniformity_loss


MULTILAYER_VERSION = 'last3_shared_norm_decoder_final_layer_lp_v3'


def validate_multilayer_options(depth, lambda_aux_diffusion, one_person=True,
                                diff_prediction='noise', lambda_loss_uni=.02,
                                layer_mask_ratio=0., uncond_ratio=.1,
                                multilayer_version=MULTILAYER_VERSION):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 3:
        raise ValueError('Multilayer diffusion requires encoder depth >= 3')
    for name, value in (('lambda_aux_diffusion', lambda_aux_diffusion),
                        ('lambda_loss_uni', lambda_loss_uni)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(name + ' must be finite and non-negative')
    if one_person is not True or diff_prediction != 'noise':
        raise ValueError('This experiment requires person0 and epsilon prediction')
    if layer_mask_ratio != 0.:
        raise ValueError('Use layer_mask_ratio=0; condition dropout is shared across branches')
    if (isinstance(uncond_ratio, bool) or not isinstance(uncond_ratio, (int, float))
            or not math.isfinite(uncond_ratio) or not 0 <= uncond_ratio <= 1):
        raise ValueError('uncond_ratio must be finite and in [0, 1]')
    if multilayer_version != MULTILAYER_VERSION:
        raise ValueError('Unsupported multilayer_version')


def combine_multilayer_losses(final_loss, auxiliary_losses, uniformity,
                              lambda_aux_diffusion, lambda_loss_uni):
    """Preserve the final-layer coefficient and add the MEAN auxiliary loss."""
    if len(auxiliary_losses) != 2:
        raise ValueError('Exactly two auxiliary diffusion losses are required')
    auxiliary_mean = (auxiliary_losses[0] + auxiliary_losses[1]) * .5
    return (final_loss + lambda_aux_diffusion * auxiliary_mean
            + lambda_loss_uni * uniformity), auxiliary_mean


def decode_skeleton(decoder, noisy, condition, time):
    """Native decoder arithmetic; the caller already applied shared condition dropout."""
    time = timestep_embedding(time, decoder.dim_t_embed)
    hidden = decoder.decoder_embed(noisy)
    batch, temporal, joints, width = hidden.shape
    hidden = hidden + decoder.decoder_pos_embed + decoder.decoder_temp_embed
    hidden = hidden.reshape(batch, temporal * joints, width)
    for block in decoder.decoder_blocks:
        hidden = block(hidden, z=condition, t=time)
    return decoder.decoder_pred(decoder.decoder_norm(hidden))


class Transformer(NativeTransformer):
    supports_pretrain_metrics = True
    pretrain_metric_prefix = 'multilayer_diffusion/'

    def __init__(self, lambda_aux_diffusion=.1,
                 multilayer_version=MULTILAYER_VERSION, **kwargs):
        kwargs.setdefault('depth', 8)
        kwargs.setdefault('decoder_depth', 3)
        validate_multilayer_options(
            kwargs['depth'], lambda_aux_diffusion,
            one_person=kwargs.get('one_person', True),
            diff_prediction=kwargs.get('diff_prediction', 'noise'),
            lambda_loss_uni=kwargs.get('lambda_loss_uni', 0.),
            layer_mask_ratio=kwargs.get('layer_mask_ratio', 0.),
            uncond_ratio=kwargs.get('uncond_ratio', .1),
            multilayer_version=multilayer_version)
        super().__init__(**kwargs)
        self.lambda_aux_diffusion = float(lambda_aux_diffusion)
        self.multilayer_version = multilayer_version
        self.supervision_layers = tuple(range(len(self.blocks) - 2, len(self.blocks) + 1))
        # No reinitialization of the root encoder/decoder: their names, shapes,
        # initial values and parameter count match native MacDiff exactly.
        # The same final LayerNorm, including affine weight/bias, serves all
        # three encoder layers. No layer-specific trainable adapters are added.
        # All branches call the SAME decoder modules, including their input
        # embedding, positional/time conditioning, modulation, norm and head.
        print('Multilayer diffusion: encoder layers {}; shared final norm and decoder; '
              'L_final + {} * mean(L_aux) + {} * uniformity_final; LP uses final layer'
              .format(self.supervision_layers, self.lambda_aux_diffusion, self.lambda_loss_uni))

    def forward_encoder_multilayer(self, x, *, x_orig, mask_ratio, motion_aware_tau):
        x = self.joints_embed(x)
        batch, temporal, joints, width = x.shape
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        x = x.reshape(batch, temporal * joints, width)
        if motion_aware_tau <= 0.:
            x, mask, ids_restore, _ = self.random_masking(x, mask_ratio)
        else:
            original = self.patchify(x_orig).reshape(batch, temporal, joints, -1)
            x, mask, ids_restore, _ = self.motion_aware_random_masking(
                x, x_orig=original, mask_ratio=mask_ratio, tau=motion_aware_tau)
        if x.shape[1] == 0:
            raise ValueError('At least one visible encoder token is required')
        features = {}
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if layer in self.supervision_layers:
                features[layer] = self.norm(x)
        return features, mask, ids_restore

    def forward_macdiff(self, source, source_aug, mask_ratio=.9,
                        motion_stride=1, motion_aware_tau=-1):
        if source.ndim != 5 or source.shape != source_aug.shape:
            raise ValueError('Expected matching [B,C,T,V,M] skeleton inputs')
        if not 0 < mask_ratio < 1:
            raise ValueError('Multilayer diffusion requires mask_ratio in (0, 1)')
        with torch.no_grad():
            raw = self._sequence_layout(source).float()
            augmented = self._sequence_layout(source_aug).float()
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center=center)
            augmented = self._normalize_sequence(augmented, center=center)
            if not torch.isfinite(clean).all() or not torch.isfinite(augmented).all():
                raise FloatingPointError('Nonfinite normalized skeleton input')
            time, _ = self.schedule_sampler.sample(clean.shape[0], clean.device)
            noise = torch.randn_like(clean)
            noisy = self.diffusion.q_sample(clean, time, noise=noise)

        features, mask, ids_restore = self.forward_encoder_multilayer(
            augmented, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        if torch.any(mask.sum(dim=-1) <= 0):
            raise ValueError('At least one masked patch per person is required')
        # Native per-sample .1 condition dropout, drawn once for all three calls.
        keep = (torch.rand(clean.shape[0], device=clean.device) > self.uncond_ratio).float()
        losses = {}
        final_prediction = None
        final_layer = self.supervision_layers[-1]
        with torch.cuda.amp.autocast(enabled=False):
            for layer in self.supervision_layers:
                latent = features[layer].float()
                if not torch.isfinite(latent).all():
                    raise FloatingPointError('Nonfinite encoder features at layer %d' % layer)
                condition = self.build_global_local_condition(
                    latent, latent.mean(dim=1, keepdim=True), ids_restore)
                condition = condition * keep[:, None, None]
                prediction = decode_skeleton(self, noisy.float(), condition, time)
                if layer == final_layer:
                    final_prediction = prediction
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite diffusion prediction at layer %d' % layer)
                losses[layer] = self.forward_loss(noise.float(), prediction.float(), mask, time)
            uniformity = token_uniformity_loss(features[final_layer].float())
            if not torch.is_tensor(uniformity):
                raise FloatingPointError('Nonfinite final-layer token uniformity')
            total, auxiliary_mean = combine_multilayer_losses(
                losses[final_layer], [losses[layer] for layer in self.supervision_layers[:-1]],
                uniformity, self.lambda_aux_diffusion, self.lambda_loss_uni)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite multilayer diffusion loss')
        metrics = {'loss_diffusion_layer%d' % layer: value.detach()
                   for layer, value in losses.items()}
        metrics.update(loss_native=(losses[final_layer] + self.lambda_loss_uni * uniformity).detach(),
                       loss_auxiliary_mean=auxiliary_mean.detach(),
                       loss_auxiliary_weighted=(self.lambda_aux_diffusion * auxiliary_mean).detach(),
                       loss_uniformity=uniformity.detach())
        return total, final_prediction, mask, metrics

    def forward(self, source, source_aug, mask_ratio=.9, motion_stride=1,
                motion_aware_tau=-1, enable_ose=False):
        if enable_ose:
            raise ValueError('Multilayer diffusion uses the pure-skeleton Stage1 task')
        return self.forward_macdiff(source, source_aug, mask_ratio=mask_ratio,
                                    motion_stride=motion_stride, motion_aware_tau=motion_aware_tau)
