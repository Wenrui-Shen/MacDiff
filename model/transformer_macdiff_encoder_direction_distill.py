"""Staged representation distillation from a fixed native MacDiff encoder.

The online encoder is randomly initialized and first matches a frozen teacher.
At the phase boundary, its current encoder is frozen as an internal anchor.
Training then extrapolates beyond the teacher along the anchor-to-teacher
representation residual.  No external base checkpoint is used.
"""
import math

import torch
from torch.nn import functional as F

from .transformer_macdiff import (
    MomentumSkeletonEncoder,
    Transformer as NativeTransformer,
)


METHOD_VERSION = 'native_t20_random_student_phase_anchor_v1'


def staged_lambda(epoch, match_epochs, ramp_epochs, lambda_max):
    """Teacher matching, cosine ramp, then fixed representation extrapolation."""
    if epoch < match_epochs:
        return 1.0
    if ramp_epochs <= 1 or epoch >= match_epochs + ramp_epochs:
        return float(lambda_max)
    progress = float(epoch - match_epochs) / float(ramp_epochs - 1)
    return 1.0 + (float(lambda_max) - 1.0) * 0.5 * (
        1.0 - math.cos(math.pi * progress))


def _load_encoder_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    expected = encoder.state_dict()
    missing = [name for name in expected if name not in state]
    if missing:
        raise ValueError(
            'Encoder checkpoint {} is missing keys: {}'.format(
                checkpoint_path, missing[:8]))
    encoder_state = {name: state[name] for name in expected}
    encoder.load_state_dict(encoder_state, strict=True)
    del checkpoint


class FixedLayerEncoder(MomentumSkeletonEncoder):
    """Frozen native encoder returning one representation per encoder layer."""

    @torch.no_grad()
    def forward_layers(self, skeleton, ids_keep):
        hidden = self.joints_embed(skeleton)
        batch, temporal, joints, width = hidden.shape
        hidden = hidden + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        hidden = hidden.reshape(batch, temporal * joints, width)
        hidden = torch.gather(
            hidden, 1, ids_keep.unsqueeze(-1).expand(-1, -1, width))
        features = []
        for block in self.blocks[:-1]:
            hidden = block(hidden)
            features.append(hidden)
        hidden = self.blocks[-1](hidden)
        features.append(self.norm(hidden))
        return features


class Transformer(NativeTransformer):
    supports_pretrain_metrics = True
    pretrain_metric_prefix = 'encoder_direction_distill/'

    def __init__(self, teacher_checkpoint,
                 match_epochs=200, ramp_epochs=40, lambda_max=1.25,
                 direction_distill_version=METHOD_VERSION, **kwargs):
        if direction_distill_version != METHOD_VERSION:
            raise ValueError('Unsupported encoder direction-distillation version')
        if kwargs.get('one_person', True) is not True:
            raise ValueError('Encoder direction distillation requires one_person=True')
        if not teacher_checkpoint:
            raise ValueError('teacher_checkpoint is required')
        if isinstance(match_epochs, bool) or not isinstance(match_epochs, int) or match_epochs < 0:
            raise ValueError('match_epochs must be a non-negative integer')
        if isinstance(ramp_epochs, bool) or not isinstance(ramp_epochs, int) or ramp_epochs <= 0:
            raise ValueError('ramp_epochs must be a positive integer')
        if (isinstance(lambda_max, bool) or not isinstance(lambda_max, (int, float))
                or not math.isfinite(lambda_max) or lambda_max < 1.0):
            raise ValueError('lambda_max must be finite and at least 1')

        super().__init__(**kwargs)

        # This experiment trains only the encoder.  Removing the unused native
        # decoder keeps every remaining trainable parameter on the loss path.
        for name in ('decoder_embed', 'decoder_blocks', 'decoder_norm',
                     'decoder_temp_embed', 'decoder_pos_embed', 'decoder_pred'):
            delattr(self, name)

        self.teacher_encoder = FixedLayerEncoder(self)
        self.anchor_encoder = FixedLayerEncoder(self)
        _load_encoder_checkpoint(self.teacher_encoder, teacher_checkpoint)
        self.teacher_encoder.eval()
        self.anchor_encoder.eval()

        self.teacher_checkpoint = str(teacher_checkpoint)
        self.match_epochs = int(match_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.lambda_max = float(lambda_max)
        self.direction_distill_version = direction_distill_version
        self.current_epoch = 0

        print(
            'Encoder direction distillation: random student; teacher={}; '
            'snapshot student at epoch {}; cosine ramp for {} epochs, then lambda={}'
            .format(self.teacher_checkpoint, self.match_epochs,
                    self.ramp_epochs, self.lambda_max))

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.anchor_encoder.eval()
        return self

    @torch.no_grad()
    def update_diffusion_sampler(self, epoch, total_epoch):
        del total_epoch
        self.current_epoch = int(epoch)
        if self.current_epoch == self.match_epochs:
            self.anchor_encoder.update_from(self, 0.0)
            self.anchor_encoder.eval()
            print('Frozen student encoder snapshot at epoch {}'.format(
                self.current_epoch))

    def _student_layers(self, skeleton, mask_ratio):
        hidden = self.joints_embed(skeleton)
        batch, temporal, joints, width = hidden.shape
        hidden = hidden + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        hidden = hidden.reshape(batch, temporal * joints, width)
        hidden, mask, _, ids_keep = self.random_masking(hidden, mask_ratio)
        features = []
        for block in self.blocks[:-1]:
            hidden = block(hidden)
            features.append(hidden)
        hidden = self.blocks[-1](hidden)
        features.append(self.norm(hidden))
        return features, mask, ids_keep

    def forward(self, source, source_aug, mask_ratio=.9, motion_stride=1,
                motion_aware_tau=-1, enable_ose=False):
        del motion_stride
        if enable_ose:
            raise ValueError('Encoder direction distillation does not use OSE')
        if motion_aware_tau > 0:
            raise ValueError('Encoder direction distillation uses shared random masking')
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError('mask_ratio must be in (0, 1)')
        if source.ndim != 5 or source.shape != source_aug.shape:
            raise ValueError('Expected matching [B,C,T,V,M] skeleton inputs')

        with torch.no_grad():
            _, augmented = self._prepare_source(source, source_aug)

        student_features, mask, ids_keep = self._student_layers(augmented, mask_ratio)
        with torch.no_grad():
            teacher_features = self.teacher_encoder.forward_layers(augmented, ids_keep)

        coefficient = staged_lambda(
            self.current_epoch, self.match_epochs, self.ramp_epochs, self.lambda_max)
        if coefficient == 1.0:
            targets = teacher_features
        else:
            with torch.no_grad():
                anchor_features = self.anchor_encoder.forward_layers(augmented, ids_keep)
                targets = [
                    teacher + (coefficient - 1.0) * (teacher - anchor)
                    for teacher, anchor in zip(teacher_features, anchor_features)
                ]

        with torch.cuda.amp.autocast(enabled=False):
            layer_losses = [
                (student.float() - target.detach().float()).square().mean()
                for student, target in zip(student_features, targets)
            ]
            raw_loss = torch.stack(layer_losses).mean()
            loss = raw_loss / (coefficient * coefficient)
            final_cosine = F.cosine_similarity(
                student_features[-1].float(), teacher_features[-1].float(), dim=-1).mean()

        if not torch.isfinite(loss):
            raise FloatingPointError('Encoder direction-distillation loss is non-finite')
        metrics = {
            'loss_direction_raw': raw_loss.detach(),
            'loss_direction_scaled': loss.detach(),
            'distill_lambda': loss.new_tensor(coefficient),
            'student_teacher_cosine': final_cosine.detach(),
        }
        return loss, student_features[-1], mask, metrics
