"""A fresh encoder/head trained through the frozen native MacDiff decoder body.

Import this module through main_pretrain_decoder_distill.py. Native, text and
OSE entry points do not import or modify this implementation.
"""
import inspect
from collections.abc import Mapping
from pathlib import Path

import torch

from guided_diffusion.resample import MaskedDiffusionSampler
from .transformer_macdiff import Transformer as NativeTransformer


MODEL_NAME = 'model.transformer_macdiff_decoder_distill.Transformer'
NATIVE_MODEL_NAME = 'model.transformer_macdiff.Transformer'


def _is_decoder_body_key(name):
    return name.startswith('decoder_') and not name.startswith('decoder_pred.')


def _decoder_body_state(state):
    """Select only the body; neither encoder nor prediction head can be loaded."""
    if not isinstance(state, Mapping):
        raise ValueError('Decoder checkpoint must contain a model state dictionary')
    selected = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError('Checkpoint state keys must be strings')
        key = key[len('module.'):] if key.startswith('module.') else key
        if _is_decoder_body_key(key):
            if key in selected:
                raise ValueError('Duplicate decoder checkpoint key: ' + key)
            selected[key] = value
    return selected


def _validate_decoder_state(expected, source):
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise ValueError('Decoder body keys do not match (check decoder_depth): '
                         'missing=%s, unexpected=%s' % (missing, extra))
    for name, value in source.items():
        if not hasattr(value, 'shape') or tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError('Decoder body shape mismatch: %s, expected=%s, found=%s'
                             % (name, tuple(expected[name].shape),
                                getattr(value, 'shape', None)))


def _argument(arguments, name, default=None):
    if isinstance(arguments, Mapping):
        return arguments.get(name, default)
    return getattr(arguments, name, default)


def _canonical(value):
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    return value


def _validate_source_definition(arguments, expected):
    """Check non-tensor diffusion/normalization settings when metadata exists."""
    source_model = _argument(arguments, 'model')
    if source_model is not None and source_model not in (NATIVE_MODEL_NAME, MODEL_NAME):
        raise ValueError('Use a native MacDiff decoder checkpoint; found model=%s'
                         % source_model)
    options = _argument(arguments, 'model_args')
    if options is None:
        return False
    if not isinstance(options, Mapping):
        raise ValueError('Checkpoint model_args must be a mapping')
    # These are native constructor defaults for fields sometimes omitted in YAML.
    defaults = dict(diff_steps=1000, diff_noise_schedule='cosine', num_heads=8,
                    act_layer='nn.GELU',
                    input_mean=[0., 0., 0.], input_var=[1., 1., 1.],
                    self_shift=False, diff_prediction='noise')
    prediction = options.get('diff_prediction', defaults['diff_prediction'])
    if prediction not in ('noise', 'noise2joint'):
        raise ValueError('Source decoder must use epsilon prediction, found %s' % prediction)
    for name, value in expected.items():
        actual = options.get(name, defaults[name])
        if name == 'act_layer':
            actual = actual.rsplit('.', 1)[-1] if isinstance(actual, str) else getattr(actual, '__name__', None)
        if _canonical(actual) != _canonical(value):
            raise ValueError('Decoder checkpoint definition mismatch: %s, source=%r, requested=%r'
                             % (name, actual, value))
    return True


def load_checkpoint_file(path):
    """Read a trusted local training checkpoint on both Torch 1.8 and newer Torch."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError('Decoder checkpoint not found: %s' % path)
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        # Native checkpoints also contain argparse.Namespace, not just tensors.
        options['weights_only'] = False
    checkpoint = torch.load(str(path), **options)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a checkpoint/state dictionary: %s' % path)
    return checkpoint


class Transformer(NativeTransformer):
    """Keep native keys/forward so existing linear probing loads the new encoder."""

    def __init__(self, decoder_checkpoint, distill_t_min=300, distill_t_max=700,
                 **kwargs):
        if not decoder_checkpoint:
            raise ValueError('A pretrained native decoder checkpoint is required')
        if (not isinstance(distill_t_min, int) or isinstance(distill_t_min, bool)
                or not isinstance(distill_t_max, int) or isinstance(distill_t_max, bool)):
            raise ValueError('Distillation timestep bounds must be integers')
        steps = kwargs.get('diff_steps', 1000)
        if not 0 <= distill_t_min <= distill_t_max < steps:
            raise ValueError('Require 0 <= distill_t_min <= distill_t_max < diff_steps')
        if kwargs.get('diff_prediction', 'noise2joint') != 'noise2joint':
            raise ValueError('Decoder distillation requires diff_prediction=noise2joint')
        if kwargs.get('uncond_ratio', 0.) != 0. or kwargs.get('layer_mask_ratio', 0.) != 0.:
            raise ValueError('Decoder distillation requires condition/layer dropout=0')
        if not kwargs.get('one_person', True):
            raise ValueError('This initial distillation protocol requires one_person=True')
        if kwargs.get('loss_reweight') is not None:
            raise ValueError('This protocol uses unweighted masked coordinate MSE')
        kwargs['diff_prediction'] = 'noise2joint'
        kwargs['uncond_ratio'] = 0.
        kwargs['layer_mask_ratio'] = 0.
        kwargs['one_person'] = True
        super().__init__(**kwargs)

        self.distill_t_min = distill_t_min
        self.distill_t_max = distill_t_max
        self.decoder_checkpoint = str(Path(decoder_checkpoint).expanduser())
        self.schedule_sampler = MaskedDiffusionSampler(
            self.diffusion, distill_t_min, distill_t_max + 1)
        self.load_decoder_body(self.decoder_checkpoint)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not _is_decoder_body_key(name))
        self.train(self.training)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print('Decoder distillation: fresh encoder + fresh trainable decoder_pred; '
              't=[%d,%d], trainable=%d, frozen=%d'
              % (distill_t_min, distill_t_max, trainable, frozen))

    def load_decoder_body(self, path):
        checkpoint = load_checkpoint_file(path)
        source = _decoder_body_state(checkpoint.get('model', checkpoint))
        expected = _decoder_body_state(self.state_dict())
        _validate_decoder_state(expected, source)
        if any(not torch.is_tensor(value) for value in source.values()):
            raise ValueError('Decoder body entries must be tensors')
        has_metadata = _validate_source_definition(checkpoint.get('args'), dict(
            diff_steps=self.diff_steps, diff_noise_schedule=self.diff_noise_schedule,
            num_heads=self.decoder_blocks[0].attn.num_heads,
            act_layer=type(self.decoder_blocks[0].mlp.act).__name__,
            input_mean=self.input_mean, input_var=self.input_var,
            self_shift=self.self_shift))
        # Loading this allowlist leaves all encoder parameters and decoder_pred
        # at the native random initialization performed by super().__init__.
        result = self.load_state_dict(source, strict=False)
        expected_missing = set(self.state_dict()) - set(source)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError('Unexpected selective decoder loading result: %s' % result)
        print('Loaded %d decoder-body tensors from %s; source definition metadata=%s'
              % (len(source), path, 'checked' if has_metadata else 'unavailable'))

    def train(self, mode=True):
        super().train(mode)
        # engine_pretrain calls model.train(True) at every epoch. Keep frozen
        # dropout/drop-path deterministic while the new head follows mode.
        self.decoder_embed.eval()
        self.decoder_blocks.eval()
        self.decoder_norm.eval()
        return self

    def initialize_ose(self, *args, **kwargs):
        raise ValueError('Decoder distillation is an independent native-only protocol')

    def forward(self, source, source_aug, mask_ratio=.9, motion_stride=1,
                motion_aware_tau=-1, enable_ose=False):
        if enable_ose:
            raise ValueError('Decoder distillation cannot enable OSE')
        length = self.joints_embed.t_grid_size * self.joints_embed.grid_size
        keep = round(length * (1. - mask_ratio))
        if not 0. < mask_ratio < 1. or not 0 < keep < length:
            raise ValueError('Masking must leave both visible and masked patches')
        # The inherited forward keeps autograd through decoder/head and uses
        # noise2joint: epsilon -> x0 -> masked MSE + skeleton uniformity.
        return super().forward(
            source, source_aug, mask_ratio=mask_ratio, motion_stride=motion_stride,
            motion_aware_tau=motion_aware_tau, enable_ose=False)
