"""Check the S->T output space on CPU; real autograd checks require Torch."""
import ast
import importlib.util
from pathlib import Path
import re
import types
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None


class Module:
    def __call__(self, values):
        return self.forward(values)


class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self.weight = np.zeros((output_dim, input_dim), dtype=np.float64)
        self.bias = np.zeros(output_dim, dtype=np.float64)

    def forward(self, values):
        return values @ self.weight.T + self.bias


class LayerNorm(Module):
    def __init__(self, dim):
        self.weight = np.ones(dim, dtype=np.float64)
        self.bias = np.zeros(dim, dtype=np.float64)

    def forward(self, values):
        centered = values - values.mean(axis=-1, keepdims=True)
        return (centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True)
                                  + 1e-5)) * self.weight + self.bias


class Identity(Module):
    def forward(self, values):
        return values


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __getitem__(self, index):
        return self.layers[index]

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class Embedding:
    def __init__(self, count, dim):
        self.weight = np.zeros((count, dim), dtype=np.float64)


def decoder_constructor():
    """Run the production constructor, substituting only the tensor primitives."""
    path = PROJECT / 'model/transformer_macdiff_text.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    decoder = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'TextNoiseDecoder')
    namespace = {
        'nn': types.SimpleNamespace(
            Module=Module, Linear=Linear, LayerNorm=LayerNorm, Identity=Identity,
            Sequential=Sequential, Embedding=Embedding, Parameter=lambda values: values,
            ModuleList=list, init=types.SimpleNamespace(normal_=lambda *args, **kwargs: None)),
        'torch': types.SimpleNamespace(zeros=lambda *shape: np.zeros(shape)),
        'TextNoiseBlock': lambda *args: types.SimpleNamespace(),
    }
    module = ast.Module(body=[decoder], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['TextNoiseDecoder']


class DecoderOutputSpaceTests(unittest.TestCase):
    def test_512_output_can_preserve_every_noise_direction(self):
        decoder = decoder_constructor()(512, 64, 512, 1, 8, 7,
                                        skeleton_dim=256, output_norm='none')
        decoder.output[1].weight[:] = np.eye(512)
        # Every basis direction must survive, including their common mean direction.
        basis = np.eye(512)
        np.testing.assert_array_equal(decoder.output(basis), basis)
        common = np.ones((2, 7, 512))
        np.testing.assert_array_equal(decoder.output(common), common)
        noise = np.random.RandomState(41).normal(size=(2, 7, 512))
        np.testing.assert_array_equal(decoder.output(noise), noise)

    def test_default_retains_legacy_layernorm_and_linear_index(self):
        decoder = decoder_constructor()(512, 64, 256, 1, 8, 7, skeleton_dim=256)
        self.assertIsInstance(decoder.output[0], LayerNorm)
        self.assertEqual(decoder.output[1].weight.shape, (512, 256))
        # An unmodified LayerNorm loses a constant hidden noise component.
        decoder.output[1].weight[:, :256] = np.eye(512, 256)
        np.testing.assert_array_equal(decoder.output(np.ones((2, 7, 256))),
                                      np.zeros((2, 7, 512)))
        full_width = decoder_constructor()(512, 64, 512, 1, 8, 7,
                                           skeleton_dim=256, output_norm='layernorm')
        full_width.output[1].weight[:] = np.eye(512)
        np.testing.assert_array_equal(full_width.output(np.ones((1, 512))),
                                      np.zeros((1, 512)))

    def test_unknown_output_normalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'output_norm'):
            decoder_constructor()(512, 64, 512, 1, 8, 7, output_norm='batchnorm')

    def test_sentence_configs_remove_bottleneck_and_isolate_new_outputs(self):
        for name in ('pretrain_madiff_text_sentence_fixed_rms.yaml',
                     'pretrain_madiff_text_sentence_sample_target_blend_shared.yaml'):
            with self.subTest(config=name):
                text = (PROJECT / 'config/ntu60_xsub_joint' / name).read_text(encoding='utf-8')
                width = re.search(r'^  text_decoder_hidden_dim:\s*(\d+)\s*(?:#.*)?$', text, re.M)
                norm = re.search(r'^  text_decoder_output_norm:\s*([\w\'\"]+)\s*(?:#.*)?$', text, re.M)
                output = re.search(r'^output_dir:\s*(\S+)\s*$', text, re.M)
                log = re.search(r'^log_dir:\s*(\S+)\s*$', text, re.M)
                self.assertIsNotNone(width)
                self.assertIsNotNone(norm)
                self.assertIsNotNone(output)
                self.assertIsNotNone(log)
                self.assertEqual(int(width.group(1)), 512)
                self.assertEqual(norm.group(1).strip('\'\"'), 'none')
                self.assertTrue(output.group(1).endswith('_st512'))
                self.assertEqual(log.group(1), output.group(1) + '/tensorboard')


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for real decoder/autograd checks')
class DecoderTorchTests(unittest.TestCase):
    def assert_tensor_close(self, actual, expected):
        import torch
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    @staticmethod
    def decoder(hidden_dim=8, output_norm='none'):
        import torch
        from model.transformer_macdiff_text import TextNoiseDecoder
        torch.set_num_threads(1)
        return TextNoiseDecoder(8, 64, hidden_dim, 1, 2, 7,
                                skeleton_dim=6, output_norm=output_norm)

    def test_full_forward_masks_padding_and_backpropagates_to_condition(self):
        import torch
        torch.manual_seed(13)
        decoder = self.decoder()
        noise = torch.randn(2, 4, 8, requires_grad=True)
        skeleton = torch.randn(2, 5, 6, requires_grad=True)
        valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        visible = torch.tensor([[True, True, True, False, False],
                                [True, True, True, True, False]])
        persons = torch.tensor([[0, 0, 999], [0, 1, 1]])
        positions = torch.tensor([[1, 2, 999], [1, 2, 3]])
        time = torch.tensor([2, 6])
        output = decoder(noise, time, skeleton, visible, valid, persons, positions)
        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assert_tensor_close(output[~valid], torch.zeros_like(output[~valid]))
        with torch.no_grad():
            padded_noise = noise.detach().clone()
            padded_skeleton = skeleton.detach().clone()
            padded_noise[~valid] = float('nan')
            padded_skeleton[~visible] = float('nan')
            changed = decoder(padded_noise, time, padded_skeleton, visible,
                              valid, persons, positions)
            self.assert_tensor_close(changed, output)
        target = torch.randn_like(output)
        (output[valid] - target[valid]).square().mean().backward()
        for values, mask in ((noise, valid), (skeleton, visible)):
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertGreater(values.grad[mask].abs().sum().item(), 0.)
            self.assert_tensor_close(values.grad[~mask], torch.zeros_like(values.grad[~mask]))

    def test_strict_checkpoint_requires_matching_width_and_output_norm(self):
        import torch
        source = self.decoder()
        weights = source.state_dict()
        self.assertIn('output.1.weight', weights)
        self.assertNotIn('output.0.weight', weights)
        restored = self.decoder()
        restored.load_state_dict(weights, strict=True)
        for name, values in restored.state_dict().items():
            self.assert_tensor_close(values, weights[name])
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            self.decoder(hidden_dim=16).load_state_dict(weights, strict=True)
        legacy = self.decoder(output_norm='layernorm')
        self.assertIn('output.0.weight', legacy.state_dict())
        self.assertIn('output.1.weight', legacy.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            legacy.load_state_dict(weights, strict=True)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            restored.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
