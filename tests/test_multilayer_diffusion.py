"""Production objective/routing, entry/resume checks, and optional Torch regression."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model/transformer_macdiff_multilayer.py'
NATIVE = ROOT / 'model/transformer_macdiff.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None
VERSION = 'last3_shared_norm_decoder_final_layer_lp_v3'


def production(name, namespace=None, owner=None, path=MODEL):
    nodes = ast.parse(path.read_text(encoding='utf-8')).body
    if owner:
        nodes = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == owner).body
    node = copy.deepcopy(next(x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == name))
    node.decorator_list = []
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[name]


def native_parser():
    return production('get_args_parser', {'argparse': argparse}, path=ROOT / 'main_pretrain.py')()


def validator():
    return production('validate_multilayer_options', {'math': math, 'MULTILAYER_VERSION': VERSION})


class DefinitionTests(unittest.TestCase):
    def args(self, argv=None):
        import main_pretrain_multilayer as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=native_parser)}):
            return entry.get_args_parser().parse_args(argv or [])

    def prepare(self, args):
        import main_pretrain_multilayer as entry
        with mock.patch.object(entry.importlib, 'import_module', return_value=types.SimpleNamespace(
                validate_multilayer_options=validator())):
            return entry.prepare_args(args)

    def test_reject_invalid_depth_weights_targets_and_version(self):
        valid = dict(depth=8, lambda_aux_diffusion=.1)
        validator()(**valid)
        for change in (dict(depth=2), dict(depth=True), dict(depth=8.),
                       dict(lambda_aux_diffusion=-.1), dict(lambda_aux_diffusion=True),
                       dict(lambda_aux_diffusion=float('nan')), dict(lambda_loss_uni=-1),
                       dict(diff_prediction='v'), dict(one_person=False),
                       dict(layer_mask_ratio=.1), dict(uncond_ratio=1.1),
                       dict(multilayer_version='last3_independent_diffusion_final_layer_lp_v1'),
                       dict(multilayer_version='last3_shared_decoder_final_layer_lp_v2'),
                       dict(multilayer_version='other')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validator()(**dict(valid, **change))

    def test_loss_preserves_main_and_averages_auxiliary_only(self):
        combine = production('combine_multilayer_losses')
        # Deliberately unequal heads reveal an accidental sum, /3, or repeated uni.
        total, auxiliary = combine(2., [4., 8.], .5, .1, .02)
        self.assertAlmostEqual(auxiliary, 6.)
        self.assertAlmostEqual(total, 2.61)
        self.assertEqual(combine(2., [4., 8.], .5, 0., .02)[0], 2.01)
        with self.assertRaises(ValueError):
            combine(2., [4.], .5, .1, .02)

    def test_cli_override_reaches_saved_model_definition(self):
        args = self.prepare(self.args(['--lambda_aux_diffusion', '.3', '--disable_amp']))
        self.assertEqual(args.model_args['lambda_aux_diffusion'], .3)
        self.assertEqual(args.model_args['depth'], 8)
        self.assertEqual(args.model_args['decoder_depth'], 3)
        self.assertEqual(args.model_args['multilayer_version'], VERSION)
        self.assertFalse(args.enable_amp)

    def test_yaml_defaults_and_cli_precision_override(self):
        import main_pretrain_multilayer as entry
        defaults = dict(model=entry.MODEL_NAME, enable_amp=False,
                        model_args=dict(depth=8, lambda_aux_diffusion=.1))
        with mock.patch.dict(sys.modules, {
                'main_pretrain': types.SimpleNamespace(get_args_parser=native_parser),
                'yaml': types.SimpleNamespace(safe_load=lambda stream: defaults)}):
            args = entry.parse_args(['--lambda_aux_diffusion', '.2', '--enable_amp'])
        args = self.prepare(args)
        self.assertEqual(args.model_args['lambda_aux_diffusion'], .2)
        self.assertTrue(args.enable_amp)

    def test_reject_text_ose_and_different_task(self):
        for key, value in (('enable_ose', True), ('text_cache', 'old_cache'),
                           ('lambda_text_to_skeleton', 0.), ('task', '2dto3d')):
            args = self.args()
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(args)

    def test_resume_binds_model_objective_data_and_training_protocol(self):
        import main_pretrain_multilayer as entry
        args = self.prepare(self.args())
        saved = copy.deepcopy(args)
        saved.mask_ratio, args.mask_ratio = [.9], .9
        entry._validate_resume_definition(saved, args)
        for key, value in (('model', 'model.transformer_macdiff.Transformer'),
                           ('model_args', dict(args.model_args, lambda_aux_diffusion=.2)),
                           ('train_feeder_args', {'data_path': 'another.npz'}),
                           ('mask_ratio', .8), ('epochs', 500), ('enable_amp', True)):
            changed = copy.deepcopy(saved)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry._validate_resume_definition(changed, args)

    def test_resume_requires_optimizer_scaler_epoch_and_identity(self):
        args = self.prepare(self.args())
        args.resume = 'checkpoint-80.pth'
        saved = copy.deepcopy(args)
        checkpoint = dict(model={}, args=saved, optimizer={}, scaler={}, epoch=80)
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **k: checkpoint)}):
            self.prepare(args)
            del checkpoint['optimizer']
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                self.prepare(args)

    def test_fresh_pt_directory_cannot_overwrite_existing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args()
            args.output_dir = directory
            (Path(directory) / 'log.txt').write_text('existing run', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                self.prepare(args)


class Array(np.ndarray):
    def float(self):
        return self.astype(np.float32)

    def detach(self):
        return self

    def clone(self):
        return self.copy()

    def to(self, device):
        return self

    @property
    def device(self):
        return 'cpu'

    def mean(self, dim=None, keepdim=False, **kwargs):
        return array(np.asarray(self).mean(axis=kwargs.pop('axis', dim),
                                          keepdims=kwargs.pop('keepdims', keepdim), **kwargs))

    def sum(self, dim=None, keepdim=False, **kwargs):
        return array(np.asarray(self).sum(axis=kwargs.pop('axis', dim),
                                         keepdims=kwargs.pop('keepdims', keepdim), **kwargs))

    def reshape(self, *shape, **kwargs):
        return np.asarray(self).reshape(kwargs.get('shape', shape[0] if len(shape) == 1 else shape)).view(Array)

    def permute(self, *dimensions):
        return np.asarray(self).transpose(dimensions).view(Array)

    def contiguous(self):
        return self

    def view(self, *shape):
        return np.asarray(self).reshape(shape).view(Array)

    def unsqueeze(self, dimension):
        return array(np.expand_dims(self, dimension))

    def expand(self, *shape):
        shape = tuple(self.shape[i] if size == -1 else size for i, size in enumerate(shape))
        return array(np.broadcast_to(self, shape))

    def repeat(self, *repeats):
        return array(np.tile(np.asarray(self), repeats))


def array(value):
    return np.asarray(value).view(Array)


def array_runtime():
    return types.SimpleNamespace(
        no_grad=contextlib.nullcontext, isfinite=np.isfinite, any=np.any,
        is_tensor=lambda x: isinstance(x, Array),
        cat=lambda xs, dim: array(np.concatenate(xs, axis=dim)),
        gather=lambda x, dim, index: array(np.take_along_axis(x, index, axis=dim)),
        einsum=lambda equation, x: array(np.einsum(equation, x)),
        cuda=types.SimpleNamespace(amp=types.SimpleNamespace(
            autocast=lambda **kwargs: contextlib.nullcontext())))


class RoutingTests(unittest.TestCase):
    def test_decoder_arithmetic_matches_native_after_condition_dropout(self):
        runtime = array_runtime()
        runtime.rand = lambda *shape, **kwargs: array(np.ones(shape))
        observed = []
        def embedding(x):
            return array(np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4) / 10)
        def block(x, *, z, t):
            observed.append(1)
            return x + .2 * z + t[:, None, :1]
        model = types.SimpleNamespace(dim_t_embed=4, dim_feat=4, uncond_ratio=0.,
            joints_embed=types.SimpleNamespace(t_grid_size=2, grid_size=3),
            decoder_embed=embedding, decoder_blocks=[block, block],
            decoder_pos_embed=array(np.ones((1, 1, 3, 4))),
            decoder_temp_embed=array(np.ones((1, 2, 1, 4))) * 2,
            decoder_norm=lambda x: x * .5, decoder_pred=lambda x: x[..., :3])
        time_embed = lambda time, width: array(np.repeat(np.asarray(time)[:, None], width, axis=1))
        namespace = dict(torch=runtime, timestep_embedding=time_embed)
        native = production('forward_decoder', namespace, owner='Transformer', path=NATIVE)
        candidate = production('decode_skeleton', namespace)
        noisy, condition, time = array(np.zeros((2, 8, 3, 3))), array(np.ones((2, 6, 4))), array([2, 7])
        expected = native(model, noisy, z=condition, t=time)
        actual = candidate(model, noisy, condition, time)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (2, 6, 3))
        self.assertEqual(len(observed), 4)

    def test_encoder_runs_once_and_returns_exact_last_three_block_outputs(self):
        calls = []
        masks = []
        norm_inputs = []
        def block(index):
            def forward(x):
                calls.append(index)
                return x + index
            return forward
        x = array(np.zeros((2, 2, 3, 4)))
        def masking(tokens, ratio):
            masks.append(ratio)
            return tokens[:, :3], 'shared_mask', 'shared_restore', None
        def shared_norm(x):
            norm_inputs.append(x.copy())
            return x + 100
        model = types.SimpleNamespace(joints_embed=lambda x: x, pos_embed=array(np.zeros((1, 1, 3, 4))),
            temp_embed=array(np.zeros((1, 2, 1, 4))), random_masking=masking,
            blocks=[block(i) for i in range(1, 9)], supervision_layers=(6, 7, 8),
            norm=shared_norm)
        encode = production('forward_encoder_multilayer', owner='Transformer')
        features, mask, restore = encode(model, x, x_orig=x, mask_ratio=.5, motion_aware_tau=-1)
        self.assertEqual(calls, list(range(1, 9)))
        self.assertEqual(masks, [.5])
        self.assertEqual((mask, restore), ('shared_mask', 'shared_restore'))
        self.assertEqual(len(norm_inputs), 3)
        for observed, expected in zip(norm_inputs, (21, 28, 36)):
            np.testing.assert_array_equal(observed, np.full((2, 3, 4), expected))
        for layer, expected in ((6, 121), (7, 128), (8, 136)):
            np.testing.assert_array_equal(features[layer], np.full((2, 3, 4), expected))

    def test_three_supervisions_use_one_decoder_and_share_diffusion_state(self):
        runtime = array_runtime()
        events, sample_events, uniformity_inputs = [], [], []
        runtime.rand = lambda *shape, **kwargs: array([.05, .9])
        runtime.randn_like = lambda x: array(np.full_like(x, .4))
        model = types.SimpleNamespace(one_person=True, self_shift=False,
            input_mean=[0, 0, 0], input_std=[1, 1, 1], t_patch_size=4, patch_size=1,
            joints_embed=types.SimpleNamespace(t_grid_size=2, grid_size=3),
            uncond_ratio=.1, supervision_layers=(6, 7, 8),
            lambda_aux_diffusion=.1, lambda_loss_uni=.02, loss_reweight=array(np.ones(10)))
        for name in ('_sequence_layout', '_normalize_sequence', 'patchify',
                     'build_global_local_condition', 'forward_loss'):
            method = production(name, {'torch': runtime, 'np': np}, owner='Transformer', path=NATIVE)
            setattr(model, name, types.MethodType(method, model))
        mask = array([[0, 0, 0, 1, 1, 1]] * 2)
        restore = array(np.tile([0, 3, 1, 4, 2, 5], (2, 1)))
        features = {i: array(np.full((2, 3, 4), i)) for i in (6, 7, 8)}
        encoder_calls = []
        def encode(*args, **kwargs):
            encoder_calls.append(1)
            return features, mask, restore
        model.forward_encoder_multilayer = encode
        time = array([2, 7])
        def sample(*args):
            sample_events.append(1)
            return time, None
        def q_sample(clean, t, noise):
            self.assertIs(t, time)
            return array(.7 * clean + .3 * noise)
        model.schedule_sampler = types.SimpleNamespace(sample=sample)
        model.diffusion = types.SimpleNamespace(q_sample=q_sample)
        def decode(decoder, noisy, condition, t):
            self.assertIs(decoder, model)
            layer = int(condition[1, 0, 0])
            events.append((layer, noisy.copy(), condition.copy(), t, mask))
            return model.patchify(noisy) + layer / 100
        def uniformity(x):
            uniformity_inputs.append(x)
            return array(.3)
        forward = production('forward_macdiff', {
            'torch': runtime, 'token_uniformity_loss': uniformity,
            'decode_skeleton': decode,
            'combine_multilayer_losses': production('combine_multilayer_losses')}, owner='Transformer')
        source = array(np.arange(2 * 3 * 8 * 3 * 2).reshape(2, 3, 8, 3, 2) / 100)
        original = source.copy()
        # Native debug printing uses NumPy RNG only; suppress it for deterministic output.
        with mock.patch.object(np.random, 'rand', return_value=np.array([.5])):
            loss, prediction, observed_mask, metrics = forward(model, source, source + .1, mask_ratio=.5)
        self.assertEqual(len(sample_events), 1)
        self.assertEqual(len(encoder_calls), 1)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(uniformity_inputs), 1)
        self.assertIs(observed_mask, mask)
        for layer, noisy, condition, t, observed in events:
            np.testing.assert_array_equal(noisy, events[0][1])
            self.assertIs(t, time)
            self.assertIs(observed, mask)
            np.testing.assert_array_equal(condition[0], np.zeros((6, 4)))
            np.testing.assert_array_equal(condition[1], np.full((6, 4), layer))
        expected = metrics['loss_diffusion_layer8'] + .05 * (
            metrics['loss_diffusion_layer6'] + metrics['loss_diffusion_layer7']) + .006
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertEqual(prediction.shape, (2, 6, 12))
        np.testing.assert_array_equal(source, original)
        # The second person is not an input to this PT method.
        changed = source.copy()
        changed[..., 1] += 10000
        with mock.patch.object(np.random, 'rand', return_value=np.array([.5])):
            changed_loss = forward(model, changed, changed + .1, mask_ratio=.5)[0]
        self.assertAlmostEqual(float(loss), float(changed_loss), places=6)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for real shared-decoder gradients and LP loading')
class TorchTests(unittest.TestCase):
    def options(self):
        return dict(dim_feat=8, dim_t_embed=64, depth=8, decoder_depth=1, num_heads=2,
                    num_frames=8, num_joints=25, t_patch_size=4, one_person=True,
                    lambda_loss_uni=.02, uncond_ratio=0.)

    def model(self, **kwargs):
        import torch
        from model.transformer_macdiff_multilayer import Transformer
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**dict(self.options(), **kwargs))

    def test_native_parameter_names_count_and_initialization_are_identical(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        torch.manual_seed(41)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(**self.options())
        torch.manual_seed(41)
        model = self.model()
        for name, value in native.state_dict().items():
            torch.testing.assert_allclose(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(list(model.state_dict()), list(native.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in native.parameters()))
        self.assertFalse(hasattr(model, 'auxiliary_norms'))
        self.assertFalse(hasattr(model, 'auxiliary_decoders'))

    def test_forward_all_parameters_receive_finite_gradients(self):
        import torch
        model = self.model()
        source = torch.randn(2, 3, 8, 25, 2)
        loss, prediction, mask, metrics = model(source, source + .01, mask_ratio=.5)
        torch.testing.assert_allclose(loss.detach(), metrics['loss_native'] + metrics['loss_auxiliary_weighted'])
        self.assertEqual(tuple(prediction.shape), (2, 50, 12))
        self.assertEqual(tuple(mask.shape), (2, 50))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.decoder_pred.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[-1].attn.qkv.weight.grad.abs().sum().item(), 0)

    def test_auxiliary_layer6_gradient_stops_before_layers7_and8(self):
        import torch
        import model.transformer_macdiff_multilayer as module
        model = self.model()
        x = torch.randn(2, 8, 25, 3)
        features, mask, restore = model.forward_encoder_multilayer(
            x, x_orig=x, mask_ratio=.5, motion_aware_tau=-1)
        condition = model.build_global_local_condition(features[6], features[6].mean(1, keepdim=True), restore)
        prediction = module.decode_skeleton(model, x, condition, torch.tensor([3, 7]))
        model.forward_loss(torch.zeros_like(x), prediction, mask, torch.tensor([3, 7])).backward()
        self.assertGreater(model.blocks[5].attn.qkv.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.blocks[6].attn.qkv.weight.grad)
        self.assertIsNone(model.blocks[7].attn.qkv.weight.grad)

    def test_zero_aux_weight_preserves_native_prediction_mask_and_objective(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        torch.manual_seed(47)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(**self.options())
        torch.manual_seed(47)
        model = self.model(lambda_aux_diffusion=0.)
        source = torch.randn(2, 3, 8, 25, 2)
        results = []
        for network in (native, model):
            torch.manual_seed(53)
            np.random.seed(53)
            results.append(network(source.clone(), (source + .01).clone(), mask_ratio=.5))
        for a, b in zip(results[0], results[1][:3]):
            torch.testing.assert_allclose(a, b, rtol=1.e-6, atol=1.e-7)
        results[1][0].backward()
        self.assertIsNotNone(model.norm.weight.grad)
        self.assertTrue(torch.isfinite(model.norm.weight.grad).all())

    def test_shared_dropout_and_diffusion_state_in_real_decoders(self):
        import torch
        import model.transformer_macdiff_multilayer as module
        model = self.model(uncond_ratio=.1)
        source = torch.randn(2, 3, 8, 25, 2)
        original = module.decode_skeleton
        observed = []
        decoder_instances = []
        def capture(decoder, noisy, condition, time):
            decoder_instances.append(decoder)
            observed.append((noisy.detach().clone(), condition.detach().clone(), time.detach().clone()))
            return original(decoder, noisy, condition, time)
        native_rand = torch.rand
        def rand(*shape, **kwargs):
            if shape == (2,):
                return torch.tensor([.05, .9], device=kwargs.get('device', 'cpu'))
            return native_rand(*shape, **kwargs)
        with mock.patch.object(module, 'decode_skeleton', side_effect=capture), \
                mock.patch.object(torch, 'rand', side_effect=rand):
            model(source, source + .01, mask_ratio=.5)
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(decoder is model for decoder in decoder_instances))
        for noisy, condition, time in observed:
            torch.testing.assert_allclose(noisy, observed[0][0], rtol=0, atol=0)
            torch.testing.assert_allclose(time, observed[0][2], rtol=0, atol=0)
            self.assertEqual(condition[0].abs().sum().item(), 0.)
            self.assertGreater(condition[1].abs().sum().item(), 0.)

    def test_shared_norm_and_decoder_gradients_accumulate_all_weighted_losses(self):
        import torch
        import model.transformer_macdiff_multilayer as module
        model = self.model(lambda_aux_diffusion=.4)
        source = torch.randn(2, 3, 8, 25, 2)
        original_decode, original_noise = module.decode_skeleton, torch.randn_like
        predictions, noises, normalized = [], [], []
        def decode(decoder, noisy, condition, time):
            self.assertIs(decoder, model)
            prediction = original_decode(decoder, noisy, condition, time)
            predictions.append((prediction, time))
            return prediction
        def noise_like(x):
            noise = original_noise(x)
            noises.append(noise)
            return noise
        hook = model.norm.register_forward_hook(lambda _, inputs, output: normalized.append(output))
        try:
            with mock.patch.object(module, 'decode_skeleton', side_effect=decode), \
                    mock.patch.object(torch, 'randn_like', side_effect=noise_like):
                total, _, mask, _ = model(source, source + .01, mask_ratio=.5)
        finally:
            hook.remove()
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(noises), 1)
        parameter = model.decoder_pred.weight
        gradients = []
        for prediction, time in predictions:
            branch_loss = model.forward_loss(noises[0], prediction, mask, time)
            gradients.append(torch.autograd.grad(branch_loss, parameter, retain_graph=True)[0])
        self.assertEqual(len(normalized), 3)
        actual = torch.autograd.grad(total, parameter, retain_graph=True)[0]
        torch.testing.assert_allclose(actual, gradients[2] + .2 * (gradients[0] + gradients[1]),
                                      rtol=1.e-5, atol=1.e-6)
        norm_gradients = []
        for prediction, time in predictions:
            branch_loss = model.forward_loss(noises[0], prediction, mask, time)
            norm_gradients.append(torch.autograd.grad(branch_loss, model.norm.weight, retain_graph=True)[0])
        uniformity = module.token_uniformity_loss(normalized[-1].float())
        uniformity_gradient = torch.autograd.grad(uniformity, model.norm.weight, retain_graph=True)[0]
        actual = torch.autograd.grad(total, model.norm.weight)[0]
        torch.testing.assert_allclose(actual, norm_gradients[2] + .2 * (
            norm_gradients[0] + norm_gradients[1]) + .02 * uniformity_gradient,
            rtol=1.e-5, atol=1.e-6)

    def test_existing_lp_loads_encoder_and_shared_norm_and_ignores_decoder(self):
        import torch
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        options = self.options()
        for name in ('decoder_depth', 'one_person', 'lambda_loss_uni', 'uncond_ratio', 'dim_t_embed'):
            options.pop(name)
        with contextlib.redirect_stdout(io.StringIO()):
            target = Downstream(num_classes=60, protocol='linprobe2', **options).eval()
        message = target.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all('head' in key for key in message.missing_keys), message)
        for name, parameter in target.named_parameters():
            parameter.requires_grad = 'head' in name
            if not name.startswith('head.'):
                torch.testing.assert_allclose(parameter, model.state_dict()[name], rtol=0, atol=0)
        source = torch.randn(2, 3, 8, 25, 2)
        before = target(source)
        with torch.no_grad():
            model.decoder_pred.weight.add_(100)
        target.load_state_dict(model.state_dict(), strict=False)
        torch.testing.assert_allclose(target(source), before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
