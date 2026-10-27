"""FM mathematics, production control flow, resume, and optional real Torch checks."""
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
SOURCE = ROOT / 'model/transformer_macdiff_flow_matching.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def production_function(name, namespace=None, owner=None, path=SOURCE):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = tree.body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == owner).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[name]


def validator():
    return production_function('validate_flow_options', {
        'math': math, 'FLOW_MATCHING_VERSION': 'conditional_ot_noise_to_data_v2',
        'FLOW_TIME_SAMPLERS': ('uniform', 'inverse_cosine_snr_v1'),
        'build_inverse_cosine_snr_times': inverse_cosine_times()})


def native_beta_schedule():
    path = ROOT / 'guided_diffusion/gaussian_diffusion.py'
    discretize = production_function('betas_for_alpha_bar', {'np': np}, path=path)
    return production_function('get_named_beta_schedule', {
        'math': math, 'np': np, 'betas_for_alpha_bar': discretize}, path=path)


def inverse_cosine_times():
    return production_function('build_inverse_cosine_snr_times', {
        'np': np, 'get_named_beta_schedule': native_beta_schedule()})


def native_parser():
    return production_function('get_args_parser', {'argparse': argparse},
                               path=ROOT / 'main_pretrain.py')()


class Tensor(np.ndarray):
    """Only the tensor primitives used by the extracted production methods."""
    def float(self):
        return self.astype(np.float32)

    def clone(self):
        return self.copy()

    def detach(self):
        return self

    @property
    def device(self):
        return 'cpu'

    def square(self):
        return self * self

    def numel(self):
        return self.size

    def mean(self, dim=None, keepdim=False, **kwargs):
        axis = kwargs.pop('axis', dim)
        keepdims = kwargs.pop('keepdims', keepdim)
        return tensor(np.asarray(self).mean(axis=axis, keepdims=keepdims, **kwargs))

    def sum(self, dim=None, keepdim=False, **kwargs):
        axis = kwargs.pop('axis', dim)
        keepdims = kwargs.pop('keepdims', keepdim)
        return tensor(np.asarray(self).sum(axis=axis, keepdims=keepdims, **kwargs))

    def permute(self, *dimensions):
        return np.asarray(self).transpose(dimensions).view(Tensor)

    def contiguous(self):
        return self

    def view(self, *shape):
        return np.asarray(self).reshape(shape).view(Tensor)

    def reshape(self, *shape, **kwargs):
        shape = kwargs.get('shape', shape[0] if len(shape) == 1 else shape)
        return np.asarray(self).reshape(shape).view(Tensor)


def tensor(values):
    return np.asarray(values).view(Tensor)


def fake_torch():
    return types.SimpleNamespace(
        isfinite=np.isfinite, any=np.any, no_grad=contextlib.nullcontext,
        float32=np.float32, einsum=lambda equation, values: tensor(np.einsum(equation, values)))


class FlowDefinitionTests(unittest.TestCase):
    def args(self):
        import main_pretrain_flow_matching as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=native_parser)}):
            args = entry.get_args_parser().parse_args([])
        args.output_dir = ''
        return args

    def prepare(self, args):
        import main_pretrain_flow_matching as entry
        with mock.patch.object(entry.importlib, 'import_module', return_value=types.SimpleNamespace(
                validate_flow_options=validator())):
            return entry.prepare_args(args)

    def test_definition_validation(self):
        valid = dict(flow_time_scale=999., flow_matching_version='conditional_ot_noise_to_data_v2',
                     one_person=True, lambda_loss_uni=.02, flow_sigma_min=1.e-4)
        validator()(**valid)
        for change in (dict(flow_time_scale=0), dict(flow_time_scale=float('nan')),
                       dict(flow_time_scale=True), dict(one_person=False),
                       dict(flow_matching_version='linear_noise_to_data_v1'),
                       dict(flow_matching_version='diffusion_v'), dict(lambda_loss_uni=-1),
                       dict(flow_sigma_min=-1.e-4), dict(flow_sigma_min=1.),
                       dict(flow_sigma_min=float('nan')), dict(flow_sigma_min=True),
                       dict(flow_time_sampling='cosine'),
                       dict(flow_time_sampling='inverse_cosine_snr_v1', flow_sigma_min=.2)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validator()(**dict(valid, **change))

    def test_native_parser_defaults_and_cli_override_are_saved(self):
        args = self.args()
        self.assertFalse(args.enable_amp)
        args.flow_time_scale = 500.
        self.prepare(args)
        self.assertEqual(args.model_args['flow_time_scale'], 500.)
        self.assertEqual(args.model_args['flow_matching_version'], 'conditional_ot_noise_to_data_v2')
        self.assertEqual(args.model_args['flow_sigma_min'], 1.e-4)
        self.assertEqual(args.model_args['flow_time_sampling'], 'uniform')
        self.assertEqual(args.config, 'config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml')

    def test_cli_overrides_yaml_time_scale_and_precision(self):
        import main_pretrain_flow_matching as entry
        for configured, cli, expected in ((True, ['--disable_amp'], False),
                                           (False, [], False),
                                           (False, ['--enable_amp'], True)):
            with self.subTest(configured=configured, cli=cli), mock.patch.dict(sys.modules, {
                    'main_pretrain': types.SimpleNamespace(get_args_parser=native_parser),
                    'yaml': types.SimpleNamespace(safe_load=lambda stream: dict(
                        enable_amp=configured, model_args=dict(flow_time_scale=700.)))}):
                args = entry.parse_args(cli + ['--flow_time_scale', '500', '--flow_sigma_min', '.002',
                                              '--flow_time_sampling', 'inverse_cosine_snr_v1'])
                args.output_dir = ''
                self.prepare(args)
            self.assertEqual(args.enable_amp, expected)
            self.assertEqual(args.model_args['flow_time_scale'], 500.)
            self.assertEqual(args.model_args['flow_sigma_min'], .002)
            self.assertEqual(args.model_args['flow_time_sampling'], 'inverse_cosine_snr_v1')

    def test_common_engine_selects_and_logs_fm_metrics_without_teacher_flags(self):
        tree = ast.parse((ROOT / 'engine_pretrain.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'train_one_epoch_macdiff')
        loop = next(node for node in function.body if isinstance(node, ast.For))
        forward = next(node for node in loop.body if isinstance(node, ast.With))
        select = next(node for node in forward.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'auxiliary_metrics')
        logging = function.body[-2]
        events = []
        env = dict(result=(0, 0, 0, {'loss_flow_matching': 1.5}), stats=dict(
            loss=1.5, lr=.001, loss_flow_matching=1.5), epoch=0,
            model_without_ddp=types.SimpleNamespace(pretrain_metric_prefix='flow_matching/'),
            log_writer=types.SimpleNamespace(add_scalar=lambda *args: events.append(args)))
        for text, teacher, generic, prefix in (
                (None, False, True, 'flow_matching/'),
                (None, False, False, None),
                (object(), False, False, 'text_stage1/'),
                (None, True, False, 'skeleton_teacher/')):
            events.clear()
            env.update(text_features=text, uses_skeleton_teacher=teacher,
                       uses_pretrain_metrics=generic)
            exec(compile(ast.fix_missing_locations(ast.Module(
                body=[select, logging], type_ignores=[])), 'engine_metrics', 'exec'), env)
            self.assertEqual(env['auxiliary_metrics'], {} if prefix is None else env['result'][3])
            self.assertEqual(events, [] if prefix is None else [(prefix + 'loss_flow_matching', 1.5, 0)])

    def test_diffusion_text_and_ose_settings_are_rejected(self):
        for key, value in (('enable_ose', True), ('text_cache', 'old_cache'),
                           ('lambda_skeleton_to_text', 0.), ('task', '2dto3d')):
            args = self.args()
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(args)
        for key, value in (('diff_prediction', 'v'), ('diff_steps', 1000),
                           ('loss_reweight', None), ('diff_noise_schedule', 'cosine')):
            args = self.args()
            args.model_args = {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'diffusion-only'):
                self.prepare(args)

    def test_full_resume_matches_objective_data_and_training_protocol(self):
        import main_pretrain_flow_matching as entry
        args = self.prepare(self.args())
        saved = copy.deepcopy(args)
        entry._validate_resume_definition(saved, args)
        saved.mask_ratio = [.9]
        args.mask_ratio = .9
        entry._validate_resume_definition(saved, args)
        changes = [('model', 'model.transformer_macdiff.Transformer'),
                   ('model_args', dict(args.model_args, flow_time_scale=100.)),
                   ('model_args', dict(args.model_args, flow_sigma_min=0.)),
                   ('model_args', dict(args.model_args, flow_time_sampling='inverse_cosine_snr_v1')),
                   ('model_args', dict(args.model_args, flow_matching_version='linear_noise_to_data_v1')),
                   ('train_feeder_args', {'data_path': 'other.npz'}),
                   ('mask_ratio', .8), ('epochs', 500), ('enable_amp', True)]
        for key, value in changes:
            changed = copy.deepcopy(saved)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry._validate_resume_definition(changed, args)

    def test_old_uniform_checkpoints_resume_only_as_uniform(self):
        import main_pretrain_flow_matching as entry
        args = self.prepare(self.args())
        legacy = copy.deepcopy(args)
        del legacy.model_args['flow_time_sampling']
        entry._validate_resume_definition(legacy, args)
        args.flow_time_sampling = 'inverse_cosine_snr_v1'
        self.prepare(args)
        with self.assertRaisesRegex(ValueError, 'model_args'):
            entry._validate_resume_definition(legacy, args)
        entry._validate_resume_definition(copy.deepcopy(args), args)

    def test_existing_pt_logs_and_checkpoints_require_explicit_resume(self):
        for filename in ('log.txt', 'checkpoint-399.pth'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                args = self.args()
                args.output_dir = directory
                (Path(directory) / filename).write_text('existing run', encoding='utf-8')
                with self.assertRaises(FileExistsError):
                    self.prepare(args)

    def test_prepare_checks_complete_checkpoint_before_native_main(self):
        args = self.prepare(self.args())
        args.resume = 'checkpoint-80.pth'
        saved = copy.deepcopy(args)
        checkpoint = {'model': {}, 'args': saved, 'optimizer': {}, 'scaler': {}, 'epoch': 80}
        fake = types.SimpleNamespace(load=lambda *a, **kw: checkpoint)
        with mock.patch.dict(sys.modules, {'torch': fake}):
            self.prepare(args)
            del checkpoint['optimizer']
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                self.prepare(args)


class FlowNumpyTests(unittest.TestCase):
    def test_inverse_cosine_snr_mapping_matches_every_native_step(self):
        alpha_bar = np.cumprod(1 - native_beta_schedule()(['inverse_cosine', 1], 1000))
        expected_snr = alpha_bar / (1 - alpha_bar)
        for sigma_min in (0., 1.e-4, .02):
            time = inverse_cosine_times()(sigma_min)
            self.assertEqual(time.shape, (1000,))
            self.assertTrue(np.all((time > 0) & (time < 1)))
            self.assertTrue(np.all(np.diff(time) < 0))
            observed_snr = (time / (1 - (1 - sigma_min) * time)) ** 2
            np.testing.assert_allclose(observed_snr, expected_snr, rtol=2.e-12)
        time = inverse_cosine_times()(1.e-4).astype(np.float32)
        snr = (time / (1 - (1 - 1.e-4) * time)) ** 2
        self.assertAlmostEqual(np.mean((snr >= .1) & (snr <= 10)), .902)

    def test_time_sampler_uses_uniform_reference_indices_and_preserves_uniform_branch(self):
        runtime = fake_torch()
        indices = np.array([0, 100, 500, 900, 999])
        runtime.randint = mock.Mock(return_value=indices)
        runtime.rand = mock.Mock(return_value=tensor([.1234, .6789]))
        sample = production_function('sample_flow_time', {'torch': runtime}, owner='Transformer')
        model = types.SimpleNamespace(flow_time_sampling='inverse_cosine_snr_v1',
            _flow_snr_time_lookup=tensor(inverse_cosine_times()(1.e-4).astype(np.float32)))
        observed = sample(model, 5, 'cpu')
        np.testing.assert_array_equal(observed, model._flow_snr_time_lookup[indices])
        runtime.randint.assert_called_once_with(1000, (5,), device='cpu')
        runtime.rand.assert_not_called()
        model.flow_time_sampling = 'uniform'
        np.testing.assert_array_equal(sample(model, 2, 'cpu'), [.1234, .6789])
        runtime.rand.assert_called_once_with(2, device='cpu', dtype=np.float32)

    def path_function(self):
        return production_function('conditional_ot_flow_path', {'torch': fake_torch()})

    def test_endpoints_derivative_and_constant_paired_velocity(self):
        rng = np.random.RandomState(51)
        clean, noise = [tensor(rng.normal(size=(3, 8, 25, 3))) for _ in range(2)]
        path = self.path_function()
        for sigma_min in (0., 1.e-4, .2):
            for time in (0., .25, .75, 1.):
                mixed, velocity = path(clean, noise, tensor([time] * 3), sigma_min)
                expected = (1 - (1 - sigma_min) * time) * noise + time * clean
                np.testing.assert_allclose(mixed, expected)
                np.testing.assert_array_equal(velocity, clean - (1 - sigma_min) * noise)
            np.testing.assert_allclose(path(clean, noise, tensor([0.] * 3), sigma_min)[0], noise)
            np.testing.assert_allclose(path(clean, noise, tensor([1.] * 3), sigma_min)[0],
                                       clean + sigma_min * noise)
            first = path(clean, noise, tensor([.3] * 3), sigma_min)[0]
            second = path(clean, noise, tensor([.300001] * 3), sigma_min)[0]
            np.testing.assert_allclose((second - first) / 1.e-6,
                                       clean - (1 - sigma_min) * noise, atol=1.e-9)
        for time in (tensor([-1., .5, .5]), tensor([np.nan, .5, .5]), tensor([.5])):
            with self.assertRaises(ValueError):
                path(clean, noise, time, 1.e-4)

    def test_ot_sampled_derivative_matches_eulerian_field_and_zero_noise_limit(self):
        rng = np.random.RandomState(97)
        clean, noise = [tensor(rng.normal(size=(3, 8, 25, 3))) for _ in range(2)]
        time = tensor([0., .37, 1.])
        sigma_min = 1.e-4
        mixed, velocity = self.path_function()(clean, noise, time, sigma_min)
        weight = time[:, None, None, None]
        eulerian = (clean - (1 - sigma_min) * mixed) / (1 - (1 - sigma_min) * weight)
        np.testing.assert_allclose(velocity, eulerian, rtol=1.e-8, atol=1.e-8)
        mixed_zero, velocity_zero = self.path_function()(clean, noise, time, 0.)
        np.testing.assert_array_equal(mixed_zero, (1 - weight) * noise + weight * clean)
        np.testing.assert_array_equal(velocity_zero, clean - noise)

    def loss_model(self):
        runtime = fake_torch()
        model = types.SimpleNamespace(t_patch_size=4, patch_size=1)
        patchify = production_function('patchify', {'torch': runtime}, owner='Transformer',
                                      path=ROOT / 'model/transformer_macdiff.py')
        model.patchify = types.MethodType(patchify, model)
        loss = production_function('forward_loss', {'torch': runtime}, owner='Transformer')
        model.forward_loss = types.MethodType(loss, model)
        return model

    def test_masked_mse_is_per_person_then_mean_and_has_no_time_weight(self):
        model = self.loss_model()
        target = tensor(np.zeros((2, 8, 25, 3)))
        prediction = tensor(np.full((2, 50, 12), 10000.))
        mask = tensor(np.zeros((2, 50)))
        mask[0, :1], mask[1, :3] = 1, 1
        prediction[0, :1], prediction[1, :3] = 2, 3
        loss = model.forward_loss(target, prediction, mask, t=tensor([0., 1.]))
        self.assertAlmostEqual(float(loss), (4 + 9) / 2)
        with self.assertRaisesRegex(ValueError, 'masked patch'):
            model.forward_loss(target, prediction, mask * 0)

    def forward_fixture(self):
        runtime = fake_torch()
        runtime.rand = lambda batch, **kw: tensor(np.array([.25, .75], dtype=np.float32)[:batch])
        runtime.randn_like = lambda values: tensor(np.full(values.shape, .4, dtype=np.float32))
        model = self.loss_model()
        model.one_person, model.self_shift = True, False
        model.input_mean, model.input_std = [1., 2., 3.], [2., 3., 4.]
        model.flow_time_scale, model.flow_sigma_min, model.lambda_loss_uni = 999., .2, .02
        model.flow_time_sampling = 'uniform'
        model.sample_flow_time = types.MethodType(production_function('sample_flow_time',
            {'torch': runtime}, owner='Transformer'), model)
        captures = {}
        for name in ('_sequence_layout', '_normalize_sequence'):
            method = production_function(name, owner='Transformer',
                                         path=ROOT / 'model/transformer_macdiff.py')
            setattr(model, name, types.MethodType(method, model))
        def encoder(values, **kwargs):
            captures['encoder_input'], captures['encoder_kwargs'] = values.copy(), kwargs
            latent = tensor(np.ones((2, 25, 8)))
            mask = tensor(np.tile([0., 1.], (2, 25)))
            return latent, latent.mean(dim=1, keepdim=True), mask, tensor(np.tile(np.arange(50), (2, 1)))
        model.forward_encoder = encoder
        model.build_global_local_condition = lambda latent, pooled, restore: tensor(np.ones((2, 50, 8)))
        def decoder(values, **kwargs):
            captures['decoder_input'], captures['decoder_kwargs'] = values.copy(), kwargs
            return tensor(np.zeros((2, 50, 12)))
        model.forward_decoder = decoder
        forward = production_function('forward_macdiff', {
            'torch': runtime, 'conditional_ot_flow_path': self.path_function(),
            'token_uniformity_loss': lambda values: tensor(.25)}, owner='Transformer')
        return model, captures, lambda source, aug: forward(model, source, aug, mask_ratio=.5)

    def test_production_forward_keeps_native_pairing_and_uses_continuous_scaled_time(self):
        model, captured, forward = self.forward_fixture()
        source = tensor(np.arange(2 * 3 * 8 * 25 * 2).reshape(2, 3, 8, 25, 2) / 100.)
        original = source.copy()
        augmented = source + .1
        loss, prediction, mask, metrics = forward(source, augmented)
        raw = source[..., :1].permute(0, 4, 2, 3, 1).reshape(2, 8, 25, 3).float()
        clean = (raw - np.array(model.input_mean)) / np.array(model.input_std)
        weight = np.array([.25, .75])[:, None, None, None]
        expected = .4 * (1 - (1 - model.flow_sigma_min) * weight) + clean * weight
        np.testing.assert_allclose(captured['decoder_input'], expected, rtol=1.e-6, atol=1.e-6)
        np.testing.assert_array_equal(captured['decoder_kwargs']['t'], [.25 * 999., .75 * 999.])
        np.testing.assert_array_equal(captured['encoder_kwargs']['x_orig'], raw)
        np.testing.assert_allclose(captured['encoder_input'],
            clean + .1 / np.array(model.input_std), rtol=1.e-6, atol=1.e-6)
        self.assertAlmostEqual(float(loss), float(metrics['loss_flow_matching']) + .02 * .25, places=5)
        target = model.patchify(tensor(clean - (1 - model.flow_sigma_min) * .4))
        expected_mse = ((np.asarray(target) ** 2).mean(axis=-1) * np.asarray(mask)).sum(axis=-1) / np.asarray(mask).sum(axis=-1)
        self.assertAlmostEqual(float(metrics['loss_flow_matching']), expected_mse.mean(), places=5)
        np.testing.assert_array_equal(source, original)
        changed = source.copy()
        changed[..., 1] += 10000
        np.testing.assert_allclose(forward(changed, changed + .1)[0], loss)

    def test_mapped_time_reaches_path_embedding_and_unmodified_velocity_loss(self):
        model, captured, forward = self.forward_fixture()
        model.flow_sigma_min = 1.e-4
        mapped = tensor(inverse_cosine_times()(model.flow_sigma_min)[[100, 900]].astype(np.float32))
        model.sample_flow_time = lambda batch_size, device: mapped
        source = tensor(np.arange(2 * 3 * 8 * 25 * 2).reshape(2, 3, 8, 25, 2) / 100.)
        loss, _, mask, metrics = forward(source, source + .1)
        raw = model._sequence_layout(source).float()
        clean = model._normalize_sequence(raw)
        mixed, velocity = self.path_function()(clean, tensor(np.full(clean.shape, .4)), mapped,
                                               model.flow_sigma_min)
        np.testing.assert_allclose(captured['decoder_input'], mixed, rtol=1.e-6)
        np.testing.assert_array_equal(captured['decoder_kwargs']['t'], mapped * 999.)
        expected = model.forward_loss(velocity, tensor(np.zeros((2, 50, 12))), mask)
        self.assertAlmostEqual(float(metrics['loss_flow_matching']), float(expected), places=5)
        self.assertAlmostEqual(float(loss), float(expected) + .02 * .25, places=5)
        self.assertEqual(float(metrics['snr_middle_fraction']), 1.)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for real FM/autograd/LP checks')
class FlowTorchTests(unittest.TestCase):
    @staticmethod
    def options():
        return dict(dim_feat=8, dim_t_embed=64, depth=1, decoder_depth=1, num_heads=2,
                    num_frames=8, num_joints=25, t_patch_size=4, one_person=True,
                    lambda_loss_uni=.02, uncond_ratio=0.)

    def model(self, **overrides):
        import torch
        from model.transformer_macdiff_flow_matching import Transformer
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**dict(self.options(), **overrides))

    def test_mapped_sampler_keeps_weights_and_supports_autograd_and_device_moves(self):
        import torch
        torch.manual_seed(41)
        uniform = self.model()
        torch.manual_seed(41)
        mapped = self.model(flow_time_sampling='inverse_cosine_snr_v1')
        self.assertEqual(list(uniform.state_dict()), list(mapped.state_dict()))
        for name, value in uniform.state_dict().items():
            torch.testing.assert_allclose(value, mapped.state_dict()[name], rtol=0, atol=0,
                                          equal_nan=False)
        self.assertEqual(tuple(mapped._flow_snr_time_lookup.shape), (1000,))
        for device in ([torch.device('cpu'), torch.device('cuda')]
                       if torch.cuda.is_available() else [torch.device('cpu')]):
            mapped.to(device)
            source = torch.randn(2, 3, 8, 25, 2, device=device)
            loss, _, _, metrics = mapped(source, source + .01, mask_ratio=.5)
            self.assertEqual(mapped._flow_snr_time_lookup.device, source.device)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(mapped.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
            self.assertGreater(mapped.decoder_pred.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in mapped.parameters()))
            self.assertIn('snr_middle_fraction', metrics)
            mapped.zero_grad()

    def test_native_weights_and_initialization_are_identical(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        torch.manual_seed(41)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(**self.options())
        torch.manual_seed(41)
        flow = self.model()
        self.assertEqual(list(native.state_dict()), list(flow.state_dict()))
        for name, value in native.state_dict().items():
            torch.testing.assert_allclose(value, flow.state_dict()[name], rtol=0, atol=0,
                                          equal_nan=False)
        for name in ('diffusion', 'schedule_sampler', 'loss_reweight'):
            self.assertFalse(hasattr(flow, name))

    def test_forward_gradient_reaches_native_encoder_and_decoder(self):
        import torch
        model = self.model()
        source = torch.randn(2, 3, 8, 25, 2)
        loss, prediction, mask, metrics = model(source, source + .01, mask_ratio=.5)
        self.assertEqual(tuple(prediction.shape), (2, 50, 12))
        torch.testing.assert_allclose(loss.detach(), metrics['loss_flow_matching']
                                      + .02 * metrics['loss_uniformity'], equal_nan=False)
        loss.backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder_pred.weight.grad.abs().sum().item(), 0)

    def test_exact_path_decoder_time_and_person0(self):
        import torch
        model = self.model().eval()
        source = torch.randn(2, 3, 8, 25, 2)
        times = torch.tensor([.1234, .6789])
        original_decoder = model.forward_decoder
        observed = []
        def capture(values, *, z, t):
            observed.append((values.detach().clone(), t.detach().clone()))
            return original_decoder(values, z=z, t=t)
        model.forward_decoder = capture
        # Patch only the first torch.rand call (continuous path t); native masking
        # and condition dropout must keep using the real random generator.
        original_rand = torch.rand
        calls = [0]
        def rand(*shape, **kwargs):
            calls[0] += 1
            return times.to(kwargs.get('device', 'cpu')) if calls[0] == 1 else original_rand(*shape, **kwargs)
        with mock.patch.object(torch, 'rand', side_effect=rand), \
                mock.patch.object(torch, 'randn_like', side_effect=lambda x: torch.full_like(x, .4)):
            model(source, source + .01, mask_ratio=.5)
        raw = model._sequence_layout(source)
        clean = model._normalize_sequence(raw)
        weight = times[:, None, None, None]
        torch.testing.assert_allclose(observed[0][0], (1 - (1 - model.flow_sigma_min) * weight) * .4
                                      + weight * clean,
                                      equal_nan=False)
        torch.testing.assert_allclose(observed[0][1], times * 999., equal_nan=False)

    def test_checkpoint_encoder_loads_into_existing_linear_probe(self):
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        options = self.options()
        for name in ('decoder_depth', 'one_person', 'lambda_loss_uni', 'uncond_ratio', 'dim_t_embed'):
            options.pop(name)
        with contextlib.redirect_stdout(io.StringIO()):
            target = Downstream(num_classes=60, protocol='linprobe2', **options)
        message = target.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all('head' in name for name in message.missing_keys), message)

    def test_common_engine_records_fm_metrics_without_teacher_hooks(self):
        import torch
        from engine_pretrain import train_one_epoch
        model = self.model()
        source = torch.randn(2, 3, 8, 25, 2)
        dataset = torch.utils.data.TensorDataset(source, source + .01,
                                                torch.zeros(2), torch.arange(2))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        class Scaler:
            _scaler = types.SimpleNamespace(get_scale=lambda: 1.)
            def __call__(self, loss, optimizer, parameters, update_grad):
                loss.backward()
                if update_grad:
                    optimizer.step()
        args = types.SimpleNamespace(enable_ose=False, epochs=2, warmup_epochs=0,
            min_lr_epochs=0, lr=.001, min_lr=.0001, accum_iter=1, max_train_steps=0,
            mask_ratio=.5, motion_stride=1, motion_aware_tau=-1, enable_amp=False)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(model, loader, optimizer, torch.device('cpu'),
                                    0, Scaler(), args=args)
        self.assertIn('loss_flow_matching', stats)
        self.assertIn('loss_uniformity', stats)


if __name__ == '__main__':
    unittest.main()
