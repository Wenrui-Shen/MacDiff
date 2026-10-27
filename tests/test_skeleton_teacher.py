"""Definition/update-contract tests; real tensor/gradient tests require Torch."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/transformer_macdiff_skeleton_teacher.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def option_validator():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'validate_teacher_options')
    scope = {'math': math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['validate_teacher_options']


def native_parser():
    tree = ast.parse((ROOT / 'main_pretrain.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'get_args_parser')
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'native_parser', 'exec'), scope)
    return scope['get_args_parser']()


class DefinitionTests(unittest.TestCase):
    def validate(self, **kwargs):
        options = dict(ema_momentum=.999, contrast_temperature=.07,
                       contrast_queue_size=65536, alignment_teacher_layers=5,
                       depth=8, lambda_contrastive=1.e-4, lambda_alignment=1.)
        options.update(kwargs)
        option_validator()(**options)

    def test_invalid_options_and_zero_weight_ablations(self):
        self.validate()
        self.validate(lambda_contrastive=0., lambda_alignment=0.)
        for overrides in [dict(ema_momentum=1.), dict(ema_momentum=-.1),
                          dict(contrast_temperature=0.), dict(contrast_temperature=float('nan')),
                          dict(contrast_queue_size=0), dict(contrast_queue_size=2.5),
                          dict(alignment_teacher_layers=9), dict(alignment_teacher_layers=True),
                          dict(lambda_alignment=-1.), dict(lambda_contrastive=float('inf'))]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.validate(**overrides)

    def test_entry_cli_overrides_are_saved_and_native_flags_still_parse(self):
        import main_pretrain_skeleton_teacher as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=native_parser)}):
            args = entry.get_args_parser().parse_args([
                '--lambda_alignment', '.02', '--lambda_contrastive', '.0002',
                '--alignment_teacher_layers', '3', '--min_lr', '.0005'])
        args.model_args = dict(depth=8)
        module = types.SimpleNamespace(validate_teacher_options=option_validator())
        with mock.patch.object(entry.importlib, 'import_module', return_value=module):
            entry.prepare_args(args)
        self.assertEqual(args.model_args['lambda_alignment'], .02)
        self.assertEqual(args.model_args['lambda_contrastive'], .0002)
        self.assertEqual(args.model_args['alignment_teacher_layers'], 3)
        self.assertEqual(args.model_args['teacher_method_version'], 'raw_full_position_student_adapter_aa1_v3')
        self.assertEqual(args.min_lr, .0005)

    def test_entry_default_alignment_weight_is_saved(self):
        import main_pretrain_skeleton_teacher as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=native_parser)}):
            args = entry.get_args_parser().parse_args([])
        args.model_args = dict(depth=8)
        module = types.SimpleNamespace(validate_teacher_options=option_validator())
        with mock.patch.object(entry.importlib, 'import_module', return_value=module):
            entry.prepare_args(args)
        self.assertEqual(args.model_args['lambda_alignment'], 1.)
        self.assertFalse(args.enable_amp)
        self.assertEqual(args.skeleton_teacher_numerics_version, 'full_fp32_no_amp_v1')

    def test_amp_cli_overrides_config_in_both_directions(self):
        import main_pretrain_skeleton_teacher as entry
        self.assertTrue(native_parser().parse_args([]).enable_amp)
        self.assertFalse(native_parser().parse_args(['--disable_amp']).enable_amp)
        for configured, cli, expected in [(True, ['--disable_amp'], False),
                                           (False, [], False),
                                           (False, ['--enable_amp'], True)]:
            with self.subTest(configured=configured, cli=cli), mock.patch.dict(sys.modules, {
                    'main_pretrain': types.SimpleNamespace(get_args_parser=native_parser),
                    'yaml': types.SimpleNamespace(safe_load=lambda stream: dict(enable_amp=configured))}):
                args = entry.parse_args(cli)
            self.assertEqual(args.enable_amp, expected)

    def test_disabled_scaler_stops_before_step_on_nonfinite_gradient_norm(self):
        class PassThroughScaler:
            def __init__(self, enabled=True):
                self.enabled = enabled

            def is_enabled(self):
                return self.enabled

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.steps += 1

            def update(self):
                pass

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(amp=types.SimpleNamespace(GradScaler=PassThroughScaler)),
            isfinite=lambda value: types.SimpleNamespace(item=lambda: math.isfinite(value)))
        tree = ast.parse((ROOT / 'util/misc.py').read_text(encoding='utf-8'))
        scaler_class = next(node for node in tree.body
                            if isinstance(node, ast.ClassDef) and node.name == 'NativeScalerWithGradNormCount')
        scope = dict(torch=fake_torch, get_grad_norm_=lambda parameters: parameters[0])
        exec(compile(ast.Module(body=[scaler_class], type_ignores=[]), 'actual_scaler', 'exec'), scope)
        scaler = scope['NativeScalerWithGradNormCount'](enabled=False)
        optimizer = types.SimpleNamespace(steps=0)
        loss = types.SimpleNamespace(backward=lambda **kwargs: None)
        self.assertEqual(scaler(loss, optimizer, parameters=[2.]), 2.)
        self.assertEqual(optimizer.steps, 1)
        for norm in (float('inf'), float('nan')):
            with self.subTest(norm=norm), self.assertRaisesRegex(FloatingPointError, 'step aborted'):
                scaler(loss, optimizer, parameters=[norm])
            self.assertEqual(optimizer.steps, 1)
        # Execute the actual main-entry setup: all PT paths follow the AMP flag.
        tree = ast.parse((ROOT / 'main_pretrain.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        setup = [node for node in main.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == 'loss_scaler']
        for teacher in (False, True):
            for amp in (False, True):
                env = dict(args=types.SimpleNamespace(enable_amp=amp),
                           model_without_ddp=types.SimpleNamespace(supports_skeleton_teacher=teacher),
                           NativeScaler=scope['NativeScalerWithGradNormCount'])
                exec(compile(ast.Module(body=setup, type_ignores=[]), 'actual_scaler_setup', 'exec'), env)
                self.assertEqual(env['loss_scaler']._scaler.is_enabled(), amp)

    def test_resume_rejects_old_methods_and_changed_teacher_definition(self):
        import main_pretrain_skeleton_teacher as entry
        options = dict(entry.TEACHER_DEFAULTS, depth=8, decoder_depth=3,
                       diff_noise_schedule=['inverse_cosine', 1])
        saved = argparse.Namespace(model=entry.MODEL_NAME, model_args=options)
        entry._validate_resume_definition(saved, dict(options))
        for changes in [dict(lambda_alignment=.2), dict(ema_momentum=.99),
                        dict(contrast_queue_size=16), dict(decoder_depth=5),
                        dict(teacher_method_version='different')]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'model_args'):
                entry._validate_resume_definition(saved, dict(options, **changes))
        with self.assertRaisesRegex(ValueError, 'this skeleton-teacher method'):
            entry._validate_resume_definition(dict(model='model.transformer_macdiff.Transformer'), options)
        for version in ('raw_full_position_aa_v1', 'raw_full_position_paper_aa_v2'):
            old_args = dict(model=entry.MODEL_NAME, model_args=dict(
                options, teacher_method_version=version))
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'model_args'):
                entry._validate_resume_definition(old_args, options)

    @unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'NumPy is required')
    def test_actual_alignment_uses_signed_weights_and_unprojected_teacher_targets(self):
        # Execute the real method with NumPy arithmetic. This checks the loss
        # without Torch; autograd and AMP still require the TensorTests below.
        import numpy as np

        class Array(np.ndarray):
            def float(self):
                return self.astype(np.float32)

            def detach(self):
                return self.copy()

        def array(values):
            return np.asarray(values, dtype=np.float32).view(Array)

        def smooth_l1(prediction, target, reduction):
            self.assertEqual(reduction, 'mean')
            error = np.abs(prediction - target)
            return np.where(error < 1., .5 * error ** 2, error - .5).mean()

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(amp=types.SimpleNamespace(
                autocast=lambda **kwargs: contextlib.nullcontext())),
            einsum=np.einsum,
            stack=lambda values, dim=0: array(np.stack(values, axis=dim)))
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        model_class = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == 'Transformer')
        function = next(node for node in model_class.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'alignment_loss')
        scope = dict(torch=fake_torch, F=types.SimpleNamespace(smooth_l1_loss=smooth_l1))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), scope)
        model = types.SimpleNamespace(
            alignment_weights=array([[-1., 2.], [2., 0.], [0., -.5]]),
            alignment_adapters=[lambda value: value, lambda value: value])
        layers = [array(np.full((1, 1, 8), value)) for value in (1., 2., 4.)]
        targets = [array(np.full((1, 1, 8), value)) for value in (.5, 1.5)]
        # Student mixtures = 3, 0; raw teacher targets = .5, 1.5.
        # SmoothL1 = 2, 1. A softmax or extra LN cannot give this value.
        self.assertAlmostEqual(float(scope['alignment_loss'](model, layers, targets)), 1.5)
        # A zero student projection cannot erase these nonzero teacher targets.
        model.alignment_adapters = [lambda value: 0. * value] * 2
        self.assertAlmostEqual(float(scope['alignment_loss'](model, layers, targets)), .5625)

    @unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'NumPy is required')
    def test_native_precision_wrappers_disable_outer_amp_and_preserve_targets(self):
        import numpy as np
        state = types.SimpleNamespace(enabled=True)

        @contextlib.contextmanager
        def autocast(enabled):
            previous, state.enabled = state.enabled, enabled
            try:
                yield
            finally:
                state.enabled = previous

        class Array(np.ndarray):
            def float(self):
                return self.astype(np.float32)

            def detach(self):
                return self.copy()

        def array(values, dtype=np.float16):
            return np.asarray(values, dtype=dtype).view(Array)

        class Native:
            def forward_decoder(inner, x, *, z, t):
                self.assertFalse(state.enabled)
                self.assertEqual(x.dtype, np.float32)
                self.assertEqual(z.dtype, np.float32)
                # Finite Q/K may overflow in FP16 before attention scaling.
                return (x @ z.T) * .5

            def forward_loss(inner, imgs, pred, mask, t):
                self.assertFalse(state.enabled)
                for value in (imgs, pred, mask):
                    self.assertEqual(value.dtype, np.float32)
                return array(np.mean((pred - imgs) ** 2), np.float32)

        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'Transformer')
        methods = [node for node in model.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('forward_decoder', 'forward_loss')]
        wrapper = ast.ClassDef(name='Wrapper', bases=[ast.Name(id='Native', ctx=ast.Load())],
                               keywords=[], body=methods, decorator_list=[])
        scope = dict(Native=Native, torch=types.SimpleNamespace(
            isfinite=np.isfinite, cuda=types.SimpleNamespace(amp=types.SimpleNamespace(autocast=autocast))))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     str(SOURCE), 'exec'), scope)
        actual = scope['Wrapper']()
        x = array(np.full((2, 4), 128.))
        with np.errstate(over='ignore'):
            self.assertTrue(np.isinf(x @ x.T).all())
        result = actual.forward_decoder(x, z=x, t=array([0, 1], np.int64))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result == 32768.))
        loss = actual.forward_loss(array(np.zeros((2, 2))), result,
                                   array(np.ones((2, 2))), array([0, 1], np.int64))
        self.assertAlmostEqual(float(loss), 1073741824.)
        self.assertTrue(state.enabled)
        self.assertEqual(x.dtype, np.float16)

    def test_entry_requires_full_resume_and_rejects_text_and_ose(self):
        import main_pretrain_skeleton_teacher as entry
        args = argparse.Namespace(model=entry.MODEL_NAME, model_args=dict(depth=8),
                                  enable_ose=False, text_cache='', resume='saved.pth')
        module = types.SimpleNamespace(validate_teacher_options=option_validator())
        checkpoint = dict(model={}, optimizer={}, scaler={}, epoch=7,
                          args=dict(model=entry.MODEL_NAME,
                                    model_args=dict(entry.TEACHER_DEFAULTS, depth=8)))
        with mock.patch.object(entry.importlib, 'import_module', return_value=module), \
                mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(
                    load=mock.Mock(return_value=checkpoint))}):
            entry.prepare_args(args)
            del checkpoint['optimizer']
            with self.assertRaisesRegex(ValueError, 'missing'):
                entry.prepare_args(args)
        for change in [dict(text_cache='text'), dict(enable_ose=True),
                       dict(model_args=dict(depth=8, teacher_method_version='raw_full_position_aa_v1')),
                       dict(model_args=dict(depth=8, teacher_method_version='raw_full_position_paper_aa_v2'))]:
            bad_args = argparse.Namespace(**dict(vars(args), **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.prepare_args(bad_args)

    def test_teacher_initialization_precedes_resume_and_follows_ddp(self):
        source = (ROOT / 'main_pretrain.py').read_text(encoding='utf-8')
        self.assertLess(source.index('model_without_ddp = model.module'),
                        source.index('model_without_ddp.initialize_pretraining_state()'))
        self.assertLess(source.index('model_without_ddp.initialize_pretraining_state()'),
                        source.index('misc.load_model(args=args'))

    def test_engine_commits_only_successful_complete_windows_and_discards_tail(self):
        # Execute the actual engine update statements with scalar gradients.
        from test_pretrain_accumulation_policy import update_block, Model, Optimizer, Scaler

        class TeacherModel(Model):
            supports_skeleton_teacher = True

            def __init__(self):
                super().__init__()
                self.commits = []

            def complete_pretraining_step(self, pending_batches, successful):
                if successful:
                    self.commits.append(tuple(pending_batches))

        for skipped, expected in [((), [(0, 1), (2, 3)]), ((1,), [(2, 3)])]:
            model, optimizer, scaler = TeacherModel(), Optimizer(), Scaler(skip_attempts=skipped)
            env = dict(accum_iter=2, model=model, model_without_ddp=model,
                       optimizer=optimizer, loss_scaler=scaler, text_features=None,
                       text_target_bank=None, pending_target_indices=[], pending_target_batches=[],
                       pending_skeleton_teacher_batches=[])
            for step in range(5):
                env.update(data_iter_step=step, loss=float(step + 1))
                env['pending_skeleton_teacher_batches'].append(step)
                exec(update_block('train_one_epoch_macdiff'), env)
            self.assertEqual(model.commits, expected)
            self.assertEqual(env['pending_skeleton_teacher_batches'], [4])


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for real model/gradient/DDP checks')
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.options = dict(dim_feat=8, dim_t_embed=64, depth=3, decoder_depth=1,
                            num_heads=2, num_frames=8, num_joints=5, t_patch_size=4,
                            diff_steps=10, diff_noise_schedule=['inverse_cosine', 1],
                            input_mean=[-.0058, -.1333, -.0246],
                            input_var=[.0206, .0805, .0218],
                            lambda_loss_uni=.02, one_person=True,
                            uncond_ratio=.1, layer_mask_ratio=0., drop_rate=0.)

    def model(self, **overrides):
        from model.transformer_macdiff_skeleton_teacher import Transformer
        options = dict(self.options, contrast_queue_size=8, alignment_teacher_layers=2)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**options)

    def run_forward(self, model):
        import torch
        raw = torch.randn(2, 3, 8, 5, 2)
        aug = raw + .005 * torch.randn_like(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            return model(raw, aug, sample_indices=torch.tensor([1, 2]), mask_ratio=.8)

    def test_position_gather_uses_actual_mask_order_and_teacher_full_context(self):
        import torch
        from model.transformer_macdiff_skeleton_teacher import gather_token_positions
        features = torch.arange(2 * 10 * 8).reshape(2, 10, 8)
        ids = torch.tensor([[7, 2], [4, 9]])
        gathered = gather_token_positions(features, ids)
        self.assertTrue(torch.equal(gathered[0], features[0, [7, 2]]))
        self.assertTrue(torch.equal(gathered[1], features[1, [4, 9]]))
        model = self.model()
        raw = torch.randn(2, 3, 8, 5, 2)
        prepared = model._prepare_target(raw)
        teacher = model.ema_teacher
        x = teacher.joints_embed(prepared) + teacher.pos_embed + teacher.temp_embed
        x = x.reshape(2, 10, 8)
        expected = []
        for index, block in enumerate(teacher.blocks):
            x = block(x)
            if index in model.target_layers:
                expected.append(gather_token_positions(x, ids))
        keys, actual = teacher(prepared, ids, model.target_layers)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(tuple(a.shape), (2, 2, 8))
        self.assertEqual(tuple(keys.shape), (2, 8))
        self.assertTrue(all(not value.requires_grad for value in actual))

    def test_native_objective_is_identical_when_auxiliary_weights_are_zero_without_amp(self):
        import numpy as np
        import torch
        from model.transformer_macdiff import Transformer as Native
        torch.manual_seed(71)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(**self.options)
        torch.manual_seed(71)
        student = self.model(lambda_contrastive=0., lambda_alignment=0.)
        for name, value in native.state_dict().items():
            self.assertTrue(torch.equal(value, student.state_dict()[name]), name)
        raw = torch.randn(2, 3, 8, 5, 2)
        aug = raw + .005 * torch.randn_like(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(84)
            np.random.seed(84)
            baseline = native(raw.clone(), aug.clone(), mask_ratio=.8)
            torch.manual_seed(84)
            np.random.seed(84)
            result = student(raw.clone(), aug.clone(), sample_indices=torch.tensor([1, 2]), mask_ratio=.8)
        for a, b in zip(baseline, result[:3]):
            self.assertTrue(torch.equal(a, b))

    def test_native_diagnostics_reject_nonfinite_inputs_with_sample_identity(self):
        import torch
        model = self.model()
        model._native_sample_ids = torch.tensor([17, 23])
        x = torch.zeros(2, 8, 5, 3)
        x[0, 0, 0, 0] = float('nan')
        z = torch.zeros(2, 10, 8)
        with self.assertRaisesRegex(FloatingPointError, 'decoder_prediction.*17, 23'):
            model.forward_decoder(x, z=z, t=torch.tensor([0, 1]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_disabled_scaler_resumes_amp_state_and_uses_unscaled_gradients(self):
        import torch
        from util.misc import NativeScalerWithGradNormCount
        scaler = NativeScalerWithGradNormCount(enabled=False)
        scaler.load_state_dict(dict(scale=65536., growth_factor=2., backoff_factor=.5,
                                    growth_interval=2000, _growth_tracker=7))
        self.assertFalse(scaler._scaler.is_enabled())
        self.assertEqual(scaler._scaler.get_scale(), 1.)
        self.assertEqual(scaler.state_dict(), {})
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler(parameter.pow(2), optimizer, parameters=[parameter])
        self.assertAlmostEqual(parameter.item(), .8)
        optimizer.zero_grad()
        with self.assertRaisesRegex(FloatingPointError, 'step aborted'):
            scaler(parameter * float('nan'), optimizer, parameters=[parameter])
        self.assertAlmostEqual(parameter.item(), .8)

    def test_cuda_decoder_fp32_avoids_finite_qk_overflow_under_outer_amp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for a real autocast overflow regression')
        from model.transformer_macdiff import Transformer as Native
        model = self.model().cuda()
        with torch.no_grad():
            qkv = model.decoder_blocks[0].attn.qkv
            qkv.weight.zero_()
            qkv.bias[:16].fill_(128.)  # head_dim=4: QK=65536 before scaling
            qkv.bias[16:].fill_(1.)
        x = torch.randn(2, 8, 5, 3, device='cuda')
        z = torch.randn(2, 10, 8, device='cuda', requires_grad=True)
        t = torch.tensor([0, 1], device='cuda')
        with torch.cuda.amp.autocast():
            unsafe = Native.forward_decoder(model, x, z=z, t=t)
            actual = model.forward_decoder(x, z=z, t=t)
        self.assertFalse(torch.isfinite(unsafe).all())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.dtype, torch.float32)
        actual.pow(2).mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.grad is not None))

    def test_each_auxiliary_loss_trains_encoder_but_not_teacher_or_decoder(self):
        import torch
        for name in ('contrastive', 'alignment'):
            model = self.model()
            latent, _, _, _ = model.forward_encoder(
                torch.randn(2, 8, 5, 3), x_orig=None, mask_ratio=.8, motion_aware_tau=-1)
            layers, latent, ids = model._student_alignment
            keys, targets = model.ema_teacher(torch.randn(2, 8, 5, 3), ids, model.target_layers)
            if name == 'contrastive':
                model.complete_pretraining_step([
                    dict(teacher_keys=torch.nn.functional.normalize(torch.randn(3, 8), dim=-1),
                         teacher_ids=torch.tensor([20, 21, 22]))], True)
                loss = model.contrastive_loss(latent, keys, torch.tensor([1, 2]))
            else:
                loss = model.alignment_loss(layers, targets)
            loss.backward()
            grad = model.blocks[0].attn.qkv.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)
            self.assertTrue(all(p.grad is None for p in model.ema_teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in model.decoder_blocks.parameters()))
            if name == 'alignment':
                self.assertGreater(model.alignment_weights.grad.abs().sum().item(), 0.)
                self.assertGreater(model.alignment_adapters[0].weight.grad.abs().sum().item(), 0.)

    def test_alignment_raw_teacher_targets_are_detached_and_student_adapters_train(self):
        import torch
        model = self.model()
        with torch.no_grad():
            model.alignment_weights.copy_(torch.tensor([[-1., 2.], [2., 0.], [0., -.5]]))
            for adapter in model.alignment_adapters:
                torch.nn.init.eye_(adapter.weight)
                torch.nn.init.zeros_(adapter.bias)
        self.assertFalse(hasattr(model, 'teacher_alignment_adapters'))
        self.assertFalse(any('teacher_alignment_adapters' in name for name in model.state_dict()))
        layers = [torch.full((1, 1, 8), value, requires_grad=True) for value in (1., 2., 4.)]
        targets = [torch.full((1, 1, 8), value, requires_grad=True) for value in (.5, 1.5)]
        loss = model.alignment_loss(layers, targets)
        self.assertAlmostEqual(loss.item(), 1.5)
        loss.backward()
        self.assertTrue(all(target.grad is None for target in targets))
        self.assertTrue(all(layer.grad is not None for layer in layers))
        for adapter in model.alignment_adapters:
            self.assertIsNotNone(adapter.bias.grad)
            self.assertGreater(adapter.bias.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(model.alignment_weights.grad).all())
        self.assertTrue(all(p.grad is None for p in model.ema_teacher.parameters()))

    def test_ema_update_does_not_update_trainable_alignment_parameters(self):
        import torch
        model = self.model(ema_momentum=.5)
        parameters = [model.alignment_weights]
        parameters += list(model.alignment_adapters.parameters())
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        before = [parameter.detach().clone() for parameter in parameters]
        model.complete_pretraining_step([
            dict(teacher_keys=torch.randn(2, 8), teacher_ids=torch.tensor([1, 2]))], True)
        for expected, actual in zip(before, parameters):
            self.assertTrue(torch.equal(expected, actual))

    def test_queue_cold_start_same_id_exclusion_ring_wrap_and_ema_skip(self):
        import torch
        model = self.model(contrast_queue_size=3, ema_momentum=.5)
        model.train()
        self.assertFalse(model.ema_teacher.training)
        result = self.run_forward(model)
        self.assertTrue(torch.isfinite(result[0]))
        self.assertEqual(model.lambda_alignment, 1.)
        self.assertTrue(torch.allclose(result[0].detach(), result[3]['loss_native']
                                      + 1.e-4 * result[3]['loss_contrastive']
                                      + result[3]['loss_alignment']))
        self.assertEqual(result[3]['loss_contrastive'].item(), 0.)
        before = copy.deepcopy(model.state_dict())
        model.complete_pretraining_step([result[4]], successful=False)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        target_before = model.ema_teacher.pos_embed.clone()
        with torch.no_grad():
            model.pos_embed.add_(2.)
        model.complete_pretraining_step([result[4]], successful=True)
        self.assertTrue(torch.allclose(model.ema_teacher.pos_embed, target_before + 1.))
        self.assertEqual(model.contrast_queue_count.item(), 2)
        self.assertEqual(model.teacher_updates.item(), 1)
        model.complete_pretraining_step([
            dict(teacher_keys=torch.randn(2, 8), teacher_ids=torch.tensor([3, 4]))], True)
        self.assertEqual(model.contrast_queue_count.item(), 3)
        self.assertEqual(set(model.contrast_queue_ids.tolist()), {2, 3, 4})
        model.contrast_queue_ids.fill_(1)
        loss = model.contrastive_loss(torch.randn(1, 2, 8, requires_grad=True),
                                      torch.randn(1, 8), torch.tensor([1]))
        self.assertEqual(loss.item(), 0.)

    def test_queue_all_gather_keeps_same_order_on_all_ranks(self):
        import torch
        from model import transformer_macdiff_skeleton_teacher as module
        model = self.model()
        keys = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        payload = dict(teacher_keys=keys, teacher_ids=torch.tensor([1, 2]))

        def gather(outputs, tensor):
            outputs[0].copy_(tensor)
            outputs[1].copy_(tensor + (10 if tensor.dtype == torch.long else 0))

        with mock.patch.object(module.dist, 'is_initialized', return_value=True), \
                mock.patch.object(module.dist, 'is_available', return_value=True), \
                mock.patch.object(module.dist, 'get_world_size', return_value=2), \
                mock.patch.object(module.dist, 'all_gather', side_effect=gather):
            model.complete_pretraining_step([payload], True)
        self.assertEqual(model.contrast_queue_ids[:4].tolist(), [1, 2, 11, 12])

    def test_full_training_state_roundtrip_and_lp_loads_only_student(self):
        import torch
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        result = self.run_forward(model)
        result[0].backward()
        self.assertGreater(model.decoder_pred.weight.grad.abs().sum().item(), 0.)
        model.complete_pretraining_step([result[4]], True)
        restored = self.model()
        restored.load_state_dict(model.state_dict(), strict=True)
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        for name in ('joints_embed.proj.weight', 'blocks.0.attn.qkv.weight',
                     'norm.weight', 'temp_embed', 'pos_embed'):
            self.assertIn(name, model.state_dict())
        options = {key: self.options[key] for key in
                   ('dim_feat', 'depth', 'num_heads', 'num_frames', 'num_joints', 't_patch_size')}
        with contextlib.redirect_stdout(io.StringIO()):
            downstream = Downstream(num_classes=60, **options)
        message = downstream.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all('head' in name for name in message.missing_keys))
        self.assertTrue(torch.equal(downstream.blocks[0].attn.qkv.weight,
                                    model.blocks[0].attn.qkv.weight))


if __name__ == '__main__':
    unittest.main()
