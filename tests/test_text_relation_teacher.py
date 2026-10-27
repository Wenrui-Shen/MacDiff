"""Full EMA keys, one shared paired queue, and masked-GAP InfoNCE/KL controls."""
import ast
import contextlib
import copy
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests import test_text_relation as base_tests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/transformer_macdiff_text_relation_teacher.py'
VERSION = 'full_ema_gap_shared_queue_global_kl_v2'
COARSE_VERSION = 'coarse3_mean_random70_ema_gap_shared_queue_kl_v3'
array, normalize = base_tests.array, base_tests.normalize


def production(name, namespace=None, owner=None):
    return base_tests.production_function(name, namespace, owner, path=SOURCE)


def teacher_validation():
    return production('validate_relation_teacher_options', {
        'math': math, 'RELATION_TEACHER_VERSION': VERSION,
        'RELATION_COARSE_TEACHER_VERSION': COARSE_VERSION})


def view_validation():
    return production('validate_teacher_view_options', {
        'math': math, 'RELATION_TEACHER_VERSION': VERSION,
        'RELATION_COARSE_TEACHER_VERSION': COARSE_VERSION})


def runtime():
    value = base_tests.runtime()
    value.long = np.dtype('int64')
    value.zeros = lambda *shape, **kw: array(np.zeros(shape, dtype=kw.get('dtype', np.float32)))
    return value


def info_nce():
    def cross_entropy(logits, labels):
        logp = base_tests.log_softmax(logits)
        return array(-logp[np.arange(len(labels)), labels].mean())
    return production('shared_queue_info_nce', dict(torch=runtime(),
        F=types.SimpleNamespace(normalize=normalize, cross_entropy=cross_entropy)))


class DefinitionTests(unittest.TestCase):
    def args(self):
        import main_pretrain_text_relation_teacher as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=base_tests.native_parser)}):
            args = entry.get_args_parser().parse_args([])
        args.text_cache, args.output_dir = 'sentence_cache_v3', ''
        return args

    def prepare(self, args):
        import main_pretrain_text_relation_teacher as entry
        module = types.SimpleNamespace(validate_relation_teacher_options=teacher_validation(),
                                       validate_relation_options=base_tests.validation(),
                                       validate_teacher_view_options=view_validation())
        with mock.patch.object(entry.importlib, 'import_module', return_value=module):
            return entry.prepare_args(args)

    def test_defaults_use_one_65536_queue_gap_and_old_small_cl_weight(self):
        args = self.prepare(self.args())
        self.assertEqual(args.model_args['relation_bank_size'], 65536)
        self.assertEqual(args.model_args['relation_feature_mode'], 'gap')
        self.assertEqual(args.model_args['lambda_contrastive'], 1.e-4)
        self.assertEqual(args.model_args['ema_momentum'], .999)
        self.assertEqual(args.model_args['relation_teacher_version'], VERSION)
        self.assertFalse(args.enable_amp)
        self.assertNotIn('teacher_view', args.model_args)
        self.assertEqual(args.model_args['lambda_skeleton_to_text'], 0.)
        self.assertEqual(args.model_args['lambda_text_to_skeleton'], 1.)

    def test_overrides_and_invalid_teacher_definitions(self):
        args = self.args()
        args.relation_bank_size, args.lambda_contrastive, args.ema_momentum = 128, .0002, .99
        args = self.prepare(args)
        self.assertEqual(args.model_args['relation_bank_size'], 128)
        self.assertEqual(args.model_args['lambda_contrastive'], .0002)
        self.assertEqual(args.model_args['ema_momentum'], .99)
        valid = dict(relation_teacher_version=VERSION, ema_momentum=.999,
                     lambda_contrastive=1.e-4, contrast_temperature=.07, relation_feature_mode='gap')
        for change in (dict(ema_momentum=1.), dict(lambda_contrastive=-1.),
                       dict(contrast_temperature=0.), dict(relation_feature_mode='joint_concat'),
                       dict(relation_teacher_version='old'), dict(ema_momentum=float('nan'))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                teacher_validation()(**dict(valid, **change))

    def test_resume_rejects_masked_queue_method_and_changed_ema_cl_or_bank(self):
        import main_pretrain_text_relation as base
        import main_pretrain_text_relation_teacher as entry
        args = self.prepare(self.args())
        saved = copy.deepcopy(args)
        base._validate_resume_definition(saved, args, model_name=entry.MODEL_NAME)
        for key, value in (
            ('model', base.MODEL_NAME),
            ('model_args', dict(args.model_args, lambda_contrastive=.001)),
            ('model_args', dict(args.model_args, ema_momentum=.9)),
            ('model_args', dict(args.model_args, relation_bank_size=4096)),
            ('enable_amp', True)):
            changed = copy.deepcopy(saved)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                base._validate_resume_definition(changed, args, model_name=entry.MODEL_NAME)
        args.resume = 'checkpoint-20.pth'
        checkpoint = dict(model={}, optimizer={}, scaler={}, epoch=20, args=saved)
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **kw: checkpoint)}):
            self.prepare(args)
            checkpoint['args'] = copy.deepcopy(saved)
            checkpoint['args'].model = base.MODEL_NAME
            with self.assertRaises(ValueError): self.prepare(args)


class NumpyTests(unittest.TestCase):
    def test_production_forward_combines_masked_query_full_keys_shared_kl_and_cl(self):
        torch = runtime()
        torch.bool = np.dtype('bool')
        torch.randn_like = lambda value: array(np.full(value.shape, .4, dtype=np.float32))
        torch.ones = lambda *shape, **kw: array(np.ones(shape, dtype=kw.get('dtype', np.float32)))
        model = types.SimpleNamespace(one_person=True, self_shift=False, input_mean=[1., 2., 3.],
            input_std=[2., 3., 4.], text_input_dim=3, lambda_loss_uni=.02,
            lambda_text_to_skeleton=1., lambda_text_uniformity=0., lambda_relation=.1,
            lambda_contrastive=1.e-4, contrast_temperature=.07,
            relation_temperature_skeleton=.1, relation_temperature_text=.1,
            relation_skeleton_keys=array([[1., 0.], [0., 1.], [-1., 0.]]),
            relation_text_keys=array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]),
            relation_sample_ids=array([100, 200, 300]))
        for name in ('_sequence_layout', '_normalize_sequence'):
            setattr(model, name, types.MethodType(base_tests.production_function(name, owner='Transformer',
                path=ROOT / 'model/transformer_macdiff.py'), model))
        model.schedule_sampler = types.SimpleNamespace(sample=lambda n, device: (array(np.arange(n)), None))
        model.diffusion = types.SimpleNamespace(q_sample=lambda clean, time, noise: clean * .7 + noise * .3)
        encoder_calls, teacher_inputs = [], []
        def encoder(values, **kwargs):
            encoder_calls.append(kwargs)
            latent = array(np.tile([[[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]]],
                                   (values.shape[0], 1, 1)))
            return latent, latent.mean(dim=1, keepdim=True), array(np.ones((values.shape[0], 50))), None
        model.forward_encoder = encoder
        model.build_global_local_condition = lambda *args: None
        model.forward_decoder = lambda noisy, **kwargs: array(np.zeros((noisy.shape[0], 50, 12)))
        model.forward_loss = lambda *args: array(2.)
        model.validate_text_tokens = lambda *args: None
        model.fixed_clip_target = lambda value, *args: value
        model.text_remap = lambda value: value + 10.
        model.text_person_embedding = lambda index: array(np.zeros(index.shape + (3,)))
        model.text_position_embedding = lambda index: array(np.zeros(index.shape + (3,)))
        model.text_to_skeleton_loss = lambda *args: array(3.)
        def full_encoder(clean):
            teacher_inputs.append(clean.copy())
            return array(np.tile([[0., 1.]], (clean.shape[0], 1)))
        model.ema_teacher = full_encoder
        model.make_relation_keys = types.MethodType(production('make_relation_keys', owner='Transformer'), model)
        model.skeleton_auxiliary_loss = types.MethodType(production('skeleton_auxiliary_loss',
            {'shared_queue_info_nce': info_nce()}, owner='Transformer'), model)
        forward = base_tests.production('forward', dict(torch=torch, paired_relation_kl=base_tests.relation_loss(),
            token_uniformity_loss=lambda latent: array(.25)), 'Transformer')
        source = array(np.arange(2 * 3 * 8 * 25 * 2, dtype=np.float32).reshape(2, 3, 8, 25, 2) + 1)
        source[1, ..., 0] = 0
        targets, tokens = array([[1., 0., 0.], [0., 0., 1.]]), array(np.ones((2, 6, 3)))
        result = forward(model, source, source + .01, text_features=targets,
            text_tokens=tokens, text_token_mask=array(np.ones((2, 6), dtype=bool)),
            text_person_ids=array(np.zeros((2, 6), dtype=np.int64)),
            text_positions=array(np.tile(np.arange(1, 7), (2, 1))), text_target_global=targets,
            text_target_tokens=tokens, sample_indices=array(np.array([10, 20], dtype=np.int64)), mask_ratio=.9)
        loss, _, _, metrics, pending = result
        query, full = array([[.6, .4]]), array([[0., 1.]])
        cl, _ = info_nce()(query, full, array([10]), model.relation_skeleton_keys, model.relation_sample_ids, .07)
        kl, _ = base_tests.relation_loss()(query, targets[:1], array([10]), model.relation_skeleton_keys,
            model.relation_text_keys, model.relation_sample_ids, .1, .1)
        self.assertAlmostEqual(float(loss), 2. + 3. + .02 * .25 + .1 * float(kl) + 1.e-4 * float(cl), places=5)
        self.assertEqual(len(encoder_calls), 1)
        self.assertEqual(encoder_calls[0]['mask_ratio'], .9)
        np.testing.assert_array_equal(pending['skeleton'], full)
        np.testing.assert_array_equal(pending['sample_ids'], [10])
        np.testing.assert_array_equal(pending['text'], targets[:1])
        expected_clean = model._normalize_sequence(model._sequence_layout(source))[:1]
        np.testing.assert_array_equal(teacher_inputs[0], expected_clean)
        self.assertAlmostEqual(float(metrics['loss_contrastive']), float(cl), places=5)

    def test_full_encoder_processes_every_token_in_original_order(self):
        values = array(np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4))
        seen = []
        def block(x):
            seen.append(x.copy())
            return x + 1.
        encoder = types.SimpleNamespace(joints_embed=lambda clean: values,
            pos_embed=array(np.zeros((1, 1, 5, 4))), temp_embed=array(np.zeros((1, 3, 1, 4))),
            blocks=[block, block], norm=lambda x: x)
        forward = production('forward', {'F': types.SimpleNamespace(normalize=normalize)},
                             owner='FullContextRelationKeyEncoder')
        observed = forward(encoder, None)
        np.testing.assert_array_equal(seen[0], values.reshape(2, 15, 4))
        self.assertEqual(seen[1].shape[1], 15)
        np.testing.assert_allclose(observed, normalize((values.reshape(2, 15, 4) + 2.).mean(dim=1), eps=1.e-6))

    def test_info_nce_uses_current_full_positive_and_shared_bank_with_all_self_excluded(self):
        query = array([[1., 0.]])
        positive = array([[0., 1.]])
        bank = array([[1., 0.], [0., -1.], [-1., 0.], [1., 0.]])
        ids = array([9, 2, -1, 9])
        observed, cosine = info_nce()(query, positive, array([9]), bank, ids, .5)
        # Only historical ID2 is a negative; ID9 copies and the empty slot are excluded.
        self.assertAlmostEqual(float(observed), math.log(2.), places=6)
        self.assertEqual(float(cosine), 0.)
        empty, _ = info_nce()(query, positive, array([9]), bank, array([-1] * 4), .5)
        self.assertEqual(float(empty), 0.)

    def test_key_hook_returns_full_encoder_result_instead_of_masked_query(self):
        observed = []
        full_key = array([[0., 1.]])
        model = types.SimpleNamespace(ema_teacher=lambda clean: observed.append(clean.copy()) or full_key)
        key = production('make_relation_keys', owner='Transformer')(model,
            array(np.ones((1, 8, 25, 3))), array([[1., 0.]]))
        np.testing.assert_array_equal(key, full_key)
        self.assertEqual(observed[0].shape, (1, 8, 25, 3))

    def test_contrastive_hook_uses_the_same_queue_objects_and_weight(self):
        captures = []
        model = types.SimpleNamespace(relation_skeleton_keys=object(), relation_sample_ids=object(),
            contrast_temperature=.07, lambda_contrastive=1.e-4)
        def capture(*args): captures.append(args); return array(2.), array(.8)
        auxiliary = production('skeleton_auxiliary_loss', {'shared_queue_info_nce': capture}, owner='Transformer')
        weighted, metrics = auxiliary(model, 'query', 'full key', 'ids')
        self.assertIs(captures[0][3], model.relation_skeleton_keys)
        self.assertIs(captures[0][4], model.relation_sample_ids)
        self.assertAlmostEqual(float(weighted), .0002)
        self.assertEqual(float(metrics['loss_contrastive']), 2.)

    def test_ema_commits_once_after_queue_and_never_on_skipped_or_empty_windows(self):
        events = []
        class Base:
            def complete_relation_step(self, batches, successful):
                if successful and batches: events.append('queue')
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Transformer')
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'complete_relation_step')
        method.decorator_list = []
        wrapper = ast.ClassDef(name='Wrapper', bases=[ast.Name(id='Base', ctx=ast.Load())],
                              keywords=[], body=[method], decorator_list=[])
        scope = {'Base': Base}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     str(SOURCE), 'exec'), scope)
        model = scope['Wrapper']()
        model.ema_momentum = .999
        model.ema_teacher = types.SimpleNamespace(update_from=lambda source, momentum: events.append(('ema', momentum)))
        model.teacher_updates = types.SimpleNamespace(add_=lambda count: events.append(('count', count)))
        model.complete_relation_step([1], successful=False)
        model.complete_relation_step([], successful=True)
        self.assertEqual(events, [])
        model.complete_relation_step([1, 2], successful=True)
        self.assertEqual(events, ['queue', ('ema', .999), ('count', 1)])
        events.clear()
        production('initialize_pretraining_state', owner='Transformer')(model)
        self.assertEqual(events, [('ema', 0.)])


@unittest.skipUnless(base_tests.HAS_TORCH, 'PyTorch required for actual full-key/gradient/EMA/LP checks')
class TorchTests(unittest.TestCase):
    batch = base_tests.RelationTorchTests.batch

    def model(self):
        import torch
        from model.transformer_macdiff_text_relation_teacher import Transformer
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1, num_heads=2,
                num_frames=8, num_joints=25, t_patch_size=4, one_person=True,
                text_input_dim=8, text_hidden_dim=16, text_context_length=7,
                relation_bank_size=8, lambda_loss_uni=.02, uncond_ratio=0., diff_prediction='noise')

    def test_masked_query_full_key_and_only_one_shared_skeleton_queue(self):
        import torch
        model = self.model().train()
        self.assertFalse(model.ema_teacher.training)
        self.assertFalse(hasattr(model, 'contrast_queue'))
        self.assertFalse(hasattr(model, 'text_noise_decoder'))
        self.assertFalse(any('alignment_' in name for name, _ in model.named_parameters()))
        sizes = []
        online = model.blocks[0].register_forward_pre_hook(lambda module, inputs: sizes.append(('q', inputs[0].shape[1])))
        teacher = model.ema_teacher.blocks[0].register_forward_pre_hook(lambda module, inputs: sizes.append(('k', inputs[0].shape[1])))
        source, kwargs = self.batch()
        kwargs['mask_ratio'] = .9
        result = model(source, source + .01, **kwargs)
        online.remove(); teacher.remove()
        self.assertEqual(sizes, [('q', 5), ('k', 50)])
        with torch.no_grad():
            expected = model.ema_teacher(model._normalize_sequence(model._sequence_layout(source)))
        torch.testing.assert_allclose(result[4]['skeleton'], expected)
        self.assertFalse(result[4]['skeleton'].requires_grad)

    def test_gradient_optimizer_ema_and_shared_pair_enqueue(self):
        import torch
        model = self.model()
        source, kwargs = self.batch()
        first = model(source, source + .01, **kwargs)
        model.complete_relation_step([first[4]], successful=True)
        source, kwargs = self.batch((30, 40))
        result = model(source, source + .01, **kwargs)
        self.assertGreater(result[3]['loss_contrastive'].item(), 0.)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
        result[0].backward()
        for name, parameter in model.named_parameters():
            if name.startswith('ema_teacher.'):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        before = model.ema_teacher.blocks[0].attn.qkv.weight.detach().clone()
        optimizer.step()
        model.complete_relation_step([result[4]], successful=False)
        torch.testing.assert_allclose(model.ema_teacher.blocks[0].attn.qkv.weight, before, rtol=0, atol=0)
        model.complete_relation_step([result[4]], successful=True)
        expected = before * .999 + model.blocks[0].attn.qkv.weight.detach() * .001
        torch.testing.assert_allclose(model.ema_teacher.blocks[0].attn.qkv.weight, expected)
        self.assertEqual(model.teacher_updates.item(), 2)
        torch.testing.assert_allclose(model.relation_skeleton_keys[2:4], result[4]['skeleton'])
        torch.testing.assert_allclose(model.relation_sample_ids[:4], torch.tensor([10, 20, 30, 40]))

    def test_checkpoint_restores_ema_queue_counter_and_keeps_existing_lp(self):
        import torch
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        source, kwargs = self.batch()
        model.complete_relation_step([model(source, source + .01, **kwargs)[4]], successful=True)
        restored = self.model()
        restored.load_state_dict(model.state_dict())
        for name, value in model.state_dict().items():
            torch.testing.assert_allclose(value, restored.state_dict()[name], rtol=0, atol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            probe = Downstream(dim_feat=8, depth=2, num_heads=2, num_frames=8, num_joints=25,
                               t_patch_size=4, num_classes=60, protocol='linprobe2')
        message = probe.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all('head' in name for name in message.missing_keys), message)


if __name__ == '__main__':
    unittest.main()
