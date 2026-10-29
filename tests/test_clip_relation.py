"""Frozen global CLIP teacher, shared queues, and removal of text reconstruction."""
import ast
import contextlib
import copy
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from tests import test_text_relation as base
from tests import test_text_relation_teacher as teacher_tests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/transformer_macdiff_clip_relation.py'
VERSION = 'fixed_clip_full_ema_gap_shared_queue_kl_v1'
array, normalize = base.array, base.normalize


def production(name, namespace=None, owner=None):
    return base.production_function(name, namespace, owner, path=SOURCE)


def validation():
    return production('validate_clip_relation_options',
                      dict(math=math, CLIP_RELATION_VERSION=VERSION))


class DefinitionTests(unittest.TestCase):
    def args(self, cli=()):
        import main_pretrain_clip_relation as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(
                get_args_parser=base.native_parser)}):
            args = entry.get_args_parser().parse_args(cli)
        args.output_dir, args.text_cache = '', 'sentence_cache_v3'
        return args

    def prepare(self, args):
        import main_pretrain_clip_relation as entry
        with mock.patch.object(entry.importlib, 'import_module', return_value=types.SimpleNamespace(
                validate_clip_relation_options=validation())):
            return entry.prepare_args(args)

    def test_fixed_source_defaults_cli_overrides_and_no_text_loss(self):
        args = self.prepare(self.args(['--relation_bank_size', '128', '--lambda_contrastive', '.0002',
                                      '--relation_temperature_text', '.05']))
        self.assertEqual(args.model_args['text_target_mode'], 'fixed_clip')
        self.assertEqual(args.model_args['clip_relation_version'], VERSION)
        self.assertEqual(args.model_args['relation_bank_size'], 128)
        self.assertEqual(args.model_args['lambda_contrastive'], .0002)
        self.assertEqual(args.model_args['relation_temperature_text'], .05)
        self.assertEqual(args.model_args['lambda_text_to_skeleton'], 0.)
        self.assertEqual(args.model_args['lambda_skeleton_to_text'], 0.)
        self.assertFalse(args.enable_amp)
        for change in (dict(text_target_mode='sample_target_blend'), dict(one_person=False),
                       dict(relation_feature_mode='joint_concat'), dict(clip_relation_version='old'),
                       dict(relation_bank_size=1), dict(ema_momentum=1.),
                       dict(lambda_text_to_skeleton=1.), dict(lambda_skeleton_to_text=.1),
                       dict(lambda_relation=float('nan')), dict(relation_temperature_text=0.)):
            args = self.args()
            args.model_args.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.prepare(args)

    def test_resume_binds_source_queues_precision_and_complete_state(self):
        import main_pretrain_clip_relation as entry
        import main_pretrain_text_relation as relation_entry
        args = self.prepare(self.args())
        saved = copy.deepcopy(args)
        relation_entry._validate_resume_definition(saved, args, entry.MODEL_NAME)
        for field, value in (
                ('model', 'model.transformer_macdiff_text_relation_teacher.Transformer'),
                ('model_args', dict(args.model_args, text_target_mode='sample_target_blend')),
                ('model_args', dict(args.model_args, relation_temperature_text=.04)),
                ('model_args', dict(args.model_args, ema_momentum=.9)),
                ('text_cache', 'different_cache'), ('enable_amp', True)):
            changed = copy.deepcopy(saved)
            setattr(changed, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                relation_entry._validate_resume_definition(changed, args, entry.MODEL_NAME)
        args.resume = 'checkpoint-38.pth'
        checkpoint = dict(args=saved, model={}, optimizer={}, scaler={}, epoch=38)
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **kw: checkpoint)}):
            self.prepare(args)
            del checkpoint['optimizer']
            with self.assertRaises(ValueError): self.prepare(args)
        with tempfile.TemporaryDirectory() as folder:
            args = self.args()
            args.output_dir = folder
            (Path(folder) / 'checkpoint-38.pth').write_bytes(b'')
            with self.assertRaises(FileExistsError): self.prepare(args)

    def test_fixed_mode_satisfies_cache_schema_without_creating_target_bank(self):
        from util.structured_text_cache import PROTOCOL, validate_training_definition
        validate_training_definition({'protocol': PROTOCOL}, self.prepare(self.args()).model_args)
        tree = ast.parse((ROOT / 'main_pretrain.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        block = next(node for node in main.body if isinstance(node, ast.If)
                     and any(isinstance(item, ast.Name) and item.id == 'target_bank_class'
                             for item in ast.walk(node)))
        # Run the real target-bank gate; its body must not be entered for fixed_clip.
        scope = dict(supports_text=True, model_without_ddp=types.SimpleNamespace(text_target_mode='fixed_clip'))
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'target-bank-gate', 'exec'), scope)
        self.assertNotIn('target_bank_class', scope)


class CacheTests(unittest.TestCase):
    def test_global_reader_preserves_ids_person0_and_never_reads_local_arrays(self):
        from util.person_text_cache import PersonTokenFeatureCache
        reader = object.__new__(PersonTokenFeatureCache)
        reader.global_text = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        # No local arrays exist on this reader; touching features/mask would fail.
        batch = reader.get_global_batch(np.array([3, 1]), one_person=True)
        self.assertEqual(set(batch), {'text_features'})
        np.testing.assert_array_equal(batch['text_features'], reader.global_text[[3, 1], 0])
        for invalid in ([], [-1], [4], [1.5], [[1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): reader.get_global_batch(invalid)

    def test_engine_selects_global_reader_and_keeps_legacy_reader(self):
        class Tensor:
            def __init__(self, values): self.values = np.asarray(values)
            def cpu(self): return self
            def numpy(self): return self.values
            def to(self, *args, **kwargs): return self
        tree = ast.parse((ROOT / 'engine_pretrain.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'train_one_epoch_macdiff')
        loop = next(node for node in function.body if isinstance(node, ast.For))
        read = next(node for node in loop.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == 'text_features')
        block = compile(ast.Module(body=[read], type_ignores=[]), 'cache-read', 'exec')
        for global_only in (True, False):
            calls = []
            def get_global(indices, **kw):
                calls.append('global'); return {'text_features': np.ones((2, 3))}
            def get_all(indices, **kw):
                calls.append('all'); return {'text_features': np.ones((2, 3)), 'text_tokens': np.ones((2, 6, 3))}
            scope = dict(text_features=types.SimpleNamespace(get_batch=get_all, get_global_batch=get_global),
                model_without_ddp=types.SimpleNamespace(text_global_only=global_only, one_person=True),
                sample_indices=Tensor([3, 1]), text_target_bank=None, text_kwargs={}, device='cpu',
                torch=types.SimpleNamespace(from_numpy=Tensor))
            exec(block, scope)
            self.assertEqual(calls, ['global' if global_only else 'all'])
            self.assertEqual('text_tokens' in scope['text_kwargs'], not global_only)


class NumpyTests(unittest.TestCase):
    def fixture(self):
        torch = teacher_tests.runtime()
        torch.randn_like = lambda value: array(np.full(value.shape, .4, dtype=np.float32))
        captures = dict(online=[], teacher=[], decoder=[])
        model = types.SimpleNamespace(one_person=True, self_shift=False, input_mean=[1., 2., 3.], input_std=[2., 3., 4.],
            text_input_dim=3, lambda_loss_uni=.02, lambda_relation=.1, lambda_contrastive=1.e-4,
            contrast_temperature=.07, relation_temperature_skeleton=.1, relation_temperature_text=.1,
            relation_skeleton_keys=array([[1., 0.], [0., 1.], [-1., 0.]]),
            relation_text_keys=array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]),
            relation_sample_ids=array([100, 200, 300]))
        for name in ('_sequence_layout', '_normalize_sequence'):
            setattr(model, name, types.MethodType(base.production_function(name, owner='Transformer',
                path=ROOT / 'model/transformer_macdiff.py'), model))
        model.schedule_sampler = types.SimpleNamespace(sample=lambda n, device: (array(np.arange(n)), None))
        model.diffusion = types.SimpleNamespace(q_sample=lambda clean, time, noise: clean * .7 + noise * .3)
        def encoder(values, **kwargs):
            captures['online'].append((values.copy(), kwargs))
            latent = array(np.tile([[[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]]],
                                   (values.shape[0], 1, 1)))
            return latent, latent.mean(dim=1, keepdim=True), array(np.ones((values.shape[0], 50))), None
        def key_encoder(clean):
            captures['teacher'].append(clean.copy())
            return array(np.tile([[0., 1.]], (clean.shape[0], 1)))
        def decoder(noisy, **kwargs):
            captures['decoder'].append(noisy.copy())
            return array(np.zeros((noisy.shape[0], 50, 12)))
        model.forward_encoder, model.ema_teacher, model.forward_decoder = encoder, key_encoder, decoder
        model.build_global_local_condition = lambda *args: None
        model.forward_loss = lambda *args: array(2.)
        forward = production('forward', dict(torch=torch, F=types.SimpleNamespace(normalize=normalize),
            shared_queue_info_nce=teacher_tests.info_nce(), paired_relation_kl=base.relation_loss(),
            token_uniformity_loss=lambda latent: array(.25)), owner='Transformer')
        source = array(np.arange(2 * 3 * 8 * 25 * 2, dtype=np.float32).reshape(2, 3, 8, 25, 2) + 1)
        source[1, ..., 0] = 0
        kwargs = dict(text_features=array([[0., 2., 0.], [0., 0., 0.]]),
            sample_indices=array(np.array([10, 20], dtype=np.int64)), mask_ratio=.9)
        return model, forward, source, kwargs, captures

    def test_forward_uses_fixed_global_clip_exact_objective_and_single_native_decoder(self):
        model, forward, source, kwargs, captures = self.fixture()
        original = source.copy()
        # Old remapped targets must have no influence, even if an external caller supplies them.
        result = forward(model, source, source + .01, text_target_global=array([[9., 0., 0.]] * 2), **kwargs)
        query, keys, text = array([[.6, .4]]), array([[0., 1.]]), array([[0., 1., 0.]])
        cl, _ = teacher_tests.info_nce()(query, keys, array([10]), model.relation_skeleton_keys,
                                       model.relation_sample_ids, .07)
        kl, _ = base.relation_loss()(query, text, array([10]), model.relation_skeleton_keys,
                                    model.relation_text_keys, model.relation_sample_ids, .1, .1)
        self.assertAlmostEqual(float(result[0]), 2. + .02 * .25 + 1.e-4 * float(cl) + .1 * float(kl), places=5)
        np.testing.assert_array_equal(result[4]['text'], text)
        np.testing.assert_array_equal(result[4]['skeleton'], keys)
        np.testing.assert_array_equal(result[4]['sample_ids'], [10])
        np.testing.assert_array_equal(captures['teacher'][0],
            model._normalize_sequence(model._sequence_layout(source))[:1])
        np.testing.assert_array_equal(source, original)
        self.assertEqual([len(captures[key]) for key in ('online', 'teacher', 'decoder')], [1, 1, 1])
        self.assertNotIn('loss_text_to_skeleton', result[3])
        self.assertAlmostEqual(float(result[3]['loss_contrastive']), float(cl), places=5)

    def test_forward_rejects_missing_ids_bad_clip_vectors_and_unsupported_views(self):
        for change in (dict(sample_indices=None), dict(sample_indices=array([-1, 20])),
                       dict(text_features=None), dict(text_features=array(np.zeros((2, 3)))),
                       dict(text_features=array(np.full((2, 3), np.nan))),
                       dict(enable_ose=True), dict(mask_ratio=1.)):
            model, forward, source, kwargs, _ = self.fixture()
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                forward(model, source, source + .01, **kwargs)

    def test_queue_skip_wrap_pairing_and_postcommit_ema(self):
        events = []
        class Counter:
            value = 0
            def add_(self, count): self.value += count
        model = types.SimpleNamespace(relation_bank_size=4, ema_momentum=.999,
            relation_pointer=array(0), relation_sample_ids=array(np.full(4, -1)),
            relation_skeleton_keys=array(np.zeros((4, 2))), relation_text_keys=array(np.zeros((4, 3))),
            teacher_updates=Counter())
        def update_ema(source, momentum):
            events.append((momentum, source.relation_sample_ids.copy()))
        model.ema_teacher = types.SimpleNamespace(update_from=update_ema)
        update = production('complete_relation_step', dict(torch=base.runtime(),
            F=types.SimpleNamespace(normalize=normalize), gather_relation_pairs=lambda *values: values), 'Transformer')
        pending = [dict(sample_ids=array(np.arange(7)), skeleton=array(np.column_stack((np.arange(7), np.ones(7)))),
                        text=array(np.column_stack((np.arange(7), np.ones(7), np.ones(7)))))]
        update(model, pending, successful=False)
        update(model, [], successful=True)
        self.assertEqual(events, [])
        update(model, pending, successful=True)
        np.testing.assert_array_equal(model.relation_sample_ids, [4, 5, 6, 3])
        self.assertEqual(int(model.relation_pointer), 3)
        self.assertEqual(model.teacher_updates.value, 1)
        self.assertEqual(events[0][0], .999)
        np.testing.assert_array_equal(events[0][1], model.relation_sample_ids)
        for slot, index in enumerate(model.relation_sample_ids):
            np.testing.assert_allclose(model.relation_skeleton_keys[slot], normalize(array([index, 1.])))
            np.testing.assert_allclose(model.relation_text_keys[slot], normalize(array([index, 1., 1.])))
        production('initialize_pretraining_state', owner='Transformer')(model)
        self.assertEqual(events[-1][0], 0.)


@unittest.skipUnless(base.HAS_TORCH, 'PyTorch required for actual gradients, parameter inventory and LP checks')
class TorchTests(unittest.TestCase):
    def model(self):
        import torch
        from model.transformer_macdiff_clip_relation import Transformer
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1, num_heads=2,
                num_frames=8, num_joints=25, t_patch_size=4, one_person=True, text_input_dim=8,
                relation_bank_size=8, lambda_loss_uni=.02, uncond_ratio=0., diff_prediction='noise')

    def batch(self, ids=(10, 20)):
        import torch
        return torch.randn(2, 3, 8, 25, 2), dict(text_features=torch.randn(2, 8, requires_grad=True),
            sample_indices=torch.tensor(ids), mask_ratio=.9)

    def test_only_native_trainables_full_key_masked_query_and_frozen_clip(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        model = self.model().train()
        self.assertFalse(model.ema_teacher.training)
        for name in ('text_remap', 'text_skeleton_decoder', 'text_noise_decoder',
                     'text_person_embedding', 'contrast_queue'):
            self.assertFalse(hasattr(model, name), name)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1, num_heads=2,
                            num_frames=8, num_joints=25, t_patch_size=4, one_person=True)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {name for name, p in native.named_parameters() if p.requires_grad})
        shapes = []
        handles = [model.blocks[0].register_forward_pre_hook(lambda m, inputs: shapes.append(('q', inputs[0].shape[1]))),
                   model.ema_teacher.blocks[0].register_forward_pre_hook(lambda m, inputs: shapes.append(('k', inputs[0].shape[1])))]
        source, kwargs = self.batch()
        result = model(source, source + .01, **kwargs)
        for handle in handles: handle.remove()
        self.assertEqual(shapes, [('q', 5), ('k', 50)])
        self.assertEqual(result[3]['loss_relation'].item(), 0.)
        torch.testing.assert_allclose(result[4]['text'], torch.nn.functional.normalize(kwargs['text_features'].detach(), dim=-1))
        self.assertFalse(result[4]['text'].requires_grad)
        result[0].backward()
        self.assertIsNone(kwargs['text_features'].grad)

    def test_nonempty_queue_gradients_optimizer_and_successful_ema(self):
        import torch
        model = self.model()
        source, kwargs = self.batch()
        model.complete_relation_step([model(source, source + .01, **kwargs)[4]], successful=True)
        source, kwargs = self.batch((30, 40))
        result = model(source, source + .01, **kwargs)
        self.assertGreater(result[3]['loss_contrastive'].item(), 0.)
        self.assertGreater(result[3]['relation_anchor_count'].item(), 1.)
        result[0].backward()
        self.assertIsNone(kwargs['text_features'].grad)
        for name, parameter in model.named_parameters():
            if name.startswith('ema_teacher.'):
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        before = model.ema_teacher.blocks[0].attn.qkv.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
        optimizer.step()
        model.complete_relation_step([result[4]], successful=False)
        torch.testing.assert_allclose(model.ema_teacher.blocks[0].attn.qkv.weight, before, rtol=0, atol=0)
        model.complete_relation_step([result[4]], successful=True)
        torch.testing.assert_allclose(model.ema_teacher.blocks[0].attn.qkv.weight,
            before * .999 + model.blocks[0].attn.qkv.weight.detach() * .001)
        torch.testing.assert_allclose(model.relation_sample_ids[:4], torch.tensor([10, 20, 30, 40]))

    def test_checkpoint_queue_ema_roundtrip_and_original_lp_compatibility(self):
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
