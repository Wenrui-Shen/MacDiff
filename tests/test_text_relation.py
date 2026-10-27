"""Relation mathematics/control flow without Torch, plus optional real Torch checks."""
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

from tests.test_pretrain_accumulation_policy import Model, Optimizer, Scaler, update_block


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/transformer_macdiff_text_relation.py'
VERSION = 'paired_global_fifo_text_to_skeleton_kl_v1'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def production_function(name, namespace=None, owner=None, path=SOURCE):
    nodes = ast.parse(path.read_text(encoding='utf-8')).body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[name]


def native_parser():
    return production_function('get_args_parser', {'argparse': argparse}, path=ROOT / 'main_pretrain.py')()


def production(name, namespace=None, owner=None):
    return production_function(name, namespace, owner, path=SOURCE)


class Array(np.ndarray):
    def float(self): return array(self.astype(np.float32))
    def detach(self): return self
    def square(self): return self * self
    def exp(self): return array(np.exp(np.asarray(self)))
    def log(self): return array(np.log(np.asarray(self)))
    def t(self): return self.T
    def abs(self): return array(np.abs(self))
    def clone(self): return self.copy()
    def permute(self, *axes): return array(np.asarray(self).transpose(axes))
    def contiguous(self): return self
    def view(self, *shape): return array(np.asarray(self).reshape(shape))
    def new_zeros(self, shape): return array(np.zeros(shape, dtype=self.dtype))
    def numel(self): return self.size
    def copy_(self, value): np.copyto(self, value); return self
    @property
    def device(self): return 'cpu'
    def sum(self, dim=None, keepdim=False, **kwargs):
        return array(np.asarray(self).sum(axis=kwargs.pop('axis', dim),
                                        keepdims=kwargs.pop('keepdims', keepdim), **kwargs))
    def mean(self, dim=None, keepdim=False, **kwargs):
        return array(np.asarray(self).mean(axis=kwargs.pop('axis', dim), keepdims=keepdim, **kwargs))
    def masked_fill(self, mask, value): return array(np.where(mask, value, self))


def array(value): return np.asarray(value).view(Array)


def normalize(value, dim=-1, eps=1.e-12):
    return array(np.asarray(value) / np.maximum(np.linalg.norm(value, axis=dim, keepdims=True), eps))


def log_softmax(value, dim=-1):
    shifted = np.asarray(value) - np.max(value, axis=dim, keepdims=True)
    return array(shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True)))


def runtime():
    return types.SimpleNamespace(no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(amp=types.SimpleNamespace(autocast=lambda **kw: contextlib.nullcontext())),
        cat=lambda values, dim=0: array(np.concatenate(values, axis=dim)),
        arange=lambda *args, **kw: array(np.arange(*args)), isfinite=np.isfinite, any=np.any)


def relation_loss():
    return production('paired_relation_kl', dict(torch=runtime(),
        F=types.SimpleNamespace(normalize=normalize, log_softmax=log_softmax)))


def validation():
    return production('validate_relation_options', dict(math=math, RELATION_VERSION=VERSION))


class RelationDefinitionTests(unittest.TestCase):
    def args(self):
        import main_pretrain_text_relation as entry
        with mock.patch.dict(sys.modules, {'main_pretrain': types.SimpleNamespace(get_args_parser=native_parser)}):
            args = entry.get_args_parser().parse_args([])
        args.output_dir, args.text_cache = '', 'sentence_cache_v3'
        return args

    def prepare(self, args):
        import main_pretrain_text_relation as entry
        with mock.patch.object(entry.importlib, 'import_module', return_value=types.SimpleNamespace(
                validate_relation_options=validation())):
            return entry.prepare_args(args)

    def test_definition_defaults_overrides_and_rejected_targets(self):
        args = self.args()
        args.lambda_relation, args.relation_bank_size = .2, 128
        self.prepare(args)
        self.assertFalse(args.enable_amp)
        self.assertEqual(args.model_args['lambda_relation'], .2)
        self.assertEqual(args.model_args['relation_bank_size'], 128)
        self.assertEqual(args.model_args['lambda_skeleton_to_text'], 0.)
        for key, value in [('lambda_skeleton_to_text', .1), ('lambda_text_to_skeleton', 0.),
                           ('relation_bank_size', 1), ('relation_temperature_text', 0.),
                           ('relation_temperature_skeleton', float('nan')), ('lambda_relation', -.1)]:
            changed = self.args()
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): self.prepare(changed)

    def test_resume_binds_bank_loss_temperatures_data_and_precision(self):
        import main_pretrain_text_relation as entry
        args = self.prepare(self.args())
        saved = copy.deepcopy(args)
        entry._validate_resume_definition(saved, args)
        for key, value in [('model', 'model.transformer_macdiff_text.Transformer'),
                           ('model_args', dict(args.model_args, relation_bank_size=32)),
                           ('model_args', dict(args.model_args, relation_temperature_text=.04)),
                           ('text_cache', 'other_cache'), ('enable_amp', True)]:
            changed = copy.deepcopy(saved)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry._validate_resume_definition(changed, args)

    def test_complete_resume_and_fresh_directory_guard(self):
        args = self.prepare(self.args())
        args.resume = 'checkpoint-20.pth'
        checkpoint = dict(model={}, optimizer={}, scaler={}, epoch=20, args=copy.deepcopy(args))
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **kw: checkpoint)}):
            self.prepare(args)
            del checkpoint['optimizer']
            with self.assertRaises(ValueError): self.prepare(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args()
            args.output_dir = directory
            (Path(directory) / 'log.txt').write_text('old run', encoding='utf-8')
            with self.assertRaises(FileExistsError): self.prepare(args)


class RelationNumpyTests(unittest.TestCase):
    def test_production_forward_uses_person0_detached_blended_global_and_exact_objective(self):
        torch = runtime()
        torch.long, torch.bool = np.dtype('int64'), np.dtype('bool')
        torch.randn_like = lambda value: array(np.full(value.shape, .4, dtype=np.float32))
        torch.ones = lambda *shape, **kw: array(np.ones(shape, dtype=kw.get('dtype', np.float32)))
        captures = {}
        model = types.SimpleNamespace(one_person=True, self_shift=False,
            input_mean=[1., 2., 3.], input_std=[2., 3., 4.], text_input_dim=3,
            text_target_norm='rms', text_context_length=7, lambda_loss_uni=.02,
            lambda_text_to_skeleton=1., lambda_text_uniformity=0., lambda_relation=.1,
            relation_temperature_skeleton=.1, relation_temperature_text=.1,
            relation_skeleton_keys=array([[1., 0.], [0., 1.], [-1., 0.]]),
            relation_text_keys=array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]),
            relation_sample_ids=array([100, 200, 300]))
        for name in ('_sequence_layout', '_normalize_sequence'):
            setattr(model, name, types.MethodType(production_function(name, owner='Transformer',
                path=ROOT / 'model/transformer_macdiff.py'), model))
        model.schedule_sampler = types.SimpleNamespace(sample=lambda n, device: (array(np.arange(n)), None))
        model.diffusion = types.SimpleNamespace(q_sample=lambda clean, time, noise: clean * .7 + noise * .3)
        def encoder(values, **kwargs):
            captures['encoder_input'] = values.copy()
            latent = array(np.tile([[[1., 0.], [0., 1.]]], (values.shape[0], 1, 1)))
            return latent, latent.mean(dim=1, keepdim=True), array(np.ones((values.shape[0], 2))), None
        model.forward_encoder = encoder
        model.build_global_local_condition = lambda *args: None
        model.forward_decoder = lambda noisy, **kwargs: array(np.zeros((noisy.shape[0], 2, 12)))
        model.forward_loss = lambda *args: array(2.)
        model.validate_text_tokens = lambda *args: None
        model.fixed_clip_target = lambda value, *args: value
        model.text_remap = lambda value: value + 10.
        model.text_person_embedding = lambda index: array(np.zeros(index.shape + (3,)))
        model.text_position_embedding = lambda index: array(np.zeros(index.shape + (3,)))
        model.text_to_skeleton_loss = lambda *args: array(3.)
        model.make_relation_keys = types.MethodType(production('make_relation_keys', owner='Transformer'), model)
        model.skeleton_auxiliary_loss = types.MethodType(production('skeleton_auxiliary_loss', owner='Transformer'), model)
        forward = production('forward', dict(torch=torch, paired_relation_kl=relation_loss(),
            token_uniformity_loss=lambda latent: array(.25)), 'Transformer')
        source = array(np.arange(2 * 3 * 8 * 25 * 2, dtype=np.float32).reshape(2, 3, 8, 25, 2) + 1)
        # Keep a nonempty second person, but person0 of row1 is inactive.
        source[1, ..., 0] = 0
        original = source.copy()
        features = array([[0., 1., 0.], [1., 0., 0.]])
        target = array([[1., 0., 0.], [0., 0., 1.]])
        tokens = array(np.ones((2, 6, 3)))
        result = forward(model, source, source + .01, text_features=features,
            text_tokens=tokens, text_token_mask=array(np.ones((2, 6), dtype=bool)),
            text_person_ids=array(np.zeros((2, 6), dtype=np.int64)),
            text_positions=array(np.tile(np.arange(1, 7), (2, 1))),
            text_target_global=target, text_target_tokens=tokens,
            sample_indices=array(np.array([10, 20], dtype=np.int64)), mask_ratio=.5)
        loss, prediction, mask, metrics, pending = result
        expected, _ = relation_loss()(array([[.5, .5]]), target[:1], array([10]),
            model.relation_skeleton_keys, model.relation_text_keys, model.relation_sample_ids, .1, .1)
        self.assertAlmostEqual(float(loss), 2. + 3. + .02 * .25 + .1 * float(expected), places=5)
        np.testing.assert_array_equal(pending['sample_ids'], [10])
        np.testing.assert_array_equal(pending['text'], target[:1])
        np.testing.assert_array_equal(pending['skeleton'], [[.5, .5]])
        np.testing.assert_array_equal(source, original)
        self.assertEqual(float(metrics['relation_bank_fill']), 3.)

    def example(self):
        return [array(values) for values in (
            [[1., 0.]], [[1., 0., 0.]], [9],
            [[1., 0.], [0., 1.], [-1., 0.]],
            [[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]], [1, 2, 3])]

    def test_matching_relations_allow_different_dimensions_and_joint_anchor_permutation(self):
        values = self.example()
        loss, metrics = relation_loss()(*values, 1., 1.)
        self.assertAlmostEqual(float(loss), 0., places=6)
        self.assertEqual(float(metrics['relation_anchor_count']), 3.)
        values[3] = values[3][[2, 0, 1]]
        changed, _ = relation_loss()(*values, 1., 1.)
        self.assertGreater(float(changed), .1)
        values[4], values[5] = values[4][[2, 0, 1]], values[5][[2, 0, 1]]
        restored, _ = relation_loss()(*values, 1., 1.)
        self.assertAlmostEqual(float(restored), 0., places=6)

    def test_kl_direction_matches_text_weighted_distribution(self):
        values = self.example()
        values[3] = values[3][[1, 0, 2]]
        loss, _ = relation_loss()(*values, .8, .6)
        teacher_log = log_softmax(values[1] @ values[4].T / .6)
        student_log = log_softmax(values[0] @ values[3].T / .8)
        expected = np.sum(np.exp(teacher_log) * (teacher_log - student_log))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_all_self_copies_and_uninitialized_entries_are_excluded(self):
        values = self.example()
        values[2] = array([1])
        base, _ = relation_loss()(*values, .7, .9)
        values[3] = array(np.concatenate((values[3], [[999., 0.], [1., 1.]])))
        values[4] = array(np.concatenate((values[4], [[-999., 0., 0.], [1., 1., 1.]])))
        values[5] = array([1, 2, 3, 1, -1])
        observed, metrics = relation_loss()(*values, .7, .9)
        self.assertTrue(np.isfinite(observed))
        self.assertAlmostEqual(float(observed), float(base), places=6)
        self.assertEqual(float(metrics['relation_anchor_count']), 2.)
        for ids in ([-1] * 5, [1, 1, 1, 1, 2]):
            values[5] = array(ids)
            loss, _ = relation_loss()(*values, .7, .9)
            self.assertEqual(float(loss), 0.)

    def test_fifo_pairs_survive_wrapping_and_skipped_optimizer_steps(self):
        model = types.SimpleNamespace(relation_bank_size=4,
            relation_skeleton_keys=array(np.zeros((4, 2))), relation_text_keys=array(np.zeros((4, 3))),
            relation_sample_ids=array(np.full(4, -1, dtype=np.int64)), relation_pointer=array(np.array(0)))
        update = production('complete_relation_step', dict(torch=runtime(),
            F=types.SimpleNamespace(normalize=normalize), gather_relation_pairs=lambda *values: values), 'Transformer')
        pending = [dict(sample_ids=array(np.arange(7)), skeleton=array(np.column_stack((np.arange(7), np.ones(7)))),
                        text=array(np.column_stack((np.arange(7), np.ones(7), np.ones(7)))))]
        update(model, pending, successful=False)
        np.testing.assert_array_equal(model.relation_sample_ids, [-1] * 4)
        update(model, pending, successful=True)
        np.testing.assert_array_equal(model.relation_sample_ids, [4, 5, 6, 3])
        self.assertEqual(int(model.relation_pointer), 3)
        for slot, index in enumerate(model.relation_sample_ids):
            np.testing.assert_allclose(model.relation_skeleton_keys[slot], normalize(array([index, 1.])))
            np.testing.assert_allclose(model.relation_text_keys[slot], normalize(array([index, 1., 1.])))

    def test_engine_commits_only_successful_complete_windows(self):
        for skipped, expected in (((), [(0, 1), (2, 3)]), ((1,), [(2, 3)])):
            model, optimizer, scaler = Model(), Optimizer(), Scaler(skipped)
            model.supports_relation_memory = True
            updates = []
            model.complete_relation_step = lambda batches, successful: updates.append(tuple(batches)) if successful else None
            env = dict(accum_iter=2, optimizer=optimizer, model=model, model_without_ddp=model,
                loss_scaler=scaler, text_features=None, pending_target_indices=[], pending_target_batches=[],
                pending_relation_batches=[])
            block = update_block('train_one_epoch_macdiff')
            for step in range(5):
                env.update(data_iter_step=step, loss=float(step + 1))
                env['pending_relation_batches'].append(step)
                exec(block, env)
            self.assertEqual(updates, expected)
            self.assertEqual(env['pending_relation_batches'], [4])


@unittest.skipUnless(HAS_TORCH, 'PyTorch required for real relation gradients/forward/checkpoint checks')
class RelationTorchTests(unittest.TestCase):
    def model(self):
        import torch
        from model.transformer_macdiff_text_relation import Transformer
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1, num_heads=2,
                num_frames=8, num_joints=25, t_patch_size=4, one_person=True,
                text_input_dim=8, text_hidden_dim=16, text_context_length=7,
                relation_bank_size=8, lambda_loss_uni=.02, uncond_ratio=0., diff_prediction='noise')

    def batch(self, ids=(10, 20)):
        import torch
        source = torch.randn(2, 3, 8, 25, 2)
        features, tokens = torch.randn(2, 8), torch.randn(2, 6, 8)
        return source, dict(text_features=features, text_tokens=tokens,
            text_token_mask=torch.ones(2, 6, dtype=torch.bool), text_person_ids=torch.zeros(2, 6, dtype=torch.long),
            text_positions=torch.arange(1, 7).expand(2, -1), text_target_global=features.clone(),
            text_target_tokens=tokens.clone(), sample_indices=torch.tensor(ids), mask_ratio=.5)

    def test_teacher_and_all_keys_are_detached_but_student_query_gets_gradient(self):
        import torch
        from model.transformer_macdiff_text_relation import paired_relation_kl
        query, teacher = torch.randn(2, 4, requires_grad=True), torch.randn(2, 6, requires_grad=True)
        keys_s, keys_t = torch.randn(5, 4, requires_grad=True), torch.randn(5, 6, requires_grad=True)
        loss, _ = paired_relation_kl(query, teacher, torch.tensor([10, 20]), keys_s, keys_t,
                                    torch.arange(5), .1, .1)
        loss.backward()
        self.assertGreater(query.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertIsNone(teacher.grad)
        self.assertIsNone(keys_s.grad)
        self.assertIsNone(keys_t.grad)

    def test_full_forward_without_s2t_decoder_reaches_encoder_and_remap(self):
        import torch
        model = self.model()
        self.assertFalse(hasattr(model, 'text_noise_decoder'))
        source, kwargs = self.batch()
        first = model(source, source + .01, **kwargs)
        self.assertEqual(first[3]['loss_relation'].item(), 0.)
        model.complete_relation_step([first[4]], successful=True)
        source, kwargs = self.batch((30, 40))
        loss, _, _, metrics, _ = model(source, source + .01, **kwargs)
        self.assertGreater(metrics['relation_anchor_count'].item(), 1)
        loss.backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.text_remap.body[0].weight.grad.abs().sum().item(), 0)

    def test_memory_checkpoint_roundtrip_and_existing_lp_compatibility(self):
        import torch
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        source, kwargs = self.batch()
        model.complete_relation_step([model(source, source + .01, **kwargs)[4]], successful=True)
        restored = self.model()
        restored.load_state_dict(model.state_dict())
        for name in ('relation_sample_ids', 'relation_pointer', 'relation_skeleton_keys', 'relation_text_keys'):
            torch.testing.assert_allclose(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            probe = Downstream(dim_feat=8, depth=2, num_heads=2, num_frames=8, num_joints=25,
                               t_patch_size=4, num_classes=60, protocol='linprobe2')
        message = probe.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all('head' in name for name in message.missing_keys), message)

    def test_variable_rank_gather_keeps_modalities_and_ids_in_the_same_order(self):
        import torch
        from model import transformer_macdiff_text_relation as module
        calls = [0]
        def gather(outputs, values):
            calls[0] += 1
            if calls[0] == 1:
                outputs[0].fill_(1); outputs[1].fill_(2)
            else:
                outputs[0].copy_(values)
                outputs[1].copy_(values + (10 if values.ndim == 1 else 100))
        with mock.patch.object(module.dist, 'is_available', return_value=True), \
                mock.patch.object(module.dist, 'is_initialized', return_value=True), \
                mock.patch.object(module.dist, 'get_world_size', return_value=2), \
                mock.patch.object(module.dist, 'all_gather', side_effect=gather):
            ids, skeleton, text = module.gather_relation_pairs(torch.tensor([1]), torch.tensor([[2., 3.]]),
                                                              torch.tensor([[4., 5., 6.]]))
        torch.testing.assert_allclose(ids, torch.tensor([1, 11, 10]))
        torch.testing.assert_allclose(skeleton, torch.tensor([[2., 3.], [102., 103.], [100., 100.]]))
        torch.testing.assert_allclose(text, torch.tensor([[4., 5., 6.], [104., 105., 106.], [100., 100., 100.]]))


if __name__ == '__main__':
    unittest.main()
