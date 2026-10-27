"""Optional 3x-time coarse-key/random-70% ablation; full keys remain the default."""
import contextlib
import copy
import io
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests import test_text_relation as base_tests
from tests import test_text_relation_teacher as teacher_tests


array = base_tests.array


def mask_runtime():
    value = teacher_tests.runtime()
    value.rand = lambda *shape, **kw: array(np.random.RandomState(91).uniform(size=shape))
    value.argsort = lambda values, dim: array(np.argsort(values, axis=dim))
    return value


def coarsen():
    return teacher_tests.production('coarsen_temporal_tokens')


def sample():
    return teacher_tests.production('sample_teacher_tokens', {'torch': mask_runtime()})


class DefinitionTests(unittest.TestCase):
    def args(self, coarse):
        args = teacher_tests.DefinitionTests().args()
        if coarse:
            args.model_args = dict(relation_teacher_version=teacher_tests.COARSE_VERSION,
                teacher_view='coarse_mask', teacher_temporal_group=3, teacher_mask_ratio=.7)
        return teacher_tests.DefinitionTests().prepare(args)

    def test_view_definition_is_explicit_and_full_remains_default(self):
        full, coarse = self.args(False), self.args(True)
        self.assertNotIn('teacher_view', full.model_args)
        self.assertEqual(coarse.model_args['teacher_view'], 'coarse_mask')
        self.assertEqual(coarse.model_args['teacher_temporal_group'], 3)
        self.assertEqual(coarse.model_args['teacher_mask_ratio'], .7)
        self.assertEqual(full.model_args['relation_bank_size'], coarse.model_args['relation_bank_size'])
        validate = teacher_tests.view_validation()
        validate(teacher_tests.VERSION)
        validate(teacher_tests.COARSE_VERSION, 'coarse_mask', 3, .7)
        for args in [(teacher_tests.VERSION, 'coarse_mask', 3, .7),
                     (teacher_tests.COARSE_VERSION, 'full', 1, 0.),
                     (teacher_tests.COARSE_VERSION, 'coarse_mask', 2, .7),
                     (teacher_tests.COARSE_VERSION, 'coarse_mask', 3, .8),
                     (teacher_tests.COARSE_VERSION, 'coarse_mask', True, .7)]:
            with self.subTest(args=args), self.assertRaises(ValueError): validate(*args)

    def test_full_and_coarse_cannot_resume_as_each_other(self):
        import main_pretrain_text_relation as base
        import main_pretrain_text_relation_teacher as entry
        full, coarse = self.args(False), self.args(True)
        base._validate_resume_definition(copy.deepcopy(coarse), coarse, model_name=entry.MODEL_NAME)
        with self.assertRaises(ValueError): base._validate_resume_definition(full, coarse, model_name=entry.MODEL_NAME)
        with self.assertRaises(ValueError): base._validate_resume_definition(coarse, full, model_name=entry.MODEL_NAME)
        coarse.resume = 'checkpoint-20.pth'
        checkpoint = dict(model={}, optimizer={}, scaler={}, epoch=20, args=copy.deepcopy(coarse))
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **kw: checkpoint)}):
            teacher_tests.DefinitionTests().prepare(coarse)
            checkpoint['args'] = full
            with self.assertRaises(ValueError): teacher_tests.DefinitionTests().prepare(coarse)


class NumpyTests(unittest.TestCase):
    def test_temporal_grouping_preserves_joint_identity_and_triples_coverage(self):
        tokens = array(np.arange(2 * 30 * 25 * 4, dtype=np.float32).reshape(2, 30, 25, 4))
        coarse = coarsen()(tokens, 3)
        self.assertEqual(coarse.shape, (2, 10, 25, 4))
        for index in range(10):
            np.testing.assert_array_equal(coarse[:, index],
                (tokens[:, 3*index] + tokens[:, 3*index+1] + tokens[:, 3*index+2]) / 3.)
        with self.assertRaises(ValueError): coarsen()(tokens[:, :29], 3)

    def test_teacher_retains_exactly_75_of_250_coarse_tokens_per_sample(self):
        tokens = array(np.arange(2 * 250 * 4, dtype=np.float32).reshape(2, 250, 4))
        visible, ids = sample()(tokens, .7)
        self.assertEqual(visible.shape, (2, 75, 4))
        for row in range(2):
            self.assertEqual(len(np.unique(ids[row])), 75)
            np.testing.assert_array_equal(visible[row], tokens[row, ids[row]])
        self.assertFalse(np.array_equal(ids[0], ids[1]))
        self.assertEqual(visible.shape[1] * 3, 225)

    def test_actual_coarse_forward_merges_positions_then_masks_before_blocks(self):
        tokens = array(np.arange(2 * 30 * 25 * 4, dtype=np.float32).reshape(2, 30, 25, 4))
        position = array(np.arange(25 * 4, dtype=np.float32).reshape(1, 1, 25, 4))
        temporal = array(np.arange(30 * 4, dtype=np.float32).reshape(1, 30, 1, 4))
        observed = []
        def block(x): observed.append(x.copy()); return x + 1.
        encoder = types.SimpleNamespace(joints_embed=lambda clean: tokens, pos_embed=position,
            temp_embed=temporal, temporal_group=3, mask_ratio=.7, blocks=[block, block], norm=lambda x: x)
        forward = teacher_tests.production('forward', dict(coarsen_temporal_tokens=coarsen(),
            sample_teacher_tokens=sample(), F=types.SimpleNamespace(normalize=base_tests.normalize)),
            owner='CoarseMaskedRelationKeyEncoder')
        key = forward(encoder, None)
        expected_grid = coarsen()(tokens + position + temporal, 3).reshape(2, 250, 4)
        selected, _ = sample()(expected_grid, .7)
        np.testing.assert_array_equal(observed[0], selected)
        self.assertEqual(observed[1].shape, (2, 75, 4))
        expected = base_tests.normalize((selected + 2.).mean(dim=1), eps=1.e-6)
        np.testing.assert_allclose(key, expected)


@unittest.skipUnless(base_tests.HAS_TORCH, 'PyTorch required for coarse-key EMA and gradient checks')
class TorchTests(unittest.TestCase):
    def model(self, coarse):
        import torch
        from model.transformer_macdiff_text_relation_teacher import Transformer
        options = dict(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1, num_heads=2,
            num_frames=120, num_joints=25, t_patch_size=4, one_person=True,
            text_input_dim=8, text_hidden_dim=16, text_context_length=7,
            relation_bank_size=8, lambda_loss_uni=.02, uncond_ratio=0., diff_prediction='noise')
        if coarse:
            options.update(relation_teacher_version=teacher_tests.COARSE_VERSION,
                teacher_view='coarse_mask', teacher_temporal_group=3, teacher_mask_ratio=.7)
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()): return Transformer(**options)

    def test_actual_query_and_coarse_teacher_both_use_75_tokens_and_gradients_are_finite(self):
        import torch
        model = self.model(True)
        source, kwargs = base_tests.RelationTorchTests().batch()
        source = torch.randn(2, 3, 120, 25, 2)
        kwargs['mask_ratio'] = .9
        sizes = []
        q = model.blocks[0].register_forward_pre_hook(lambda module, inputs: sizes.append(('q', inputs[0].shape[1])))
        k = model.ema_teacher.blocks[0].register_forward_pre_hook(lambda module, inputs: sizes.append(('k', inputs[0].shape[1])))
        result = model(source, source + .01, **kwargs)
        q.remove(); k.remove()
        self.assertEqual(sizes, [('q', 75), ('k', 75)])
        result[0].backward()
        for name, parameter in model.named_parameters():
            if name.startswith('ema_teacher.'):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.complete_relation_step([result[4]], successful=True)
        self.assertEqual(model.teacher_updates.item(), 1)
        torch.testing.assert_allclose(model.relation_skeleton_keys[:2], result[4]['skeleton'])

    def test_embedding_shapes_ema_updates_and_checkpoint_are_compatible(self):
        import torch
        torch.manual_seed(17)
        full = self.model(False)
        torch.manual_seed(17)
        coarse = self.model(True)
        self.assertEqual(list(full.state_dict()), list(coarse.state_dict()))
        for name, value in full.state_dict().items():
            torch.testing.assert_allclose(value, coarse.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            coarse.joints_embed.proj.weight.add_(.1)
        old = coarse.ema_teacher.joints_embed.proj.weight.detach().clone()
        coarse.ema_teacher.update_from(coarse, .999)
        torch.testing.assert_allclose(coarse.ema_teacher.joints_embed.proj.weight,
            old * .999 + coarse.joints_embed.proj.weight.detach() * .001)
        restored = self.model(True)
        restored.load_state_dict(coarse.state_dict())
        for name, value in coarse.state_dict().items():
            torch.testing.assert_allclose(value, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
