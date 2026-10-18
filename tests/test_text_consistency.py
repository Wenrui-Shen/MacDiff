"""Definition/resume checks plus real model/gradient checks when Torch is available."""
import argparse
import ast
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/transformer_macdiff_text_consistency.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def definition_validator():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'validate_consistency_definition']
    scope = {'math': math, 'TARGET_MODE': 'reconstruction_consistency'}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['validate_consistency_definition']


class DefinitionTests(unittest.TestCase):
    def validate(self, **overrides):
        options = dict(lambda_consistency=.003, lambda_text_to_skeleton=1.,
                       one_person=True, share_skeleton_decoder=False,
                       text_target_mode='reconstruction_consistency', text_target_norm='rms',
                       lambda_skeleton_to_text=0., lambda_text_uniformity=0.)
        options.update(overrides)
        definition_validator()(**options)

    def test_weight_validation_and_zero_consistency_ablation(self):
        self.validate(lambda_consistency=0.)
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            self.validate(lambda_consistency=None)
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(lambda_consistency=value)

    def test_both_ground_truth_objectives_and_no_share_are_required(self):
        for options in [dict(lambda_text_to_skeleton=0.), dict(one_person=False),
                        dict(share_skeleton_decoder=True)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(**options)

    def test_old_text_target_modes_and_removed_losses_are_rejected(self):
        for options in [dict(text_target_mode='sample_target_blend'),
                        dict(text_target_norm='none'), dict(lambda_skeleton_to_text=.1),
                        dict(lambda_text_uniformity=.02)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(**options)

    def test_resume_rejects_other_methods_and_changed_weights_or_depths(self):
        import main_pretrain_text_consistency as entry
        options = dict(decoder_depth=3, lambda_consistency=.003,
                       diff_noise_schedule=['inverse_cosine', 1])
        saved = argparse.Namespace(model=entry.MODEL_NAME, model_args=options)
        entry._validate_resume_definition(saved, dict(options))
        for changed in [dict(options, decoder_depth=5),
                        dict(options, lambda_consistency=.004)]:
            with self.assertRaisesRegex(ValueError, 'model_args'):
                entry._validate_resume_definition(saved, changed)
        with self.assertRaisesRegex(ValueError, 'old S->T/native'):
            entry._validate_resume_definition(
                dict(model='model.transformer_macdiff_text.Transformer'), options)

    def test_unspecified_weight_defaults_to_one(self):
        import main_pretrain_text_consistency as entry
        args = argparse.Namespace(model=entry.MODEL_NAME, enable_ose=False,
                                  text_cache='cache', model_args={},
                                  lambda_consistency=None, lambda_text_to_skeleton=None,
                                  lambda_skeleton_to_text=None)
        args.resume = ''
        module = types.SimpleNamespace(validate_consistency_definition=definition_validator())
        with mock.patch.object(entry.importlib, 'import_module', return_value=module):
            result = entry.prepare_args(args)
        self.assertEqual(result.model_args['lambda_consistency'], 1.)

    def test_entry_extends_the_actual_native_parser(self):
        import main_pretrain_text_consistency as entry
        tree = ast.parse((ROOT / 'main_pretrain.py').read_text(encoding='utf-8'))
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'get_args_parser']
        scope = {'argparse': argparse}
        exec(compile(ast.Module(body=body, type_ignores=[]), 'native_parser', 'exec'), scope)
        native = types.SimpleNamespace(get_args_parser=scope['get_args_parser'])
        with mock.patch.dict(sys.modules, {'main_pretrain': native}):
            args = entry.get_args_parser().parse_args(
                ['--lambda_consistency', '.003', '--batch_size', '32', '--min_lr', '.0005'])
        self.assertEqual(args.model, entry.MODEL_NAME)
        self.assertEqual(args.lambda_consistency, .003)
        self.assertEqual(args.min_lr, .0005)

    def test_cli_weight_is_copied_into_saved_model_definition(self):
        import main_pretrain_text_consistency as entry
        args = argparse.Namespace(model=entry.MODEL_NAME, enable_ose=False,
                                  text_cache='cache', model_args={}, resume='',
                                  lambda_consistency=.004, lambda_text_to_skeleton=1.,
                                  lambda_skeleton_to_text=None)
        module = types.SimpleNamespace(validate_consistency_definition=definition_validator())
        with mock.patch.object(entry.importlib, 'import_module', return_value=module):
            result = entry.prepare_args(args)
        self.assertEqual(result.model_args['lambda_consistency'], .004)
        self.assertEqual(result.model_args['lambda_text_to_skeleton'], 1.)

    def test_existing_sentence_cache_training_guard_accepts_this_mode(self):
        source = ROOT / 'util/structured_text_cache.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'validate_training_definition']
        scope = {'PROTOCOL': 'macdiff_clip_sentence_cache_v3'}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), scope)
        scope['validate_training_definition'](
            {'protocol': scope['PROTOCOL']},
            {'text_target_mode': 'reconstruction_consistency', 'text_target_norm': 'rms'},
            types.SimpleNamespace(flip=False))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for reconstruction/autograd checks')
class ConsistencyTorchTests(unittest.TestCase):
    def assert_close(self, actual, expected, rtol=1e-5, atol=1e-8):
        # torch.testing.assert_close is not available in the historical Torch
        # 1.8 server environment; compare tensors without that newer API.
        import torch
        self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol))

    @staticmethod
    def model(**overrides):
        import torch
        from model.transformer_macdiff_text_consistency import Transformer
        torch.set_num_threads(1)
        options = dict(dim_feat=8, dim_t_embed=64, depth=1, decoder_depth=1,
                       num_heads=2, num_frames=8, num_joints=5, t_patch_size=4,
                       diff_prediction='noise', diff_steps=10,
                       diff_noise_schedule=['inverse_cosine', 1],
                       text_input_dim=6, text_hidden_dim=12, text_context_length=7,
                       lambda_loss_uni=.02, uncond_ratio=0., lambda_consistency=.003)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**options)

    @staticmethod
    def inputs(batch=2):
        import torch
        source = torch.randn(batch, 3, 8, 5, 2)
        valid = torch.ones(batch, 6, dtype=torch.bool)
        valid[0, -2:] = False
        data = dict(text_features=torch.randn(batch, 6, requires_grad=True),
                    text_tokens=torch.randn(batch, 6, 6, requires_grad=True),
                    text_token_mask=valid,
                    text_person_ids=torch.zeros(batch, 6, dtype=torch.long),
                    text_positions=torch.arange(1, 7).expand(batch, -1))
        return source, source + .005 * torch.randn_like(source), data

    def test_independent_full_schedule_noise_and_timesteps(self):
        import torch
        model = self.model()
        source, aug, data = self.inputs()
        times = [torch.tensor([0, 9]), torch.tensor([9, 0])]
        sampled, corruptions = [], []
        original = model.diffusion.q_sample

        def sample(batch, device):
            t = times[len(sampled)].to(device)
            sampled.append(t)
            return t, torch.ones(batch, device=device)

        def corrupt(clean, t, noise=None):
            corruptions.append((clean.clone(), t.clone(), noise.clone()))
            return original(clean, t, noise=noise)

        model.schedule_sampler.sample = sample
        model.diffusion.q_sample = corrupt
        loss, _, _, metrics = model(source, aug, mask_ratio=.5, **data)
        self.assertEqual(len(corruptions), 2)
        self.assert_close(corruptions[0][0], corruptions[1][0])
        self.assertFalse(torch.equal(corruptions[0][1], corruptions[1][1]))
        self.assertFalse(torch.equal(corruptions[0][2], corruptions[1][2]))
        self.assertEqual(metrics['consistency_fraction'].item(), 1.)
        self.assertTrue(torch.isfinite(loss))

    def test_consistency_alone_backpropagates_to_both_branches_and_encoder(self):
        import torch
        import model.transformer_macdiff_text_consistency as module
        model = self.model()
        source, aug, data = self.inputs()
        captured = []
        original = module.masked_reconstruction_consistency

        def record(*args):
            result = original(*args)
            captured.append(result[0])
            return result

        with mock.patch.object(module, 'masked_reconstruction_consistency', side_effect=record):
            model(source, aug, mask_ratio=.5, **data)
        captured[0].backward()
        for parameter in [model.blocks[0].attn.qkv.weight, model.decoder_pred.weight,
                          model.text_skeleton_decoder.decoder_pred.weight,
                          model.text_remap.body[-1].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        self.assertIsNone(data['text_features'].grad)
        self.assertIsNone(data['text_tokens'].grad)

    def test_t2s_supervision_does_not_directly_train_native_encoder(self):
        model = self.model()
        source, aug, data = self.inputs()
        captured = []
        original = model.forward_loss

        def record(*args):
            result = original(*args)
            captured.append(result)
            return result

        model.forward_loss = record
        model(source, aug, mask_ratio=.5, **data)
        captured[1].backward()
        self.assertIsNone(model.blocks[0].attn.qkv.weight.grad)
        self.assertIsNone(model.decoder_pred.weight.grad)
        self.assertGreater(model.text_remap.body[-1].weight.grad.abs().sum().item(), 0.)

    def test_masking_drop_condition_and_no_eligible_rows(self):
        import torch
        from model.transformer_macdiff_text_consistency import masked_reconstruction_consistency
        model = self.model()
        a = torch.randn(2, 3, 12, requires_grad=True)
        b = torch.randn(2, 3, 12, requires_grad=True)
        x, y = torch.randn_like(a), torch.randn_like(b)
        times_a, times_b = torch.tensor([0, 9]), torch.tensor([9, 0])
        mask = torch.tensor([[1., 0., 1.], [1., 1., 1.]])
        loss, fraction = masked_reconstruction_consistency(
            model.diffusion, x, a, times_a, y, b, times_b, mask, torch.tensor([True, False]))
        loss.backward()
        self.assertEqual(fraction.item(), .5)
        self.assertEqual(a.grad[1].abs().sum().item(), 0.)
        self.assertEqual(b.grad[1].abs().sum().item(), 0.)
        self.assertEqual(a.grad[0, 1].abs().sum().item(), 0.)
        self.assertGreater(a.grad[0, 0].abs().sum().item(), 0.)
        a.grad = b.grad = None
        zero, fraction = masked_reconstruction_consistency(
            model.diffusion, x, a, times_a, y, b, times_b, mask, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertEqual(a.grad.abs().sum().item(), 0.)
        self.assertEqual(b.grad.abs().sum().item(), 0.)

    def test_raw_x0_mse_matches_the_independent_error_formula_without_clipping(self):
        import torch
        from model.transformer_macdiff_text_consistency import masked_reconstruction_consistency
        model = self.model()
        clean = torch.randn(2, 8, 5, 3)
        ta, tb = torch.tensor([0, 9]), torch.tensor([9, 0])
        ea, eb = torch.randn_like(clean), torch.randn_like(clean)
        xa = model.patchify(model.diffusion.q_sample(clean, ta, noise=ea))
        xb = model.patchify(model.diffusion.q_sample(clean, tb, noise=eb))
        pa = (model.patchify(ea) + .2).requires_grad_()
        pb = (model.patchify(eb) - .4).requires_grad_()
        mask = torch.ones(2, 10)
        loss, _ = masked_reconstruction_consistency(
            model.diffusion, xa, pa, ta, xb, pb, tb, mask, torch.ones(2, dtype=torch.bool))
        k = torch.tensor(model.diffusion.sqrt_recipm1_alphas_cumprod, dtype=torch.float32)
        expected = (k[ta, None, None] * .2 + k[tb, None, None] * .4).square().mean()
        self.assert_close(loss, expected, rtol=2e-5, atol=1e-5)
        self.assertGreater(loss.item(), 1.)

    def test_native_decoder_matches_native_forward_and_only_person0_is_used(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        model = self.model(uncond_ratio=.1)
        x, z = torch.randn(2, 8, 5, 3), torch.randn(2, 10, 8)
        t = torch.tensor([2, 7])
        torch.manual_seed(81)
        actual, _ = model._native_noise_prediction(x, z, t)
        torch.manual_seed(81)
        expected = Native.forward_decoder(model, x, z=z, t=t)
        self.assert_close(actual, expected)
        source, aug, data = self.inputs()
        torch.manual_seed(12)
        first = model(source, aug, mask_ratio=.5, **data)[0]
        source[..., 1] = 900.
        aug[..., 1] = -700.
        torch.manual_seed(12)
        second = model(source, aug, mask_ratio=.5, **data)[0]
        self.assert_close(first, second)

    def test_no_text_decoder_or_target_bank_and_all_parameters_participate(self):
        import torch
        model = self.model()
        self.assertFalse(hasattr(model, 'text_noise_decoder'))
        self.assertFalse(hasattr(model, 'text_target_remap'))
        self.assertNotEqual(model.text_target_mode, 'sample_target_blend')
        source, aug, data = self.inputs()
        loss, _, _, metrics = model(source, aug, mask_ratio=.5, **data)
        self.assert_close(loss.detach(), metrics['loss_diff'] + .02 * metrics['loss_uniformity']
                          + metrics['loss_text_to_skeleton'] + .003 * metrics['loss_consistency'])
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        tracked = [model.blocks[0].attn.qkv.weight, model.decoder_pred.weight,
                   model.text_remap.body[-1].weight,
                   model.text_skeleton_decoder.decoder_pred.weight]
        before = [parameter.detach().clone() for parameter in tracked]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        for previous, parameter in zip(before, tracked):
            self.assertFalse(torch.equal(previous, parameter))
            self.assertTrue(torch.isfinite(parameter).all())

    def test_padding_is_ignored_and_zero_consistency_is_an_actual_ablation(self):
        import torch
        model = self.model(lambda_consistency=0.)
        source, aug, data = self.inputs()
        torch.manual_seed(21)
        expected = model(source, aug, mask_ratio=.5, **data)[0]
        changed = dict(data)
        changed['text_tokens'] = data['text_tokens'].masked_fill(~data['text_token_mask'][..., None], float('nan'))
        changed['text_person_ids'] = data['text_person_ids'].masked_fill(~data['text_token_mask'], 999)
        changed['text_positions'] = data['text_positions'].masked_fill(~data['text_token_mask'], -999)
        torch.manual_seed(21)
        loss, _, _, metrics = model(source, aug, mask_ratio=.5, **changed)
        self.assert_close(loss, expected)
        self.assertEqual(metrics['loss_consistency'].item(), 0.)

    def test_state_dict_roundtrip_and_existing_linear_probe_encoder_loading(self):
        import torch
        from model.transformer_downstream import Transformer as Downstream
        model = self.model()
        restored = self.model()
        restored.load_state_dict(model.state_dict(), strict=True)
        with contextlib.redirect_stdout(io.StringIO()):
            probe = Downstream(dim_feat=8, depth=1, num_heads=2, num_frames=8,
                               num_joints=5, t_patch_size=4, num_classes=60, protocol='linprobe2')
        # Same head-key removal performed by main_linprobe.py.
        result = probe.load_state_dict(model.state_dict(), strict=False)
        self.assertTrue(all(key.startswith('head.') for key in result.missing_keys))
        self.assert_close(probe.blocks[0].attn.qkv.weight, model.blocks[0].attn.qkv.weight)


if __name__ == '__main__':
    unittest.main()
