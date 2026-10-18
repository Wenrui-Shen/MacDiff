"""Checkpoint-contract tests plus real CPU/autograd tests when Torch is installed."""
import argparse
import ast
from collections.abc import Mapping
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model' / 'transformer_macdiff_decoder_distill.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def checkpoint_helpers():
    # Execute the actual dependency-free checkpoint validators. This does not
    # stand in for tensor loading/autograd, which is tested separately below.
    names = {'_is_decoder_body_key', '_decoder_body_state', '_validate_decoder_state',
             '_argument', '_canonical', '_validate_source_definition'}
    parsed = ast.parse(SOURCE.read_text(encoding='utf-8'))
    body = [node for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    environment = dict(Mapping=Mapping,
                       MODEL_NAME='model.transformer_macdiff_decoder_distill.Transformer',
                       NATIVE_MODEL_NAME='model.transformer_macdiff.Transformer')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), environment)
    return environment


class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.helpers = checkpoint_helpers()
        self.expected = dict(diff_steps=1000, diff_noise_schedule=['inverse_cosine', 1], num_heads=8,
                             act_layer='GELU',
                             input_mean=[-.0058, -.1333, -.0246],
                             input_var=[.0206, .0805, .0218], self_shift=False)

    def test_allowlist_never_loads_encoder_or_output_head_and_handles_ddp(self):
        state = {'module.decoder_embed.proj.weight': 'body',
                 'module.decoder_blocks.0.norm1.fc_z.weight': 'condition',
                 'module.decoder_norm.weight': 'norm',
                 'module.decoder_pos_embed': 'position',
                 'module.decoder_temp_embed': 'time',
                 'module.decoder_pred.weight': 'teacher_head',
                 'module.joints_embed.proj.weight': 'teacher_encoder',
                 'text_decoder.weight': 'text'}
        selected = self.helpers['_decoder_body_state'](state)
        self.assertEqual(set(selected), {'decoder_embed.proj.weight',
                                        'decoder_blocks.0.norm1.fc_z.weight',
                                        'decoder_norm.weight', 'decoder_pos_embed',
                                        'decoder_temp_embed'})

    def test_duplicate_prefixed_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.helpers['_decoder_body_state'](
                {'decoder_norm.weight': 1, 'module.decoder_norm.weight': 2})

    def test_decoder_depth_and_shape_must_match(self):
        tensor = types.SimpleNamespace(shape=(8, 8))
        validate = self.helpers['_validate_decoder_state']
        with self.assertRaisesRegex(ValueError, 'decoder_depth'):
            validate({'decoder_blocks.0.weight': tensor},
                     {'decoder_blocks.0.weight': tensor, 'decoder_blocks.1.weight': tensor})
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            validate({'decoder_norm.weight': tensor},
                     {'decoder_norm.weight': types.SimpleNamespace(shape=(12, 8))})

    def test_native_namespace_metadata_and_sequence_types_are_supported(self):
        options = dict(self.expected, diff_prediction='noise')
        options['diff_noise_schedule'] = ('inverse_cosine', 1)
        saved = argparse.Namespace(model='model.transformer_macdiff.Transformer', model_args=options)
        self.assertTrue(self.helpers['_validate_source_definition'](saved, self.expected))

    def test_normalization_and_diffusion_mismatch_are_rejected(self):
        for name, value in [('input_var', [1., 1., 1.]), ('diff_steps', 500),
                            ('diff_noise_schedule', 'cosine'), ('num_heads', 4), ('act_layer', 'nn.ReLU')]:
            options = dict(self.expected, diff_prediction='noise')
            options[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.helpers['_validate_source_definition']({'model_args': options}, self.expected)

    def test_text_and_joint_prediction_sources_are_rejected(self):
        for saved in [dict(model='model.transformer_macdiff_text.Transformer'),
                      dict(model_args=dict(self.expected, diff_prediction='joint'))]:
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                self.helpers['_validate_source_definition'](saved, self.expected)

    def test_missing_metadata_is_reported_as_unavailable(self):
        self.assertFalse(self.helpers['_validate_source_definition'](None, self.expected))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for tensor loading/autograd checks')
class DecoderDistillTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'teacher.pth'
        self.options = dict(dim_feat=8, dim_t_embed=64, depth=2, decoder_depth=1,
                            num_heads=2, num_frames=8, num_joints=5, t_patch_size=4,
                            diff_steps=10, diff_noise_schedule=['inverse_cosine', 1],
                            input_mean=[-.0058, -.1333, -.0246],
                            input_var=[.0206, .0805, .0218],
                            lambda_loss_uni=.02, one_person=True,
                            uncond_ratio=0., layer_mask_ratio=0., drop_rate=0.)
        from model.transformer_macdiff import Transformer
        torch.manual_seed(12)
        with contextlib.redirect_stdout(io.StringIO()):
            self.teacher = Transformer(diff_prediction='noise', **self.options)
        # Sentinel teacher encoder/head make accidental teacher loading visible.
        with torch.no_grad():
            self.teacher.joints_embed.proj.weight.fill_(3.)
            self.teacher.decoder_pred.weight.fill_(4.)
        self.save_teacher(self.teacher)

    def save_teacher(self, model, options=None):
        import torch
        options = dict(options or self.options, diff_prediction='noise')
        torch.save(dict(model=model.state_dict(), args=argparse.Namespace(
            model='model.transformer_macdiff.Transformer', model_args=options)), str(self.path))

    def student(self, **overrides):
        from model.transformer_macdiff_decoder_distill import Transformer
        options = dict(self.options, decoder_checkpoint=str(self.path),
                       distill_t_min=4, distill_t_max=6)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**options)

    def test_only_body_is_loaded_student_and_head_remain_fresh(self):
        import torch
        from model.transformer_macdiff import Transformer
        from model.transformer_macdiff_decoder_distill import _is_decoder_body_key
        torch.manual_seed(25)
        with contextlib.redirect_stdout(io.StringIO()):
            fresh = Transformer(diff_prediction='noise2joint', **self.options)
        torch.manual_seed(25)
        student = self.student()
        self.assertEqual(set(student.state_dict()), set(fresh.state_dict()))
        for key, value in student.state_dict().items():
            expected = self.teacher.state_dict()[key] if _is_decoder_body_key(key) else fresh.state_dict()[key]
            self.assertTrue(torch.equal(value, expected), key)
        for key, parameter in student.named_parameters():
            self.assertEqual(parameter.requires_grad, not _is_decoder_body_key(key), key)

    def test_reconstruction_gradient_reaches_encoder_and_head_body_stays_fixed(self):
        import torch
        from model.transformer_macdiff_decoder_distill import _is_decoder_body_key
        student = self.student(lambda_loss_uni=0.)  # Isolate reconstruction gradient.
        student.train()
        before = {name: value.detach().clone() for name, value in student.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
        clean = torch.randn(2, 3, 8, 5, 2)
        aug = clean.clone() + .005 * torch.randn_like(clean)
        with contextlib.redirect_stdout(io.StringIO()):
            loss, predicted_x0, mask = student(clean.clone(), aug, mask_ratio=.8)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(predicted_x0.shape), (2, 10, 12))
        self.assertEqual(mask.sum().item(), 16)
        loss.backward()
        for name, parameter in student.named_parameters():
            if _is_decoder_body_key(name):
                self.assertIsNone(parameter.grad, name)
            elif parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in [student.joints_embed.proj.weight, student.blocks[0].attn.qkv.weight,
                          student.decoder_pred.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        optimizer.step()
        for name, parameter in student.named_parameters():
            if _is_decoder_body_key(name):
                self.assertTrue(torch.equal(before[name], parameter), name)
        self.assertFalse(torch.equal(before['decoder_pred.weight'], student.decoder_pred.weight))
        self.assertFalse(torch.equal(before['joints_embed.proj.weight'], student.joints_embed.proj.weight))

    def test_frozen_body_stays_eval_after_epoch_train_calls(self):
        student = self.student(drop_rate=.2)
        for _ in range(2):
            student.eval()
            student.train(True)
            self.assertTrue(student.blocks.training)
            self.assertTrue(student.decoder_pred.training)
            for module in [student.decoder_embed, student.decoder_blocks, student.decoder_norm]:
                self.assertTrue(all(not child.training for child in module.modules()))

    def test_timestep_sampling_includes_both_endpoints(self):
        student = self.student()
        times, _ = student.schedule_sampler.sample(4096, 'cpu')
        self.assertEqual(set(times.tolist()), {4, 5, 6})
        fixed = self.student(distill_t_min=5, distill_t_max=5)
        times, _ = fixed.schedule_sampler.sample(20, 'cpu')
        self.assertEqual(set(times.tolist()), {5})

    def test_noise_to_joint_loss_matches_analytic_masked_mse(self):
        import torch
        student = self.student()
        clean = torch.randn(2, 8, 5, 3)
        noise = torch.randn_like(clean)
        t = torch.tensor([4, 6])
        noisy = student.diffusion.q_sample(clean, t, noise=noise)
        predicted_noise = student.patchify(noise) + .1 * torch.randn(2, 10, 12)
        predicted_x0 = student.diffusion._predict_xstart_from_eps(student.patchify(noisy), t, predicted_noise)
        mask = torch.tensor([[0., 1.] * 5, [1., 0.] * 5])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = student.forward_loss(clean, predicted_x0, mask, t)
        per_patch = (predicted_noise - student.patchify(noise)).square().mean(-1)
        alpha = torch.tensor(student.diffusion.alphas_cumprod[t.numpy()], dtype=torch.float32)
        expected = (((per_patch * mask).sum(-1) / mask.sum(-1)) * (1 - alpha) / alpha).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_wrong_decoder_depth_and_normalization_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, 'decoder_depth'):
            self.student(decoder_depth=2)
        with self.assertRaisesRegex(ValueError, 'input_var'):
            self.student(input_var=[1., 1., 1.])
        with self.assertRaisesRegex(ValueError, 'num_heads'):
            self.student(num_heads=1)

    def test_existing_linear_probe_loads_student_encoder(self):
        from model.transformer_downstream import Transformer
        student = self.student()
        with contextlib.redirect_stdout(io.StringIO()):
            probe = Transformer(dim_feat=8, depth=2, num_heads=2, num_frames=8,
                                num_joints=5, t_patch_size=4, num_classes=3,
                                protocol='linprobe2')
        result = probe.load_state_dict(student.state_dict(), strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(key.startswith('head.') for key in result.missing_keys))

    def test_resume_uses_saved_body_without_original_teacher_file(self):
        import torch
        import main_pretrain_decoder_distill as entry
        from model.transformer_macdiff_decoder_distill import MODEL_NAME
        student = self.student()
        options = dict(self.options, diff_prediction='noise2joint',
                       distill_t_min=4, distill_t_max=6, decoder_checkpoint=str(self.path))
        saved_args = argparse.Namespace(model=MODEL_NAME, model_args=options,
                                        decoder_source_checkpoint=str(self.path))
        resume = Path(self.directory.name) / 'student.pth'
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad])
        torch.save(dict(model=student.state_dict(), args=saved_args,
                        optimizer=optimizer.state_dict(), scaler={}, epoch=3), str(resume))
        self.path.unlink()
        args = argparse.Namespace(model=MODEL_NAME, model_args=options,
                                  enable_ose=False, text_cache='', decoder_checkpoint=None,
                                  resume=str(resume))
        prepared = entry.prepare_args(args)
        self.assertEqual(prepared.model_args['decoder_checkpoint'], str(resume))
        restored = self.student(**prepared.model_args)
        restored.load_state_dict(student.state_dict())
        for name, value in student.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
        wrong = dict(options, distill_t_max=7)
        with self.assertRaisesRegex(ValueError, 'distill_t_max'):
            entry._validate_resume_definition(saved_args, wrong)

    def test_invalid_masks_and_protocols_fail_explicitly(self):
        import torch
        student = self.student()
        clean = torch.randn(2, 3, 8, 5, 2)
        for mask_ratio in (0., 1., .999):
            with self.subTest(mask_ratio=mask_ratio), self.assertRaises(ValueError):
                student(clean, clean.clone(), mask_ratio=mask_ratio)
        with self.assertRaises(ValueError):
            student(clean, clean.clone(), enable_ose=True)
        with self.assertRaises(ValueError):
            self.student(uncond_ratio=.1)


if __name__ == '__main__':
    unittest.main()
