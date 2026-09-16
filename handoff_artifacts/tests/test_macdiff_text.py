"""Real CPU/autograd checks; skip only when PyTorch is unavailable."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for diffusion/gradient checks')
class TextStage1Tests(unittest.TestCase):
    @staticmethod
    def model(**overrides):
        import torch
        from model.transformer_macdiff_text import Transformer
        torch.set_num_threads(1)
        options = dict(dim_feat=8, dim_t_embed=64, depth=1, decoder_depth=1,
                       num_heads=2, num_frames=8, num_joints=25, t_patch_size=4,
                       diff_prediction='noise', diff_steps=10,
                       diff_noise_schedule=['inverse_cosine', 1],
                       text_input_dim=6, text_hidden_dim=12,
                       text_decoder_hidden_dim=12, text_decoder_depth=2,
                       lambda_loss_uni=.02, uncond_ratio=0.)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return Transformer(**options)

    def test_all_three_losses_backward_and_one_encoder_forward(self):
        import torch
        model = self.model()
        calls = []
        handle = model.norm.register_forward_hook(lambda *args: calls.append(1))
        source = torch.randn(3, 3, 8, 25, 2)
        text = torch.randn(3, 6)
        loss, pred, mask, metrics = model(source, source + .01, text_features=text, mask_ratio=.5)
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(pred.shape, (3, 50, 12))
        self.assertEqual(mask.shape, (3, 50))
        expected = (metrics['loss_diff'] + .02 * metrics['loss_uniformity']
                    + metrics['loss_text_to_skeleton'] + metrics['loss_skeleton_to_text'])
        torch.testing.assert_close(loss.detach(), expected)
        self.assertAlmostEqual(metrics['text_energy'].item(), 1., places=3)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for prefix in ('blocks.', 'text_remap.', 'text_skeleton_decoder.', 'text_noise_decoder.'):
            self.assertGreater(sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                                   if n.startswith(prefix) and p.grad is not None), 0, prefix)

    def test_reverse_loss_detaches_before_corruption_but_trains_encoder(self):
        import torch
        model = self.model()
        x = torch.randn(3, 8, 25, 3)
        _, pooled, _, _ = model.forward_encoder(x, x_orig=x, mask_ratio=.5, motion_aware_tau=-1)
        r = model.text_remap(torch.randn(3, 6))
        recorded = []
        q_sample = model.diffusion.q_sample

        def capture(target, t, noise=None):
            recorded.append(target.requires_grad)
            return q_sample(target, t, noise=noise)

        model.diffusion.q_sample = capture
        loss = model.skeleton_to_text_loss(r, pooled.squeeze(1))
        loss.backward()
        self.assertEqual(recorded, [False])
        self.assertTrue(all(p.grad is None for p in model.text_remap.parameters()))
        self.assertTrue(all(p.grad is None for p in model.text_skeleton_decoder.parameters()))
        self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.text_noise_decoder.input.weight.grad.abs().sum().item(), 0)

    def test_forward_text_loss_trains_remap_without_encoder_shortcut(self):
        import torch
        model = self.model()
        x = torch.randn(3, 8, 25, 3)
        noise = torch.randn_like(x)
        t = torch.tensor([2, 4, 7])
        noisy = model.diffusion.q_sample(x, t, noise=noise)
        r = model.text_remap(torch.randn(3, 6))
        loss = model.text_to_skeleton_loss(noisy, noise, t, torch.ones(3, 50),
                                          r, torch.ones(3, dtype=torch.bool), people=1)
        loss.backward()
        self.assertGreater(model.text_remap[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.blocks.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder_blocks.parameters()))

    def test_global_pool_excludes_empty_second_person(self):
        import torch
        model = self.model(one_person=False)
        source = torch.randn(2, 3, 8, 25, 2)
        source[0, ..., 1] = 0
        captured = {}

        def capture_tokens(module, inputs, output):
            captured['pooled'] = output.mean(dim=1).detach()

        def capture_condition(module, inputs):
            captured['condition'] = inputs[2].detach()

        first = model.norm.register_forward_hook(capture_tokens)
        second = model.text_noise_decoder.register_forward_pre_hook(capture_condition)
        loss, pred, _, _ = model(source, source.clone(), text_features=torch.randn(2, 6), mask_ratio=.5)
        first.remove()
        second.remove()
        self.assertEqual(pred.shape[0], 4)
        pooled = captured['pooled'].reshape(2, 2, 8)
        torch.testing.assert_close(captured['condition'][0], pooled[0, 0])
        torch.testing.assert_close(captured['condition'][1], pooled[1].mean(dim=0))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_zero_weights_reproduce_native_and_keep_encoder_transfer(self):
        import torch
        from model.transformer_macdiff import Transformer as Native
        from model.transformer_downstream import Transformer as Downstream
        from compare_stage1_readouts import load_encoder
        model = self.model(lambda_text_to_skeleton=0., lambda_skeleton_to_text=0.)
        with contextlib.redirect_stdout(io.StringIO()):
            native = Native(dim_feat=8, dim_t_embed=64, depth=1, decoder_depth=1,
                num_heads=2, num_frames=8, num_joints=25, t_patch_size=4,
                diff_prediction='noise', diff_steps=10, diff_noise_schedule=['inverse_cosine', 1],
                lambda_loss_uni=.02, uncond_ratio=0.)
        native.load_state_dict({k: model.state_dict()[k] for k in native.state_dict()})
        x, text = torch.randn(2, 3, 8, 25, 2), torch.randn(2, 6)
        torch.manual_seed(23)
        np.random.seed(23)
        baseline = native(x.clone(), x.clone(), mask_ratio=.5)
        torch.manual_seed(23)
        np.random.seed(23)
        actual = model(x.clone(), x.clone(), text_features=text, mask_ratio=.5)
        for a, b in zip(baseline, actual[:3]):
            torch.testing.assert_close(a, b)
        downstream = Downstream(dim_feat=8, depth=1, num_heads=2,
                               num_frames=8, num_joints=25, t_patch_size=4, protocol='linprobe2')
        downstream.head.fc = torch.nn.Identity()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save({'model': model.state_dict()}, path)
            result = load_encoder(downstream, str(path))
            self.assertGreater(result['encoder_tensors'], 0)

    def test_disabled_branch_parameters_are_frozen_and_active_parameters_have_gradients(self):
        import torch
        for t2s, s2t in ((0., 1.), (1., 0.)):
            model = self.model(lambda_text_to_skeleton=t2s, lambda_skeleton_to_text=s2t)
            x = torch.randn(2, 3, 8, 25, 2)
            loss = model(x, x.clone(), text_features=torch.randn(2, 6), mask_ratio=.5)[0]
            loss.backward()
            for name, p in model.named_parameters():
                self.assertEqual(p.grad is not None, p.requires_grad, name)

    def test_training_engine_looks_up_raw_indices_and_updates_weights(self):
        import torch
        from engine_pretrain import train_one_epoch
        model = self.model()
        ids = torch.tensor([3, 0, 4, 1, 2])
        x = torch.randn(5, 3, 8, 25, 2)
        data = torch.utils.data.TensorDataset(x, x.clone(), torch.zeros(5), ids)
        loader = torch.utils.data.DataLoader(data, batch_size=2)
        text = torch.randn(5, 6)
        observed = []
        hook = model.text_remap[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0].detach().clone()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.joints_embed.proj.weight.detach().clone()

        class Scaler:
            def __init__(self):
                self.steps = 0

            def __call__(self, loss, optimizer, parameters, update_grad):
                loss.backward()
                if update_grad:
                    optimizer.step()
                    self.steps += 1

        scaler = Scaler()
        args = types.SimpleNamespace(enable_ose=False, epochs=2, warmup_epochs=0,
            min_lr_epochs=0, lr=.001, min_lr=.0001, accum_iter=2, max_train_steps=0,
            mask_ratio=.5, motion_stride=1, motion_aware_tau=-1, enable_amp=False)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(model, loader, optimizer, torch.device('cpu'),
                                    0, scaler, args=args, text_features=text)
        hook.remove()
        torch.testing.assert_close(torch.cat(observed), text[ids])
        self.assertEqual(scaler.steps, 2)  # includes the final incomplete accumulation window
        self.assertFalse(torch.equal(before, model.joints_embed.proj.weight))
        self.assertTrue(np.isfinite(stats['loss_skeleton_to_text']))

    def test_resume_checks_cache_identity(self):
        import torch
        from util.misc import load_model
        model = self.model()
        saved_args = types.SimpleNamespace(text_cache_identity={'hash': 'old'},
                                          text_training_weights=(1., 1.))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pth'
            torch.save({'model': model.state_dict(), 'args': saved_args}, checkpoint)
            args = types.SimpleNamespace(resume=str(checkpoint), text_cache='cache',
                text_cache_identity={'hash': 'new'}, text_training_weights=(1., 1.))
            with self.assertRaisesRegex(ValueError, 'text_cache_identity'):
                load_model(args, model, None, None)


if __name__ == '__main__':
    unittest.main()
