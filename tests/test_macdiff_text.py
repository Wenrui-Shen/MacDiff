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
    def tokens(batch, dim=6):
        import torch
        valid = torch.ones(batch, 6, dtype=torch.bool)
        valid[0, -2:] = False
        return dict(text_tokens=torch.randn(batch, 6, dim), text_token_mask=valid,
                    text_person_ids=torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(batch, -1),
                    text_positions=torch.tensor([[1, 2, 3, 1, 2, 3]]).expand(batch, -1))

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
        loss, pred, mask, metrics = model(source, source + .01, text_features=text,
                                          mask_ratio=.5, **self.tokens(3))
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
        latent, pooled, _, _ = model.forward_encoder(x, x_orig=x, mask_ratio=.5, motion_aware_tau=-1)
        r = model.text_remap(torch.randn(3, 6))
        data = self.tokens(3)
        local = model.remap_tokens(data["text_tokens"], data["text_token_mask"],
                                   data["text_person_ids"], data["text_positions"])
        memory = torch.cat([r[:, None], local], dim=1)
        valid = torch.cat([torch.ones(3, 1, dtype=torch.bool), data["text_token_mask"]], dim=1)
        h = torch.cat([pooled, latent], dim=1)
        hmask = torch.ones(h.shape[:2], dtype=torch.bool)
        recorded = []
        q_sample = model.diffusion.q_sample

        def capture(target, t, noise=None):
            recorded.append(target.requires_grad)
            return q_sample(target, t, noise=noise)

        model.diffusion.q_sample = capture
        loss = model.skeleton_to_text_loss(memory, h, valid, hmask, data["text_person_ids"], data["text_positions"])
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
        tokens = self.tokens(3)
        token_condition = model.remap_tokens(tokens['text_tokens'], tokens['text_token_mask'],
                                             tokens['text_person_ids'], tokens['text_positions'])
        loss = model.text_to_skeleton_loss(noisy, noise, t, torch.ones(3, 50),
                                          r, torch.ones(3, dtype=torch.bool), people=1,
                                          token_condition=token_condition,
                                          token_mask=tokens['text_token_mask'])
        loss.backward()
        self.assertGreater(model.text_remap[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.blocks.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder_blocks.parameters()))

    def test_reverse_conditions_are_per_person_and_exclude_empty_person(self):
        import torch
        model = self.model(one_person=False)
        source = torch.randn(2, 3, 8, 25, 2)
        source[0, ..., 1] = 0
        captured = {}

        def capture_tokens(module, inputs, output):
            captured['pooled'] = output.mean(dim=1).detach()

        def capture_condition(module, inputs):
            captured['condition'] = inputs[2].detach()
            captured['valid'] = inputs[3].detach()

        first = model.norm.register_forward_hook(capture_tokens)
        second = model.text_noise_decoder.register_forward_pre_hook(capture_condition)
        loss, pred, _, _ = model(source, source.clone(), text_features=torch.randn(4, 6),
                                mask_ratio=.5, **self.tokens(4))
        first.remove()
        second.remove()
        self.assertEqual(pred.shape[0], 4)
        pooled = captured['pooled'].reshape(2, 2, 8)
        torch.testing.assert_close(captured['condition'][0, 0], pooled[0, 0])
        torch.testing.assert_close(captured['condition'][1, 0], pooled[1, 0])
        self.assertEqual(captured["valid"][0].sum().item(), 26)
        self.assertEqual(captured["valid"][1].sum().item(), 26)
        self.assertEqual(captured["condition"].shape, (3, 26, 8))
        torch.testing.assert_close(captured["condition"][2, 0], pooled[1, 1])
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
            loss = model(x, x.clone(), text_features=torch.randn(2, 6),
                         mask_ratio=.5, **self.tokens(2))[0]
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
        def record_global(module, inputs):
            if inputs[0].ndim == 2:
                observed.append(inputs[0].detach().clone())

        hook = model.text_remap[0].register_forward_pre_hook(record_global)
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
        from util.person_text_cache import PersonTokenFeatureCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.zeros((5, 2, 7, 6), dtype=np.float16)
            masks = np.zeros((5, 2, 7), dtype=bool)
            features[:, 0, 1:4] = np.random.randn(5, 3, 6)
            masks[:, 0, 1:4] = True
            np.save(root / 'token_features.npy', features)
            np.save(root / 'token_mask.npy', masks)
            np.save(root / "person_features.npy", np.stack([text.numpy(), np.zeros_like(text.numpy())], axis=1))
            storage = PersonTokenFeatureCache(root, {})
            with contextlib.redirect_stdout(io.StringIO()):
                stats = train_one_epoch(model, loader, optimizer, torch.device('cpu'),
                                        0, scaler, args=args, text_features=storage)
            storage.global_text._mmap.close()
            storage.features._mmap.close()
            storage.mask._mmap.close()
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

    def test_padding_is_invisible_and_valid_tokens_affect_prediction(self):
        import torch
        model = self.model().eval()
        x, t, r = torch.randn(2, 8, 25, 3), torch.tensor([2, 6]), torch.randn(2, 8)
        data = self.tokens(2)
        data['text_token_mask'][:, -2:] = False

        def predict(values):
            memory = model.remap_tokens(values['text_tokens'], values['text_token_mask'],
                                        values['text_person_ids'], values['text_positions'])
            return model.text_skeleton_decoder(x, t, r, memory, values['text_token_mask'])

        with torch.no_grad():
            expected = predict(data)
            changed = {k: v.clone() for k, v in data.items()}
            changed['text_tokens'][:, -2:] = float('nan')
            changed['text_person_ids'][:, -2:] = 999
            changed['text_positions'][:, -2:] = -999
            torch.testing.assert_close(predict(changed), expected)
            changed['text_tokens'][:, 0] += torch.arange(6).float() * 3
            self.assertFalse(torch.allclose(predict(changed), expected))
        for name in ('text_token_mask',):
            invalid = dict(data)
            invalid[name] = torch.zeros_like(data[name])
            with self.assertRaises(ValueError):
                predict(invalid)

    def test_token_order_and_person_metadata_travel_together(self):
        import torch
        model = self.model().eval()
        data = self.tokens(2)
        x, t, r = torch.randn(2, 8, 25, 3), torch.tensor([2, 6]), torch.randn(2, 8)

        def predict(values):
            memory = model.remap_tokens(values['text_tokens'], values['text_token_mask'],
                                        values['text_person_ids'], values['text_positions'])
            return model.text_skeleton_decoder(x, t, r, memory, values['text_token_mask'])

        with torch.no_grad():
            expected = predict(data)
            order = torch.tensor([4, 0, 2, 5, 1, 3])
            # Packing order is irrelevant when original positions/person IDs follow tokens.
            torch.testing.assert_close(predict({k: v[:, order] for k, v in data.items()}), expected)
            changed = {k: v.clone() for k, v in data.items()}
            changed['text_person_ids'] = 1 - changed['text_person_ids']
            self.assertFalse(torch.allclose(predict(changed), expected))


    def test_reverse_padding_is_invisible_and_loss_is_single_valid_mean(self):
        import torch
        model = self.model().eval()
        data = self.tokens(2)
        valid = torch.cat([torch.ones(2, 1, dtype=torch.bool), data['text_token_mask']], dim=1)
        memory = torch.randn(2, 7, 8, requires_grad=True)
        h = torch.randn(2, 5, 8, requires_grad=True)
        hm = torch.tensor([[True, True, True, False, False], [True]*5])
        t = torch.tensor([2, 4])
        args = (t, h, hm, valid, data['text_person_ids'], data['text_positions'])
        with torch.no_grad():
            expected = model.text_noise_decoder(memory, *args)
            changed = memory.detach().clone().masked_fill(~valid[..., None], float('nan'))
            changed_h = h.detach().clone().masked_fill(~hm[..., None], float('nan'))
            actual = model.text_noise_decoder(changed, t, changed_h, hm, valid,
                                              data['text_person_ids'], data['text_positions'])
            torch.testing.assert_close(actual[valid], expected[valid])
            changed_h[:, 0] += torch.arange(8).float()
            actual = model.text_noise_decoder(memory, t, changed_h, hm, valid,
                                              data['text_person_ids'], data['text_positions'])
            self.assertFalse(torch.allclose(actual[valid], expected[valid]))
        recorded = {}
        original = model.diffusion.q_sample
        def capture(target, times, noise=None):
            recorded['noise'] = noise.detach()
            return original(target, times, noise=noise)
        model.diffusion.q_sample = capture
        hook = model.text_noise_decoder.register_forward_hook(
            lambda module, inputs, output: recorded.update(prediction=output))
        loss = model.skeleton_to_text_loss(memory, h, valid, hm,
            data['text_person_ids'], data['text_positions'])
        hook.remove()
        expected_loss = (recorded['prediction'][valid].float() - recorded['noise'][valid]).square().mean()
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        self.assertIsNone(memory.grad)
        self.assertGreater(h.grad[hm].abs().sum().item(), 0)
        self.assertEqual(h.grad[~hm].abs().sum().item(), 0)

    def test_global_token_is_read_by_attention_and_gets_gradient(self):
        import torch
        model = self.model()
        x = torch.randn(2, 8, 25, 3)
        r = torch.randn(2, 8, requires_grad=True)
        local = torch.randn(2, 4, 8, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        observed = []
        hook = model.text_skeleton_decoder.text_readers[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[1].shape))
        out = model.text_skeleton_decoder(x, torch.tensor([2, 4]), r, local, mask)
        hook.remove()
        self.assertEqual(observed, [torch.Size([5, 2, 8])])
        out.square().mean().backward()
        self.assertGreater(r.grad.abs().sum().item(), 0)
        self.assertGreater(local.grad.abs().sum().item(), 0)

    def test_shared_decoder_gradients_freezing_and_checkpoint(self):
        import torch
        for enabled in (False, True):
            model = self.model(share_skeleton_decoder=enabled)
            keys = model.state_dict()
            self.assertEqual(any(k.startswith('text_skeleton_decoder.decoder_blocks.') for k in keys), not enabled)
            x = torch.randn(2, 8, 25, 3)
            t = torch.tensor([2, 4])
            noise = torch.randn_like(x)
            r = model.text_remap(torch.randn(2, 6))
            local = torch.randn(2, 4, 8)
            loss = model.text_to_skeleton_loss(x, noise, t, torch.ones(2, 50), r,
                torch.ones(2, dtype=torch.bool), 1, local, torch.ones(2, 4, dtype=torch.bool))
            loss.backward()
            self.assertEqual(model.decoder_pred.weight.grad is not None, enabled)
            self.assertTrue(all(p.grad is None for p in model.blocks.parameters()))
            if enabled:
                self.assertGreater(model.decoder_pred.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.decoder_blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.decoder_embed.proj.weight.grad.abs().sum().item(), 0)
                self.assertIsNotNone(model.decoder_pos_embed.grad)
            restored = self.model(share_skeleton_decoder=enabled)
            restored.load_state_dict(keys)
        for weights in ((1., 1.), (0., 1.), (1., 0.), (0., 0.)):
            model = self.model(share_skeleton_decoder=True,
                lambda_text_to_skeleton=weights[0], lambda_skeleton_to_text=weights[1])
            self.assertTrue(all(p.requires_grad for p in model.decoder_blocks.parameters()))
            x = torch.randn(2, 3, 8, 25, 2)
            model(x, x.clone(), text_features=torch.randn(2, 6), mask_ratio=.5,
                  **self.tokens(2))[0].backward()
            for name, p in model.named_parameters():
                self.assertEqual(p.grad is not None, p.requires_grad, name)

    def test_resume_rejects_changed_decoder_sharing(self):
        import torch
        from util.misc import load_model
        model = self.model(share_skeleton_decoder=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            saved = types.SimpleNamespace(text_cache_identity={}, text_training_weights=(1., 1.))
            torch.save({'model': model.state_dict(), 'args': saved}, path)
            args = types.SimpleNamespace(resume=str(path), text_cache='cache',
                text_cache_identity={}, text_training_weights=(1., 1.), text_share_skeleton_decoder=True)
            with self.assertRaisesRegex(ValueError, 'share_skeleton_decoder'):
                load_model(args, model, None, None)


    def test_person_cache_keeps_people_separate_and_one_person_ignores_second(self):
        import torch
        from util.person_text_cache import PersonTokenFeatureCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentence = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
            features = np.arange(2*2*7*6, dtype=np.float16).reshape(2, 2, 7, 6)
            mask = np.zeros((2,2,7), dtype=bool)
            mask[:,0,1:3] = True
            mask[:,1,1:6] = True
            for name, value in [('person_features',sentence),('token_features',features),('token_mask',mask)]:
                np.save(root / (name+'.npy'), value)
            storage = PersonTokenFeatureCache(root, {})
            single = storage.get_batch(np.array([1,0]), one_person=True)
            dual = storage.get_batch(np.array([1,0]), one_person=False)
            self.assertEqual(single['text_tokens'].shape, (2,2,6))
            self.assertEqual(dual['text_tokens'].shape, (4,5,6))
            np.testing.assert_array_equal(single['text_features'], sentence[[1,0],0])
            np.testing.assert_array_equal(dual['text_features'], sentence[[1,0]].reshape(4,6))
            np.testing.assert_array_equal(dual['text_tokens'][1], features[1,1,1:6])
            np.testing.assert_array_equal(dual['text_token_mask'].sum(1), [2,5,2,5])
            model = self.model().eval()
            x = torch.randn(2,3,8,25,2)
            kwargs = {k:torch.from_numpy(v) for k,v in single.items()}
            torch.manual_seed(99); np.random.seed(99)
            before = model(x,x.clone(),mask_ratio=.5,**kwargs)
            changed = x.clone(); changed[...,1] = torch.randn_like(changed[...,1])*100
            torch.manual_seed(99); np.random.seed(99)
            after = model(changed,changed.clone(),mask_ratio=.5,**kwargs)
            torch.testing.assert_close(before[0],after[0])
            torch.testing.assert_close(before[1],after[1])
            for array in (storage.global_text,storage.features,storage.mask): array._mmap.close()

    def test_active_person_without_caption_fails_instead_of_using_other_person(self):
        import torch
        model = self.model(one_person=False)
        data = self.tokens(4)
        data['text_token_mask'][1] = False
        x = torch.randn(2,3,8,25,2)
        with self.assertRaisesRegex(ValueError, 'no matching cached description'):
            model(x,x.clone(),text_features=torch.randn(4,6),mask_ratio=.5,**data)


if __name__ == '__main__':
    unittest.main()
