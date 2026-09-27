"""Paired-input and EMA-target checks for the read-only diagnostic."""
import importlib.util
import unittest


HAS_RUNTIME = (importlib.util.find_spec('torch') is not None
               and importlib.util.find_spec('numpy') is not None)


@unittest.skipUnless(HAS_RUNTIME, 'PyTorch and NumPy are required')
class EmaDiagnosticTests(unittest.TestCase):
    @staticmethod
    def fixture():
        import numpy as np
        import torch
        import diagnose_ema_bidirectional as diag
        from tests.test_macdiff_text import TextStage1Tests

        torch.set_num_threads(1)
        model = TextStage1Tests.model(
            text_target_mode='ema_remap', text_target_norm='rms',
            share_skeleton_decoder=True, lambda_text_to_skeleton=.1,
            lambda_skeleton_to_text=.1, lambda_text_uniformity=.02)
        model.eval().requires_grad_(False)
        rng = np.random.RandomState(17)
        global_text = rng.randn(3, 6).astype(np.float32)
        global_text /= np.linalg.norm(global_text, axis=-1, keepdims=True)
        local_text = rng.randn(3, 4, 6).astype(np.float32)
        local_text /= np.linalg.norm(local_text, axis=-1, keepdims=True)

        class Cache:
            def get_batch(self, indices, one_person=True):
                assert one_person
                return {
                    'text_features': global_text[indices],
                    'text_tokens': local_text[indices],
                    'text_token_mask': np.ones((len(indices), 4), dtype=bool),
                    'text_person_ids': np.zeros((len(indices), 4), dtype=np.int64),
                    'text_positions': np.tile(np.arange(1, 5), (len(indices), 1)),
                }

        data, views = diag.text_views(model, Cache(), [0, 1, 2], 'cpu')
        return diag, model, data, views

    def test_teacher_starts_at_fixed_clip(self):
        import torch

        diag, model, data, views = self.fixture()
        torch.testing.assert_close(views['global_teacher'], views['global_clip'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(views['local_teacher'], views['local_clip'],
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(views['local_condition'].shape, data['text_tokens'].shape)
        self.assertAlmostEqual(diag.paired_distance(
            views['global_clip'].numpy(), views['global_teacher'].numpy())['mse'], 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_shuffles_change_only_the_intended_condition(self):
        import numpy as np
        import torch

        diag, model, data, views = self.fixture()
        raw = torch.randn(3, 8, 25, 3)
        aug = raw + .01 * torch.randn_like(raw)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        captured = []
        handle = model.text_skeleton_decoder.register_forward_pre_hook(
            lambda module, values: captured.append(tuple(value.clone() for value in values)))
        text_captured = []
        text_handle = model.text_noise_decoder.register_forward_pre_hook(
            lambda module, values: text_captured.append(
                tuple(value.clone() for value in values)))
        order = np.asarray([1, 2, 0])
        result = diag.t2s_trial(model, raw, aug, data, views, timestep=5,
                                mask_ratio=.5, motion_tau=-1,
                                permutation=order, mask_seed=13, noise_seed=29)
        handle.remove()
        text_handle.remove()
        self.assertEqual(set(result), {
            't2s_global_shuffled', 't2s_local_shuffled', 't2s_all_shuffled',
            's2t_all', 's2t_global', 's2t_local'})
        self.assertEqual(len(captured), 4)
        for values in captured[1:]:
            for position in (0, 1, 4):
                torch.testing.assert_close(values[position], captured[0][position],
                                           rtol=0, atol=0)
        torch.testing.assert_close(captured[1][2], captured[0][2][order], rtol=0, atol=0)
        torch.testing.assert_close(captured[2][3], captured[0][3][order], rtol=0, atol=0)
        torch.testing.assert_close(captured[3][2], captured[0][2][order], rtol=0, atol=0)
        torch.testing.assert_close(captured[3][3], captured[0][3][order], rtol=0, atol=0)
        self.assertEqual(len(text_captured), 2)
        for position in (0, 1, 3, 4, 5, 6):
            torch.testing.assert_close(text_captured[1][position],
                                       text_captured[0][position], rtol=0, atol=0)
        torch.testing.assert_close(text_captured[1][2], text_captured[0][2][order],
                                   rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
