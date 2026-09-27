"""Persistent target values follow samples, not EMA network parameters."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_NUMPY, 'NumPy is required')
class SampleTargetBankTests(unittest.TestCase):
    @staticmethod
    def cache_fixture():
        import numpy as np

        class Cache:
            manifest = {'protocol': 'test', 'identity': {'data': 'small'}, 'files': {}}
            global_text = np.zeros((2, 1, 4), dtype=np.float32)
            features = np.zeros((2, 1, 4, 4), dtype=np.float32)

            def get_batch(self, indices, one_person=True):
                assert one_person
                base = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=np.float32)
                indices = np.asarray(indices, dtype=int)
                global_text = base[indices]
                local = global_text[:, None, :].copy()
                return dict(text_features=global_text, text_tokens=local,
                            text_token_mask=np.ones((len(indices), 1), dtype=bool))

        return Cache()

    def test_sample_outputs_blend_at_fixed_ratio_and_survive_snapshot(self):
        import numpy as np
        from util.sample_text_target_bank import SampleTextTargetBank

        cache = self.cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'current.sqlite'
            bank = SampleTextTargetBank.prepare(path, cache, .1)
            indices = np.asarray([0, 1])
            arrays = cache.get_batch(indices)
            initial = bank.get_batch(indices, arrays)
            np.testing.assert_allclose(initial['text_target_global'],
                                       arrays['text_features'] * 2, rtol=0, atol=0)
            online_global = np.asarray([[0., 2., 0., 0.],
                                        [2., 0., 0., 0.]], dtype=np.float32)
            online_local = online_global[:, None, :].copy()
            bank.apply_updates(indices, arrays, online_global, online_local)
            first = bank.get_batch(indices, arrays)
            np.testing.assert_allclose(first['text_target_global'],
                                       .9 * initial['text_target_global'] + .1 * online_global)
            bank.apply_updates(indices, arrays, online_global, online_local)
            second = bank.get_batch(indices, arrays)
            np.testing.assert_allclose(second['text_target_global'],
                                       .9 * first['text_target_global'] + .1 * online_global)
            np.testing.assert_allclose(second['text_target_tokens'][:, 0],
                                       second['text_target_global'])
            checkpoint = Path(directory) / 'checkpoint-0.pth'
            bank.snapshot(checkpoint)
            bank.close()
            resumed = SampleTextTargetBank.prepare(
                Path(directory) / 'resumed.sqlite', cache, .1, resume=checkpoint)
            np.testing.assert_allclose(resumed.get_batch(indices, arrays)['text_target_global'],
                                       second['text_target_global'])
            resumed.close()
            with self.assertRaisesRegex(ValueError, 'metadata'):
                SampleTextTargetBank(path, cache, .2)

    def test_each_sample_keeps_its_own_previous_target(self):
        import numpy as np
        from util.sample_text_target_bank import SampleTextTargetBank

        cache = self.cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'current.sqlite'
            first_reader = SampleTextTargetBank.prepare(path, cache, .1)
            second_reader = SampleTextTargetBank(path, cache, .1)
            indices = np.asarray([0, 1])
            before = first_reader.get_batch(indices, cache.get_batch(indices))
            one = np.asarray([0])
            one_arrays = cache.get_batch(one)
            replacement = np.asarray([[0., 2., 0., 0.]], dtype=np.float32)
            first_reader.apply_updates(one, one_arrays, replacement, replacement[:, None])
            after = second_reader.get_batch(indices, cache.get_batch(indices))
            np.testing.assert_allclose(after['text_target_global'][0],
                                       .9 * before['text_target_global'][0]
                                       + .1 * replacement[0])
            np.testing.assert_array_equal(after['text_target_global'][1],
                                          before['text_target_global'][1])
            first_reader.close()
            second_reader.close()

    def test_empty_inactive_person_does_not_create_a_target(self):
        import numpy as np
        from util.sample_text_target_bank import SampleTextTargetBank

        cache = self.cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            bank = SampleTextTargetBank.prepare(
                Path(directory) / 'current.sqlite', cache, .1)
            arrays = cache.get_batch(np.asarray([0, 1]))
            arrays['text_token_mask'][1] = False
            targets = bank.get_batch(np.asarray([0, 1]), arrays)
            np.testing.assert_array_equal(targets['text_target_global'][1], 0)
            np.testing.assert_array_equal(targets['text_target_tokens'][1], 0)
            self.assertEqual(bank.initialized_samples(), 0)
            bank.close()


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, 'PyTorch and NumPy are required')
class SampleTargetModelTests(unittest.TestCase):
    def test_model_uses_supplied_target_without_teacher_network(self):
        import torch
        from tests.test_macdiff_text import TextStage1Tests

        model = TextStage1Tests.model(
            text_target_mode='sample_target_blend', text_target_norm='rms',
            text_target_update_ratio=.1, one_person=True,
            lambda_text_to_skeleton=.1, lambda_skeleton_to_text=.1,
            lambda_text_uniformity=.02)
        self.assertFalse(any(key.startswith('text_target_remap.')
                             for key in model.state_dict()))
        source = torch.randn(2, 3, 8, 25, 2)
        source_aug = source + .01 * torch.randn_like(source)
        tokens = TextStage1Tests.tokens(2, dim=6)
        features = torch.randn(2, 6)
        target_global = model.fixed_clip_target(features)
        target_local = model.fixed_clip_target(tokens['text_tokens'],
                                               tokens['text_token_mask'])
        loss, _, _, metrics = model(
            source, source_aug, text_features=features, mask_ratio=.5,
            text_target_global=target_global,
            text_target_tokens=target_local, **tokens)
        self.assertAlmostEqual(metrics['text_target_drift_mse'].item(), 0., places=6)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.text_remap.parameters()))
        self.assertTrue(all(parameter.grad is None
                            for parameter in (target_global, target_local)))


if __name__ == '__main__':
    unittest.main()
