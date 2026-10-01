"""Persistent target values follow samples, not EMA network parameters."""
import importlib.util
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


class SnapshotRecoveryChecks:
    """Run the same interrupted-save checks against both bank implementations."""

    def snapshot_fixture(self, directory):
        cache = SampleTargetBankTests.cache_fixture()
        bank = self.snapshot_bank_class().prepare(
            Path(directory) / 'snapshot-current', cache, .1)
        arrays = cache.get_batch([0])
        bank.apply_updates([0], arrays, arrays['text_features'], arrays['text_tokens'])
        return bank, arrays

    def assert_snapshot_values(self, bank, checkpoint, arrays):
        import numpy as np
        destination = bank.checkpoint_path(checkpoint)
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            blob = connection.execute('SELECT vectors FROM targets WHERE sample_id=0').fetchone()[0]
        values = bank._decode(blob, 1)
        expected = bank.get_batch([0], arrays)
        np.testing.assert_array_equal(values[0], expected['text_target_global'][0])
        np.testing.assert_array_equal(values[1], expected['text_target_tokens'][0, 0])

    def test_snapshot_retries_orphan_temporary_and_complete_files(self):
        import numpy as np
        for orphan in ('temporary', 'complete', 'both'):
            with self.subTest(orphan=orphan), tempfile.TemporaryDirectory() as directory:
                bank, arrays = self.snapshot_fixture(directory)
                try:
                    checkpoint = Path(directory) / 'checkpoint-10.pth'
                    destination = bank.checkpoint_path(checkpoint)
                    legacy_temporary = destination.with_name(destination.name + '.tmp')
                    if orphan in ('complete', 'both'):
                        bank.snapshot(checkpoint)
                    previous = destination.read_bytes() if destination.exists() else None
                    if orphan in ('temporary', 'both'):
                        legacy_temporary.write_bytes(b'interrupted SQLite snapshot')
                    replacement = np.asarray([[0., 2., 0., 0.]], dtype=np.float32)
                    bank.apply_updates([0], arrays, replacement, replacement[:, None])
                    original_replace = os.replace

                    def publish(source, target):
                        if previous is not None:
                            self.assertEqual(destination.read_bytes(), previous)
                        return original_replace(source, target)

                    with patch('util.sample_text_target_bank.os.replace', side_effect=publish):
                        bank.snapshot(checkpoint)
                    self.assert_snapshot_values(bank, checkpoint, arrays)
                    self.assertFalse(legacy_temporary.exists())
                    self.assertEqual(list(destination.parent.glob(destination.name + '.tmp-*')), [])
                finally:
                    bank.close()

    def test_snapshot_rejects_overwriting_existing_model_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, arrays = self.snapshot_fixture(directory)
            try:
                checkpoint = Path(directory) / 'checkpoint-10.pth'
                destination = bank.checkpoint_path(checkpoint)
                bank.snapshot(checkpoint)
                previous = destination.read_bytes()
                checkpoint.write_bytes(b'saved model checkpoint')
                with self.assertRaisesRegex(FileExistsError, 'existing checkpoint'):
                    bank.snapshot(checkpoint)
                self.assertEqual(destination.read_bytes(), previous)
                self.assertEqual(list(destination.parent.glob(destination.name + '.tmp-*')), [])
            finally:
                bank.close()

    def test_failed_snapshot_publish_preserves_previous_complete_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, arrays = self.snapshot_fixture(directory)
            try:
                checkpoint = Path(directory) / 'checkpoint-10.pth'
                destination = bank.checkpoint_path(checkpoint)
                bank.snapshot(checkpoint)
                previous = destination.read_bytes()
                legacy_temporary = destination.with_name(destination.name + '.tmp')
                legacy_temporary.write_bytes(b'older incomplete bank')
                with patch('util.sample_text_target_bank.os.replace',
                           side_effect=OSError('simulated interrupted publication')):
                    with self.assertRaisesRegex(OSError, 'interrupted publication'):
                        bank.snapshot(checkpoint)
                self.assertEqual(destination.read_bytes(), previous)
                self.assertEqual(legacy_temporary.read_bytes(), b'older incomplete bank')
                self.assertEqual(list(destination.parent.glob(destination.name + '.tmp-*')), [])
            finally:
                bank.close()

    def test_snapshot_rechecks_model_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, arrays = self.snapshot_fixture(directory)
            try:
                checkpoint = Path(directory) / 'checkpoint-10.pth'
                destination = bank.checkpoint_path(checkpoint)
                bank.snapshot(checkpoint)
                previous = destination.read_bytes()
                original_fsync = os.fsync

                def model_appears(file_descriptor):
                    original_fsync(file_descriptor)
                    checkpoint.write_bytes(b'model saved during snapshot construction')

                with patch('util.sample_text_target_bank.os.fsync', side_effect=model_appears):
                    with self.assertRaisesRegex(FileExistsError, 'existing checkpoint'):
                        bank.snapshot(checkpoint)
                self.assertEqual(destination.read_bytes(), previous)
                self.assertEqual(list(destination.parent.glob(destination.name + '.tmp-*')), [])
            finally:
                bank.close()


@unittest.skipUnless(HAS_NUMPY, 'NumPy is required')
class SampleTargetBankTests(SnapshotRecoveryChecks, unittest.TestCase):
    @staticmethod
    def snapshot_bank_class():
        from util.sample_text_target_bank import SampleTextTargetBank
        return SampleTextTargetBank

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
        self.assertNotIn('text_target_drift_mse', metrics)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.text_remap.parameters()))
        self.assertTrue(all(parameter.grad is None
                            for parameter in (target_global, target_local)))


if __name__ == '__main__':
    unittest.main()
