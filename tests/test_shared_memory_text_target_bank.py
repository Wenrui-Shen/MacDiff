"""Shared sample histories, concurrent updates, and legacy snapshot migration."""
import importlib.util
from contextlib import closing
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import traceback
import unittest
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


class FixtureCache:
    manifest = {'protocol': 'test', 'identity': {'data': 'variable-tokens'}, 'files': {}}

    def __init__(self):
        import numpy as np
        self.global_text = np.asarray([[[1., 0., 0., 0.]],
                                       [[0., 1., 0., 0.]],
                                       [[0., 0., 0., 0.]]], dtype=np.float32)
        self.features = np.zeros((3, 1, 4, 4), dtype=np.float32)
        self.mask = np.asarray([[[1, 0, 0, 0]], [[1, 1, 0, 0]],
                                [[0, 0, 0, 0]]], dtype=bool)
        self.features[0, 0, 0] = self.global_text[0, 0]
        self.features[1, 0, 0] = self.global_text[1, 0]
        self.features[1, 0, 1, 2] = 1.

    def get_batch(self, indices, one_person=True):
        import numpy as np
        assert one_person
        indices = np.asarray(indices, dtype=np.int64)
        length = max(1, int(self.mask[indices, 0].sum(axis=1).max()))
        return {'text_features': self.global_text[indices, 0].copy(),
                'text_tokens': self.features[indices, 0, :length].copy(),
                'text_token_mask': self.mask[indices, 0, :length].copy()}


def shared_worker(path, rank, barrier, queue):
    import numpy as np
    from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank
    bank = None
    try:
        cache = FixtureCache()
        bank = SharedMemoryTextTargetBank.prepare(path, cache, .1, rank=rank,
                                                  barrier=barrier.wait)
        indices = np.asarray([0])
        arrays = cache.get_batch(indices)
        replacement = np.asarray([[0., 2., 0., 0.]], dtype=np.float32)
        if rank == 0:
            bank.apply_updates(indices, arrays, replacement, replacement[:, None])
        barrier.wait(timeout=15)
        visible = bank.get_batch(indices, arrays)['text_target_global'].copy()
        barrier.wait(timeout=15)
        for _ in range(10):
            bank.apply_updates(indices, arrays, replacement, replacement[:, None])
        barrier.wait(timeout=15)
        queue.put(('ok', rank, visible.tolist(),
                   bank.get_batch(indices, arrays)['text_target_global'].tolist(),
                   bank.initialized_samples()))
        barrier.wait(timeout=15)
    except Exception:
        queue.put(('error', rank, traceback.format_exc()))
        barrier.abort()
    finally:
        if bank is not None:
            bank.close()


@unittest.skipUnless(HAS_NUMPY, 'NumPy is required')
class SharedMemoryTargetBankTests(unittest.TestCase):
    def test_matches_sqlite_for_repeated_indices_padding_and_empty_rows(self):
        import numpy as np
        from util.sample_text_target_bank import SampleTextTargetBank
        from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank

        cache = FixtureCache()
        with tempfile.TemporaryDirectory() as directory:
            sql = SampleTextTargetBank.prepare(Path(directory) / 'current.sqlite', cache, .1)
            ram = SharedMemoryTextTargetBank.prepare(Path(directory) / 'shared.json', cache, .1)
            try:
                indices = np.asarray([0, 1, 2])
                arrays = cache.get_batch(indices)
                for key, values in sql.get_batch(indices, arrays).items():
                    np.testing.assert_array_equal(ram.get_batch(indices, arrays)[key], values)
                self.assertEqual(ram.initialized_samples(), 0)
                rng = np.random.RandomState(7)
                for batch in ([1], [0, 1, 0], [1, 0]):
                    batch = np.asarray(batch)
                    packed = cache.get_batch(batch)
                    global_online = rng.randn(*packed['text_features'].shape).astype(np.float32)
                    local_online = rng.randn(*packed['text_tokens'].shape).astype(np.float32)
                    sql.apply_updates(batch, packed, global_online, local_online)
                    # No training-step SQLite access in the RAM backend.
                    with patch('sqlite3.connect', side_effect=AssertionError('per-step disk I/O')):
                        ram.apply_updates(batch, packed, global_online, local_online)
                        result = ram.get_batch(indices, arrays)
                    for key, values in sql.get_batch(indices, arrays).items():
                        np.testing.assert_array_equal(result[key], values)
                self.assertEqual(ram.initialized_samples(), 2)
                empty = cache.get_batch(np.asarray([2]))
                ram.apply_updates([2], empty, empty['text_features'], empty['text_tokens'])
                self.assertEqual(ram.initialized_samples(), 2)
                np.testing.assert_array_equal(result['text_target_global'][2], 0)
                copied = ram.get_batch(indices, arrays)
                copied['text_target_global'][:] = 123
                np.testing.assert_array_equal(ram.get_batch(indices, arrays)['text_target_global'],
                                              result['text_target_global'])
            finally:
                sql.close()
                ram.close()

    def test_legacy_snapshot_resumes_in_ram_and_new_snapshot_resumes_in_sqlite(self):
        import numpy as np
        from util.sample_text_target_bank import SampleTextTargetBank
        from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank

        cache = FixtureCache()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            indices = np.asarray([0, 1, 2])
            arrays = cache.get_batch(indices)
            sql = SampleTextTargetBank.prepare(directory / 'old.sqlite', cache, .1)
            online = np.asarray([[0., 2., 0., 0.]], dtype=np.float32)
            sql.apply_updates([0], cache.get_batch([0]), online, online[:, None])
            expected = sql.get_batch(indices, arrays)
            old_checkpoint = directory / 'checkpoint-10.pth'
            sql.snapshot(old_checkpoint)
            sql.close()
            ram = SharedMemoryTextTargetBank.prepare(directory / 'shared.json', cache, .1,
                                                     resume=old_checkpoint)
            try:
                self.assertEqual(ram.initialized_samples(), 1)
                for key in expected:
                    np.testing.assert_array_equal(ram.get_batch(indices, arrays)[key], expected[key])
                new_checkpoint = directory / 'checkpoint-20.pth'
                ram.snapshot(new_checkpoint)
            finally:
                ram.close()
            with closing(sqlite3.connect(str(SampleTextTargetBank.checkpoint_path(new_checkpoint)))) as db:
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            restored = SampleTextTargetBank.prepare(directory / 'restored.sqlite', cache, .1,
                                                   resume=new_checkpoint)
            try:
                for key in expected:
                    np.testing.assert_array_equal(restored.get_batch(indices, arrays)[key], expected[key])
            finally:
                restored.close()
            again = SharedMemoryTextTargetBank.prepare(directory / 'again.json', cache, .1,
                                                       resume=new_checkpoint)
            try:
                for key in expected:
                    np.testing.assert_array_equal(again.get_batch(indices, arrays)[key], expected[key])
            finally:
                again.close()
            with self.assertRaisesRegex(ValueError, 'metadata'):
                SharedMemoryTextTargetBank.prepare(directory / 'bad.json', cache, .2,
                                                   resume=old_checkpoint)
            with self.assertRaises(FileNotFoundError):
                SharedMemoryTextTargetBank.prepare(directory / 'missing.json', cache, .1,
                                                   resume=directory / 'checkpoint-99.pth')

    def test_two_processes_share_history_and_do_not_lose_updates(self):
        import numpy as np
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as directory:
            barrier = context.Barrier(2)
            queue = context.Queue()
            processes = [context.Process(target=shared_worker,
                         args=(str(Path(directory) / 'shared.json'), rank, barrier, queue))
                         for rank in (0, 1)]
            try:
                for process in processes:
                    process.start()
                results = [queue.get(timeout=25) for _ in processes]
                for result in results:
                    self.assertEqual(result[0], 'ok', result)
                    np.testing.assert_allclose(result[2], [[1.8, .2, 0., 0.]], atol=1e-6)
                    remaining = .9 ** 21
                    np.testing.assert_allclose(result[3],
                        [[2 * remaining, 2 * (1 - remaining), 0., 0.]], atol=1e-6)
                    self.assertEqual(result[4], 1)
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
                queue.close()
                queue.join_thread()


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, 'PyTorch and NumPy are required')
class CachedWindowTests(unittest.TestCase):
    def test_reuses_inputs_with_post_step_weights_and_variable_padding(self):
        import numpy as np
        import torch
        from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank

        class Model:
            text_remap = torch.nn.Linear(4, 4, bias=False)

            @staticmethod
            def fixed_clip_target(features, valid=None):
                values = features.detach().float()
                values = values * torch.rsqrt(values.square().mean(-1, keepdim=True).clamp_min(1e-12))
                return values if valid is None else values.masked_fill(~valid[..., None], 0)

        cache, model = FixtureCache(), Model()
        with tempfile.TemporaryDirectory() as directory:
            bank = SharedMemoryTextTargetBank.prepare(Path(directory) / 'shared.json', cache, .1)
            try:
                batches = []
                for indices, active in (([0], [True]), ([1, 2], [True, False])):
                    arrays = cache.get_batch(indices)
                    tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
                    batches.append((np.asarray(indices), arrays, tensors, np.asarray(active)))
                arrays = cache.get_batch([0, 1, 2])
                initial = bank.get_batch([0, 1, 2], arrays)
                # Inputs were collected earlier; outputs must use these latest weights.
                with torch.no_grad():
                    model.text_remap.weight.copy_(torch.eye(4).roll(1, dims=0))
                expected_global = model.text_remap(model.fixed_clip_target(
                    torch.from_numpy(arrays['text_features'][:2]))).detach().numpy()
                expected_local = model.text_remap(model.fixed_clip_target(
                    torch.from_numpy(arrays['text_tokens'][:2]),
                    torch.from_numpy(arrays['text_token_mask'][:2]))).detach().numpy()
                with patch.object(cache, 'get_batch', side_effect=AssertionError('cache reread')):
                    bank.update_from_model([0, 1], model, torch.device('cpu'), batches=batches)
                result = bank.get_batch([0, 1, 2], arrays)
                np.testing.assert_allclose(result['text_target_global'][:2],
                    .9 * initial['text_target_global'][:2] + .1 * expected_global)
                valid = arrays['text_token_mask'][:2]
                np.testing.assert_allclose(result['text_target_tokens'][:2][valid],
                    .9 * initial['text_target_tokens'][:2][valid] + .1 * expected_local[valid])
                np.testing.assert_array_equal(result['text_target_global'][2], 0)
                self.assertEqual(bank.initialized_samples(), 2)
            finally:
                bank.close()


if __name__ == '__main__':
    unittest.main()
