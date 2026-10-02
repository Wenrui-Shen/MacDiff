"""Execute actual bank updates with a small NumPy-backed Torch surface on CPU."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from util.sample_text_target_bank import SampleTextTargetBank
from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank


class ArrayTensor:
    def __init__(self, values, operations):
        self.values = np.asarray(values)
        self.operations = operations
        self.device = 'cpu-fake'

    def to(self, device):
        return self

    def index_select(self, dimension, indices):
        self.operations.append(('index_select', self.values.shape, dimension))
        return ArrayTensor(np.take(self.values, indices.values, axis=dimension), self.operations)

    def reshape(self, *shape):
        return ArrayTensor(self.values.reshape(*shape), self.operations)

    def index_copy_(self, dimension, indices, source):
        assert dimension == 0
        self.operations.append(('index_copy', self.values.shape, dimension))
        self.values[indices.values] = source.values
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ArrayTorch(types.ModuleType):
    float32 = np.float32
    no_grad = staticmethod(nullcontext)

    def __init__(self):
        super().__init__('torch')
        self.operations = []

    def from_numpy(self, values):
        return ArrayTensor(values, self.operations)

    def zeros_like(self, tensor, dtype=None):
        self.operations.append(('zeros_like', tensor.values.shape))
        return ArrayTensor(np.zeros_like(tensor.values, dtype=dtype), self.operations)


class ArrayModel:
    def __init__(self, torch):
        self.torch = torch
        self.weight = np.eye(4, dtype=np.float32)
        self.bias = np.asarray([.05, -.03, .02, .01], dtype=np.float32)
        self.calls = []

    @staticmethod
    def unit_rms(values):
        values = np.asarray(values, dtype=np.float32)
        return values / np.sqrt(np.maximum(np.mean(values * values, axis=-1,
                                                  keepdims=True), 1e-12))

    def fixed_clip_target(self, features, valid=None):
        values = features.values.astype(np.float32, copy=True)
        if valid is not None:
            values = np.where(valid.values[..., None], values, 0)
        values = self.unit_rms(values)
        return ArrayTensor(values, self.torch.operations)

    def remap_values(self, values):
        return self.unit_rms(values + np.tanh(values @ self.weight + self.bias))

    def text_remap(self, features):
        if not np.isfinite(features.values).all():
            raise AssertionError('Inactive/NaN padding reached the remap')
        self.calls.append(features.values.shape)
        return ArrayTensor(self.remap_values(features.values), self.torch.operations)


class UpdateCache:
    manifest = {'protocol': 'test', 'identity': {'data': 'packed-updates'}, 'files': {}}

    def __init__(self):
        self.global_text = np.asarray([[[1., 0., 0., 0.]], [[0., 1., 0., 0.]],
                                       [[0., 0., 0., 0.]]], dtype=np.float32)
        self.features = np.zeros((3, 1, 3, 4), dtype=np.float32)
        self.mask = np.asarray([[[1, 0, 0]], [[1, 1, 0]], [[0, 0, 0]]], dtype=bool)
        self.features[0, 0, 0] = self.global_text[0, 0]
        self.features[1, 0, 0] = self.global_text[1, 0]
        self.features[1, 0, 1, 2] = 1.

    def get_batch(self, indices, one_person=True):
        assert one_person
        indices = np.asarray(indices, dtype=np.int64)
        length = max(1, int(self.mask[indices, 0].sum(axis=1).max()))
        return {'text_features': self.global_text[indices, 0].copy(),
                'text_tokens': self.features[indices, 0, :length].copy(),
                'text_token_mask': self.mask[indices, 0, :length].copy()}


class CaptureBank:
    dim = 4
    update_from_model = SampleTextTargetBank.update_from_model

    def __init__(self):
        self.updates = []

    def apply_updates(self, *values):
        self.updates.append(values)


class PackedTargetUpdateTests(unittest.TestCase):
    @staticmethod
    def batch(cache, torch, indices, active):
        arrays = cache.get_batch(indices)
        tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
        return (np.asarray(indices, dtype=np.int64), arrays, tensors,
                np.asarray(active, dtype=bool))

    @staticmethod
    def dense_reference(model, batches):
        """Old whole-microbatch calculation, reduced to its observable active outputs."""
        pieces = []
        for indices, arrays, _, active in batches:
            global_values = model.remap_values(model.unit_rms(arrays['text_features']))
            tokens = np.where(arrays['text_token_mask'][..., None], arrays['text_tokens'], 0)
            local_values = model.remap_values(model.unit_rms(tokens))
            local_values = np.where(arrays['text_token_mask'][..., None], local_values, 0)
            pieces.append((indices[active], {key: values[active] for key, values in arrays.items()},
                           global_values[active], local_values[active]))
        return pieces

    def test_variable_lengths_repeats_and_post_step_weights_match_dense_history(self):
        cache, torch = UpdateCache(), ArrayTorch()
        model = ArrayModel(torch)
        batches = [self.batch(cache, torch, [0, 0], [1, 1]),
                   self.batch(cache, torch, [1, 2, 0], [1, 0, 1])]
        # Inputs were collected before the step; targets must use the latest weights.
        model.weight[:] = np.eye(4, dtype=np.float32)[[2, 0, 3, 1]]
        expected = self.dense_reference(model, batches)
        with tempfile.TemporaryDirectory() as directory:
            reference = SampleTextTargetBank.prepare(Path(directory) / 'reference.sqlite', cache, .1)
            banks = [SampleTextTargetBank.prepare(Path(directory) / 'updated.sqlite', cache, .1),
                     SharedMemoryTextTargetBank.prepare(Path(directory) / 'updated.json', cache, .1)]
            try:
                for values in expected:
                    reference.apply_updates(*values)
                arrays = cache.get_batch([0, 1, 2])
                for bank in banks:
                    with self.subTest(backend=type(bank).__name__):
                        model.calls.clear()
                        with patch.dict(sys.modules, {'torch': torch}):
                            bank.update_from_model([0, 0, 1, 0], model, 'cpu-fake', batches=batches)
                        self.assertEqual(model.calls, [(2, 4), (2, 1, 4), (2, 4), (3, 4)])
                        self.assertEqual(sum(np.prod(shape[:-1]) for shape in model.calls), 9)
                        for key, values in reference.get_batch([0, 1, 2], arrays).items():
                            np.testing.assert_allclose(bank.get_batch([0, 1, 2], arrays)[key],
                                                       values, rtol=0, atol=2e-7)
            finally:
                reference.close()
                for bank in banks:
                    bank.close()

    def test_nan_inactive_rows_and_padding_never_reach_remap(self):
        cache, torch = UpdateCache(), ArrayTorch()
        model, bank = ArrayModel(torch), CaptureBank()
        batch = self.batch(cache, torch, [1, 2, 0], [1, 0, 1])
        indices, arrays, tensors, active = batch
        arrays['text_features'][1] = np.nan
        arrays['text_tokens'][~arrays['text_token_mask']] = np.nan
        expected = self.dense_reference(model, [batch])[0]
        with patch.dict(sys.modules, {'torch': torch}):
            bank.update_from_model([1, 0], model, 'cpu-fake', batches=[batch])
        self.assertEqual(model.calls, [(2, 4), (3, 4)])
        actual = bank.updates[0]
        np.testing.assert_array_equal(actual[0], [1, 0])
        np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=0)
        np.testing.assert_allclose(actual[3], expected[3], rtol=0, atol=0)
        self.assertTrue(np.isfinite(actual[2]).all())
        self.assertTrue(np.isfinite(actual[3]).all())
        np.testing.assert_array_equal(actual[3][~actual[1]['text_token_mask']], 0)

    def test_all_valid_dense_path_avoids_gather_and_zero_allocation(self):
        cache, torch = UpdateCache(), ArrayTorch()
        model, bank = ArrayModel(torch), CaptureBank()
        batch = self.batch(cache, torch, [1, 1], [1, 1])
        with patch.dict(sys.modules, {'torch': torch}):
            bank.update_from_model([1, 1], model, 'cpu-fake', batches=[batch])
        self.assertEqual(model.calls, [(2, 4), (2, 2, 4)])
        self.assertEqual(torch.operations, [])

    def test_all_empty_local_skips_local_remap_and_keeps_zero_layout(self):
        for width in (0, 1, 3):
            with self.subTest(width=width):
                torch, bank = ArrayTorch(), CaptureBank()
                model = ArrayModel(torch)
                arrays = {'text_features': np.asarray([[1., 0., 0., 0.]], dtype=np.float32),
                          'text_tokens': np.full((1, width, 4), np.nan, dtype=np.float32),
                          'text_token_mask': np.zeros((1, width), dtype=bool)}
                tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
                batch = (np.asarray([2]), arrays, tensors, np.asarray([True]))
                with patch.dict(sys.modules, {'torch': torch}):
                    bank.update_from_model([2], model, 'cpu-fake', batches=[batch])
                self.assertEqual(model.calls, [(1, 4)])
                self.assertEqual(bank.updates[0][3].shape, (1, width, 4))
                np.testing.assert_array_equal(bank.updates[0][3], 0)

    def test_all_inactive_window_does_not_remap_or_update(self):
        cache, torch = UpdateCache(), ArrayTorch()
        model, bank = ArrayModel(torch), CaptureBank()
        batch = self.batch(cache, torch, [0, 1], [0, 0])
        with patch.dict(sys.modules, {'torch': torch}):
            bank.update_from_model([0, 1], model, 'cpu-fake', batches=[batch])
        self.assertEqual(model.calls, [])
        self.assertEqual(bank.updates, [])


if __name__ == '__main__':
    unittest.main()
