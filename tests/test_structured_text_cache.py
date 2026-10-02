"""New captions -> RMS sentence cache -> training batch -> persistent targets."""
import argparse
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import cache_clip_motion_text as cache
from tests.test_clip_text_cache import Encoder, Tokenizer
from util.person_text_cache import PersonTokenFeatureCache, load_person_token_cache
from util.structured_text_cache import (DEFINITION, LOCAL_PARTS, PROTOCOL, train_person_valid,
                                       validate_cache, validate_training_definition)


class SentenceTokenizer(Tokenizer):
    def __call__(self, texts, **kwargs):
        kwargs.setdefault('max_length', max(len(text) + 2 for text in texts))
        return super().__call__(texts, **kwargs)


class SentenceReaderTests(unittest.TestCase):
    """Compare the fixed-slot fast path with the original generic packing."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        globals_ = np.arange(6 * 2 * 8, dtype=np.float32).reshape(6, 2, 8)
        features = np.arange(6 * 2 * 6 * 8, dtype=np.float32).reshape(6, 2, 6, 8)
        present = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 0]], dtype=bool)
        np.save(root / 'person_features.npy', globals_)
        # Nonzero masked contents check that the reader itself still zeros padding.
        np.save(root / 'token_features.npy', features)
        np.save(root / 'token_mask.npy', np.repeat(present[..., None], 6, axis=-1))
        self.storage = PersonTokenFeatureCache(root, {'protocol': PROTOCOL})
        self.addCleanup(lambda: [array._mmap.close() for array in
                               (self.storage.global_text, self.storage.features, self.storage.mask)
                               if hasattr(array, '_mmap')])

    @staticmethod
    def generic_batch(storage, indices, one_person):
        people = 1 if one_person else 2
        valid = storage.mask[indices, :people]
        length = max(1, int(valid.sum(axis=-1).max()))
        rows, dim = len(indices) * people, storage.features.shape[-1]
        tokens = np.zeros((rows, length, dim), dtype=np.float32)
        mask = np.zeros((rows, length), dtype=bool)
        persons = np.zeros((rows, length), dtype=np.int64)
        positions = np.zeros_like(persons)
        for sample, index in enumerate(indices):
            for person in range(people):
                row = sample * people + person
                pos = np.flatnonzero(valid[sample, person])
                count = len(pos)
                tokens[row, :count] = storage.features[index, person, pos]
                mask[row, :count] = True
                persons[row, :count] = person
                positions[row, :count] = pos + storage.position_offset
        return dict(text_features=np.asarray(storage.global_text[indices, :people],
                                             dtype=np.float32).reshape(rows, dim),
                    text_tokens=tokens, text_token_mask=mask,
                    text_person_ids=persons, text_positions=positions)

    def assert_batch_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            self.assertEqual(actual[key].dtype, expected[key].dtype, key)
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            self.assertTrue(actual[key].flags.c_contiguous, key)

    def test_vectorized_rows_match_generic_for_shuffle_repeats_and_empty_people(self):
        indices = np.array([2, 1, 0, 2, 3, 4, 1])
        for one_person in (True, False):
            with self.subTest(one_person=one_person):
                expected = self.generic_batch(self.storage, indices, one_person)
                with patch('util.person_text_cache.np.flatnonzero',
                           side_effect=AssertionError('Fixed sentence rows need no compaction')):
                    actual = self.storage.get_batch(indices, one_person)
                self.assert_batch_equal(actual, expected)
                actual['text_tokens'][0] = -1
                np.testing.assert_array_equal(self.storage.features[2, 0],
                    np.arange(6 * 2 * 6 * 8, dtype=np.float32).reshape(6, 2, 6, 8)[2, 0])

    def test_all_empty_batches_keep_one_zero_slot(self):
        for one_person, indices in ((True, np.array([2, 3, 5, 2])),
                                    (False, np.array([5, 3, 5]))):
            with self.subTest(one_person=one_person):
                expected = self.generic_batch(self.storage, indices, one_person)
                with patch('util.person_text_cache.np.flatnonzero',
                           side_effect=AssertionError('Empty sentence rows need no compaction')):
                    actual = self.storage.get_batch(indices, one_person)
                self.assert_batch_equal(actual, expected)
                self.assertEqual(actual['text_tokens'].shape[1], 1)

    def test_partial_masks_and_v2_keep_generic_compaction(self):
        original = self.storage.mask
        self.storage.mask = np.array(original)
        original._mmap.close()
        self.storage.mask[1, 0] = [1, 0, 1, 0, 0, 1]
        indices = np.array([1, 2, 1, 0])
        for offset in (1, 0):
            self.storage.position_offset = offset
            for one_person in (True, False):
                with self.subTest(offset=offset, one_person=one_person):
                    expected = self.generic_batch(self.storage, indices, one_person)
                    actual = self.storage.get_batch(indices, one_person)
                    self.assert_batch_equal(actual, expected)

    def test_single_person_gathers_only_person_zero(self):
        arrays = self.storage.features
        calls = []

        class RetainedSlotArray:
            shape = arrays.shape

            def __getitem__(self, key):
                calls.append(key)
                return arrays[key]

        self.storage.features = RetainedSlotArray()
        self.addCleanup(arrays._mmap.close)
        actual = self.storage.get_batch(np.array([1, 0, 1]), one_person=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], slice(None, 1))
        self.assertEqual(actual['text_tokens'].shape, (3, 6, 8))


class StructuredCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = np.zeros((3, 4, 150), dtype=np.float32)
        self.data[:, :, :75] = 1
        self.data[2, :, 75:] = 2
        # No y_train in this fixture: the pipeline must be label-free.
        self.data_path = self.root / 'data.npz'
        np.savez_compressed(self.data_path, x_train=self.data)
        clip = self.root / 'clip'
        clip.mkdir()
        (clip / 'config.json').write_text('{}', encoding='utf-8')
        (clip / 'pytorch_model.bin').write_bytes(b'fake fixture')
        self.path = self.root / 'captions.json'
        self.samples = [self.sample(2, 2), self.sample(0), self.sample(1)]
        self.write_captions()
        self.args = argparse.Namespace(data_path=str(self.data_path), captions=[str(self.path)],
            clip_model=str(clip), output_dir=str(self.root / 'cache'), batch_size=1,
            device='cpu', resume=True)

    @staticmethod
    def sample(index, people=1):
        return {'sample_index': index, 'persons': [
            {'person_index': person, 'global': 'Person {} moves {}.'.format(person, index),
             'local': {part: '{} moves {} {}.'.format(part, index, person) for part in LOCAL_PARTS}}
            for person in range(people)]}

    def write_captions(self, path=None, rows=None, model='qwen', prompt='prompt'):
        path = path or self.path
        rows = rows if rows is not None else self.samples
        content = (json.dumps(rows) if path.suffix == '.json'
                   else '\n'.join(json.dumps(row) for row in rows))
        path.write_text(content, encoding='utf-8')
        path.with_suffix('.metadata.json').write_text(json.dumps({
            'schema_version': 'macdiff.caption_text.v2', 'caption_schema': 'global_local',
            'model': model, 'prompt': {'sha256': prompt}, 'requested_revision': None}), encoding='utf-8')

    def run_fake(self, encoder=None, context=77):
        encoder = encoder or Encoder()
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        config = types.SimpleNamespace(projection_dim=3, max_position_embeddings=context)
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(cache, 'load_encoder',
                return_value=(encoder, SentenceTokenizer(), config)):
            cache.run(self.args)
        return encoder

    def storage(self, **kwargs):
        result = load_person_token_cache(self.args.output_dir, self.data_path, 3, **kwargs)
        self.addCleanup(lambda: [array._mmap.close() for array in
                               (result.global_text, result.features, result.mask)])
        return result

    def test_sentence_encoding_rms_and_fixed_regions_in_raw_sample_order(self):
        encoder = self.run_fake()
        self.assertEqual(encoder.calls, 4)  # four persons; each encodes seven sentences
        storage = self.storage()
        manifest = storage.manifest
        self.assertEqual(manifest['protocol'], PROTOCOL)
        self.assertEqual(manifest['definition'], DEFINITION)
        self.assertEqual(manifest['context_length'], 7)
        self.assertEqual(storage.features.dtype, np.float32)
        self.assertEqual(storage.features.shape, (3, 2, 6, 3))
        batch = storage.get_batch(np.array([2, 0, 2]))
        self.assertEqual(batch['text_tokens'].shape, (3, 6, 3))
        np.testing.assert_array_equal(batch['text_positions'], [[1, 2, 3, 4, 5, 6]] * 3)
        np.testing.assert_array_equal(batch['text_token_mask'], True)
        np.testing.assert_array_equal(batch['text_person_ids'], 0)
        np.testing.assert_allclose(np.mean(batch['text_tokens'] ** 2, axis=-1), 1, atol=2e-6)
        # Compare every local slot with that entire sentence's pooled encoder output.
        texts = [self.sample(2)['persons'][0]['local'][part] for part in LOCAL_PARTS]
        expected = cache.encode_sentences(Encoder(), SentenceTokenizer(), texts, 'cpu')
        np.testing.assert_allclose(batch['text_tokens'][0], expected, atol=2e-6)
        global_expected = cache.encode_sentences(Encoder(), SentenceTokenizer(),
                            [self.sample(2)['persons'][0]['global']], 'cpu')[0]
        np.testing.assert_allclose(batch['text_features'][0], global_expected, atol=2e-6)
        np.testing.assert_array_equal(batch['text_tokens'][0], batch['text_tokens'][2])
        np.testing.assert_array_equal(storage.features[0, 1], 0)

    def test_two_people_are_separate_and_absent_person_never_borrows_text(self):
        self.run_fake()
        storage = self.storage()
        batch = storage.get_batch(np.array([0, 2]), one_person=False)
        self.assertEqual(batch['text_features'].shape, (4, 3))
        np.testing.assert_array_equal(batch['text_token_mask'].sum(1), [6, 0, 6, 6])
        np.testing.assert_array_equal(batch['text_features'][1], 0)
        np.testing.assert_array_equal(batch['text_person_ids'][2:], [[0] * 6, [1] * 6])
        np.testing.assert_array_equal(batch['text_tokens'][2], storage.features[2, 0])
        np.testing.assert_array_equal(batch['text_tokens'][3], storage.features[2, 1])

    def test_interruption_resume_and_complete_reuse_without_loading_torch(self):
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_fake(Encoder(fail_on_call=2))
        manifest = json.loads((self.root / 'cache/manifest.json').read_text())
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['completed_persons'], 1)
        before = np.load(self.root / 'cache/person_features.npy')[0].copy()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.storage()
        resumed = self.run_fake()
        self.assertEqual(resumed.calls, 3)
        np.testing.assert_array_equal(np.load(self.root / 'cache/person_features.npy')[0], before)
        with patch.object(cache, 'load_encoder', side_effect=AssertionError('already complete')):
            cache.run(self.args)

    def test_invalid_caption_coverage_duplicate_regions_and_provenance_fail_early(self):
        cases = []
        cases.append((self.samples[:1], 'Missing captions'))
        cases.append((self.samples + [self.samples[0]], 'Duplicate'))
        missing = copy.deepcopy(self.samples)
        del missing[0]['persons'][0]['local']['head']
        cases.append((missing, 'alignment'))
        bad_person = copy.deepcopy(self.samples)
        bad_person[0]['persons'][1]['person_index'] = 0
        cases.append((bad_person, 'alignment'))
        for rows, message in cases:
            with self.subTest(message=message):
                self.write_captions(rows=rows)
                with patch.object(cache, 'load_encoder', side_effect=AssertionError('preflight only')):
                    with self.assertRaisesRegex(ValueError, message):
                        cache.run(self.args)
        self.write_captions()
        second = self.root / 'second.jsonl'
        self.write_captions(path=second, rows=[], model='different')
        self.args.captions.append(str(second))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            cache.run(self.args)

    def test_raw_skeleton_slots_are_checked_without_labels(self):
        np.testing.assert_array_equal(train_person_valid(self.data_path, batch_size=2),
                                      [[1, 0], [1, 0], [1, 1]])
        missing_person = copy.deepcopy(self.samples)
        missing_person[0]['persons'] = missing_person[0]['persons'][:1]
        self.write_captions(rows=missing_person)
        with self.assertRaisesRegex(ValueError, 'raw skeleton slots'):
            cache.run(self.args)
        self.write_captions()
        self.data[1] = 0
        np.savez(self.data_path, x_train=self.data)
        with self.assertRaisesRegex(ValueError, 'without any skeleton'):
            cache.run(self.args)

    def test_jsonl_input_and_overlength_rejection_without_truncation(self):
        path = self.root / 'captions.jsonl'
        self.write_captions(path=path)
        self.args.captions = [str(path)]
        with self.assertRaisesRegex(ValueError, 'token limit'):
            self.run_fake(context=8)
        self.assertFalse((self.root / 'cache/manifest.json').exists())
        self.run_fake()
        self.assertEqual(self.storage().get_batch(np.array([1]))['text_tokens'].shape[1], 6)

    def test_integrity_semantics_and_fast_validation(self):
        self.run_fake()
        full = self.storage()
        with patch('util.structured_text_cache.file_identity', side_effect=AssertionError('no hash')):
            fast = self.storage(skip_full_validation=True)
        for key, value in full.get_batch(np.array([2, 0])).items():
            np.testing.assert_array_equal(fast.get_batch(np.array([2, 0]))[key], value)
        for storage in (full, fast):
            for array in (storage.global_text, storage.features, storage.mask):
                array._mmap.close()
        with self.assertRaisesRegex(ValueError, 'sample count'):
            validate_cache(self.args.output_dir, expected_count=4)
        manifest_path = self.root / 'cache/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        changed = copy.deepcopy(manifest)
        changed['definition']['local_parts'].reverse()
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'region order'):
            self.storage(skip_full_validation=True)
        manifest_path.write_text(json.dumps(manifest))
        # Header-only reuse still catches same-size reshapes.
        path = self.root / 'cache/token_features.npy'
        original = path.read_bytes()
        features = np.load(path)
        np.save(path, features.reshape(2, 3, 6, 3))
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            self.storage(skip_full_validation=True)
        path.write_bytes(original)
        values = np.load(path, mmap_mode='r+')
        values[0, 0, 0, 0] = 99
        values.flush()
        values._mmap.close()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.storage()

    def test_source_changes_and_unit_rms_training_guard(self):
        self.run_fake()
        manifest = self.storage().manifest
        validate_training_definition(manifest, {'text_target_mode': 'sample_target_blend', 'text_target_norm': 'rms'})
        with self.assertRaisesRegex(ValueError, 'flip=False'):
            validate_training_definition(manifest,
                {'text_target_mode': 'sample_target_blend', 'text_target_norm': 'rms'},
                types.SimpleNamespace(flip=True))
        with self.assertRaisesRegex(ValueError, 'already uses unit RMS'):
            validate_training_definition(manifest, {'text_target_mode': 'fixed_clip', 'text_target_norm': 'none'})
        validate_training_definition({'protocol': 'macdiff_clip_token_cache_v2'}, {})
        self.write_captions(prompt='new prompt')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.run_fake()

    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'PyYAML not available')
    def test_new_training_configs_use_six_slots_rms_and_separate_outputs(self):
        import yaml
        project = Path(__file__).resolve().parents[1]
        for name, target_mode in (
                ('pretrain_madiff_text_sentence_fixed_rms.yaml', 'fixed_clip'),
                ('pretrain_madiff_text_sentence_sample_target_blend_shared.yaml', 'sample_target_blend')):
            with self.subTest(config=name):
                config = yaml.safe_load((project / 'config/ntu60_xsub_joint' / name).read_text(encoding='utf-8'))
                self.assertEqual(config['model_args']['text_context_length'], 7)
                self.assertEqual(config['model_args']['text_target_mode'], target_mode)
                self.assertEqual(config['model_args']['text_target_norm'], 'rms')
                self.assertFalse(config['train_feeder_args']['flip'])
                self.assertEqual(config['text_cache'], 'vlm_pilot/ntu60_xsub_clip_sentence_cache_v3')
                self.assertIn('sentence', config['output_dir'])
                self.assertEqual(config['lambda_skeleton_to_text'], .1)

    def test_six_region_targets_work_with_sqlite_ram_and_snapshot_resume(self):
        from util.sample_text_target_bank import SampleTextTargetBank
        from util.shared_memory_text_target_bank import SharedMemoryTextTargetBank
        self.run_fake()
        storage = self.storage()
        indices = np.array([2, 0])
        arrays = storage.get_batch(indices)
        sql = SampleTextTargetBank.prepare(self.root / 'sql.sqlite', storage, .1)
        ram = SharedMemoryTextTargetBank.prepare(self.root / 'ram.json', storage, .1)
        self.addCleanup(sql.close)
        self.addCleanup(ram.close)
        initial = sql.get_batch(indices, arrays)
        np.testing.assert_allclose(initial['text_target_tokens'], arrays['text_tokens'], atol=2e-6)
        replacement = np.roll(arrays['text_tokens'], 1, axis=-1)
        global_replacement = np.roll(arrays['text_features'], 1, axis=-1)
        for bank in (sql, ram):
            bank.apply_updates(indices, arrays, global_replacement, replacement)
            after = bank.get_batch(indices, arrays)
            np.testing.assert_allclose(after['text_target_tokens'],
                .9 * initial['text_target_tokens'] + .1 * replacement, atol=2e-6)
        checkpoint = self.root / 'checkpoint-0.pth'
        ram.snapshot(checkpoint)
        restored = SampleTextTargetBank.prepare(self.root / 'restored.sqlite', storage, .1, resume=checkpoint)
        self.addCleanup(restored.close)
        np.testing.assert_array_equal(restored.get_batch(indices, arrays)['text_target_tokens'],
                                      ram.get_batch(indices, arrays)['text_target_tokens'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not available')
class StructuredTrainingTests(unittest.TestCase):
    def test_seven_sentence_vectors_reach_skeleton_encoder_through_s2t(self):
        import torch
        from model.transformer_macdiff_text import Transformer
        model = Transformer(dim_feat=16, depth=1, decoder_depth=1, num_heads=4,
            num_frames=8, num_joints=25, t_patch_size=4, patch_size=1, one_person=True,
            diff_prediction='noise', diff_steps=10, uncond_ratio=0.,
            text_input_dim=8, text_context_length=7, text_decoder_hidden_dim=16,
            text_decoder_depth=1, text_target_mode='fixed_clip', text_target_norm='rms',
            lambda_text_to_skeleton=0., lambda_skeleton_to_text=.1)
        source = torch.randn(2, 3, 8, 25, 2)
        global_values = torch.randn(2, 8)
        local_values = torch.randn(2, 6, 8)
        losses = []
        original = model.skeleton_to_text_loss
        def capture(*args, **kwargs):
            loss = original(*args, **kwargs)
            losses.append(loss)
            return loss
        with patch.object(model, 'skeleton_to_text_loss', side_effect=capture):
            result = model(source, source.clone(), text_features=global_values,
                text_tokens=local_values, text_token_mask=torch.ones(2, 6, dtype=torch.bool),
                text_person_ids=torch.zeros(2, 6, dtype=torch.long),
                text_positions=torch.arange(1, 7).expand(2, -1))
        self.assertEqual(result[3]['text_valid_tokens'].item(), 6)
        losses[0].backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in model.blocks.parameters()))


if __name__ == '__main__':
    unittest.main()
