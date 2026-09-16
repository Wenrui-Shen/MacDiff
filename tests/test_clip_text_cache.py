"""Cache alignment/integrity/resume tests, with no model downloads or GPU."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import cache_clip_text as cache
from util.clip_text_cache import global_features, load_cache, load_token_cache


class ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def float(self):
        return self

    def norm(self, dim=-1, keepdim=False):
        return ArrayTensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim))

    def __lt__(self, value):
        return self.values < value

    def __truediv__(self, other):
        return ArrayTensor(self.values / other.values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    bos_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, **kwargs):
        ids = [[0] + [ord(c) for c in text] + [1] for text in texts]
        if kwargs.get('return_tensors') != 'pt':
            return Batch(input_ids=ids)
        context = kwargs['max_length']
        return Batch(input_ids=ArrayTensor([row + [1] * (context - len(row)) for row in ids]),
                     attention_mask=ArrayTensor([[1] * len(row) + [0] * (context - len(row)) for row in ids]))


class Encoder:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    @staticmethod
    def text_projection(hidden):
        return hidden

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError('simulated interruption')
        ids = input_ids.values
        hidden = np.stack([np.broadcast_to(np.arange(ids.shape[1]) + 1, ids.shape),
                           np.cumsum(ids, axis=1) % 127 + 1, np.ones_like(ids)], axis=-1)
        vectors = hidden[np.arange(len(ids)), attention_mask.values.sum(axis=1).astype(int) - 1]
        return types.SimpleNamespace(text_embeds=ArrayTensor(vectors), last_hidden_state=ArrayTensor(hidden))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        np.savez(self.root / 'data.npz', x_train=np.zeros((3, 4, 150), dtype=np.float32))
        self.clip = self.root / 'clip'
        self.clip.mkdir()
        (self.clip / 'config.json').write_text('{}')
        (self.clip / 'pytorch_model.bin').write_bytes(b'fake test weights')
        self.caption_path = self.root / 'captions.jsonl'
        self.rows = [self.caption(2, ['walk', 'wave']), self.caption(0, ['sit']),
                     self.caption(1, ['old']), self.caption(1, ['jump']),
                     dict(self.caption(1, []), status='pipeline_error')]
        self.write_captions()
        self.args = argparse.Namespace(data_path=str(self.root / 'data.npz'),
            captions=[str(self.caption_path)], clip_model=str(self.clip),
            output_dir=str(self.root / 'out'), batch_size=1, device='cpu', resume=True)

    @staticmethod
    def caption(index, texts):
        return {'status': 'accepted', 'sample_index': index, 'source_split': 'train',
                'model': 'test', 'prompt': {'sha256': 'fixed'}, 'actor_count': len(texts),
                'texts': [{'person_index': i, 'text': text} for i, text in enumerate(texts)]}

    def write_captions(self):
        self.caption_path.write_text('\n'.join(map(json.dumps, self.rows)), encoding='utf-8')

    def run_fake(self, encoder=None, max_length=77):
        encoder = encoder or Encoder()
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                          isfinite=lambda x: np.isfinite(x.values))
        config = types.SimpleNamespace(projection_dim=3, max_position_embeddings=max_length)
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(
                cache, 'load_encoder', return_value=(encoder, Tokenizer(), config)):
            cache.run(self.args)
        return encoder

    def test_full_coverage_last_accepted_and_person_alignment(self):
        self.run_fake()
        features, manifest = load_cache(self.args.output_dir, self.args.data_path, 3)
        samples = json.loads((self.root / 'out/samples.json').read_text())
        self.assertEqual(samples, [dict(sample_index=0, texts=['sit']),
                                  dict(sample_index=1, texts=['jump']),
                                  dict(sample_index=2, texts=['walk', 'wave'])])
        valid = np.load(self.root / 'out/person_valid.npy')
        np.testing.assert_array_equal(valid, [[True, False], [True, False], [True, True]])
        persons = np.load(self.root / 'out/person_features.npy')
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1., atol=1e-6)
        expected = persons[2].mean(axis=0)
        np.testing.assert_allclose(features[2], expected / np.linalg.norm(expected))
        self.assertTrue(manifest['complete'])

    def test_interrupted_run_resumes_and_complete_run_needs_no_encoder(self):
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_fake(Encoder(fail_on_call=2))
        before = np.load(self.root / 'out/person_features.npy').copy()
        manifest = json.loads((self.root / 'out/manifest.json').read_text())
        self.assertEqual(manifest['completed_persons'], 1)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            load_cache(self.args.output_dir)
        encoder = self.run_fake()
        self.assertEqual(encoder.calls, 3)
        np.testing.assert_array_equal(np.load(self.root / 'out/person_features.npy')[0], before[0])
        with patch.object(cache, 'load_encoder', side_effect=AssertionError('Must reuse')):
            cache.run(self.args)

    def test_cache_rejects_changed_sources_and_corrupted_features(self):
        self.run_fake()
        self.rows.append(self.caption(0, ['different']))
        self.write_captions()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            cache.run(self.args)
        with self.assertRaisesRegex(ValueError, 'count'):
            load_cache(self.args.output_dir, expected_count=4)
        np.savez(self.root / 'other.npz', x_train=np.ones((3, 4, 150)))
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            load_cache(self.args.output_dir, self.root / 'other.npz')
        values = np.load(self.root / 'out/person_features.npy', mmap_mode='r+')
        values[0, 0, 0] = 123
        values.flush()
        del values
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_cache(self.args.output_dir)

    def test_missing_captions_fail_before_loading_clip(self):
        self.rows = [self.caption(0, ['sit'])]
        self.write_captions()
        with self.assertRaisesRegex(ValueError, 'Missing accepted'):
            cache.run(self.args)

    def test_overlength_captions_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, 'token limit'):
            self.run_fake(max_length=2)
        self.assertFalse((self.root / 'out/manifest.json').exists())

    def test_missing_people_do_not_dilute_global_feature(self):
        features = np.array([[[1., 0.], [0., 0.]], [[1., 0.], [0., 1.]]])
        result = global_features(features, np.array([[True, False], [True, True]]))
        np.testing.assert_allclose(result, [[1, 0], [2 ** -.5, 2 ** -.5]])
        with self.assertRaises(ValueError):
            global_features(features, np.zeros((2, 2), dtype=bool))

    def test_tokens_keep_person_positions_eos_and_exclude_bos_padding(self):
        self.run_fake()
        storage = load_token_cache(self.args.output_dir)
        self.assertIsInstance(storage.features, np.memmap)
        batch = storage.get_batch(np.array([2, 0, 2]))
        # 'walk' and 'wave': four content tokens plus EOS per person.
        self.assertEqual(batch['text_tokens'].shape, (3, 10, 3))
        np.testing.assert_array_equal(batch['text_person_ids'][0], [0] * 5 + [1] * 5)
        np.testing.assert_array_equal(batch['text_positions'][0], [1, 2, 3, 4, 5] * 2)
        self.assertEqual(int(batch['text_token_mask'][1].sum()), 4)  # 'sit' + EOS
        np.testing.assert_array_equal(batch['text_tokens'][1, 4:], 0)
        np.testing.assert_array_equal(batch['text_tokens'][0], batch['text_tokens'][2])
        for sample in range(3):
            for person, text in enumerate(json.loads((self.root / 'out/samples.json').read_text())[sample]['texts']):
                np.testing.assert_allclose(storage.features[sample, person, len(text) + 1],
                    np.load(self.root / 'out/person_features.npy')[sample, person], atol=5e-4)
        with self.assertRaises(ValueError):
            storage.get_batch(np.array([-1]))

    def test_token_file_corruption_is_rejected(self):
        self.run_fake()
        values = np.load(self.root / 'out/token_features.npy', mmap_mode='r+')
        values[0, 0, 1, 0] = 12
        values.flush()
        del values
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_token_cache(self.args.output_dir)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None
                     and importlib.util.find_spec('transformers') is not None,
                     'Torch and Transformers are required for the tiny real CLIP test')
class RealClipTests(unittest.TestCase):
    def test_real_clip_projection_and_cache_roundtrip_without_downloads(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizerFast
        from transformers.models.clip.tokenization_clip import bytes_to_unicode
        torch.set_num_threads(1)
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alphabet = list(bytes_to_unicode().values())
            vocab = {token: i for i, token in enumerate(
                alphabet + [token + '</w>' for token in alphabet] + ['<|startoftext|>', '<|endoftext|>'])}
            (root / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
            (root / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
            tokenizer = CLIPTokenizerFast(vocab_file=str(root / 'vocab.json'),
                merges_file=str(root / 'merges.txt'), model_max_length=16)
            config = CLIPTextConfig(vocab_size=len(vocab), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=16, projection_dim=8,
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
            model = CLIPTextModelWithProjection(config).eval()
            np.savez(root / 'data.npz', x_train=np.zeros((2, 4, 150), dtype=np.float32))
            captions = [CacheTests.caption(0, ['sit']), CacheTests.caption(1, ['walk', 'wave'])]
            (root / 'captions.jsonl').write_text('\n'.join(map(json.dumps, captions)), encoding='utf-8')
            for safe in (True, False):
                clip = root / ('clip_safe' if safe else 'clip_bin')
                model.save_pretrained(clip, safe_serialization=safe)
                tokenizer.save_pretrained(clip)
                args = argparse.Namespace(data_path=str(root / 'data.npz'),
                    captions=[str(root / 'captions.jsonl')], clip_model=str(clip),
                    output_dir=str(root / ('out_safe' if safe else 'out_bin')),
                    batch_size=2, device='cpu', resume=True)
                cache.run(args)
                storage = load_token_cache(args.output_dir, args.data_path, 2)
                batch = storage.get_batch(np.array([1, 0]))
                self.assertEqual(batch['text_tokens'].shape[-1], 8)
                with torch.inference_mode():
                    expected = model(**tokenizer(['sit'], return_tensors='pt')).text_embeds
                    expected = torch.nn.functional.normalize(expected, dim=-1).numpy()[0]
                np.testing.assert_allclose(storage.global_text[0], expected, atol=1e-6)
                length = len(tokenizer('sit')['input_ids'])
                np.testing.assert_allclose(storage.features[0, 0, length - 1], expected, atol=5e-4)
                self.assertFalse(storage.mask[0, 0, 0])
                self.assertFalse(storage.mask[0, 1].any())
                storage.features._mmap.close()
                storage.mask._mmap.close()


if __name__ == '__main__':
    unittest.main()
