"""Cache alignment/integrity/resume tests, with no model downloads or GPU."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import cache_clip_text as cache
from util.clip_text_cache import global_features, load_cache


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
    def __call__(self, texts, **kwargs):
        return Batch(input_ids=[[ord(c) for c in text] for text in texts])


class Encoder:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, input_ids):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError('simulated interruption')
        vectors = [[len(ids), sum(ids) % 127 + 1, 1] for ids in input_ids]
        return types.SimpleNamespace(text_embeds=ArrayTensor(vectors))


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


if __name__ == '__main__':
    unittest.main()
