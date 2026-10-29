"""Pair selection/identity/report math plus optional real-Torch CLI coverage."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import diagnose_clip_remap_pair as diag


class PairDiagnosticTests(unittest.TestCase):
    def test_default_seed_is_fresh_and_pair_is_distinct_person0(self):
        valid = np.array([[True, False], [False, True], [True, True], [True, False]])
        entropy = mock.Mock(side_effect=[32, 65])
        with mock.patch.object(diag, 'secrets', argparse.Namespace(randbits=entropy)):
            first, seed_a, eligible = diag.select_pair(valid)
            second, seed_b, _ = diag.select_pair(valid)
        self.assertEqual(entropy.call_count, 2)
        self.assertEqual((seed_a, seed_b, eligible), (32, 65, 3))
        for pair in (first, second):
            self.assertEqual(len(set(pair)), 2)
            self.assertNotIn(1, pair)  # person1 never replaces an absent person0
        np.testing.assert_array_equal(first, diag.select_pair(valid, 32)[0])

    def test_bad_seed_and_insufficient_people_fail(self):
        valid = np.ones((3, 2), dtype=bool)
        for seed in (-1, 2 ** 32, True):
            with self.assertRaises(ValueError):
                diag.select_pair(valid, seed)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            diag.select_pair(np.array([[False, True], [True, False]]))

    def test_preview_reports_raw_signed_and_absolute_head_tail(self):
        summary = diag.vector_summary(np.array([-3., 1., 7., -2., 4.]), 2)
        self.assertEqual(summary['head'], [-3., 1.])
        self.assertEqual(summary['head_abs'], [3., 1.])
        self.assertEqual(summary['tail'], [-2., 4.])
        self.assertEqual(summary['tail_abs'], [2., 4.])
        self.assertEqual(summary['tail_start'], 3)

    def test_cosine_uses_signed_full_vectors_and_distances(self):
        result = diag.pair_metrics([1., -1.], [-1., 1.])
        self.assertAlmostEqual(result['cosine'], -1.)
        self.assertAlmostEqual(result['unit_l2'], 2.)
        self.assertAlmostEqual(result['l2'], np.sqrt(8.))
        self.assertEqual(result['mean_abs_diff'], 2.)
        scaled = diag.pair_metrics([1., 2.], [3., 6.])
        self.assertAlmostEqual(scaled['cosine'], 1.)
        self.assertGreater(scaled['l2'], 0.)
        for a, b in (([0., 0.], [1., 2.]), ([np.nan], [1.]), ([1.], [1., 2.])):
            with self.assertRaises(ValueError):
                diag.pair_metrics(a, b)

    def test_labels_read_only_y_train_and_match_feeder_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.npz'
            # Accessing this object x_train with allow_pickle=False would fail.
            np.savez(path, x_train=np.array([object()], dtype=object),
                     y_train=np.eye(3)[[2, 0, 1]])
            np.testing.assert_array_equal(diag.read_labels(path, 3), [2, 0, 1])
            with self.assertRaises(ValueError):
                diag.read_labels(path, 4)
            np.savez(path, y_train=np.array([[1, 1], [0, 1]]))
            with self.assertRaises(ValueError):
                diag.read_labels(path, 2)

    def test_descriptions_are_selected_by_raw_index_and_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.json'
            rows = [dict(sample_index=1, persons=[{'person_index': 0, 'global': 'one'}]),
                    dict(sample_index=0, persons=[{'person_index': 1, 'global': 'other'},
                                                  {'person_index': 0, 'global': 'zero'}])]
            path.write_text(json.dumps(rows), encoding='utf-8')
            self.assertEqual(diag.read_descriptions(path, [0, 1], 2), ['zero', 'one'])
            rows[1]['sample_index'] = 1
            path.write_text(json.dumps(rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                diag.read_descriptions(path, [0, 1], 2)

    def checkpoint(self):
        state = {'module.text_remap.' + k: np.zeros(1) for k in (
            'body.0.weight', 'body.0.bias', 'body.2.weight', 'body.2.bias')}
        state['module.text_target_remap.body.0.weight'] = np.ones(1)
        return dict(model=state, epoch=39, args=argparse.Namespace(
            model_args=dict(text_target_mode='sample_target_blend'),
            text_cache_identity={'saved': True}))

    def test_only_checkpoint_online_remap_is_extracted(self):
        checkpoint = self.checkpoint()
        _, _, weights = diag.validate_checkpoint(checkpoint)
        self.assertEqual(set(weights), {'body.0.weight', 'body.0.bias', 'body.2.weight', 'body.2.bias'})
        self.assertIs(weights['body.0.weight'], checkpoint['model']['module.text_remap.body.0.weight'])

    def test_fixed_clip_missing_remap_and_wrong_cache_are_rejected(self):
        checkpoint = self.checkpoint()
        checkpoint['args'].model_args['text_target_mode'] = 'fixed_clip'
        with self.assertRaisesRegex(ValueError, 'no remap'):
            diag.validate_checkpoint(checkpoint)
        checkpoint = self.checkpoint()
        del checkpoint['model']['module.text_remap.body.2.bias']
        with self.assertRaisesRegex(ValueError, 'weights'):
            diag.validate_checkpoint(checkpoint)
        manifest = dict(protocol='macdiff_clip_sentence_cache_v3', identity={'a': 1}, files={})
        self.assertEqual(len(diag.check_cache_identity(manifest, manifest)), 64)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            diag.check_cache_identity(manifest, dict(manifest, files={'changed': 1}))
        self.assertEqual(diag.clip_weight_name({'pytorch_model.bin': {}}), 'pytorch_model.bin')
        for files in ({}, {'pytorch_model.bin': {}, 'model.safetensors': {}}):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                diag.clip_weight_name(files)

    def test_cli_is_random_by_default_and_read_only(self):
        args = diag.get_parser().parse_args(['--checkpoint', 'checkpoint-39.pth'])
        self.assertIsNone(args.seed)
        self.assertIsNone(args.clip_model)
        self.assertEqual(args.show_dims, 8)
        self.assertNotIn('output_dir', vars(args))


HAS_TORCH = (importlib.util.find_spec('torch') is not None
             and importlib.util.find_spec('timm') is not None)


@unittest.skipUnless(HAS_TORCH, 'Requires the existing Torch/timm training environment')
class TorchPairDiagnosticTests(unittest.TestCase):
    def test_real_checkpoint_cache_cli_and_no_files_changed(self):
        import torch
        from model.transformer_macdiff_text import ResidualTextRemap
        from util.clip_text_cache import file_identity
        from util.structured_text_cache import DEFINITION, unit_rms
        with tempfile.TemporaryDirectory() as tmp:
            root, count, dim = Path(tmp), 5, 4
            cache = root / 'cache'
            cache.mkdir()
            data = root / 'data.npz'
            np.savez(data, x_train=np.zeros((count, 2, 150), dtype=np.float32),
                     y_train=np.eye(3)[[0, 1, 2, 0, 1]])
            valid = np.array([[True, False], [False, True], [True, False],
                              [True, False], [True, False]])
            features = np.zeros((count, 2, dim), dtype=np.float32)
            features[valid] = unit_rms(np.random.RandomState(12).randn(valid.sum(), dim))
            np.save(cache / 'person_features.npy', features)
            np.save(cache / 'person_valid.npy', valid)
            np.save(cache / 'token_features.npy', np.repeat(features[:, :, None], 6, axis=2))
            np.save(cache / 'token_mask.npy', np.repeat(valid[:, :, None], 6, axis=2))
            samples = [dict(sample_index=i, persons=[{'person_index': int(p), 'global': 'sample ' + str(i)}
                       for p in np.flatnonzero(valid[i])]) for i in range(count)]
            (cache / 'samples.json').write_text(json.dumps(samples), encoding='utf-8')
            manifest = dict(protocol='macdiff_clip_sentence_cache_v3', complete=True,
                            sample_count=count, feature_dim=dim, context_length=7, definition=DEFINITION,
                            identity=dict(data=file_identity(data), definition=DEFINITION),
                            files={p.name: file_identity(p) for p in cache.glob('*.npy')})
            (cache / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            identity = {k: manifest[k] for k in ('protocol', 'identity', 'files')}
            torch.manual_seed(5)
            remap = ResidualTextRemap(dim, 8).eval()
            checkpoint = root / 'checkpoint-39.pth'
            torch.save(dict(model={'text_remap.' + k: v for k, v in remap.state_dict().items()}, epoch=39,
                            args=argparse.Namespace(model='model.transformer_macdiff_text_relation_teacher.Transformer',
                                model_args=dict(text_target_mode='sample_target_blend', text_input_dim=dim,
                                                text_hidden_dim=8), text_cache=str(cache),
                                train_feeder_args=dict(data_path=str(data)), text_cache_identity=identity)), checkpoint)
            before = {str(p): file_identity(p) for p in root.rglob('*') if p.is_file()}
            args = diag.get_parser().parse_args(['--checkpoint', str(checkpoint), '--seed', '27',
                                                '--show-dims', '2', '--full-validation'])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                diag.run(args)
            output = stream.getvalue()
            ids, _, _ = diag.select_pair(valid, 27)
            for i in ids:
                self.assertIn('sample_id=train_{} sample_index={} class_id_0based={}'.format(i, i, [0, 1, 2, 0, 1][i]), output)
            expected = remap(torch.from_numpy(features[ids, 0])).detach().numpy()
            cosine = diag.pair_metrics(*expected)['cosine']
            self.assertIn('  remap: cosine={:.9f}'.format(cosine), output)
            self.assertIn('remap=online_checkpoint', output)
            self.assertIn('abs=', output)
            self.assertEqual(before, {str(p): file_identity(p) for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
