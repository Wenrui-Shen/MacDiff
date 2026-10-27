"""Train-only statistics, storage streaming, person/padding and config audits."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'normalization_audit', ROOT / 'tools/compute_skeleton_normalization.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_YAML = importlib.util.find_spec('yaml') is not None


def config(data_path, **model_options):
    return dict(feeder='feeder.feeder_ntu.Feeder', train_feeder_args=dict(
        data_path=str(data_path), split='train', window_size=120, p_interval=[.5, 1.],
        random_rot=False, flip=False, normalization=False),
        model='model.transformer_macdiff_flow_matching.Transformer',
        model_args=dict(input_mean=[1., 2., 3.], input_var=[4., 9., 16.], **model_options))


class StorageTests(unittest.TestCase):
    def test_streams_only_train_in_compressed_and_uncompressed_npz(self):
        values = np.arange(5 * 7 * 150, dtype=np.float32).reshape(5, 7, 150)
        for save in (np.savez, np.savez_compressed):
            with self.subTest(save=save.__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'data.npz'
                # Accessing this test array would require allow_pickle=True.
                save(path, x_train=values, x_test=np.array([object()], dtype=object))
                with mock.patch.object(np, 'load', side_effect=AssertionError('whole-array load')):
                    with audit.TrainArray(path) as source:
                        chunks = [batch.copy() for _, batch in source.batches(2)]
                        self.assertEqual(source.metadata['member'], 'x_train.npy')
                        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
                np.testing.assert_array_equal(np.concatenate(chunks), values)

    def test_limit_and_fortran_npy_without_loading_the_whole_array(self):
        values = np.asfortranarray(np.arange(4 * 5 * 150).reshape(4, 5, 150))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.npy'
            np.save(path, values)
            with audit.TrainArray(path) as source:
                self.assertIsInstance(source.array, np.memmap)
                chunks = [batch.copy() for _, batch in source.batches(2, limit=3)]
            np.testing.assert_array_equal(np.concatenate(chunks), values[:3])
            np.savez(Path(tmp) / 'fortran.npz', x_train=values)
            with self.assertRaisesRegex(ValueError, 'Fortran-order'):
                with audit.TrainArray(Path(tmp) / 'fortran.npz'):
                    self.fail('should fail before iteration')

    def test_layouts_match_native_flattening_and_nonfinite_values_are_rejected(self):
        tmvc = np.arange(7 * 2 * 25 * 3).reshape(7, 2, 25, 3)
        expected = tmvc.transpose(3, 0, 2, 1)
        for source, layout in ((tmvc.reshape(7, 150), 'auto'), (tmvc, 'auto'),
                               (expected, 'auto'), (expected, 'ctvm')):
            np.testing.assert_array_equal(audit.sample_ctvm(source, layout=layout), expected)
        bad = tmvc.astype(float)
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            audit.sample_ctvm(bad)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            audit.sample_ctvm(np.ones((3, 2, 25, 3)))


class StatisticsTests(unittest.TestCase):
    def test_stable_streaming_population_moments(self):
        rng = np.random.RandomState(19)
        values = 1.e8 + rng.normal(size=(101, 3))
        moments = audit.Moments()
        for chunk in np.array_split(values, 13):
            moments.update(chunk)
        result = moments.result()
        self.assertEqual(result['count'], 101)
        np.testing.assert_allclose(result['mean'], values.mean(axis=0), rtol=0, atol=1.e-7)
        np.testing.assert_allclose(result['var'], values.var(axis=0), rtol=1.e-7)
        self.assertIsNone(audit.Moments().result()['mean'])

    def test_person_selection_padding_legitimate_zeros_and_centering(self):
        sample = np.zeros((3, 3, 2, 2))
        sample[:, 0, :, 0] = [[0, 2], [2, 4], [4, 6]]
        sample[:, 1, :, 0] = [[4, 6], [6, 8], [8, 10]]
        sample[:, 0, :, 1] = [[10, 12], [20, 22], [30, 32]]
        original = sample.copy()
        groups = {}
        audit.update_groups(groups, sample, 'model_input', ('person0', 'all'))
        def result(people, shifted, padding):
            return groups[('model_input', people, shifted, padding)].result()
        person0 = result('person0', False, 'active_person_frames')
        self.assertEqual(person0['count'], 4)  # x=0 at a real joint is retained.
        np.testing.assert_allclose(person0['mean'], [3., 5., 7.])
        all_valid = result('all', False, 'active_person_frames')
        self.assertEqual(all_valid['count'], 6)
        np.testing.assert_allclose(all_valid['mean'], [34/6, 62/6, 90/6])
        all_positions = result('all', False, 'model_positions')
        self.assertEqual(all_positions['count'], 12)
        np.testing.assert_allclose(all_positions['mean'], [34/12, 62/12, 90/12])
        for people in ('person0', 'all'):
            np.testing.assert_allclose(result(people, True, 'model_positions')['mean'], 0, atol=1.e-14)
        self.assertEqual(result('all', True, 'active_person_frames')['count'], 6)
        np.testing.assert_array_equal(sample, original)

    def test_empty_person0_is_never_replaced_by_person1(self):
        sample = np.zeros((3, 2, 25, 2))
        sample[..., 1] = 10
        groups = {}
        audit.update_groups(groups, sample, 'raw_valid', ('person0', 'all'))
        zero = groups[('raw_valid', 'person0', False, 'model_positions')].result()
        np.testing.assert_array_equal(zero['mean'], [0, 0, 0])
        self.assertEqual(groups[('raw_valid', 'person0', False, 'active_person_frames')].count, 0)
        self.assertEqual(groups[('raw_valid', 'all', False, 'active_person_frames')].count, 50)

    def test_existing_normalization_comparison_and_self_shift_effective_mean(self):
        cfg = config('train.npz', self_shift=True)
        ref = audit.config_reference(cfg, {'path': 'pt.json'})
        self.assertEqual(ref['configured_mean'], [1, 2, 3])
        self.assertEqual(ref['effective_mean'], [0, 0, 0])
        stats = dict(count=2, mean=[1, 2, 3], var=[4, 9, 16])
        comparison = audit.compare_statistics(stats, ref)
        np.testing.assert_allclose(comparison['normalized_mean'], [.5, 2/3, .75])
        np.testing.assert_allclose(comparison['normalized_std'], [1, 1, 1])
        self.assertTrue(ref['person_mode'] == 'person0')
        cfg['model'] = 'model.transformer_downstream.Transformer'
        self.assertEqual(audit.config_reference(cfg, {'path': 'lp.json'})['person_mode'], 'all')


class IntegrationTests(unittest.TestCase):
    def test_real_cli_raw_report_uses_train_only_and_compares_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = np.ones((3, 4, 2, 25, 3), dtype=np.float32)
            data[0, :, 1] = 0
            data[1, :, 0] = 0
            data[1, :, 1] = 10
            data[2] = 0  # Explicitly reported empty sample, not silently included.
            path = directory / 'train.npz'
            np.savez_compressed(path, x_train=data.reshape(3, 4, 150), x_test=np.full((1, 4, 150), np.nan))
            pt, lp = directory / 'pt.json', directory / 'lp.json'
            cfg = config(path)
            pt.write_text(json.dumps(cfg), encoding='utf-8')
            cfg['model'] = 'model.transformer_downstream.Transformer'
            cfg['model_args']['input_var'] = [1, 1, 1]
            lp.write_text(json.dumps(cfg), encoding='utf-8')
            output = directory / 'results'
            with contextlib.redirect_stdout(io.StringIO()):
                audit.main(['--configs', str(pt), '--compare-configs', str(lp), '--view', 'raw',
                            '--output-dir', str(output), '--chunk-size', '1', '--progress-every', '0'])
            report = json.loads(next(output.glob('*.json')).read_text(encoding='utf-8'))
            self.assertTrue(report['protocol']['train_only'])
            self.assertTrue(report['protocol']['complete_train_scan'])
            self.assertEqual(report['counters']['skipped_all_zero'], 1)
            self.assertEqual(report['counters']['person0_empty'], 1)
            self.assertEqual(report['normalization_mismatch_configs'], [str(lp.resolve())])
            group = next(item for item in report['results'] if item['person_mode'] == 'person0'
                         and not item['self_shift'] and item['padding'] == 'active_person_frames')
            self.assertEqual(group['statistics']['count'], 100)
            np.testing.assert_array_equal(group['statistics']['mean'], [1, 1, 1])
            self.assertFalse(group['candidate_usable'])
            self.assertIsNone(group['proposed_input_var'])
            self.assertTrue((output / 'comparison.csv').is_file())
            self.assertEqual(json.loads(pt.read_text())['model_args']['input_var'], [4, 9, 16])

    def test_preview_and_train_split_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'train.npy'
            np.save(path, np.ones((3, 4, 150)))
            cfg_path = directory / 'pt.json'
            cfg = config(path)
            cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
            args = audit.get_parser().parse_args(['--configs', str(cfg_path), '--view', 'raw',
                                                  '--max-samples', '1', '--progress-every', '0'])
            result = audit.audit(cfg_path, args)
            self.assertFalse(result['protocol']['complete_train_scan'])
            cfg['train_feeder_args']['split'] = 'test'
            cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Only train'):
                audit.audit(cfg_path, args)

    @unittest.skipUnless(HAS_YAML, 'PyYAML is required for actual repository YAML checks')
    def test_actual_repository_yaml_references(self):
        cfg, identity = audit.load_config(ROOT / audit.DEFAULT_CONFIG)
        reference = audit.config_reference(cfg, identity)
        self.assertEqual(reference['person_mode'], 'person0')
        self.assertFalse(reference['self_shift'])
        self.assertEqual(reference['var'], [.0206, .0805, .0218])

    @unittest.skipUnless(HAS_TORCH, 'PyTorch is required for native preprocessing execution')
    def test_model_view_matches_real_feeder_clean_transform(self):
        import torch
        from feeder import tools
        cfg = config('train.npz')
        cfg['train_feeder_args'].update(random_rot=True, flip=True)
        sample = np.random.RandomState(15).normal(size=(3, 140, 25, 2)).astype(np.float32)
        process, metadata = audit.model_processor(cfg, seed=4, threads=1)
        actual = process(sample)
        np.random.seed(4)
        random.seed(4)
        torch.manual_seed(4)
        expected = tools.valid_crop_resize(sample, 140, [.5, 1.], 120)
        expected = tools.random_spatial_flip(expected, p=.5, dataset='ntu')
        expected = tools.random_rot(expected)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (3, 120, 25, 2))
        self.assertFalse(metadata['diffusion_or_flow_noise'])


if __name__ == '__main__':
    unittest.main()
