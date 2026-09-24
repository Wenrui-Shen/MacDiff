"""End-to-end checks for the read-only S->T group comparison."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import compare_fixed_clip_conditions as comparison


def small_model(**overrides):
    from model.transformer_macdiff_text import Transformer

    options = {
        'dim_feat': 8, 'dim_t_embed': 64, 'depth': 1,
        'decoder_depth': 1, 'num_heads': 2, 'num_frames': 8,
        'num_joints': 25, 't_patch_size': 4,
        'diff_prediction': 'noise', 'diff_steps': 10,
        'diff_noise_schedule': ['inverse_cosine', 1],
        'text_input_dim': 6, 'text_hidden_dim': 12,
        'text_decoder_hidden_dim': 12, 'text_decoder_depth': 2,
        'lambda_loss_uni': .02, 'uncond_ratio': 0.,
    }
    options.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return Transformer(**options)


class FixedClipComparisonTests(unittest.TestCase):
    def test_pair_changes_only_skeleton_and_reuses_noisy_text(self):
        model = small_model(
            text_target_mode='fixed_clip', lambda_text_to_skeleton=0.).eval()
        model.requires_grad_(False)
        raw = torch.randn(3, 8, 25, 3)
        aug = raw + .001 * torch.randn_like(raw)
        data = {
            'text_person_ids': torch.zeros(3, 3, dtype=torch.long),
            'text_positions': torch.arange(1, 4)[None].expand(3, -1),
        }
        valid = torch.ones(3, 4, dtype=torch.bool)
        memory = torch.randn(3, 4, 6)
        captured = []
        hook = model.text_noise_decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(tuple(value.clone() for value in inputs)))
        try:
            result = comparison.paired_trial(
                model, raw, aug, data, memory, valid, 5, .5, -1,
                np.array([1, 2, 0]), 17, 23)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 2)
        for index in (0, 1, 3, 4, 5, 6):
            torch.testing.assert_close(captured[0][index], captured[1][index], rtol=0, atol=0)
        torch.testing.assert_close(
            captured[1][2], captured[0][2][[1, 2, 0]], rtol=0, atol=0)
        self.assertEqual(set(result), {'s2t_all', 's2t_global', 's2t_local'})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_full_cli_checks_groups_and_writes_comparable_rows(self):
        from util.clip_text_cache import CACHE_FILES, file_identity

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(17)
            data_path = root / 'ntu.npz'
            skeleton = np.zeros((8, 8, 150), dtype=np.float32)
            skeleton[:, :, :75] = rng.randn(8, 8, 75).astype(np.float32)
            np.savez(data_path, x_train=skeleton,
                     y_train=np.eye(2)[np.arange(8) % 2])
            cache_dir = root / 'cache'
            cache_dir.mkdir()
            global_text = rng.randn(8, 2, 6).astype(np.float32)
            global_text[:, 1] = 0
            global_text[:, 0] /= np.linalg.norm(global_text[:, 0], axis=-1, keepdims=True)
            tokens = np.zeros((8, 2, 8, 6), dtype=np.float16)
            local = rng.randn(8, 3, 6).astype(np.float32)
            local /= np.linalg.norm(local, axis=-1, keepdims=True)
            tokens[:, 0, 1:4] = local.astype(np.float16)
            mask = np.zeros((8, 2, 8), dtype=bool)
            mask[:, 0, 1:4] = True
            ids = np.full((8, 2, 8), -1, dtype=np.int32)
            ids[:, 0, 1:4] = [1, 2, 7]
            for name, array in (
                    ('person_features', global_text),
                    ('person_valid', mask.any(axis=-1)),
                    ('token_features', tokens),
                    ('token_mask', mask),
                    ('token_ids', ids)):
                np.save(cache_dir / (name + '.npy'), array)
            manifest = {
                'protocol': 'macdiff_clip_token_cache_v2', 'complete': True,
                'sample_count': 8, 'feature_dim': 6, 'context_length': 8,
                'eos_token_id': 7, 'identity': {'data': file_identity(data_path)},
                'files': {name: file_identity(cache_dir / name) for name in CACHE_FILES},
            }
            (cache_dir / 'manifest.json').write_text(json.dumps(manifest))
            identity = {key: manifest[key] for key in ('protocol', 'identity', 'files')}
            common = {
                'dim_feat': 8, 'dim_t_embed': 64, 'depth': 1,
                'decoder_depth': 1, 'num_heads': 2, 'num_frames': 8,
                'num_joints': 25, 't_patch_size': 4,
                'diff_prediction': 'noise', 'diff_steps': 10,
                'diff_noise_schedule': ['inverse_cosine', 1],
                'text_input_dim': 6, 'text_hidden_dim': 12,
                'text_context_length': 8, 'text_decoder_hidden_dim': 12,
                'text_decoder_depth': 2, 'text_target_mode': 'fixed_clip',
                'one_person': True,
            }
            directories = {}
            for group, norm, weight in comparison.GROUPS + (comparison.OPTIONAL_GROUP,):
                directory = root / group
                directory.mkdir()
                directories[group] = directory
                options = dict(common, text_target_norm=norm)
                model = small_model(
                    **options, lambda_text_to_skeleton=0.,
                    lambda_skeleton_to_text=weight)
                saved = argparse.Namespace(
                    model='model.transformer_macdiff_text.Transformer',
                    model_args=options, text_target_mode='fixed_clip',
                    text_target_norm=norm, text_person_alignment=('per_person_v1', True),
                    text_share_skeleton_decoder=False,
                    text_training_weights=(0., weight),
                    feeder='feeder.feeder_ntu.Feeder',
                    train_feeder_args={'data_path': str(data_path), 'split': 'train',
                                       'window_size': 8, 'p_interval': [.95]},
                    text_cache=str(cache_dir), text_cache_identity=identity,
                    mask_ratio=.5, motion_aware_tau=-1)
                if group == 'rms':
                    # The model config still records RMS if older checkpoint
                    # args omitted the additional provenance field.
                    delattr(saved, 'text_target_norm')
                torch.save({'model': model.state_dict(), 'args': saved, 'epoch': 0},
                           directory / 'checkpoint-0.pth')
            with self.assertRaisesRegex(ValueError, 'wrong fixed-CLIP target mode'):
                comparison.checkpoint_args(directories['rms'] / 'checkpoint-0.pth', 'fixed')
            output = root / 'result'
            argv = [
                '--fixed-dir', str(directories['fixed']),
                '--st01-dir', str(directories['st01']),
                '--rms-dir', str(directories['rms']),
                '--epochs', '0', '--samples', '4', '--batch-size', '2',
                '--repeats', '1', '--timesteps', '1', '5', '--device', 'cpu',
                '--skip-text-cache-validation', '--output-dir', str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                comparison.main(argv)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary['sample_indices']), 4)
            self.assertEqual(len(summary['results']), 18)
            self.assertEqual({row['group'] for row in summary['results']},
                             {'fixed', 'st01', 'rms'})
            records = [json.loads(line) for line in
                       (output / 'paired_records.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 72)
            self.assertTrue(all(row['donor'] != row['sample'] for row in records))
            self.assertTrue(all(row['correct'] >= 0 and row['shuffled'] >= 0
                                for row in records))
            rows = (output / 'comparison.csv').read_text().splitlines()
            self.assertEqual(len(rows), 19)
            with self.assertRaises(FileExistsError):
                comparison.main(argv)
            four_group_output = root / 'four_group_result'
            four_group_argv = argv[:-2] + [
                '--rms-st01-dir', str(directories['rms_st01']),
                '--output-dir', str(four_group_output)]
            with contextlib.redirect_stdout(io.StringIO()):
                comparison.main(four_group_argv)
            four_group_summary = json.loads(
                (four_group_output / 'summary.json').read_text())
            self.assertEqual(len(four_group_summary['results']), 24)
            self.assertEqual({row['group'] for row in four_group_summary['results']},
                             {'fixed', 'st01', 'rms', 'rms_st01'})


if __name__ == '__main__':
    unittest.main()
