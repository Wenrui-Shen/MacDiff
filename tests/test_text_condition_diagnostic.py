"""Checks for paired controls and the complete diagnostic CLI on synthetic data."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_text_conditioning as diag
from tests import test_macdiff_text as model_tests


class DiagnosticTests(unittest.TestCase):
    def test_small_variance_need_not_destroy_centered_relations(self):
        x = np.random.RandomState(3).randn(30, 8)
        y = 10 + .001 * x
        ids, labels = np.arange(30), np.arange(30) % 3
        before, after = diag.geometry(x, ids, labels), diag.geometry(y, ids, labels)
        self.assertLess(after['mean_channel_variance'], before['mean_channel_variance'] * 1e-5)
        retention = diag.relation_retention(x, y, ids)
        self.assertAlmostEqual(retention['centered']['cosine_pearson'], 1.)
        self.assertAlmostEqual(retention['centered']['neighbor_overlap'], 1.)
        self.assertAlmostEqual(before['centered_effective_rank'], after['centered_effective_rank'])

    def test_matching_and_derangement_preserve_structure(self):
        masks = np.zeros((11, 2, 8), dtype=bool)
        masks[:9, 0, 1:4] = True
        masks[9:, 0, 1:6] = True
        batches, excluded = diag.paired_batches(np.arange(11), masks, 4)
        self.assertEqual(excluded, [])
        self.assertEqual(sorted(i for b in batches for i in b), list(range(11)))
        for batch in batches:
            p = diag.derangement(len(batch), np.random.RandomState(4))
            self.assertTrue(np.all(p != np.arange(len(batch))))
            np.testing.assert_array_equal(np.sort(p), np.arange(len(batch)))
            for i in batch:
                np.testing.assert_array_equal(masks[i], masks[batch[0]])

    @staticmethod
    def trial_fixture():
        model = model_tests.TextStage1Tests.model(share_skeleton_decoder=True).eval().requires_grad_(False)
        raw, aug = torch.randn(3, 8, 25, 3), torch.randn(3, 8, 25, 3)
        data = dict(text_token_mask=torch.ones(3, 4, dtype=torch.bool),
                    text_person_ids=torch.zeros(3, 4, dtype=torch.long),
                    text_positions=torch.arange(1, 5)[None].expand(3, -1).clone())
        r = model.text_remap(torch.randn(3, 6))
        local = model.remap_tokens(torch.randn(3, 4, 6), data['text_token_mask'],
                                   data['text_person_ids'], data['text_positions'])
        return model, raw, aug, data, r, local, torch.cat([r[:, None], local], 1), torch.ones(3, 5, dtype=torch.bool)

    def test_identity_control_has_exactly_zero_delta_and_no_mutation(self):
        diag.seed_all(7)
        args = self.trial_fixture()
        original = {k: v.clone() for k, v in args[0].state_dict().items()}
        result = diag.condition_trial(*args, timestep=5, mask_ratio=.5, motion_tau=-1,
                                      permutation=np.arange(3))
        self.assertEqual(len(result), 6)
        for good, bad, weight, change in result.values():
            np.testing.assert_array_equal(good, bad)
            np.testing.assert_array_equal(change, np.zeros(3))
        for key, value in args[0].state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in args[0].parameters()))

    def test_real_shuffle_changes_conditions_but_not_noisy_inputs(self):
        diag.seed_all(8)
        args = self.trial_fixture()
        captured = []
        handle = args[0].text_noise_decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(tuple(x.clone() for x in inputs)))
        result = diag.condition_trial(*args, timestep=5, mask_ratio=.5, motion_tau=-1,
                                      permutation=np.array([1, 2, 0]))
        handle.remove()
        self.assertEqual(len(captured), 2)
        for k in (0, 1, 3, 4, 5, 6):
            torch.testing.assert_close(captured[0][k], captured[1][k], rtol=0, atol=0)
        torch.testing.assert_close(captured[1][2], captured[0][2][[1, 2, 0]], rtol=0, atol=0)
        self.assertGreater(result['s2t_all'][3].sum(), 0)
        args[3]['text_positions'][0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'structure'):
            diag.condition_trial(*args, timestep=5, mask_ratio=.5, motion_tau=-1,
                                  permutation=np.array([1, 2, 0]))

    def test_weighted_summary_preserves_token_average(self):
        rows = [dict(test='s2t_all', t=5, sample=i, batch=i, repeat=0, weight=w,
                     correct=c, shuffled=c+2, prediction_change=1., same_label=False)
                for i, (w, c) in enumerate([(1, 1), (3, 3)])]
        result = diag.summarize_pairs(rows, 3)[0]
        self.assertEqual(result['correct_mse'], 2.5)
        self.assertEqual(result['delta'], 2.)
        self.assertEqual(result['delta_cluster_bootstrap_95ci'], [2., 2.])

    def test_complete_cli_with_checkpoint_and_verified_cache(self):
        from util.clip_text_cache import CACHE_FILES, file_identity
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            rng = np.random.RandomState(12)
            data = root / 'ntu.npz'
            np.savez(data, x_train=rng.randn(8, 12, 150).astype(np.float32),
                     y_train=np.eye(2)[np.arange(8) % 2])
            glob = np.zeros((8, 2, 6), dtype=np.float32)
            glob[:, 0] = diag.unit(rng.randn(8, 6)).astype(np.float32)
            tok = np.zeros((8, 2, 8, 6), dtype=np.float16)
            tok[:, 0, 1:4] = diag.unit(rng.randn(24, 6)).reshape(8, 3, 6).astype(np.float16)
            mask = np.zeros((8, 2, 8), dtype=bool)
            mask[:, 0, 1:4] = True
            ids = np.full((8, 2, 8), -1, dtype=np.int32)
            ids[:, 0, 1:4] = [1, 2, 7]
            for name, value in [('person_features', glob), ('person_valid', mask.any(-1)),
                                ('token_features', tok), ('token_mask', mask), ('token_ids', ids)]:
                np.save(cache / (name + '.npy'), value)
            manifest = dict(protocol='macdiff_clip_token_cache_v2', complete=True, sample_count=8,
                            feature_dim=6, context_length=8, eos_token_id=7,
                            identity={'data': file_identity(data)},
                            files={name: file_identity(cache / name) for name in CACHE_FILES})
            (cache / 'manifest.json').write_text(json.dumps(manifest))
            options = dict(dim_feat=8, dim_t_embed=64, depth=1, decoder_depth=1,
                           num_heads=2, num_frames=8, num_joints=25, t_patch_size=4,
                           diff_prediction='noise', diff_steps=10, diff_noise_schedule=['inverse_cosine', 1],
                           text_input_dim=6, text_hidden_dim=12, text_context_length=8,
                           text_decoder_hidden_dim=12, text_decoder_depth=2,
                           share_skeleton_decoder=True, one_person=True)
            model = model_tests.TextStage1Tests.model(**options)
            saved = argparse.Namespace(model='model.transformer_macdiff_text.Transformer', model_args=options,
                    text_person_alignment=('per_person_v1', True), text_share_skeleton_decoder=True,
                    text_training_weights=(1., 1.), text_cache=str(cache),
                    text_cache_identity={k: manifest[k] for k in ('protocol', 'identity', 'files')},
                    feeder='feeder.feeder_ntu.Feeder', mask_ratio=.5,
                    train_feeder_args=dict(data_path=str(data), split='train', window_size=8, p_interval=[.95]))
            checkpoint = root / 'checkpoint-200.pth'
            torch.save(dict(model=model.state_dict(), args=saved, epoch=200), checkpoint)
            outputs = []
            for trial in range(2):
                output = root / ('out' + str(trial))
                argv = ['diagnose', '--checkpoint', str(checkpoint), '--device', 'cpu', '--samples', '8',
                        '--batch-size', '4', '--repeats', '1', '--timesteps', '1', '5', '8', '--output-dir', str(output)]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    diag.main()
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['conditioning_sample_count'], 8)
                self.assertEqual(len(summary['conditioning']), 18)
                self.assertEqual(summary['geometry']['local_clip']['vectors'], 16)
                outputs.append((output / 'paired_records.jsonl').read_text())
            self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
