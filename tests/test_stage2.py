import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from main_pretrain_stage2 import (
    ExemplarProvider,
    checkpoint_schedule,
    load_or_create_exemplars,
    prepare_output,
    stage2_training_mode,
)
from feeder.feeder_stage2 import FeederStage2
from model.transformer_stage2 import (
    MacDiffDenseOSE,
    MacDiffStage2,
    transfer_macdiff_stage1,
)
from util.dense_ose_diagnostics import (
    DenseOSEJsonlLogger,
    assignment_distribution,
    prototype_geometry,
    select_balanced_indices,
)
from model.transformer_downstream import Transformer as DownstreamTransformer


class _LabelDataset(object):
    def __init__(self):
        self.label = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)


class _AugmentedExemplarDataset(object):
    def __init__(self):
        self.augmentation_calls = 0

    def get_base_sample(self, index):
        return np.full((3, 4, 25, 1), float(index), dtype=np.float32)

    def augment(self, sample):
        self.augmentation_calls += 1
        return sample + float(self.augmentation_calls)


class MacDiffStage2Test(unittest.TestCase):
    def _model(self):
        return MacDiffStage2(
            dim_in=3,
            dim_feat=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
            num_frames=8,
            num_joints=5,
            patch_size=1,
            t_patch_size=2,
            mask_ratio=0.5,
            one_person=True,
            input_mean=(0.0, 0.0, 0.0),
            input_var=(1.0, 1.0, 1.0),
            feature_dim=4,
            projector_hidden_dim=8,
            projector_layers=2,
            ose_separate_projector=True,
            cluster_temperature=0.4,
            sinkhorn_temperature=0.05,
            sinkhorn_iterations=3,
        )

    def _dense_model(self):
        return MacDiffDenseOSE(
            dim_in=3,
            dim_feat=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
            num_frames=8,
            num_joints=5,
            patch_size=1,
            t_patch_size=2,
            mask_ratio=0.0,
            one_person=True,
            input_mean=(0.0, 0.0, 0.0),
            input_var=(1.0, 1.0, 1.0),
            feature_dim=4,
            projector_hidden_dim=8,
            projector_layers=2,
            ose_separate_projector=True,
        )

    def test_stage2_training_mode_labels_ose_only(self):
        args = SimpleNamespace(
            resa_weight=0.0,
            ose_lambda=1.0,
            ose_mix_proto_weight=1.0,
            ose_mix_ins_weight=1.0,
        )
        self.assertEqual(
            stage2_training_mode(args), 'ose_only_separate_projector')

    def test_stage2_training_mode_labels_dense_ose(self):
        args = SimpleNamespace(
            mask_protocol='dense_ose_proto_ema_v1',
            resa_weight=0.0,
            ose_lambda=1.0,
            ose_mix_proto_weight=0.0,
            ose_mix_ins_weight=0.0,
        )
        self.assertEqual(
            stage2_training_mode(args), 'dense_ose_proto_ema_v1')

    def test_lp_backbone_schedule_is_independent_of_full_checkpoints(self):
        args = SimpleNamespace(
            epochs=100,
            save_interval=10,
            lp_checkpoint_epochs=[1, 2, 3, 5, 8, 10, 15, 20],
        )
        self.assertEqual(checkpoint_schedule(args, 1), (False, True))
        self.assertEqual(checkpoint_schedule(args, 8), (False, True))
        self.assertEqual(checkpoint_schedule(args, 10), (True, True))
        self.assertEqual(checkpoint_schedule(args, 11), (False, False))
        self.assertEqual(checkpoint_schedule(args, 100), (True, True))

    def test_transfer_loads_only_stage1_online_encoder(self):
        model = self._model()
        source = {}
        for index, (name, value) in enumerate(
                model.encoder_q.state_dict().items()):
            source[name] = torch.full_like(value, float(index + 1))
        source['decoder_pred.weight'] = torch.randn(3, 3)
        source['ose_memory.queue_features'] = torch.randn(4, 4)

        report = transfer_macdiff_stage1(model, {'model': source})

        self.assertEqual(
            report['encoder_tensors'], len(model.encoder_q.state_dict()))
        self.assertEqual(report['ignored_tensors'], 2)
        for name, value in model.encoder_q.state_dict().items():
            self.assertTrue(torch.equal(value, source[name]))
            self.assertTrue(torch.equal(
                value, model.encoder_k.state_dict()[name]))

    def test_transfer_rejects_missing_or_incompatible_tensor(self):
        model = self._model()
        source = {
            name: value.clone()
            for name, value in model.encoder_q.state_dict().items()
        }
        missing = dict(source)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, 'missing encoder tensor'):
            transfer_macdiff_stage1(model, {'model': missing})

        incompatible = dict(source)
        first_name = next(iter(incompatible))
        incompatible[first_name] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'has shape'):
            transfer_macdiff_stage1(model, {'model': incompatible})

    def test_dual_heads_and_ema_start_equal(self):
        model = self._model()
        for shared, semantic in zip(
                model.projector_q.state_dict().values(),
                model.ose_projector_q.state_dict().values()):
            self.assertTrue(torch.equal(shared, semantic))
        for online, teacher in zip(
                model.encoder_q.state_dict().values(),
                model.encoder_k.state_dict().values()):
            self.assertTrue(torch.equal(online, teacher))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.encoder_k.parameters()))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.projector_k.parameters()))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.ose_projector_k.parameters()))

    def test_joint_aware_pooling_preserves_joint_positions(self):
        model = self._model()
        tokens = torch.tensor([[[1.0], [3.0], [2.0], [4.0]]])
        # Flattening is temporal-major, so ids modulo five recover joints.
        visible_indices = torch.tensor([[0, 5, 1, 6]])
        pooled = model.encoder_q._joint_pool(
            tokens, visible_indices, joint_patches=5)
        expected = torch.tensor([[[2.0], [3.0], [0.0], [0.0], [0.0]]])
        self.assertTrue(torch.equal(pooled, expected))
        sample = torch.randn(2, 3, 8, 5, 1)
        features = model.encoder_q.forward_features(sample)
        self.assertEqual(tuple(features.shape), (2, 5 * 8))

    def test_per_joint_mask_keeps_equal_temporal_tokens_per_joint(self):
        model = self._model()
        encoder = model.encoder_q
        encoder.mask_strategy = 'per_joint_random'
        sample = torch.randn(4, 3, 8, 5, 1)
        indices = encoder.sample_mask_indices(sample)
        self.assertEqual(tuple(indices.shape), (4, 10))
        counts = torch.zeros(4, 5, dtype=torch.long)
        counts.scatter_add_(
            1, indices.remainder(5), torch.ones_like(indices))
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))
        self.assertTrue(all(
            torch.unique(row).numel() == row.numel() for row in indices))

        tokens = torch.randn(4, 20, 8)
        _, internal_indices = encoder._random_mask(tokens)
        internal_counts = torch.zeros(4, 5, dtype=torch.long)
        internal_counts.scatter_add_(
            1, internal_indices.remainder(5),
            torch.ones_like(internal_indices))
        self.assertTrue(torch.equal(
            internal_counts, torch.full_like(internal_counts, 2)))

    def test_exported_online_encoder_matches_downstream_keys(self):
        model = self._model()
        downstream = DownstreamTransformer(
            dim_in=3,
            num_classes=3,
            dim_feat=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
            num_frames=8,
            num_joints=5,
            patch_size=1,
            t_patch_size=2,
            protocol='linprobe',
            input_mean=(0.0, 0.0, 0.0),
            input_var=(1.0, 1.0, 1.0),
        )
        message = downstream.load_state_dict(
            model.encoder_q.state_dict(), strict=False)
        self.assertEqual(message.unexpected_keys, [])
        self.assertTrue(message.missing_keys)
        self.assertTrue(all(
            name.startswith('head.') for name in message.missing_keys))

    def test_joint_only_forward_and_gradient_isolation(self):
        torch.manual_seed(7)
        model = self._model()
        model.train()
        view_a = torch.randn(4, 3, 8, 5, 1)
        view_b = torch.randn(4, 3, 8, 5, 1)
        exemplar_joint = torch.randn(3, 3, 8, 5, 1)
        second_joint = torch.randn(3, 3, 8, 5, 1)
        mix_index = torch.tensor([1, 0, 3, 2])
        mix_beta = 0.3
        mixed = mix_beta * view_b + (1.0 - mix_beta) * view_a[mix_index]

        online_masks = []
        teacher_masks = []
        online_forward = model.encoder_q.forward_features
        teacher_forward = model.encoder_k.forward_features

        def record_online(skeleton, mask_indices=None):
            online_masks.append(
                None if mask_indices is None else mask_indices.clone())
            return online_forward(skeleton, mask_indices=mask_indices)

        def record_teacher(skeleton, mask_indices=None):
            teacher_masks.append(
                None if mask_indices is None else mask_indices.clone())
            return teacher_forward(skeleton, mask_indices=mask_indices)

        with mock.patch.object(
                model.encoder_q, 'forward_features',
                side_effect=record_online), mock.patch.object(
                    model.encoder_k, 'forward_features',
                    side_effect=record_teacher):
            losses = model(
                view_a,
                view_b,
                [exemplar_joint, second_joint],
                momentum=0.996,
                mixed_view=mixed,
                mix_index=mix_index,
                mix_beta=mix_beta,
                exemplar_mask_seed=123,
            )

        # q/k see the same visible tokens for each unlabeled view.
        self.assertTrue(torch.equal(online_masks[0], teacher_masks[0]))
        self.assertTrue(torch.equal(online_masks[1], teacher_masks[1]))
        # Exemplar encoding is Joint-only: the teacher has no exemplar calls.
        self.assertEqual(len(teacher_masks), 2)
        self.assertEqual(len(online_masks), 5)
        self.assertIsNotNone(online_masks[2])
        self.assertIsNotNone(online_masks[3])
        self.assertIsNone(online_masks[4])

        self.assertNotIn('queue_features', losses)
        self.assertTrue(torch.isfinite(losses['proto']))
        self.assertTrue(torch.allclose(
            losses['cluster'],
            losses['cluster_entropy'] + losses['cluster_kl']))
        joint_views = torch.randn(3, 2, 4)
        ensemble = model.ensemble_labeled_exemplars(joint_views)
        expected_ensemble = torch.nn.functional.normalize(
            torch.nn.functional.normalize(
                joint_views, dim=2).mean(dim=1), dim=1)
        self.assertTrue(torch.allclose(
            ensemble.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.allclose(
            ensemble, expected_ensemble, atol=1e-6))

        ose_objective = (
            losses['proto'] + losses['mix_proto'] + losses['mix_ins'])
        resa_to_ose = torch.autograd.grad(
            losses['cluster'], list(model.ose_projector_q.parameters()),
            retain_graph=True, allow_unused=True)
        ose_to_resa = torch.autograd.grad(
            ose_objective,
            list(model.projector_q.parameters())
            + list(model.predictor.parameters()),
            retain_graph=True, allow_unused=True)
        self.assertTrue(all(value is None for value in resa_to_ose))
        self.assertTrue(all(value is None for value in ose_to_resa))

        encoder_resa = torch.autograd.grad(
            losses['cluster'], list(model.encoder_q.parameters()),
            retain_graph=True, allow_unused=True)
        encoder_ose = torch.autograd.grad(
            ose_objective, list(model.encoder_q.parameters()),
            retain_graph=True, allow_unused=True)
        self.assertTrue(any(value is not None for value in encoder_resa))
        self.assertTrue(any(value is not None for value in encoder_ose))

    def test_dense_ose_has_one_online_graph_and_ema_prototypes(self):
        torch.manual_seed(11)
        model = self._dense_model()
        model.train()
        exemplar_views = [
            torch.randn(3, 3, 8, 5, 1),
            torch.randn(3, 3, 8, 5, 1),
        ]
        prototypes = model.refresh_ema_prototypes(exemplar_views)
        self.assertEqual(tuple(prototypes.shape), (3, 4))
        self.assertFalse(prototypes.requires_grad)
        self.assertTrue(torch.allclose(
            prototypes.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertFalse(hasattr(model, 'predictor'))
        self.assertFalse(hasattr(model, 'projector_q'))

        online_calls = []
        teacher_calls = []
        online_forward = model.encoder_q.forward_features
        teacher_forward = model.encoder_k.forward_features

        def record_online(skeleton, mask_indices=None):
            online_calls.append(mask_indices)
            return online_forward(skeleton, mask_indices=mask_indices)

        def record_teacher(skeleton, mask_indices=None):
            teacher_calls.append(mask_indices)
            return teacher_forward(skeleton, mask_indices=mask_indices)

        with mock.patch.object(
                model.encoder_q, 'forward_features',
                side_effect=record_online), mock.patch.object(
                    model.encoder_k, 'forward_features',
                    side_effect=record_teacher):
            losses = model(
                torch.randn(4, 3, 8, 5, 1),
                torch.randn(4, 3, 8, 5, 1),
                prototypes,
                labels=torch.tensor([0, 1, 2, 0]),
            )

        self.assertEqual(online_calls, [None])
        self.assertEqual(teacher_calls, [None])
        self.assertTrue(torch.isfinite(losses['proto']))
        self.assertEqual(float(losses['mix_proto']), 0.0)
        self.assertEqual(float(losses['mix_ins']), 0.0)
        for name in (
                'ose_target_entropy_p10',
                'ose_target_entropy_p50',
                'ose_target_entropy_p90',
                'ose_target_confidence_p10',
                'ose_target_confidence_p50',
                'ose_target_confidence_p90',
                'ose_teacher_accuracy',
                'ose_student_accuracy',
                'ose_teacher_student_agreement',
                'ose_teacher_true_class_probability',
                'ose_teacher_margin'):
            self.assertTrue(torch.isfinite(losses[name]))
        self.assertEqual(
            tuple(losses['_ose_teacher_assignment'].shape), (4,))
        self.assertEqual(
            tuple(losses['_ose_teacher_probability_sum'].shape), (3,))
        losses['proto'].backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.encoder_q.parameters()))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.ose_projector_q.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder_k.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.ose_projector_k.parameters()))

    def test_dense_diagnostic_helpers_are_balanced_and_track_drift(self):
        labels = np.repeat(np.arange(3), 4)
        selected = select_balanced_indices(
            labels, excluded_indices=[0, 4, 8], max_samples=6,
            seed=3, num_classes=3)
        selected_labels = labels[selected]
        self.assertEqual(
            np.bincount(selected_labels, minlength=3).tolist(), [2, 2, 2])

        prototypes = torch.eye(3)
        first = prototype_geometry(prototypes)
        second = prototype_geometry(prototypes, prototypes.clone())
        self.assertIsNone(first['epoch_drift_cosine_mean'])
        self.assertAlmostEqual(second['epoch_drift_cosine_mean'], 1.0)
        distribution = assignment_distribution(torch.tensor([2, 0, 2]))
        self.assertAlmostEqual(distribution['used_fraction'], 2.0 / 3.0)
        self.assertEqual(distribution['histogram'], [2, 0, 2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostics.jsonl'
            DenseOSEJsonlLogger(path).write(
                'test_event', finite=1.0, nonfinite=float('inf'))
            with path.open('r', encoding='utf-8') as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record['event'], 'test_event')
            self.assertEqual(record['finite'], 1.0)
            self.assertIsNone(record['nonfinite'])

    def test_extra_joint_view_preserves_bn_and_retains_gradient(self):
        # Keep the forward test on ordinary CPU BatchNorm. PyTorch 1.8
        # rejects SyncBatchNorm CPU forwards even without initialized DDP;
        # production SyncBatchNorm runs on one CUDA device per DDP process.
        model = self._model()
        model.train()
        projector = model.ose_online_projector
        before = {
            name: value.clone()
            for name, value in projector.state_dict().items()
            if ('running_' in name or 'num_batches_tracked' in name)
        }
        projected = model._online_exemplar_projection(
            torch.randn(3, 3, 8, 5, 1), preserve_bn=True)
        gradients = torch.autograd.grad(
            projected.sum(), list(projector.parameters()),
            allow_unused=True)
        self.assertTrue(any(value is not None for value in gradients))
        after = projector.state_dict()
        for name, value in before.items():
            self.assertTrue(torch.equal(value, after[name]))

    def test_stage2_heads_convert_to_sync_batch_norm(self):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model())
        self.assertTrue(any(
            isinstance(module, torch.nn.SyncBatchNorm)
            for module in model.ose_online_projector.modules()))
        self.assertTrue(any(
            isinstance(module, torch.nn.SyncBatchNorm)
            for module in model.projector_q.modules()))

    def test_k2_exemplar_provider_builds_two_joint_views(self):
        dataset = _AugmentedExemplarDataset()
        provider = ExemplarProvider(dataset, [0, 1, 2])
        views = provider.joint_views(torch.device('cpu'), num_views=2)
        self.assertEqual(dataset.augmentation_calls, 6)
        self.assertEqual(len(views), 2)
        self.assertTrue(all(tuple(view.shape) == (3, 3, 4, 25, 1)
                            for view in views))
        self.assertFalse(torch.equal(views[0], views[1]))

    def test_two_views_draw_augmentations_independently(self):
        feeder = FeederStage2.__new__(FeederStage2)
        feeder.augmentation_methods = (
            'temporal_crop', 'shear', 'rotation')
        feeder.augmentation_probability = 0.5
        increments = {
            'temporal_crop': 1.0,
            'shear': 10.0,
            'rotation': 100.0,
        }
        feeder._apply_augmentation = (
            lambda sample, name: sample + increments[name])
        sample = np.zeros((3, 8, 5, 1), dtype=np.float32)
        with mock.patch(
                'feeder.feeder_stage2.random.random',
                side_effect=[0.1, 0.9, 0.9, 0.9, 0.1, 0.9]):
            first = feeder.augment(sample)
            second = feeder.augment(sample)
        self.assertTrue(np.all(first == 1.0))
        self.assertTrue(np.all(second == 10.0))

    def test_exemplar_cache_records_and_validates_seed(self):
        dataset = _LabelDataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seed0.json'
            classes, indices = load_or_create_exemplars(
                dataset, path, seed=0, num_classes=3)
            self.assertEqual(classes, [0, 1, 2])
            self.assertEqual(len(indices), 3)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['seed'], 0)
            self.assertEqual(payload['num_samples'], 6)
            self.assertEqual(
                load_or_create_exemplars(
                    dataset, path, seed=0, num_classes=3),
                (classes, indices))
            with self.assertRaisesRegex(ValueError, 'seed mismatch'):
                load_or_create_exemplars(
                    dataset, path, seed=1, num_classes=3)

    def test_fresh_output_replaces_only_a_named_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            output_root = workspace / 'output_dir'
            run = output_root / 'stage2_run'
            run.mkdir(parents=True)
            stale = run / 'stale.txt'
            stale.write_text('old', encoding='utf-8')
            args = SimpleNamespace(
                output_dir=str(run),
                log_dir=str(run / 'tensorboard'),
                resume='',
            )
            with mock.patch.object(Path, 'cwd', return_value=workspace):
                prepare_output(args)
            self.assertTrue(run.is_dir())
            self.assertTrue((run / 'tensorboard').is_dir())
            self.assertFalse(stale.exists())

            protected = SimpleNamespace(
                output_dir=str(output_root),
                log_dir=str(output_root / 'tensorboard'),
                resume='',
            )
            with mock.patch.object(Path, 'cwd', return_value=workspace):
                with self.assertRaisesRegex(
                        RuntimeError, 'only allowed for a child directory'):
                    prepare_output(protected)

            preserved = run / 'resume.txt'
            preserved.write_text('keep', encoding='utf-8')
            args.resume = str(run / 'checkpoint-010.pth')
            with mock.patch.object(Path, 'cwd', return_value=workspace):
                prepare_output(args)
            self.assertTrue(preserved.exists())


if __name__ == '__main__':
    unittest.main()
