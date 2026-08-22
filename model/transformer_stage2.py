"""MacDiff adapter for the RSDG ReSA+OSE Stage2 objective.

Stage2 deliberately owns a fresh online/EMA encoder pair and fresh heads.  A
Stage1 MacDiff checkpoint is transferred by :func:`transfer_macdiff_stage1`;
the diffusion decoder, Stage1 momentum encoder, OSE memory and optimizer state
are never imported.  The current prototype path uses independently augmented
Joint exemplars only and ensembles their normalized anchors.
"""

import copy
import math
from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .transformer_downstream import Block, SkeleEmbed, trunc_normal_


@torch.no_grad()
def _concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def _distributed_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _distributed_world_size():
    return (dist.get_world_size()
            if dist.is_available() and dist.is_initialized() else 1)


@contextmanager
def _fixed_rng(seed, device):
    if seed is None:
        yield
        return
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        if device.type == 'cuda':
            torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))
        yield


def _build_projector(input_dim, hidden_dim, output_dim, num_layers):
    if num_layers < 2:
        raise ValueError('Stage2 projector requires at least two layers')
    layers = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ])
        in_dim = hidden_dim
    layers.extend([
        nn.Linear(in_dim, output_dim),
        nn.BatchNorm1d(output_dim, affine=False),
    ])
    return nn.Sequential(*layers)


def _build_predictor(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, input_dim),
    )


class MacDiffStage2Encoder(nn.Module):
    """MacDiff skeleton encoder with a joint-aware feature adapter.

    Stage2 keeps 10% of skeleton tokens, pools visible temporal tokens within
    each joint, and flattens the joint grid as downstream ``linprobe2`` does.
    Explicit mask indices align related online/EMA paths.
    """

    def __init__(
        self,
        dim_in=3,
        dim_feat=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        num_frames=120,
        num_joints=25,
        patch_size=1,
        t_patch_size=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        mask_ratio=0.9,
        one_person=True,
        input_mean=(-0.0058, -0.1333, -0.0246),
        input_var=(0.0206, 0.0805, 0.0218),
        self_shift=False,
    ):
        super().__init__()
        if not 0.0 <= float(mask_ratio) < 1.0:
            raise ValueError('Stage2 mask_ratio must be in [0, 1)')
        if len(input_mean) != dim_in or len(input_var) != dim_in:
            raise ValueError('input_mean/input_var must match dim_in')
        if any(float(value) <= 0 for value in input_var):
            raise ValueError('input_var entries must be positive')

        self.dim_feat = int(dim_feat)
        self.mask_ratio = float(mask_ratio)
        self.one_person = bool(one_person)
        self.input_mean = tuple(float(value) for value in input_mean)
        self.input_std = tuple(math.sqrt(float(value)) for value in input_var)
        self.self_shift = bool(self_shift)
        self.num_joint_patches = int(num_joints) // int(patch_size)
        self.output_dim = self.num_joint_patches * self.dim_feat
        self.num_tokens = (
            (int(num_frames) // int(t_patch_size))
            * self.num_joint_patches)

        self.joints_embed = SkeleEmbed(
            dim_in, dim_feat, num_frames, num_joints, patch_size,
            t_patch_size)
        dpr = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[index],
                norm_layer=nn.LayerNorm,
            )
            for index in range(depth)
        ])
        self.norm = nn.LayerNorm(dim_feat)
        self.temp_embed = nn.Parameter(torch.zeros(
            1, num_frames // t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1, num_joints // patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def sample_mask_indices(self, skeleton):
        """Draw one visible-token set that can be reused by q/k branches."""
        if self.mask_ratio <= 0.0:
            return None
        if skeleton.ndim != 5:
            raise ValueError('Skeleton input must have shape [N,C,T,V,M]')
        people = 1 if self.one_person else skeleton.shape[-1]
        batch_size = skeleton.shape[0] * people
        keep = round(self.num_tokens * (1.0 - self.mask_ratio))
        if keep <= 0:
            raise ValueError('Stage2 mask_ratio leaves no visible token')
        return torch.argsort(torch.rand(
            batch_size, self.num_tokens, device=skeleton.device), dim=1
        )[:, :keep]

    def _random_mask(self, tokens, mask_indices=None):
        if self.mask_ratio <= 0.0:
            ids = torch.arange(
                tokens.shape[1], device=tokens.device, dtype=torch.long
            ).unsqueeze(0).expand(tokens.shape[0], -1)
            return tokens, ids
        batch_size, length, dim = tokens.shape
        keep = round(length * (1.0 - self.mask_ratio))
        if keep <= 0:
            raise ValueError('Stage2 mask_ratio leaves no visible token')
        if mask_indices is None:
            mask_indices = torch.argsort(torch.rand(
                batch_size, length, device=tokens.device), dim=1
            )[:, :keep]
        if tuple(mask_indices.shape) != (batch_size, keep):
            raise ValueError(
                'Stage2 mask indices must have shape [{},{}]'.format(
                    batch_size, keep))
        ids = mask_indices.to(device=tokens.device, dtype=torch.long)
        visible = torch.gather(
            tokens, 1, ids.unsqueeze(-1).expand(-1, -1, dim))
        return visible, ids

    @staticmethod
    def _joint_pool(tokens, visible_indices, joint_patches):
        """Average visible temporal tokens per original joint position."""
        if tokens.ndim != 3 or visible_indices.ndim != 2:
            raise ValueError('Joint pooling expects [N,L,D] tokens and [N,L] ids')
        if tokens.shape[:2] != visible_indices.shape:
            raise ValueError('Joint pooling token and index shapes must align')
        batch_size, _, dim = tokens.shape
        joint_ids = visible_indices.remainder(joint_patches)
        expanded_ids = joint_ids.unsqueeze(-1).expand(-1, -1, dim)
        joint_sums = tokens.new_zeros(
            batch_size, joint_patches, dim
        ).scatter_add(1, expanded_ids, tokens)
        joint_counts = tokens.new_zeros(
            batch_size, joint_patches, 1
        ).scatter_add(
            1, joint_ids.unsqueeze(-1),
            tokens.new_ones(batch_size, tokens.shape[1], 1))
        return joint_sums / joint_counts.clamp_min(1.0)

    def forward_features(self, skeleton, mask_indices=None):
        if skeleton.ndim != 5:
            raise ValueError('Skeleton input must have shape [N,C,T,V,M]')
        if self.one_person:
            skeleton = skeleton[..., :1]
        batch_size, channels, frames, joints, people = skeleton.shape
        if channels != len(self.input_mean):
            raise ValueError('Skeleton channel count does not match encoder')

        skeleton = skeleton.permute(0, 4, 2, 3, 1).contiguous().view(
            batch_size * people, frames, joints, channels)
        if self.self_shift:
            skeleton = skeleton - skeleton.mean(
                dim=(1, 2), keepdim=True)
        mean = skeleton.new_tensor(self.input_mean).view(1, 1, 1, channels)
        std = skeleton.new_tensor(self.input_std).view(1, 1, 1, channels)
        skeleton = (skeleton - mean) / std

        tokens = self.joints_embed(skeleton)
        _, temporal_patches, joint_patches, dim = tokens.shape
        tokens = (
            tokens
            + self.pos_embed[:, :, :joint_patches]
            + self.temp_embed[:, :temporal_patches]
        ).reshape(batch_size * people, temporal_patches * joint_patches, dim)
        tokens, visible_indices = self._random_mask(
            tokens, mask_indices=mask_indices)

        for block in self.blocks:
            tokens = block(tokens)
        joint_features = self._joint_pool(
            self.norm(tokens), visible_indices, joint_patches)
        features = joint_features.reshape(
            batch_size, people, joint_patches * dim)
        return features.mean(dim=1)

    def forward(self, skeleton):
        return self.forward_features(skeleton)


class MacDiffStage2(nn.Module):
    """Dual-space ReSA+OSE Stage2 using a shared MacDiff encoder."""

    def __init__(
        self,
        feature_dim=256,
        projector_hidden_dim=2048,
        projector_layers=3,
        ose_separate_projector=True,
        cluster_temperature=0.4,
        sinkhorn_temperature=0.05,
        sinkhorn_iterations=3,
        **encoder_args
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.ose_separate_projector = bool(ose_separate_projector)
        self.cluster_temperature = float(cluster_temperature)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

        self.encoder_q = MacDiffStage2Encoder(**encoder_args)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        input_dim = self.encoder_q.output_dim
        self.projector_q = _build_projector(
            input_dim, projector_hidden_dim, feature_dim, projector_layers)
        self.projector_k = copy.deepcopy(self.projector_q)
        self.predictor = _build_predictor(feature_dim, projector_hidden_dim)
        if self.ose_separate_projector:
            self.ose_projector_q = copy.deepcopy(self.projector_q)
            self.ose_projector_k = copy.deepcopy(self.projector_q)

        # Keep the reference Stage2 head initialization: every Linear/BN uses
        # PyTorch's native constructor defaults.  The Dual OSE head is then an
        # exact copy of the native ReSA head before the first optimizer step.
        self.reset_momentum_encoder()

    @property
    def ose_online_projector(self):
        return (self.ose_projector_q if self.ose_separate_projector
                else self.projector_q)

    @property
    def ose_teacher_projector(self):
        return (self.ose_projector_k if self.ose_separate_projector
                else self.projector_k)

    @torch.no_grad()
    def reset_momentum_encoder(self):
        if self.ose_separate_projector:
            self.ose_projector_q.load_state_dict(
                self.projector_q.state_dict())
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        if self.ose_separate_projector:
            self.ose_projector_k.load_state_dict(
                self.ose_projector_q.state_dict())
        for module in (
                self.encoder_k, self.projector_k,
                self.ose_teacher_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, momentum):
        momentum = float(momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('EMA momentum must be in [0, 1]')
        module_pairs = [
            (self.encoder_q, self.encoder_k),
            (self.projector_q, self.projector_k),
        ]
        if self.ose_separate_projector:
            module_pairs.append(
                (self.ose_projector_q, self.ose_projector_k))
        for online_module, teacher_module in module_pairs:
            for online, teacher in zip(
                    online_module.parameters(), teacher_module.parameters()):
                teacher.data.mul_(momentum).add_(
                    online.data, alpha=1.0 - momentum)

    @staticmethod
    def soft_cross_entropy(logits, target):
        return -(
            target * F.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()

    @torch.no_grad()
    def sinkhorn_knopp(self, scores):
        logits = scores / max(self.sinkhorn_temperature, 1e-12)
        logits = logits - logits.max()
        assignment = torch.exp(logits).t()
        assignment /= assignment.sum().clamp_min(1e-12)
        num_samples = assignment.shape[1]
        num_clusters = assignment.shape[0]
        for _ in range(self.sinkhorn_iterations):
            assignment /= assignment.sum(
                dim=1, keepdim=True).clamp_min(1e-12)
            assignment /= num_clusters
            assignment /= assignment.sum(
                dim=0, keepdim=True).clamp_min(1e-12)
            assignment /= num_samples
        assignment *= num_samples
        return assignment.t().detach()

    @staticmethod
    def ensemble_labeled_exemplars(joint_views):
        if joint_views.ndim != 3:
            raise ValueError('Joint views must have shape [C,K,D]')
        if joint_views.shape[1] < 1:
            raise ValueError('At least one Joint exemplar view is required')
        joint_views = F.normalize(joint_views, dim=2)
        return F.normalize(joint_views.mean(dim=1), dim=1)

    def _online_projection(self, skeleton, ose_branch=False,
                           mask_indices=None):
        features = self.encoder_q.forward_features(
            skeleton, mask_indices=mask_indices)
        projector = (
            self.ose_online_projector if ose_branch else self.projector_q)
        return F.normalize(projector(features), dim=1)

    def _online_exemplar_projection(self, skeleton, preserve_bn=False,
                                    mask_indices=None):
        if not preserve_bn:
            return self._online_projection(
                skeleton, ose_branch=True, mask_indices=mask_indices)
        projector = self.ose_online_projector
        bn_state = []
        for module in (self.encoder_q, projector):
            for child in module.modules():
                if isinstance(child, (
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.SyncBatchNorm)):
                    bn_state.append((child, child.track_running_stats))
                    # Use batch statistics and retain gradients without
                    # changing long-term BN buffers for extra Joint views.
                    child.track_running_stats = False
        try:
            return self._online_projection(
                skeleton, ose_branch=True, mask_indices=mask_indices)
        finally:
            for child, track_running_stats in bn_state:
                child.track_running_stats = track_running_stats

    def forward(
        self,
        view_a,
        view_b,
        exemplar_views,
        momentum=0.996,
        ose_tau_s=0.1,
        ose_tau_t=0.04,
        mixed_view=None,
        mix_index=None,
        mix_beta=None,
        exemplar_mask_seed=None,
    ):
        if not isinstance(exemplar_views, (tuple, list)):
            raise ValueError('exemplar_views must be a list or tuple')
        if not exemplar_views:
            raise ValueError('Stage2 requires at least one Joint view')
        class_count = None
        for joint in exemplar_views:
            if not torch.is_tensor(joint) or joint.ndim != 5:
                raise ValueError('Every exemplar view must be a Joint tensor')
            if class_count is None:
                class_count = joint.shape[0]
            if joint.shape[0] != class_count:
                raise ValueError('All Joint views must align by class')
        if mixed_view is None or mix_index is None or mix_beta is None:
            raise ValueError('Stage2 mixed losses require mixed inputs')

        # Each view reuses one visible-token set across online and EMA. Drawing
        # unrelated 10%-visible subsets would leave only 7.5 common tokens in
        # expectation and flatten the ReSA relation target.
        view_masks = [
            self.encoder_q.sample_mask_indices(view_a),
            self.encoder_q.sample_mask_indices(view_b),
        ]
        raw_h = [
            self.encoder_q.forward_features(
                view_a, mask_indices=view_masks[0]),
            self.encoder_q.forward_features(
                view_b, mask_indices=view_masks[1]),
        ]
        online_h = [F.normalize(value, dim=1) for value in raw_h]
        raw_z = [self.projector_q(value) for value in raw_h]
        online_z = [F.normalize(value, dim=1) for value in raw_z]
        online_q = [
            F.normalize(self.predictor(value), dim=1) for value in raw_z]
        if self.ose_separate_projector:
            online_ose_z = [
                F.normalize(self.ose_projector_q(value), dim=1)
                for value in raw_h
            ]
        else:
            online_ose_z = online_z

        with _fixed_rng(exemplar_mask_seed, view_a.device):
            # Each augmented Joint view has its own visible-token set. Online
            # processing is the only exemplar branch; Motion/Bone and their
            # EMA projections are intentionally absent.
            exemplar_view_masks = [
                self.encoder_q.sample_mask_indices(joint)
                for joint in exemplar_views
            ]
            online_joint_anchors = [
                self._online_exemplar_projection(
                    joint, preserve_bn=(index > 0),
                    mask_indices=exemplar_view_masks[index])
                for index, joint in enumerate(exemplar_views)
            ]

        with torch.no_grad():
            self.momentum_update(momentum)
            teacher_raw_h = [
                self.encoder_k.forward_features(
                    view_a, mask_indices=view_masks[0]),
                self.encoder_k.forward_features(
                    view_b, mask_indices=view_masks[1]),
            ]
            teacher_h = [
                F.normalize(value, dim=1) for value in teacher_raw_h]
            teacher_z = [
                F.normalize(self.projector_k(value), dim=1)
                for value in teacher_raw_h
            ]
            teacher_ose_z = [
                F.normalize(self.ose_teacher_projector(value), dim=1)
                for value in teacher_raw_h
            ]
        global_online_h = _concat_all_gather(online_h[0].detach())
        global_teacher_h = _concat_all_gather(teacher_h[0].detach())
        global_assignment = self.sinkhorn_knopp(torch.matmul(
            global_online_h, global_teacher_h.t()))
        local_batch = view_a.shape[0]
        expected_global_batch = local_batch * _distributed_world_size()
        if (global_online_h.shape[0] != expected_global_batch or
                global_teacher_h.shape[0] != expected_global_batch):
            raise RuntimeError(
                'Every DDP rank must contribute the same Stage2 batch size')
        row_start = _distributed_rank() * local_batch
        assignment = global_assignment[row_start:row_start + local_batch]
        global_teacher_z = [
            _concat_all_gather(value.detach()) for value in teacher_z]
        cluster_loss = online_q[0].new_zeros(())
        terms = 0
        for online_index in range(2):
            for teacher_index in range(2):
                if online_index == teacher_index:
                    continue
                logits = torch.matmul(
                    online_q[online_index],
                    global_teacher_z[teacher_index].t()
                ) / max(self.cluster_temperature, 1e-12)
                cluster_loss = cluster_loss + self.soft_cross_entropy(
                    logits, assignment)
                terms += 1
        cluster_loss = cluster_loss / max(terms, 1)
        cluster_entropy = -(
            assignment * assignment.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        cluster_kl = cluster_loss - cluster_entropy

        joint_views = torch.stack(online_joint_anchors, dim=1)
        prototypes = self.ensemble_labeled_exemplars(joint_views)
        student_logits = torch.matmul(
            online_ose_z[1], prototypes.t()) / max(float(ose_tau_s), 1e-12)
        teacher_logits = torch.matmul(
            teacher_ose_z[0].detach(), prototypes.detach().t()
        ) / max(float(ose_tau_t), 1e-12)
        teacher_target = torch.softmax(teacher_logits, dim=1).detach()
        align_loss = self.soft_cross_entropy(
            student_logits, teacher_target)
        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        if prototypes.shape[0] > 1:
            off_diagonal = ~torch.eye(
                prototypes.shape[0], dtype=torch.bool,
                device=prototypes.device)
            dispersion_loss = (
                prototype_similarity[off_diagonal].mean()
                / max(float(ose_tau_s), 1e-12))
        else:
            dispersion_loss = prototype_similarity.new_zeros(())
        prototype_loss = align_loss + dispersion_loss

        mix_index = mix_index.detach().long().view(-1)
        if mix_index.numel() != view_a.shape[0]:
            raise ValueError('mix_index must match the unlabeled batch')
        mix_beta = float(mix_beta)
        if not 0.0 <= mix_beta <= 1.0:
            raise ValueError('mix_beta must be in [0, 1]')
        mixed_z = self._online_projection(mixed_view, ose_branch=True)
        mixed_logits = torch.matmul(
            mixed_z, prototypes.t()) / max(float(ose_tau_s), 1e-12)
        student_target = torch.softmax(student_logits, dim=1).detach()
        global_teacher_target = _concat_all_gather(teacher_target)
        if (mix_index.min().item() < 0 or
                mix_index.max().item() >= global_teacher_target.shape[0]):
            raise ValueError('mix_index contains an invalid global index')
        mixed_target = (
            mix_beta * student_target
            + (1.0 - mix_beta) * global_teacher_target[mix_index])
        mix_proto_loss = self.soft_cross_entropy(
            mixed_logits, mixed_target)

        global_teacher_ose = _concat_all_gather(
            teacher_ose_z[0].detach())
        instance_logits = torch.matmul(
            mixed_z, global_teacher_ose.t()
        ) / max(float(ose_tau_s), 1e-12)
        instance_log_prob = F.log_softmax(instance_logits, dim=1)
        rows = torch.arange(mixed_z.shape[0], device=mixed_z.device)
        positive_rows = rows + row_start
        mix_ins_loss = -(
            mix_beta * instance_log_prob[rows, positive_rows]
            + (1.0 - mix_beta) * instance_log_prob[rows, mix_index]
        ).mean()

        return {
            'cluster': cluster_loss,
            'cluster_entropy': cluster_entropy.detach(),
            'cluster_kl': cluster_kl.detach(),
            'proto': prototype_loss,
            'align': align_loss.detach(),
            'disp': dispersion_loss.detach(),
            'mix_proto': mix_proto_loss,
            'mix_ins': mix_ins_loss,
        }


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Stage1 checkpoint must be a dictionary')
    for key in ('state_dict', 'model_state_dict', 'model'):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    state = OrderedDict()
    for name, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if name.startswith('module.'):
            name = name[len('module.'):]
        state[name] = value.detach().cpu()
    if not state:
        raise ValueError('Stage1 checkpoint contains no tensor weights')
    return state


def transfer_macdiff_stage1(model, checkpoint):
    """Strictly transfer the Stage1 MacDiff online skeleton encoder."""
    source = _unwrap_checkpoint(checkpoint)
    target = model.encoder_q.state_dict()
    transferred = OrderedDict()
    for name, target_value in target.items():
        if name not in source:
            raise ValueError(
                'Stage1 checkpoint is missing encoder tensor {}'.format(name))
        source_value = source[name]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                'Stage1 tensor {} has shape {}, expected {}'.format(
                    name, tuple(source_value.shape), tuple(target_value.shape)))
        transferred[name] = source_value.to(
            dtype=target_value.dtype).clone()
    model.encoder_q.load_state_dict(transferred, strict=True)
    model.reset_momentum_encoder()
    return {
        'source': 'MacDiff online encoder',
        'encoder_tensors': len(transferred),
        'ignored_tensors': len(source) - len(transferred),
    }
