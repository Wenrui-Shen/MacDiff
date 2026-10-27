"""One EMA skeleton-key queue for InfoNCE/KL; full-view default, optional coarse view."""
import math

import torch
import torch.nn.functional as F

from .transformer_macdiff import MomentumSkeletonEncoder
from .transformer_macdiff_text_relation import (
    Transformer as RelationTransformer, validate_relation_options,
)


RELATION_TEACHER_VERSION = 'full_ema_gap_shared_queue_global_kl_v2'
RELATION_COARSE_TEACHER_VERSION = 'coarse3_mean_random70_ema_gap_shared_queue_kl_v3'


def validate_relation_teacher_options(relation_teacher_version, ema_momentum,
                                      lambda_contrastive, contrast_temperature, relation_feature_mode):
    if relation_teacher_version not in (RELATION_TEACHER_VERSION, RELATION_COARSE_TEACHER_VERSION):
        raise ValueError('Unsupported relation_teacher_version')
    if relation_feature_mode != 'gap':
        raise ValueError('This control pools visible tokens to dim_feat; joint-concat is a separate readout')
    for name, value in (('ema_momentum', ema_momentum), ('lambda_contrastive', lambda_contrastive),
                        ('contrast_temperature', contrast_temperature)):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value)):
            raise ValueError(name + ' must be finite')
    if not 0 <= ema_momentum < 1:
        raise ValueError('ema_momentum must be in [0,1)')
    if lambda_contrastive < 0 or contrast_temperature <= 0:
        raise ValueError('Require lambda_contrastive>=0 and contrast_temperature>0')


def validate_teacher_view_options(relation_teacher_version, teacher_view='full',
                                  teacher_temporal_group=1, teacher_mask_ratio=0.):
    if (type(teacher_temporal_group) is not int or teacher_temporal_group < 1
            or isinstance(teacher_mask_ratio, bool) or not isinstance(teacher_mask_ratio, (int, float))
            or not math.isfinite(teacher_mask_ratio) or not 0 <= teacher_mask_ratio < 1):
        raise ValueError('Require an integer temporal group and finite teacher mask ratio in [0,1)')
    if teacher_view == 'full':
        if (relation_teacher_version != RELATION_TEACHER_VERSION
                or teacher_temporal_group != 1 or teacher_mask_ratio != 0):
            raise ValueError('The full-key control requires v2, group=1 and teacher_mask_ratio=0')
    elif teacher_view == 'coarse_mask':
        if (relation_teacher_version != RELATION_COARSE_TEACHER_VERSION
                or teacher_temporal_group != 3 or teacher_mask_ratio != .7):
            raise ValueError('The coarse-key ablation requires v3, group=3 and teacher_mask_ratio=0.7')
    else:
        raise ValueError('Unsupported teacher_view')


def coarsen_temporal_tokens(tokens, group):
    """Average adjacent fine-time embeddings at the SAME joint, before attention.

    This ties the effective large-patch projection to the EMA fine embedding;
    it does not add an independently learned 3x-wide patch projection.
    """
    batch, temporal, joints, dim = tokens.shape
    if type(group) is not int or group < 1 or temporal % group:
        raise ValueError('The temporal token grid must be divisible by teacher_temporal_group')
    return tokens.reshape(batch, temporal // group, group, joints, dim).mean(dim=2)


def sample_teacher_tokens(tokens, mask_ratio):
    keep = round(tokens.shape[1] * (1 - mask_ratio))
    if not 0 < keep <= tokens.shape[1]:
        raise ValueError('Teacher masking must retain at least one token')
    noise = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)
    indices = torch.argsort(noise, dim=1)[:, :keep]
    rows = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
    return tokens[rows, indices], indices


def shared_queue_info_nce(query, positive_keys, sample_ids, skeleton_keys, key_ids, temperature):
    """Use the current EMA key as positive and the shared historical bank as negatives."""
    with torch.cuda.amp.autocast(enabled=False):
        query = F.normalize(query.float(), dim=-1, eps=1.e-6)
        positive_keys = F.normalize(positive_keys.detach().float(), dim=-1, eps=1.e-6)
        positive = (query * positive_keys).sum(dim=-1, keepdim=True)
        selected = key_ids >= 0
        negatives = query @ skeleton_keys[selected].detach().float().t()
        # Exclude every historical copy of this sequence; it remains a current positive.
        negatives = negatives.masked_fill(sample_ids[:, None] == key_ids[selected][None, :], float('-inf'))
        logits = torch.cat((positive, negatives), dim=1) / temperature
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels), positive.mean().detach()


class FullContextRelationKeyEncoder(MomentumSkeletonEncoder):
    @torch.no_grad()
    def forward(self, clean):
        # Do not call the parent random_masking path, even with mask_ratio=0.
        x = self.joints_embed(clean)
        batch, temporal, joints, dim = x.shape
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        x = x.reshape(batch, temporal * joints, dim)
        for block in self.blocks:
            x = block(x)
        return F.normalize(self.norm(x).float().mean(dim=1), dim=-1, eps=1.e-6)


class CoarseMaskedRelationKeyEncoder(MomentumSkeletonEncoder):
    def __init__(self, online_model, temporal_group, mask_ratio):
        super().__init__(online_model)
        if online_model.joints_embed.t_grid_size % temporal_group:
            raise ValueError('Teacher temporal grid must be divisible by teacher_temporal_group')
        self.temporal_group, self.mask_ratio = temporal_group, mask_ratio

    @torch.no_grad()
    def forward(self, clean):
        x = self.joints_embed(clean)
        _, temporal, joints, _ = x.shape
        # Average temporal positions within each group; spatial identity stays intact.
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        x = coarsen_temporal_tokens(x, self.temporal_group)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x, _ = sample_teacher_tokens(x, self.mask_ratio)
        for block in self.blocks:
            x = block(x)
        return F.normalize(self.norm(x).float().mean(dim=1), dim=-1, eps=1.e-6)


class Transformer(RelationTransformer):
    def __init__(self, relation_teacher_version=RELATION_TEACHER_VERSION, ema_momentum=.999,
                 lambda_contrastive=1.e-4, contrast_temperature=.07, relation_feature_mode='gap',
                 teacher_view='full', teacher_temporal_group=1, teacher_mask_ratio=0., **kwargs):
        validate_relation_teacher_options(relation_teacher_version, ema_momentum,
            lambda_contrastive, contrast_temperature, relation_feature_mode)
        validate_teacher_view_options(relation_teacher_version, teacher_view,
                                      teacher_temporal_group, teacher_mask_ratio)
        kwargs.setdefault('relation_bank_size', 65536)
        super().__init__(**kwargs)
        self.relation_teacher_version = relation_teacher_version
        self.ema_momentum = float(ema_momentum)
        self.lambda_contrastive = float(lambda_contrastive)
        self.contrast_temperature = float(contrast_temperature)
        self.relation_feature_mode = relation_feature_mode
        self.teacher_view = teacher_view
        self.teacher_temporal_group = teacher_temporal_group
        self.teacher_mask_ratio = float(teacher_mask_ratio)
        if teacher_view == 'full':
            self.ema_teacher = FullContextRelationKeyEncoder(self)
        else:
            self.ema_teacher = CoarseMaskedRelationKeyEncoder(self, teacher_temporal_group, teacher_mask_ratio)
        self.ema_teacher.eval()
        self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.long))
        print('EMA key + shared InfoNCE/KL queue: %s; pooling=gap; dim=%d; '
              'momentum=%g; CL=%g; view=%s; temporal_group=%d; mask=%g; no alignment/local KL'
              % (relation_teacher_version, self.dim_feat, ema_momentum, lambda_contrastive,
                 teacher_view, teacher_temporal_group, teacher_mask_ratio))

    def train(self, mode=True):
        super().train(mode)
        self.ema_teacher.eval()
        return self

    @torch.no_grad()
    def initialize_pretraining_state(self):
        # Native main calls this after DDP parameter sync, before full resume.
        self.ema_teacher.update_from(self, 0.)

    @torch.no_grad()
    def make_relation_keys(self, clean, query):
        return self.ema_teacher(clean)

    def skeleton_auxiliary_loss(self, query, keys, sample_ids):
        contrastive, positive = shared_queue_info_nce(query, keys, sample_ids,
            self.relation_skeleton_keys, self.relation_sample_ids, self.contrast_temperature)
        weighted = self.lambda_contrastive * contrastive
        return weighted, dict(loss_contrastive=contrastive.detach(),
            loss_contrastive_weighted=weighted.detach(), contrast_positive_cosine=positive)

    @torch.no_grad()
    def complete_relation_step(self, batches, successful):
        super().complete_relation_step(batches, successful)
        if successful and batches:
            self.ema_teacher.update_from(self, self.ema_momentum)
            self.teacher_updates.add_(1)
