"""Native S->S + T->S diffusion, with text-to-skeleton relational distillation."""
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .transformer_macdiff import Transformer as NativeTransformer
from .transformer_macdiff_text import (
    ResidualTextRemap, TextConditionedSkeletonDecoder, Transformer as TextTransformer,
    masked_text_uniformity_loss,
)
from .util import token_uniformity_loss


RELATION_VERSION = 'paired_global_fifo_text_to_skeleton_kl_v1'


def validate_relation_options(relation_version, relation_bank_size, lambda_relation,
                              relation_temperature_text, relation_temperature_skeleton,
                              lambda_text_to_skeleton, lambda_skeleton_to_text,
                              one_person, text_target_mode, text_target_norm):
    if relation_version != RELATION_VERSION:
        raise ValueError('Unsupported relation_version')
    if type(relation_bank_size) is not int or relation_bank_size < 2:
        raise ValueError('relation_bank_size must be an integer >= 2')
    for name, value in (('lambda_relation', lambda_relation),
                        ('relation_temperature_text', relation_temperature_text),
                        ('relation_temperature_skeleton', relation_temperature_skeleton),
                        ('lambda_text_to_skeleton', lambda_text_to_skeleton)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (name != 'lambda_relation' and value == 0)):
            raise ValueError(name + ' must be finite and ' + ('non-negative' if name == 'lambda_relation' else 'positive'))
    if lambda_skeleton_to_text != 0:
        raise ValueError('S->T diffusion is replaced; lambda_skeleton_to_text must be 0')
    if one_person is not True or text_target_mode != 'sample_target_blend' or text_target_norm != 'rms':
        raise ValueError('This relation control requires person0 and RMS sample_target_blend text')


def paired_relation_kl(skeleton, text, sample_ids, skeleton_keys, text_keys, key_ids,
                       temperature_skeleton, temperature_text):
    """KL(stopgrad(text relations) || skeleton relations) on the SAME anchors.

    Feature dimensions may differ. All occurrences of the current sample ID
    are excluded on both sides. Empty/one-anchor rows provide no relation loss.
    """
    with torch.cuda.amp.autocast(enabled=False):
        skeleton, text = skeleton.float(), text.detach().float()
        selected = key_ids >= 0
        anchors = key_ids[selected]
        permitted = sample_ids[:, None] != anchors[None, :]
        usable = permitted.sum(dim=1) >= 2
        zero = skeleton.sum() * 0.
        metrics = {name: zero.detach() for name in (
            'relation_teacher_entropy', 'relation_teacher_entropy_fraction',
            'relation_student_entropy', 'relation_anchor_count', 'relation_valid_fraction')}
        if not usable.any():
            return zero, metrics
        permitted = permitted[usable]
        query = F.normalize(skeleton[usable], dim=-1)
        keys = F.normalize(skeleton_keys[selected].detach().float(), dim=-1)
        student_log = F.log_softmax((query @ keys.t() / temperature_skeleton).masked_fill(
            ~permitted, -1.e9), dim=-1)
        with torch.no_grad():
            teacher_query = F.normalize(text[usable], dim=-1)
            teacher_keys = F.normalize(text_keys[selected].detach().float(), dim=-1)
            teacher_log = F.log_softmax((teacher_query @ teacher_keys.t() / temperature_text).masked_fill(
                ~permitted, -1.e9), dim=-1)
            teacher = teacher_log.exp()
            teacher_entropy = -(teacher * teacher_log).sum(dim=-1)
            counts = permitted.sum(dim=-1).float()
            metrics.update(
                relation_teacher_entropy=teacher_entropy.mean(),
                relation_teacher_entropy_fraction=(teacher_entropy / counts.log()).mean(),
                relation_anchor_count=counts.mean(),
                relation_valid_fraction=usable.float().mean())
        loss = (teacher * (teacher_log - student_log)).sum(dim=-1).mean()
        metrics['relation_student_entropy'] = -(student_log.exp() * student_log).sum(dim=-1).mean().detach()
        return loss, metrics


def gather_relation_pairs(sample_ids, skeleton, text):
    """Gather paired rows in rank order, allowing different active-row counts."""
    if not (dist.is_available() and dist.is_initialized()):
        return sample_ids, skeleton, text
    count = torch.tensor([sample_ids.shape[0]], dtype=torch.long, device=sample_ids.device)
    sizes = [torch.zeros_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, count)
    sizes = [int(size.item()) for size in sizes]
    maximum = max(sizes)
    if not maximum:
        return sample_ids, skeleton, text
    result = []
    for values in (sample_ids, skeleton, text):
        padded = values.new_zeros((maximum,) + tuple(values.shape[1:]))
        padded[:values.shape[0]] = values
        gathered = [torch.zeros_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        result.append(torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0))
    return tuple(result)


class Transformer(NativeTransformer):
    supports_text_cache = True
    supports_relation_memory = True
    fixed_clip_target = TextTransformer.fixed_clip_target
    validate_text_tokens = TextTransformer.validate_text_tokens
    text_to_skeleton_loss = TextTransformer.text_to_skeleton_loss

    def __init__(self, relation_version=RELATION_VERSION, relation_bank_size=4096,
                 lambda_relation=.1, relation_temperature_text=.1, relation_temperature_skeleton=.1,
                 text_input_dim=512, text_hidden_dim=2048, text_context_length=7,
                 text_target_mode='sample_target_blend', text_target_norm='rms',
                 text_target_update_ratio=.1, lambda_text_uniformity=0.,
                 lambda_text_to_skeleton=1., lambda_skeleton_to_text=0.,
                 share_skeleton_decoder=False, text_conditioning_mode='concat_memory', **kwargs):
        validate_relation_options(relation_version, relation_bank_size, lambda_relation,
            relation_temperature_text, relation_temperature_skeleton,
            lambda_text_to_skeleton, lambda_skeleton_to_text, kwargs.get('one_person', True),
            text_target_mode, text_target_norm)
        if text_conditioning_mode != 'concat_memory' or not isinstance(share_skeleton_decoder, bool):
            raise ValueError('Use the existing concat_memory T->S definition')
        if min(text_input_dim, text_hidden_dim, text_context_length) < 1:
            raise ValueError('Text dimensions must be positive')
        if not math.isfinite(text_target_update_ratio) or not 0 < text_target_update_ratio <= 1:
            raise ValueError('text_target_update_ratio must be in (0,1]')
        if not math.isfinite(lambda_text_uniformity) or lambda_text_uniformity < 0:
            raise ValueError('lambda_text_uniformity must be finite and non-negative')
        super().__init__(**kwargs)
        if self.diff_prediction != 'noise' or self.dim_t_embed != 64:
            raise ValueError('This control retains native epsilon diffusion and 64-dim time embedding')
        self.relation_version = relation_version
        self.relation_bank_size = relation_bank_size
        self.lambda_relation = float(lambda_relation)
        self.relation_temperature_text = float(relation_temperature_text)
        self.relation_temperature_skeleton = float(relation_temperature_skeleton)
        self.text_input_dim, self.text_context_length = text_input_dim, text_context_length
        self.text_target_mode, self.text_target_norm = text_target_mode, text_target_norm
        self.text_target_update_ratio = float(text_target_update_ratio)
        self.text_target_momentum = .999  # unused for sample_target_blend; native metadata compatibility
        self.lambda_text_uniformity = float(lambda_text_uniformity)
        self.lambda_text_to_skeleton = float(lambda_text_to_skeleton)
        self.lambda_skeleton_to_text = 0.
        self.share_skeleton_decoder = share_skeleton_decoder
        self.text_conditioning_mode = text_conditioning_mode
        self.text_remap = ResidualTextRemap(text_input_dim, text_hidden_dim)
        self.text_remap.apply(self._init_weights)
        self.text_person_embedding = nn.Embedding(2, text_input_dim)
        self.text_position_embedding = nn.Embedding(text_context_length, text_input_dim)
        nn.init.normal_(self.text_person_embedding.weight, std=.02)
        nn.init.normal_(self.text_position_embedding.weight, std=.02)
        self.text_skeleton_decoder = TextConditionedSkeletonDecoder(
            self, share_skeleton_decoder, text_dim=text_input_dim, conditioning_mode=text_conditioning_mode)
        # Paired FIFO memories: one slot/ID means the same instance in both modalities.
        self.register_buffer('relation_skeleton_keys', torch.zeros(relation_bank_size, self.dim_feat))
        self.register_buffer('relation_text_keys', torch.zeros(relation_bank_size, text_input_dim))
        self.register_buffer('relation_sample_ids', torch.full((relation_bank_size,), -1, dtype=torch.long))
        self.register_buffer('relation_pointer', torch.zeros((), dtype=torch.long))
        print('Text relation distillation: %s; K=%d; temperatures(T/S)=%g/%g; lambda=%g; no S->T decoder'
              % (relation_version, relation_bank_size, relation_temperature_text,
                 relation_temperature_skeleton, lambda_relation))

    def make_relation_keys(self, clean, query):
        """The original control stores detached masked-online global features."""
        return query.detach()

    def skeleton_auxiliary_loss(self, query, keys, sample_ids):
        return query.sum() * 0., {}

    @torch.no_grad()
    def complete_relation_step(self, batches, successful):
        if not successful or not batches:
            return
        ids, skeleton, text = [torch.cat([batch[name] for batch in batches], dim=0)
                               for name in ('sample_ids', 'skeleton', 'text')]
        ids, skeleton, text = gather_relation_pairs(ids, skeleton, text)
        if not ids.numel():
            return
        if not torch.isfinite(skeleton).all() or not torch.isfinite(text).all() or torch.any(ids < 0):
            raise FloatingPointError('Invalid paired relation-memory update')
        total = ids.shape[0]
        start = max(0, total - self.relation_bank_size)
        positions = (torch.arange(start, total, device=ids.device)
                     + self.relation_pointer) % self.relation_bank_size
        self.relation_skeleton_keys[positions] = F.normalize(skeleton[start:].float(), dim=-1)
        self.relation_text_keys[positions] = F.normalize(text[start:].float(), dim=-1)
        self.relation_sample_ids[positions] = ids[start:]
        self.relation_pointer.copy_((self.relation_pointer + total) % self.relation_bank_size)

    def forward(self, source, source_aug, text_features=None, mask_ratio=.9,
                motion_stride=1, motion_aware_tau=-1, enable_ose=False,
                text_tokens=None, text_token_mask=None, text_person_ids=None,
                text_positions=None, text_target_global=None, text_target_tokens=None,
                sample_indices=None, **unused):
        if enable_ose or not 0 < mask_ratio < 1:
            raise ValueError('Relation PT requires masked native diffusion without OSE')
        batch_size = source.shape[0]
        if (sample_indices is None or sample_indices.shape != (batch_size,)
                or sample_indices.dtype != torch.long or torch.any(sample_indices < 0)):
            raise ValueError('Relation PT requires stable dataset sample IDs')
        if (text_features is None or text_features.shape != (batch_size, self.text_input_dim)
                or any(value is None for value in (text_tokens, text_token_mask, text_person_ids,
                    text_positions, text_target_global, text_target_tokens))
                or text_target_global.shape != text_features.shape
                or text_target_tokens.shape != text_tokens.shape):
            raise ValueError('Use the aligned sentence cache and persistent sample targets')
        with torch.no_grad():
            source, source_aug = source[..., :1], source_aug[..., :1]
            active = source.abs().sum(dim=(1, 2, 3, 4)) > 0
            if not active.any():
                raise ValueError('The entire batch has no active person0')
            raw = self._sequence_layout(source).float()
            augmented = self._sequence_layout(source_aug).float()
            center = raw.mean(dim=(1, 2), keepdim=True) if self.self_shift else None
            clean = self._normalize_sequence(raw, center)
            augmented = self._normalize_sequence(augmented, center)
            time, _ = self.schedule_sampler.sample(batch_size, clean.device)
            noise = torch.randn_like(clean)
            noisy = self.diffusion.q_sample(clean, time, noise=noise)
        latent, pooled, mask, restore = self.forward_encoder(
            augmented, x_orig=raw, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        uniformity = token_uniformity_loss(latent[active])
        condition = self.build_global_local_condition(latent, pooled, restore)
        prediction = self.forward_decoder(noisy, z=condition, t=time)
        native = self.forward_loss(noise[active], prediction[active], mask[active], time[active])
        features, tokens, valid, people, positions = [values[active] for values in
            (text_features, text_tokens, text_token_mask, text_person_ids, text_positions)]
        self.validate_text_tokens(tokens, valid, people, positions)
        r = self.text_remap(self.fixed_clip_target(features))
        local = self.text_remap(self.fixed_clip_target(tokens, valid)).masked_fill(~valid[..., None], 0)
        token_condition = (local + self.text_person_embedding(people.masked_fill(~valid, 0))
                           + self.text_position_embedding(positions.masked_fill(~valid, 0)))
        token_condition = token_condition.masked_fill(~valid[..., None], 0)
        text_uniformity = (masked_text_uniformity_loss(local, valid)
                           if self.lambda_text_uniformity else native.new_zeros(()))
        t2s = self.text_to_skeleton_loss(noisy[active], noise[active], time[active], mask[active], r,
            torch.ones(r.shape[0], dtype=torch.bool, device=r.device), 1, token_condition, valid)
        # Retain the old sample_target_blend target, now as a relation teacher.
        teacher = text_target_global[active].detach().float()
        query = pooled[active, 0].float()
        keys = self.make_relation_keys(clean[active], query)
        auxiliary, auxiliary_metrics = self.skeleton_auxiliary_loss(query, keys, sample_indices[active])
        relation, relation_metrics = paired_relation_kl(query, teacher, sample_indices[active],
            self.relation_skeleton_keys, self.relation_text_keys, self.relation_sample_ids,
            self.relation_temperature_skeleton, self.relation_temperature_text)
        loss = (native + self.lambda_loss_uni * uniformity + self.lambda_text_to_skeleton * t2s
                + self.lambda_text_uniformity * text_uniformity + self.lambda_relation * relation + auxiliary)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native/T->S/relation loss')
        metrics = dict(loss_diff=native.detach(), loss_uniformity=uniformity.detach(),
            loss_text_to_skeleton=t2s.detach(), loss_text_uniformity=text_uniformity.detach(),
            loss_relation=relation.detach(), loss_relation_weighted=(self.lambda_relation * relation).detach(),
            relation_bank_fill=(self.relation_sample_ids >= 0).float().sum(),
            **relation_metrics, **auxiliary_metrics)
        pending = dict(sample_ids=sample_indices[active].detach(), skeleton=keys.detach(), text=teacher)
        return loss, prediction, mask, metrics, pending
