"""Upload-friendly diagnostics for the full-token dense OSE protocol.

The training entry point writes the records produced here to a standalone
JSONL file.  Heavy representation diagnostics intentionally run only on a
small, fixed, class-balanced clean subset and never participate in the loss.
"""

import datetime
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


class DenseOSEJsonlLogger(object):
    """Append self-contained records that remain valid across resume runs."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **fields):
        record = {
            'schema_version': SCHEMA_VERSION,
            'event': str(event),
            'time_utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }
        record.update(fields)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(
                _json_safe(record), sort_keys=True, allow_nan=False) + '\n')


def select_balanced_indices(
        labels, excluded_indices, max_samples, seed, num_classes):
    """Choose a deterministic, near-balanced diagnostic subset."""
    labels = np.asarray(labels, dtype=np.int64)
    excluded = set(int(value) for value in excluded_indices)
    rng = np.random.RandomState(int(seed))
    per_class = []
    for class_id in range(int(num_classes)):
        candidates = np.flatnonzero(labels == class_id)
        candidates = np.asarray([
            int(index) for index in candidates if int(index) not in excluded
        ], dtype=np.int64)
        if candidates.size == 0:
            raise ValueError(
                'No diagnostic candidate for class {}'.format(class_id))
        rng.shuffle(candidates)
        per_class.append(candidates.tolist())

    limit = min(int(max_samples), sum(len(row) for row in per_class))
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for candidates in per_class:
            if offset < len(candidates) and len(selected) < limit:
                selected.append(int(candidates[offset]))
                added = True
        if not added:
            break
        offset += 1
    return selected


def _module_states(modules):
    return [module.training for module in modules]


def _restore_module_states(modules, states):
    for module, state in zip(modules, states):
        module.train(state)


def _base_batch(dataset, indices, device):
    samples = np.stack([
        dataset.get_base_sample(int(index)) for index in indices])
    return torch.from_numpy(samples).float().to(
        device, non_blocking=True)


@torch.no_grad()
def extract_stage1_reference(
        model, dataset, indices, device, batch_size, enable_amp):
    """Capture Stage1-initialized raw 6400-D features before Stage2 updates."""
    modules = (model.encoder_q,)
    states = _module_states(modules)
    for module in modules:
        module.eval()
    outputs = []
    try:
        for start in range(0, len(indices), int(batch_size)):
            chunk_indices = indices[start:start + int(batch_size)]
            batch = _base_batch(dataset, chunk_indices, device)
            with torch.cuda.amp.autocast(enabled=bool(enable_amp)):
                outputs.append(
                    model.encoder_q.forward_features(batch).float().cpu())
    finally:
        _restore_module_states(modules, states)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def extract_dense_features(
        model, dataset, indices, device, batch_size, enable_amp):
    """Extract clean online/EMA raw and projected features without mutation."""
    modules = (
        model.encoder_q,
        model.encoder_k,
        model.ose_projector_q,
        model.ose_projector_k,
    )
    states = _module_states(modules)
    for module in modules:
        module.eval()
    outputs = {
        'online_raw': [],
        'teacher_raw': [],
        'online_projected': [],
        'teacher_projected': [],
    }
    try:
        for start in range(0, len(indices), int(batch_size)):
            chunk_indices = indices[start:start + int(batch_size)]
            batch = _base_batch(dataset, chunk_indices, device)
            with torch.cuda.amp.autocast(enabled=bool(enable_amp)):
                online_raw = model.encoder_q.forward_features(batch)
                teacher_raw = model.encoder_k.forward_features(batch)
                online_projected = F.normalize(
                    model.ose_projector_q(online_raw), dim=1)
                teacher_projected = F.normalize(
                    model.ose_projector_k(teacher_raw), dim=1)
            outputs['online_raw'].append(online_raw.float().cpu())
            outputs['teacher_raw'].append(teacher_raw.float().cpu())
            outputs['online_projected'].append(
                online_projected.float().cpu())
            outputs['teacher_projected'].append(
                teacher_projected.float().cpu())
    finally:
        _restore_module_states(modules, states)
    return {
        name: torch.cat(chunks, dim=0)
        for name, chunks in outputs.items()
    }


def _effective_rank(features):
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    gram = torch.matmul(centered, centered.t())
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probability = eigenvalues / total
    probability = probability[probability > 1e-12]
    return float(torch.exp(
        -(probability * probability.log()).sum()).item())


def _class_geometry(features, labels):
    normalized = F.normalize(features.float(), dim=1)
    similarity = torch.matmul(normalized, normalized.t())
    count = similarity.shape[0]
    upper = torch.triu(torch.ones(
        count, count, dtype=torch.bool, device=similarity.device), diagonal=1)
    same = labels[:, None].eq(labels[None, :]) & upper
    different = labels[:, None].ne(labels[None, :]) & upper
    same_values = similarity[same]
    different_values = similarity[different]
    if same_values.numel() == 0 or different_values.numel() == 0:
        raise ValueError('Diagnostic subset needs same- and different-class pairs')
    dimension_std = normalized.std(dim=0, unbiased=False)
    same_mean = float(same_values.mean().item())
    different_mean = float(different_values.mean().item())
    return {
        'same_class_cosine': same_mean,
        'different_class_cosine': different_mean,
        'class_gap': same_mean - different_mean,
        'same_pair_count': int(same_values.numel()),
        'different_pair_count': int(different_values.numel()),
        'effective_rank': _effective_rank(features),
        'normalized_dimension_std_mean': float(dimension_std.mean().item()),
        'normalized_dimension_std_min': float(dimension_std.min().item()),
    }


def _linear_cka(first, second):
    first = first.float() - first.float().mean(dim=0, keepdim=True)
    second = second.float() - second.float().mean(dim=0, keepdim=True)
    first_gram = torch.matmul(first, first.t())
    second_gram = torch.matmul(second, second.t())
    numerator = (first_gram * second_gram).sum()
    denominator = torch.sqrt(
        first_gram.square().sum() * second_gram.square().sum())
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _cosine_summary(first, second):
    values = F.cosine_similarity(first.float(), second.float(), dim=1)
    return {
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
        'p10': float(torch.quantile(values, 0.1).item()),
        'p50': float(torch.quantile(values, 0.5).item()),
        'p90': float(torch.quantile(values, 0.9).item()),
    }


def _classification_metrics(features, prototypes, labels, temperature):
    logits = torch.matmul(
        F.normalize(features.float(), dim=1),
        F.normalize(prototypes.float(), dim=1).t(),
    ) / max(float(temperature), 1e-12)
    probability = torch.softmax(logits, dim=1)
    confidence, prediction = probability.max(dim=1)
    top_two = probability.topk(k=min(2, probability.shape[1]), dim=1).values
    if top_two.shape[1] == 2:
        margin = top_two[:, 0] - top_two[:, 1]
    else:
        margin = top_two[:, 0]
    entropy = -(
        probability * probability.clamp_min(1e-12).log()).sum(dim=1)
    true_probability = probability.gather(1, labels[:, None]).squeeze(1)
    return {
        'top1_accuracy': float(
            prediction.eq(labels).float().mean().item()),
        'true_class_probability': float(true_probability.mean().item()),
        'confidence': float(confidence.mean().item()),
        'entropy': float(entropy.mean().item()),
        'top1_top2_margin': float(margin.mean().item()),
    }


def _project_with_batch_statistics(projector, features, batch_size):
    """Emulate training BN locally without mutating its running buffers."""
    if int(batch_size) < 2:
        raise ValueError('BN diagnostic batch size must be at least two')
    projected = []
    start = 0
    while start < features.shape[0]:
        end = min(start + int(batch_size), features.shape[0])
        if features.shape[0] - end == 1:
            end += 1
        value = features[start:end]
        if value.shape[0] < 2:
            break
        for layer in projector:
            if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                value = F.batch_norm(
                    value,
                    running_mean=None,
                    running_var=None,
                    weight=layer.weight,
                    bias=layer.bias,
                    training=True,
                    momentum=0.0,
                    eps=layer.eps,
                )
            else:
                value = layer(value)
        projected.append(F.normalize(value, dim=1))
        start = end
    if not projected:
        raise ValueError('Diagnostic subset is too small for BN comparison')
    return torch.cat(projected, dim=0)


def prototype_geometry(prototypes, previous=None):
    normalized = F.normalize(prototypes.float(), dim=1)
    similarity = torch.matmul(normalized, normalized.t())
    count = normalized.shape[0]
    off_diagonal = ~torch.eye(
        count, dtype=torch.bool, device=normalized.device)
    off_values = similarity[off_diagonal]
    masked = similarity.masked_fill(~off_diagonal, -float('inf'))
    nearest = masked.max(dim=1).values
    result = {
        'offdiagonal_cosine_mean': float(off_values.mean().item()),
        'offdiagonal_cosine_max': float(off_values.max().item()),
        'nearest_neighbor_cosine_mean': float(nearest.mean().item()),
        'effective_rank': _effective_rank(normalized),
    }
    if previous is None:
        result['epoch_drift_cosine_mean'] = None
        result['epoch_drift_cosine_min'] = None
    else:
        drift = F.cosine_similarity(
            normalized, F.normalize(previous.float(), dim=1), dim=1)
        result['epoch_drift_cosine_mean'] = float(drift.mean().item())
        result['epoch_drift_cosine_min'] = float(drift.min().item())
    return result


@torch.no_grad()
def dense_epoch_geometry(
        model, extracted, reference_features, labels, prototypes,
        device, student_temperature, teacher_temperature, bn_batch_size):
    """Build the expensive epoch-level diagnostic record."""
    online_raw = extracted['online_raw'].to(device)
    teacher_raw = extracted['teacher_raw'].to(device)
    online_projected = extracted['online_projected'].to(device)
    teacher_projected = extracted['teacher_projected'].to(device)
    reference = reference_features.to(device)
    labels = labels.to(device=device, dtype=torch.long)
    prototypes = prototypes.to(device).float()

    batch_stat_projected = _project_with_batch_statistics(
        model.ose_projector_q, online_raw, int(bn_batch_size))
    aligned_eval_projected = online_projected[:batch_stat_projected.shape[0]]

    return {
        'online_raw_geometry': _class_geometry(online_raw, labels),
        'online_projected_geometry': _class_geometry(
            online_projected, labels),
        'teacher_projected_geometry': _class_geometry(
            teacher_projected, labels),
        'stage1_reference': {
            'sample_cosine': _cosine_summary(online_raw, reference),
            'linear_cka': _linear_cka(online_raw, reference),
        },
        'online_ema_alignment': {
            'raw_cosine': _cosine_summary(online_raw, teacher_raw),
            'projected_cosine': _cosine_summary(
                online_projected, teacher_projected),
        },
        'online_clean_prototype_classification': _classification_metrics(
            online_projected, prototypes, labels, student_temperature),
        'teacher_clean_prototype_classification': _classification_metrics(
            teacher_projected, prototypes, labels, teacher_temperature),
        'online_projector_bn_alignment': {
            'eval_vs_global_microbatch_stats_cosine': _cosine_summary(
                aligned_eval_projected, batch_stat_projected),
            'emulated_global_microbatch': int(bn_batch_size),
        },
    }


def reference_geometry(reference_features, labels, device):
    return _class_geometry(
        reference_features.to(device), labels.to(device=device, dtype=torch.long))


def assignment_distribution(counts):
    counts = counts.float()
    total = counts.sum()
    if float(total.item()) <= 0.0:
        raise ValueError('Assignment counts must contain observations')
    probability = counts / total
    nonzero = probability[probability > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    classes = counts.numel()
    return {
        'histogram': [int(value) for value in counts.cpu().tolist()],
        'used_fraction': float(counts.gt(0).float().mean().item()),
        'perplexity': float(torch.exp(entropy).item()),
        'kl_to_uniform': float(math.log(classes) - entropy.item()),
    }
