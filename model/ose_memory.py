import torch
import torch.nn as nn
import torch.nn.functional as F


class OSEMemory(nn.Module):
    """OSE prototype builder plus a synchronized FIFO feature/identity queue."""

    def __init__(
        self,
        feature_dim,
        class_ids,
        exemplar_ids,
        queue_size=32768,
        topk=4,
        alpha=0.75,
        tau_s=0.1,
        tau_t=0.04,
        assignment_confidence=0.8,
    ):
        super().__init__()
        if len(class_ids) == 0 or len(class_ids) != len(exemplar_ids):
            raise ValueError('OSE requires exactly one exemplar index per class')
        if queue_size <= 0:
            raise ValueError('ose_queue_size must be positive')
        if topk <= 0:
            raise ValueError('ose_topk must be positive')
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('ose_alpha must be in [0, 1]')
        if tau_s <= 0 or tau_t <= 0:
            raise ValueError('OSE temperatures must be positive')
        if not 0.0 <= assignment_confidence <= 1.0:
            raise ValueError('ose_assignment_confidence must be in [0, 1]')

        self.queue_size = int(queue_size)
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.tau_s = float(tau_s)
        self.tau_t = float(tau_t)
        self.assignment_confidence = float(assignment_confidence)

        num_classes = len(class_ids)
        self.register_buffer(
            'class_ids', torch.as_tensor(class_ids, dtype=torch.long))
        self.register_buffer(
            'exemplar_ids', torch.as_tensor(exemplar_ids, dtype=torch.long))
        self.register_buffer(
            'queue_features', torch.zeros(self.queue_size, feature_dim))
        self.register_buffer(
            'queue_ids', torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_count', torch.zeros(1, dtype=torch.long))

        # These epoch-frozen tensors are routing state only. Training losses use
        # differentiable prototypes rebuilt from online exemplar embeddings.
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer(
            'prototype_neighbor_ids',
            torch.full((num_classes, self.topk), -1, dtype=torch.long))
        self.register_buffer('prototype_valid', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('snapshot_version', torch.zeros(1, dtype=torch.long))
        self.register_buffer('neighbor_map_entries', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, features, dataset_ids):
        features = F.normalize(features.detach().float(), dim=-1)
        dataset_ids = dataset_ids.detach().long().view(-1)
        if features.shape[0] != dataset_ids.shape[0]:
            raise ValueError(
                'Queue features and dataset IDs must have matching order and length')
        if features.shape[0] >= self.queue_size:
            features = features[-self.queue_size:]
            dataset_ids = dataset_ids[-self.queue_size:]

        count = features.shape[0]
        ptr = int(self.queue_ptr.item())
        first = min(count, self.queue_size - ptr)
        self.queue_features[ptr:ptr + first].copy_(features[:first])
        self.queue_ids[ptr:ptr + first].copy_(dataset_ids[:first])
        rest = count - first
        if rest:
            self.queue_features[:rest].copy_(features[first:])
            self.queue_ids[:rest].copy_(dataset_ids[first:])
        self.queue_ptr[0] = (ptr + count) % self.queue_size
        self.queue_count[0] = min(
            self.queue_size, int(self.queue_count.item()) + count)

    @torch.no_grad()
    def queue_snapshot(self):
        """Return every filled FIFO slot; ordering is irrelevant for OSE Top-K."""
        count = int(self.queue_count.item())
        return self.queue_features[:count], self.queue_ids[:count]

    @torch.no_grad()
    def deduplicated_snapshot(self):
        """Return the newest feature for each immutable dataset ID."""
        count = int(self.queue_count.item())
        if count == 0:
            return self.queue_features[:0], self.queue_ids[:0]

        if count < self.queue_size:
            ordered = torch.arange(count, device=self.queue_ids.device)
        else:
            ptr = int(self.queue_ptr.item())
            ordered = torch.cat([
                torch.arange(ptr, self.queue_size, device=self.queue_ids.device),
                torch.arange(0, ptr, device=self.queue_ids.device),
            ])

        # One device transfer avoids one GPU synchronization per queue entry.
        ordered_ids = self.queue_ids[ordered].cpu().tolist()
        seen = set()
        keep = []
        for position in range(len(ordered_ids) - 1, -1, -1):
            dataset_id = int(ordered_ids[position])
            if dataset_id >= 0 and dataset_id not in seen:
                seen.add(dataset_id)
                keep.append(position)
        keep.reverse()
        keep = torch.as_tensor(keep, dtype=torch.long, device=ordered.device)
        selected = ordered[keep]
        return self.queue_features[selected], self.queue_ids[selected]

    def _discriminative_scores(self, exemplar_features, memory):
        similarity = torch.matmul(exemplar_features, memory.t())
        if exemplar_features.shape[0] == 1:
            max_other = torch.zeros_like(similarity)
        else:
            # Exact max-over-other-class computation without a C x C x K tensor.
            top_values, top_classes = torch.topk(similarity, k=2, dim=0)
            class_indices = torch.arange(
                exemplar_features.shape[0], device=similarity.device
            ).unsqueeze(1)
            max_other = torch.where(
                top_classes[0].unsqueeze(0) == class_indices,
                top_values[1].unsqueeze(0),
                top_values[0].unsqueeze(0),
            )
        return self.alpha * similarity - (1.0 - self.alpha) * max_other

    def _build_prototypes(self, exemplar_features, queue_features, queue_ids):
        """Build paper-style prototypes with AimCLR's mutually exclusive P1 pool."""
        # Keep prototype selection and low-temperature logits in fp32 under AMP.
        exemplar_features = F.normalize(exemplar_features.float(), dim=-1)
        num_classes, feature_dim = exemplar_features.shape
        neighbor_ids = torch.full(
            (num_classes, self.topk), -1, dtype=torch.long,
            device=exemplar_features.device)

        if queue_features.shape[0] == 0:
            return exemplar_features, neighbor_ids

        memory = F.normalize(queue_features.detach(), dim=-1)
        scores = self._discriminative_scores(exemplar_features, memory)
        valid_memory = queue_ids >= 0
        if queue_ids.numel():
            is_exemplar = (
                queue_ids[:, None] == self.exemplar_ids[None, :]
            ).any(dim=1)
            valid_memory = valid_memory & ~is_exemplar
        scores = scores.masked_fill(~valid_memory.unsqueeze(0), -float('inf'))

        # P1: each queue slot is owned by exactly one class before classwise top-k.
        owners = scores.argmax(dim=0)
        class_indices = torch.arange(num_classes, device=scores.device).unsqueeze(1)
        owned_scores = scores.masked_fill(
            owners.unsqueeze(0) != class_indices, -float('inf'))
        selected_count = min(self.topk, memory.shape[0])
        selected_scores, selected_indices = torch.topk(
            owned_scores, k=selected_count, dim=1)
        selected_valid = torch.isfinite(selected_scores)

        selected_features = memory[selected_indices]
        components = torch.cat([
            exemplar_features.unsqueeze(1), selected_features,
        ], dim=1)
        aggregation_scores = torch.sum(
            components * exemplar_features.unsqueeze(1), dim=2)
        aggregation_scores = torch.cat([
            aggregation_scores[:, :1],
            aggregation_scores[:, 1:].masked_fill(
                ~selected_valid, -float('inf')),
        ], dim=1)
        weights = torch.softmax(aggregation_scores, dim=1)
        prototypes = torch.sum(weights.unsqueeze(2) * components, dim=1)

        selected_ids = queue_ids[selected_indices]
        selected_ids = selected_ids.masked_fill(~selected_valid, -1)
        neighbor_ids[:, :selected_count] = selected_ids
        return prototypes, neighbor_ids

    def prototype_loss(self, student_features, teacher_features,
                       exemplar_features):
        """Paper L_proto = L_align + L_disp with differentiable prototypes."""
        queue_features, queue_ids = self.queue_snapshot()
        prototypes, _ = self._build_prototypes(
            exemplar_features, queue_features, queue_ids)

        student_features = F.normalize(student_features.float(), dim=-1)
        teacher_features = F.normalize(
            teacher_features.detach().float(), dim=-1)
        student_logits = torch.matmul(
            student_features, prototypes.t()) / self.tau_s
        teacher_logits = torch.matmul(
            teacher_features, prototypes.detach().t()) / self.tau_t
        teacher_target = torch.softmax(teacher_logits, dim=1).detach()
        align_loss = -(
            teacher_target * F.log_softmax(student_logits, dim=1)
        ).sum(dim=1).mean()

        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        if prototypes.shape[0] > 1:
            off_diagonal = ~torch.eye(
                prototypes.shape[0], dtype=torch.bool,
                device=prototypes.device)
            dispersion_loss = (
                prototype_similarity[off_diagonal].mean() / self.tau_s)
        else:
            dispersion_loss = prototype_similarity.sum() * 0.0

        target_entropy = -(
            teacher_target * teacher_target.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        confidence, assignment = teacher_target.max(dim=1)
        return {
            'proto': align_loss + dispersion_loss,
            'align': align_loss,
            'dispersion': dispersion_loss,
            'target_entropy': target_entropy,
            'align_kl': align_loss - target_entropy,
            'assignment': assignment.detach(),
            'confidence': confidence.detach(),
        }

    @torch.no_grad()
    def refresh(self, exemplar_features):
        """Refresh only the epoch-frozen prototypes and peer-routing map."""
        queue_features, queue_ids = self.deduplicated_snapshot()
        if queue_features.shape[0] == 0:
            self.prototype_valid.zero_()
            self.prototype_neighbor_ids.fill_(-1)
            self.neighbor_map_entries.zero_()
            return {}

        exemplar_features = F.normalize(exemplar_features.detach(), dim=-1)
        if exemplar_features.shape != self.prototypes.shape:
            raise ValueError(
                'Exemplar feature count must match the exemplar mapping')
        prototypes, neighbor_ids = self._build_prototypes(
            exemplar_features, queue_features, queue_ids)
        self.prototypes.copy_(prototypes)
        self.prototype_neighbor_ids.copy_(neighbor_ids)
        self.prototype_valid.fill_(True)
        self.snapshot_version.add_(1)
        neighbor_map = self.build_neighbor_map(queue_features, queue_ids)
        self.neighbor_map_entries[0] = len(neighbor_map)
        return neighbor_map

    @torch.no_grad()
    def build_neighbor_map(self, queue_features=None, queue_ids=None):
        if not bool(self.prototype_valid.item()):
            return {}
        if queue_features is None or queue_ids is None:
            queue_features, queue_ids = self.deduplicated_snapshot()
        if queue_features.shape[0] < 2:
            return {}

        probabilities = F.softmax(
            torch.matmul(queue_features, self.prototypes.t()) / self.tau_t,
            dim=1)
        confidence, assignment = probabilities.max(dim=1)
        valid = confidence >= self.assignment_confidence

        queue_ids_cpu = queue_ids.cpu().tolist()
        assignment_cpu = assignment.cpu().tolist()
        valid_cpu = valid.cpu().tolist()
        pools = [
            [int(dataset_id) for dataset_id in row if int(dataset_id) >= 0]
            for row in self.prototype_neighbor_ids.cpu().tolist()
        ]
        neighbor_map = {}
        for dataset_id, class_index, is_valid in zip(
                queue_ids_cpu, assignment_cpu, valid_cpu):
            if not is_valid:
                continue
            candidates = [
                peer_id for peer_id in pools[int(class_index)]
                if peer_id != int(dataset_id)
            ]
            if candidates:
                neighbor_map[int(dataset_id)] = candidates
        return neighbor_map

    @torch.no_grad()
    def metrics(self):
        fill = float(self.queue_count.item()) / float(self.queue_size)
        if not bool(self.prototype_valid.item()) or self.prototypes.shape[0] < 2:
            return {
                'queue_fill_ratio': fill,
                'prototype_cosine_mean': 0.0,
                'prototype_cosine_max': 0.0,
                'ose_snapshot_version': float(self.snapshot_version.item()),
                'neighbor_map_entries': float(self.neighbor_map_entries.item()),
            }
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = torch.matmul(prototypes, prototypes.t())
        off_diagonal = ~torch.eye(
            similarity.shape[0], dtype=torch.bool, device=similarity.device)
        values = similarity[off_diagonal]
        return {
            'queue_fill_ratio': fill,
            'prototype_cosine_mean': float(values.mean().item()),
            'prototype_cosine_max': float(values.max().item()),
            'ose_snapshot_version': float(self.snapshot_version.item()),
            'neighbor_map_entries': float(self.neighbor_map_entries.item()),
        }
