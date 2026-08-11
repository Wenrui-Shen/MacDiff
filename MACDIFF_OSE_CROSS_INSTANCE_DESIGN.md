# MacDiff + OSE Exemplar-Anchored Cross-Instance Diffusion

## 1. Purpose

This document is an implementation specification for modifying the official MacDiff repository after cloning it.

The change keeps MacDiff's masked encoder, diffusion decoder, AdaLN conditioning, noise schedule, prediction target, and downstream evaluation protocol. It adds OSE-style class grounding and uses a high-confidence sample from the predicted class neighborhood as an alternative diffusion reconstruction target.

The central training relation is:

```text
source skeleton x_i --90% mask--> source global representation h_i
                                             |
                                             +--> OSE class assignment and neighbor selection
                                             |
same-class neighbor x_j --add noise----------+--> conditional diffusion reconstruction
```

For self-reconstruction, retain the original MacDiff global-local condition. For cross-instance reconstruction, use a global-only condition so that local source tokens are not forced onto potentially misaligned target timestamps.

## 2. Fixed decisions

These are requirements, not suggestions:

1. Read the total number of epochs from the repository runtime configuration (`args.epochs`). Do not hardcode 500.
2. The current official pretraining default and all checked pretraining YAML files use 400 epochs.
3. For epochs before `ose_start_epoch`, use only MacDiff's native diffusion and
   token-uniformity losses. Keep peer routing disabled, but continue updating
   the EMA encoder and Queue so the OSE stage starts with a usable memory bank.
4. At `ose_start_epoch`, enable OSE and route targets with probabilities:
   - self-reconstruction: 0.9;
   - cross-instance reconstruction: 0.1.
5. Change the probabilities linearly over the active OSE stage so that at the final epoch they are:
   - self-reconstruction: 0.1;
   - cross-instance reconstruction: 0.9.
6. The routing probabilities only choose the target type. They must not dynamically reweight the loss.
7. Queue features may be updated every optimizer iteration. The momentum encoder may be updated every optimizer iteration.
8. Routing prototypes and neighbor assignments must not be recomputed every iteration. Refresh them on the slow timescale, at epoch boundaries by default. The differentiable prototypes used by the OSE training loss are rebuilt from online exemplar embeddings every iteration and are not routing state.
9. Cross-instance diffusion uses a global-only source condition.
10. Self-reconstruction keeps MacDiff's original global-local condition.
11. Do not use non-exemplar ground-truth labels for prototype construction, class assignment, neighbor selection, routing, or losses.

## 3. Existing repository behavior

The relevant official files are:

- `model/transformer_macdiff.py`
- `model/util.py`
- `engine_pretrain.py`
- `main_pretrain.py`
- `feeder/feeder_ntu.py`
- `config/*/pretrain_madiff.yaml`

MacDiff currently:

1. uses `mask_ratio: 0.9` in the official pretraining configuration;
2. extracts local representations only for visible tokens;
3. obtains `cls_token` by mean-pooling the visible local tokens;
4. restores visible local tokens to their original positions using `ids_restore`;
5. fills masked positions with the global `cls_token`;
6. conditions the diffusion decoder through AdaLN-style feature modulation;
7. adds diffusion noise to the same sample used by the encoder;
8. predicts noise by default;
9. computes diffusion loss only at encoder-masked positions;
10. adds `lambda_loss_uni * token_uniformity_loss(...)`.

The original global-local condition is:

\[
z_{i,l}^{\mathrm{self}}=
\begin{cases}
z_{i,l}^{\mathrm{local}}, & l\text{ is visible},\\
z_i^{\mathrm{global}}, & l\text{ is masked}.
\end{cases}
\]

The existing token uniformity loss normalizes visible tokens, computes their within-sample cosine-similarity matrix, and minimizes its squared entries. It prevents visible-token over-smoothing; it is not a class-balance loss.

## 4. Learning setting and exemplar input

The main design assumes one labeled exemplar per class, following the OSESSL setting. Provide a reproducible mapping:

```json
{
  "0": 182,
  "1": 904,
  "2": 1511
}
```

Keys are semantic class IDs and values are immutable dataset indices.

Add a configuration field:

```yaml
ose_exemplar_indices: path/to/exemplar_indices.json
```

The number of classes must be inferred from this mapping, not from the full training-label array.

The feeder currently returns a label for every sample. During OSE pretraining, ignore that label for every non-exemplar sample. It may remain in the returned tuple for compatibility, but must not affect the proposed method.

If an entirely label-free experiment is desired later, replace the exemplar mapping with unsupervised cluster seeds. That is a different protocol and is outside this implementation.

## 5. Feature representations

For source sample \(x_i\), retain 10% of its tokens:

\[
(z_i^{\mathrm{local}},h_i)=E_\theta(M_{0.9}(x_i)),
\qquad
h_i=\operatorname{MeanPool}(z_i^{\mathrm{local}}).
\]

Use representations as follows:

- OSE Queue: normalized momentum-encoder global representation \(\bar h_i\);
- prototype construction: normalized global representations;
- OSE alignment: online source feature \(h_i\), EMA source feature \(\bar h_i\), and online exemplar features;
- neighbor search: normalized global representations;
- self-diffusion condition: original MacDiff global-local sequence;
- peer-diffusion condition: broadcast source global representation \(h_i\) only.

Do not place source local tokens at target time positions in the peer branch.

## 6. Fast and slow state updates

### 6.1 Fast state

Update on every optimizer step:

1. online encoder parameters through gradient descent;
2. momentum encoder parameters through EMA;
3. Queue features and their immutable dataset IDs.

Use an encoder-only EMA copy:

\[
\theta_m\leftarrow m\theta_m+(1-m)\theta.
\]

Recommended initial configuration:

```yaml
ose_momentum: 0.999
ose_queue_size: 32768
```

Update the EMA encoder after an actual optimizer update, respecting gradient accumulation. Do not update it on micro-steps where the optimizer has not stepped.

Each Queue slot must store both:

```text
(normalized_momentum_feature, global_dataset_id)
```

Never treat a FIFO Queue position as a dataset index.

In distributed training, all-gather features and dataset IDs in matching order before enqueueing them.

### 6.2 Slow state

Update without gradients at epoch boundaries:

1. momentum exemplar features;
2. routing-only OSE class prototypes;
3. Queue class assignments and confidences;
4. per-sample candidate-neighbor lists (`neighbor_map`).

Recommended default:

```yaml
ose_refresh_interval: 1
```

This means once per epoch, not once per iteration. Keep it configurable so later experiments can use intervals such as 2 or 5 epochs.

Use a snapshot of the Queue for the complete next epoch. Queue entries continue to update during that epoch, but prototype and neighbor routing remain fixed until the next slow refresh. This restriction does not apply to the differentiable prototypes used by `L_proto`.

If a valid prototype/neighbor snapshot is not yet available, fall back to self-reconstruction. Log this fallback rather than inventing a ground-truth neighbor.

## 7. OSE prototype construction

For class \(k\), obtain the normalized exemplar feature \(u_k\). Use a momentum feature at a slow routing refresh and an online feature when computing the training loss.

\[
u_k=\operatorname{Normalize}(E(e_k)).
\]

Score every Queue feature \(q_r\) using the paper's discriminative rule:

\[
s_k(r)=\alpha\langle u_k,q_r\rangle-
(1-\alpha)\max_{c\ne k}\langle u_c,q_r\rangle.
\]

Following the AimCLR OSE P1 setting, assign every Queue slot to the single class with the largest discriminative score, then retain the classwise top \(K\) owned slots. Combine the exemplar and selected neighbors with soft weights based on raw similarity to the exemplar:

\[
p_k=\sum_{q\in\{u_k\}\cup\mathcal R_k}
\frac{\exp(\langle u_k,q\rangle)}
{\sum_{q'}\exp(\langle u_k,q'\rangle)}q.
\]

Recommended initial values:

```yaml
ose_topk: 4
ose_alpha: 0.75
```

Treat slow routing prototypes as stop-gradient state. Rebuild loss prototypes every iteration from online exemplar embeddings so that prototype alignment and dispersion propagate gradients. Do not make either form an `nn.Parameter`.

## 8. Soft prototype alignment and dispersion

For an online source feature \(h_i\) and its EMA-teacher feature \(h'_i\), compute student and teacher class distributions:

\[
q_i(k)=\operatorname{softmax}(h_i p_k/\tau_s),\qquad
q'_i(k)=\operatorname{softmax}(h'_i\operatorname{sg}(p_k)/\tau_t).
\]

Use the paper's soft cross-entropy alignment rather than a hard pseudo-label Anchor loss:

\[
\mathcal L_{\mathrm{align}}=-\frac{1}{B}\sum_i\sum_k
\operatorname{sg}(q'_i(k))\log q_i(k).
\]

Use the paper's prototype dispersion term:

\[
\mathcal L_{\mathrm{dispersion}}=
\frac{1}{C(C-1)}\sum_{k\ne c}
\frac{\langle p_k,p_c\rangle}{\tau_s}.
\]

Thus \(\mathcal L_{\mathrm{proto}}=\mathcal L_{\mathrm{align}}+\mathcal L_{\mathrm{dispersion}}\). The exemplar label determines only its class prototype; it is never used as a direct cross-entropy target. Teacher confidence is used only as a safety gate for peer routing.

Recommended initial configuration:

```yaml
ose_assignment_confidence: 0.8
ose_tau_s: 0.1
ose_tau_t: 0.04
lambda_ose: 1.0
```

## 9. Neighbor-map construction

At every slow refresh, assign Queue entries to their nearest routing prototype and retain only assignments above `ose_assignment_confidence`. For each valid source ID represented in the Queue, use the P1 neighbors owned by its predicted class as its candidate set. Exclude the source ID itself. If that leaves no candidate, fall back to self-reconstruction.

Store immutable dataset IDs in `neighbor_map`, never Queue slots.

The labeled exemplars are excluded from the unlabeled sampler and Queue. Random selection within the P1 neighborhood prevents a single peer from becoming a fixed reconstruction template.

## 10. Epoch-dependent target routing

Let \(E=\texttt{args.epochs}\), \(S=\texttt{ose_start_epoch}\), and epoch
index \(e\in[0,E-1]\). Before epoch \(S\), define
\(p_{\mathrm{peer}}=0\) and \(p_{\mathrm{self}}=1\). For \(e\geq S\), define:

\[
r(e)=\frac{e-S}{\max(E-S-1,1)}.
\]

Set:

\[
p_{\mathrm{peer}}(e)=0.1+0.8r(e),
\]

\[
p_{\mathrm{self}}(e)=0.9-0.8r(e)
=1-p_{\mathrm{peer}}(e).
\]

For the repository's current 400-epoch setting:

```text
epoch 0-99: native MacDiff only; self=1.000, peer=0.000
epoch 100:  self=0.900, peer=0.100
epoch 399: self=0.100, peer=0.900
```

Do not use `warmup_epochs` as the start of this routing schedule.
`warmup_epochs` belongs only to the learning-rate schedule; `ose_start_epoch`
independently controls the OSE and peer-routing stage.

At every sample:

```python
requested_peer = random_uniform() < p_peer(epoch)
use_peer = requested_peer and has_valid_neighbor
```

If `requested_peer` is true but no valid peer exists, use self-reconstruction.

Track both:

- planned peer probability;
- effective peer fraction after confidence filtering and fallback.

### Important: no dynamic loss weighting

Do not do this:

```python
loss = p_self * loss_self + p_peer * loss_peer
```

The probability schedule only selects `target_x`. Every selected sample contributes one ordinary diffusion loss with equal per-sample weight.

## 11. Feeder and pair loading

Extend the training feeder so it can receive a frozen `neighbor_map` for an epoch and return a peer target without recursive `__getitem__` calls.

Recommended return structure:

```python
source_x, source_aug, peer_x, source_id, peer_id, has_peer, source_label, peer_label
```

Requirements:

1. factor the existing sample preprocessing into a helper such as `_load_processed(index)`;
2. use `_load_processed` for both source and peer;
3. do not call `__getitem__` recursively;
4. preserve immutable dataset IDs;
5. choose one candidate randomly from `neighbor_map[source_id]`;
6. if no candidate exists, return `source_x` as a safe placeholder and `has_peer=False`;
7. keep source augmentation behavior unchanged;
8. the peer target should be the normally preprocessed clean target, not the source augmentation tensor.
9. `source_label` and `peer_label` are diagnostic-only outputs. The training
   engine may compare them after the route is realized, but must never pass
   them into the model, routing logic, loss, or backward graph.

The DataLoader currently uses `persistent_workers=False` implicitly. Update the dataset's epoch state before creating the next epoch iterator. If persistent workers are enabled later, add an explicit shared-state or sampler mechanism; otherwise worker-side copies may keep a stale neighbor map.

In DDP, construct/broadcast the same slow-state neighbor map to all ranks before the epoch begins.

## 12. Diffusion target and condition

For each source sample, select:

\[
x_i^{\mathrm{target}}=
\begin{cases}
x_i, & \text{self route},\\
x_j, & \text{peer route}.
\end{cases}
\]

Add noise exactly as MacDiff does:

\[
x_{i,t}^{\mathrm{target}}
=\sqrt{\bar\alpha_t}x_i^{\mathrm{target}}
+\sqrt{1-\bar\alpha_t}\epsilon.
\]

### 12.1 Self condition

Use the unchanged MacDiff global-local sequence:

```python
z_self = restore_visible_local_and_fill_masked_with_global(
    latent, cls_token, ids_restore
)
```

### 12.2 Peer condition

Use the source global representation at every target position:

```python
z_peer = cls_token.repeat(1, total_tokens, 1)
```

Do not apply source `ids_restore` to peer local tokens. This avoids imposing source temporal phase on a different target sequence.

For a mixed batch:

```python
z = torch.where(
    use_peer[:, None, None],
    z_peer,
    z_self,
)
```

Refactor `forward_decoder` minimally so it can accept either a preconstructed `z` or `use_peer`. Prefer passing a preconstructed `z`, because it keeps condition assembly explicit and testable.

### 12.3 Masking rule

Use random masking for this method. The checked NTU60 configuration already sets:

```yaml
motion_aware_tau: -1
```

Do not use source-derived motion-aware masks for a different peer target. If peer routing is enabled while `motion_aware_tau > 0`, fail with a clear configuration error or force the peer branch to use a separately generated random mask.

The source random mask may still determine which target positions contribute to loss. Because it is random and independent of source motion content, this does not introduce a systematic motion-phase correspondence.

## 13. Loss

Before `ose_start_epoch`, retain only the MacDiff noise-prediction loss and
token uniformity loss:

\[
\mathcal L=\mathcal L_{\mathrm{diff}}+\lambda_u\mathcal L_{\mathrm{uniformity}}.
\]

From `ose_start_epoch`, add the complete OSE prototype objective:

\[
\mathcal L=
\mathcal L_{\mathrm{diff}}
+\lambda_u\mathcal L_{\mathrm{uniformity}}
+\lambda_{\mathrm{ose}}\left(
\mathcal L_{\mathrm{align}}+\mathcal L_{\mathrm{dispersion}}
\right).
\]

All \(\lambda\) values are fixed while their corresponding objective is active.
They must not depend on `p_self`, `p_peer`, epoch progress, or effective route
counts.

The diffusion term is one batch mean over the selected target for each sample:

\[
\mathcal L_{\mathrm{diff}}
=\frac{1}{B}\sum_i
\frac{\sum_l m_{i,l}
\|\epsilon_{i,l}-\hat\epsilon_{i,l}\|^2}
{\sum_lm_{i,l}}.
\]

Do not compute separate dynamically weighted self and peer losses.

## 14. Model forward pseudocode

```python
def forward(
    self,
    source_x,
    source_aug,
    peer_x,
    source_ids,
    peer_ids,
    has_peer,
    epoch,
    total_epochs,
    mask_ratio=0.90,
    motion_aware_tau=-1,
):
    # Existing MacDiff source preprocessing.
    source_norm, source_aug_norm, source_orig = preprocess_source(
        source_x, source_aug
    )
    peer_norm = preprocess_target(peer_x)

    # Existing encoder behavior: 10% visible local tokens and their mean.
    latent, cls_token, mask, ids_restore = self.forward_encoder(
        source_aug_norm,
        x_orig=source_orig,
        mask_ratio=mask_ratio,
        motion_aware_tau=motion_aware_tau,
    )

    # The EMA/Queue warm up in both stages.
    teacher_feat = self.ose_momentum_encoder(source_norm)
    ose_active = epoch >= self.ose_start_epoch
    ose = zero_ose_losses()
    use_peer = torch.zeros_like(has_peer)
    if ose_active:
        # OSE loss exists only in the second stage.
        global_feat = F.normalize(cls_token.squeeze(1), dim=-1)
        exemplar_feat = self.online_exemplar_features()
        ose = self.ose_prototype_loss(
            global_feat, teacher_feat, exemplar_feat
        )
        confidence = ose["confidence"]

        # Runtime routing schedule; selection only, not loss weighting.
        progress = (epoch - self.ose_start_epoch) / max(
            total_epochs - self.ose_start_epoch - 1, 1)
        p_peer = 0.1 + 0.8 * progress
        requested_peer = torch.rand_like(confidence) < p_peer
        use_peer = requested_peer & has_peer & (
            confidence >= self.ose_assignment_confidence
        )

    # Pick one target per sample.
    target_gt = torch.where(
        use_peer[:, None, None, None, None],
        peer_norm,
        source_norm,
    )

    # Existing MacDiff diffusion process.
    t, _ = self.schedule_sampler.sample(target_gt.shape[0], target_gt.device)
    noise = torch.randn_like(target_gt)
    target_t = self.diffusion.q_sample(target_gt, t, noise=noise)

    # Self: global-local. Peer: global-only.
    z_self = self.build_global_local_condition(
        latent, cls_token, ids_restore
    )
    z_peer = cls_token.repeat(1, z_self.shape[1], 1)
    z = torch.where(use_peer[:, None, None], z_peer, z_self)

    pred = self.forward_decoder_with_condition(target_t, z=z, t=t)
    loss_diff = self.forward_loss(noise, pred, mask, t)
    loss_uni = token_uniformity_loss(...)

    loss = loss_diff + self.lambda_loss_uni * loss_uni
    if ose_active:
        loss = loss + self.lambda_ose * ose["proto"]

    aux = {
        "loss_diff": loss_diff.detach(),
        "loss_uniformity": loss_uni.detach(),
        "loss_ose_align": ose["align"].detach(),
        "loss_ose_dispersion": ose["dispersion"].detach(),
        "p_peer_planned": p_peer,
        "peer_fraction_effective": use_peer.float().mean().detach(),
        "ose_confidence": confidence.mean().detach(),
    }
    return loss, pred, mask, aux, teacher_feat.detach(), use_peer.detach()
```

The engine compares the returned `use_peer` mask with the two diagnostic labels
to measure realized cross-reconstruction label accuracy. This happens after the
model has selected the route and therefore cannot influence that decision.

The exact tensor shapes must respect MacDiff's `one_person` behavior. OSE identities are sequence-level dataset IDs, not flattened person IDs.

## 15. Momentum Queue update

After an optimizer step:

```python
with torch.no_grad():
    update_momentum_encoder(online_encoder, momentum_encoder, momentum)
    momentum_global = momentum_encoder.encode_global_visible(
        source_aug,
        mask_ratio=0.90,
    )
    gathered_feat = concat_all_gather(momentum_global)
    gathered_ids = concat_all_gather(source_ids)
    ose_queue.enqueue(gathered_feat, gathered_ids)
```

If it is significantly simpler, the momentum forward may be calculated before the optimizer step and enqueued after the step. Be consistent and document the choice. Do not enqueue online features that carry gradients.

Deduplicate repeated dataset IDs when building the slow snapshot. If a dataset ID appears multiple times in the FIFO Queue, retain its most recent feature.

## 16. Epoch-boundary ordering

Recommended ordering:

```text
before epoch e:
    if e >= ose_start_epoch and slow state is due and Queue is usable:
        deduplicate Queue snapshot by dataset ID
        refresh exemplar features
        rebuild prototypes
        assign Queue entries
        rebuild neighbor_map
        broadcast slow state in DDP
        install neighbor_map into training dataset

during epoch e:
    if e < ose_start_epoch:
        use native MacDiff loss and self target only
    else:
        add OSE loss and route with p_peer(e)
    train online encoder and decoder
    update EMA encoder after optimizer steps
    enqueue EMA features and IDs after optimizer steps

after epoch e:
    save normal checkpoint plus OSE state
```

When resuming, restore:

- momentum encoder;
- Queue feature buffer;
- Queue ID buffer;
- Queue pointer/full flag;
- prototype snapshot;
- exemplar feature bank;
- neighbor-map version or rebuild it before continuing.

The routing probability must use the absolute resumed epoch, so a resumed run follows the same schedule.

## 17. Required configuration additions

Add the following fields to pretraining YAML files, initially enabling the method only in a new dedicated config rather than silently changing all official baselines:

```yaml
# OSE-guided cross-instance diffusion
ose_exemplar_indices: path/to/exemplar_indices.json

# Fast state
ose_momentum: 0.999
ose_queue_size: 32768
ose_start_epoch: 100
ose_exemplar_checkpoint: false

# Prototype construction and slow routing state
ose_refresh_interval: 1
ose_topk: 4
ose_alpha: 0.75

# Soft OSE alignment and routing confidence
ose_tau_s: 0.1
ose_tau_t: 0.04
ose_assignment_confidence: 0.8

# Fixed loss weights
lambda_ose: 1.0

# Target-routing endpoints; interpolate over ose_start_epoch..args.epochs-1
self_prob_start: 0.9
self_prob_end: 0.1
peer_prob_start: 0.1
peer_prob_end: 0.9
```

Assert at startup that the self and peer probabilities sum to one at both endpoints.

`--ose_exemplar_checkpoint` checkpoints only the online exemplar encoder
blocks. It must not checkpoint or detach the prototype output, and it must not
affect the source encoder branch. This preserves alignment and dispersion
gradients while trading one exemplar encoder recomputation during backward for
lower activation memory.

Do not add a separate 500-epoch assumption. With the repository default, the
active schedule covers epochs 100-399. If a future config changes `epochs` or
`ose_start_epoch`, the same endpoint behavior must hold automatically.

## 18. Suggested code organization

### `model/ose_memory.py` (new)

Implement:

- momentum-encoder update helper;
- FIFO feature/ID Queue;
- Queue deduplication by dataset ID;
- discriminative P1 prototype construction;
- class assignment and confidence calculation;
- soft student/EMA-teacher alignment;
- differentiable paper dispersion loss;
- neighbor-map construction;
- DDP-safe slow-state broadcast;
- state-dict save/load.

### `model/transformer_macdiff.py`

Implement:

- an explicit encoder method returning visible local tokens and global token;
- refactoring of condition construction into `build_global_local_condition`;
- decoder entry accepting a preconstructed condition `z`;
- global-only peer condition;
- mixed target selection;
- fixed OSE loss additions;
- auxiliary logging values.

Keep these changes in the dedicated OSE configuration and training path; no compatibility branch is required.

### `feeder/feeder_ntu.py`

Implement:

- immutable dataset-ID handling;
- `_load_processed(index)` helper;
- epoch-level `neighbor_map` installation;
- peer target sampling;
- `has_peer` return flag;
- no use of non-exemplar labels.

Preserve the old return signature when the feature is disabled, or add a compatible wrapper/collate function.

### `engine_pretrain.py`

Implement:

- passing source IDs, peer targets, peer IDs, epoch, and total epochs;
- EMA/Queue updates only after optimizer steps;
- planned/effective route logging;
- component-loss logging;
- prototype/neighbor refresh hooks at epoch boundaries rather than iterations.

### `main_pretrain.py`

Implement:

- OSE argument parsing/config loading;
- exemplar mapping validation;
- slow-state initialization;
- DDP synchronization;
- dataset neighbor-map update before each epoch;
- OSE checkpoint state save/load.

### Config

Create a new config such as:

```text
config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml
```

Do not overwrite the official MacDiff baseline config until the new implementation is validated.

## 19. Logging requirements

Log at least:

- total loss;
- diffusion loss;
- token uniformity loss;
- OSE prototype/alignment loss;
- OSE dispersion loss;
- planned self probability;
- planned peer probability;
- effective self fraction;
- effective peer fraction;
- mean assignment confidence;
- fraction above confidence threshold;
- number of valid neighbor-map entries;
- Queue fill ratio;
- prototype pairwise cosine mean/max;
- neighbor-map/prototype version.
- offline neighbor label accuracy over all frozen source-to-candidate edges;
- offline cross-reconstruction label accuracy over realized peer targets.

The two offline accuracies may read the dataset's ground-truth labels only
after routing decisions have been made. Labels must remain outside the model
forward, prototype construction, confidence gate, neighbor selection, losses,
and backward graph. Log the corresponding edge/target counts so an empty
denominator is not mistaken for meaningful accuracy.

The effective peer fraction will usually be lower than the planned probability early in training because invalid or low-confidence routes fall back to self-reconstruction. This is expected and must be visible in logs.

## 20. Validation and tests

### Unit tests

1. **Routing endpoints**
   - `epoch < ose_start_epoch`: self 1.0, peer 0.0 and no OSE loss;
   - `epoch=ose_start_epoch`: self 0.9, peer 0.1;
   - `epoch=epochs-1`: self 0.1, peer 0.9;
   - probabilities sum to one;
   - works for arbitrary `args.epochs`, including 400.

2. **No loss scaling by routing probability**
   - verify selected self and peer samples use the same per-sample diffusion-loss weight.

3. **Queue identity correctness**
   - FIFO wraparound does not change the dataset ID associated with a feature;
   - neighbor-map values are dataset IDs, not Queue positions.

4. **DDP gathering correctness**
   - gathered features and IDs retain identical ordering;
   - all ranks receive the same prototype snapshot and neighbor map.

5. **Condition construction**
   - self samples contain restored local tokens at visible positions and global tokens elsewhere;
   - peer samples contain the identical global token at every position;
   - peer condition is independent of `ids_restore`.

6. **No time-index coupling in peer condition**
   - permuting source `ids_restore` must not alter `z_peer`.

7. **Fallback behavior**
   - missing/low-confidence peer yields self target and self condition;
   - no NaN when no sample passes the teacher-confidence threshold.

8. **Label isolation**
   - changing non-exemplar ground-truth labels does not change prototypes, assignments, neighbors, routing, or loss.

9. **Checkpoint resume**
   - Queue, IDs, pointer, EMA encoder, prototypes, and routing epoch restore correctly.

### Smoke tests

1. Run a tiny single-GPU job for two epochs.
2. Run a tiny two-GPU DDP job.
3. Verify gradients reach the online encoder from:
   - self diffusion;
   - peer diffusion through global-only condition;
   - soft OSE alignment;
   - prototype dispersion through online exemplar features.
4. Verify slow routing prototypes remain buffers without `.grad`.

## 21. Required ablations

At minimum compare:

1. official MacDiff;
2. MacDiff + OSE soft prototype loss only;
3. MacDiff + fixed Exemplar target diffusion;
4. MacDiff + randomized same-class peer diffusion;
5. peer diffusion with global-local condition;
6. peer diffusion with global-only condition (main design);
7. constant 0.5/0.5 routing;
8. scheduled 0.9/0.1 to 0.1/0.9 routing (main design).

This separates improvements from OSE class grounding, cross-instance reconstruction, avoidance of fixed-template memorization, global-only conditioning, and the routing curriculum.

## 22. Non-goals

Do not add the following in the first implementation:

- cross-attention between source local tokens and peer target tokens;
- DTW or explicit temporal alignment;
- timestep-dependent condition-reliance loss;
- dynamically changing loss coefficients;
- per-iteration prototype recomputation;
- fixed Exemplar as the only peer target;
- hardcoded 500-epoch schedules;
- use of full training labels beyond the declared exemplar map.

## 23. Final method summary

The final method is:

\[
\boxed{
\text{one-shot Exemplar}
\rightarrow
\text{differentiable OSE prototype loss + slow routing snapshot}
\rightarrow
\text{teacher class assignment}
\rightarrow
\text{random high-confidence same-class peer}
\rightarrow
\text{global-only cross-instance MacDiff}
}
\]

Training first warms the representation and memory bank with native MacDiff,
then shifts toward class-grounded cross-instance reconstruction:

```text
epoch 0-99: native MacDiff loss, self 1.0 / peer 0.0
epoch 100: OSE enabled, self 0.9 / peer 0.1
final epoch: OSE enabled, self 0.1 / peer 0.9
```

The active routing schedule uses the repository's configured `args.epochs` and
`ose_start_epoch` and changes only target-routing probabilities. The online
encoder and Queue evolve during both stages; prototypes and neighbor maps are
created only in the OSE stage and update at the configured epoch-level slow
refresh. At downstream evaluation, discard the Queue, OSE machinery, momentum
encoder, and diffusion decoder, and use the learned semantic encoder in the
same way as MacDiff.
