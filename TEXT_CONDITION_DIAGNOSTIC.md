# Stage1 text remap and conditioning diagnostics

This is a read-only diagnostic for the current `per_person_v1`, `one_person=True`
text checkpoints. It neither updates parameters nor regenerates the existing
CLIP cache. It loads the full model strictly and verifies checkpoint/cache/data
provenance. Use the existing `macdiff` environment and run from the server project
directory. No CLIP model, Transformers download, new loss, or training is needed.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u diagnose_text_conditioning.py --checkpoint output_dir/ntu60_xsub_macdiff_person_text/checkpoint-200.pth
```

The checkpoint path is the only required argument. Outputs default to
`output_dir/ntu60_xsub_macdiff_person_text/checkpoint-200_text_diagnostic/`.
Existing output directories are refused; use a new `--output-dir` for reruns.
Training settings, data path, and cache path come from the checkpoint. Override
relocated paths with `--data-path` and `--text-cache`; content checks still apply.

Defaults: 256 fixed randomly selected train descriptions, person 0 only, FP32,
batch size 8, three noise/mask/permutation repeats, timesteps 100/500/900.
These are sampling points, not averages over entire timestep ranges. Actual
alpha-cumprod values are saved. Cache validation reads existing files and may
take several minutes. GPU memory/runtime have not been measured on the server.

For an initial shorter run, add `--samples 128 --repeats 1`. To inspect text alone,
add `--geometry-only` (dataset/cache identity validation is still performed).
If memory is limited, reduce `--batch-size` to 4 and use that setting for every
checkpoint comparison. Batch size changes matched donor groups and sampling.

## What is fixed

- Same sample selection and seeds across checkpoints with identical arguments.
- One deterministic realization of the saved training crop/augmentation per sample.
- Only active cropped person-0 skeletons enter conditioning tests; no person swaps.
- Text swaps occur only among different samples with identical valid token
  positions/lengths, including the real EOS position. Masks, positions, and person
  metadata are checked for exact equality after permutation.
- Each repeat keeps targets, noisy inputs, skeleton mask and timestep identical
  across correct and shuffled conditions. No optimization or dropout is enabled.
- Timesteps within a repeat reuse base noises/masking. Repeats provide new noises,
  masks and cyclic random derangements, but not new temporal crops.
- Labels are used only for descriptive nearest-neighbor scores and donor-label
  reporting; they never select donors or provide training supervision.

Length matching restricts the diagnostic to a subset. The summary reports the
included sample count and excluded singleton/empty-person indices. Same-label
donors can remain; they may weaken the measured effect. Increase `--samples` if
coverage is low. This is not an estimate over the unrestricted training population.

## Geometry

`geometry.json` is saved before the expensive conditioning passes. Each view
reports raw mean channel variance, energy, variance/energy ratio, centered
effective rank (entropy of normalized covariance eigenvalues), cross-sample
cosine, and label agreement of the nearest neighbor versus a random neighbor.
The label metric is a descriptive train-subset diagnostic, not held-out LP.

Views:

- `global_clip`: cached person-0 EOS sentence vector.
- `global_remap`: actual `LN(MLP(sentence))` used by the model.
- `local_clip`: up to four fixed body-token identities per description, excluding
  EOS. All retained valid local tokens still participate in conditioning tests.
- `local_content_remap`: `LN(MLP(token))`, without extra person/position embeddings.
- `local_memory`: actual `LN(MLP(token)+person+position)`.

Relation retention compares the SAME selected token identities before and after
remapping. It reports pairwise cosine correlation and nearest-neighbor overlap,
both raw and after centering each representation space separately. Same-sample
neighbors are excluded, so adjacent tokens cannot trivially retrieve each other.
Local-memory statistics can contain positional diversity and are not interpreted
as pure semantic diversity. Channel-variance scales differ between CLIP and LN;
do not compare their raw variance alone. Effective rank on small samples is also
sample-limited; use fixed counts for comparisons.

## Conditioning

`summary.json` includes geometry and all conditioning results;
`paired_records.jsonl` records every target/donor/repeat and loss for auditing.

- `t2s_global_shuffled`: swap sentence vectors, retain correct local tokens.
- `t2s_local_shuffled`: swap local memory, retain correct sentence vectors.
- `t2s_all_shuffled`: swap both together from the same donor.
- `s2t_all`: retain the exact noisy text input and target; swap only skeleton
  pooled/visible-token memory. This reproduces the valid-token averaged MSE.
- `s2t_global` / `s2t_local`: separate diagnostics of the SAME reverse predictions,
  without changing the training loss or introducing separate training weights.

Forward MSE uses the original masked-patch averaging and timestep weights.
Reverse summaries weight by valid token counts, preserving the training loss
convention. The paired delta is `shuffled_mse - correct_mse`; positive values
mean correct pairing helps this denoising objective. Prediction-change MSE also
shows whether outputs changed even if objective values are nearly equal.

The approximate 95% interval uses a bootstrap over matched mini-batch clusters,
keeping repeats and within-batch donor dependencies together. It is a diagnostic
uncertainty estimate conditional on the selected samples/crops, not a significance
test over independent training runs. No universal percentage cutoff is imposed.

Interpretation:

- Low raw variance with preserved centered relations/rank and useful conditions
  does not justify declaring semantic collapse.
- Simultaneous degradation of content relations and condition utility across
  checkpoints strengthens suspicion, but still does not establish LP causality.
- Near-zero single-component effects with a positive all-text effect may indicate
  redundancy between global and local text, rather than both being unused.
- A near-zero effect at one timestep is inconclusive. Examine all noise levels,
  uncertainty, prediction changes, and sample coverage.
- Wrong-pair effects can reflect sample-specific nuisance information as well as
  action semantics. They are not proof of classification-relevant supervision.

To attribute LP changes causally to a variance intervention, subsequently compare
otherwise matched training runs with and without that single intervention and
evaluate converged LP at equal pretraining budgets. These diagnostics alone do
not establish that raising variance would improve recognition.

## Local validation

Six CPU tests in `tests/test_text_condition_diagnostic.py` cover the distinction
between scale and centered relations, exact-structure donor matching, identity
controls, reverse noisy-input invariance, loss weighting, and reproducible full
CLI runs on a synthetic NTU/cache/checkpoint fixture. Server CUDA/real checkpoint
results have not been obtained locally.
