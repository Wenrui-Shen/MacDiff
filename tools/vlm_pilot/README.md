# One-sample skeleton-to-text pilot

This pilot deliberately separates deterministic visualization from VLM inference.
The renderer reads only skeleton arrays from the processed MAMP NPZ and never
loads labels. It also bypasses the random crop, flip, rotation, and noise in the
training feeder.

## 1. Render one sample

```bash
python tools/vlm_pilot/render_skeleton_sample.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --split train \
  --sample_index 0 \
  --num_frames 16 \
  --output_dir vlm_pilot/sample_000000
```

Inspect `preview.gif` and `preview_middle.png`. The two panels are per-frame
root-centered X-Y and Z-Y projections. Together they retain all three XYZ axes
while deliberately removing the primary actor's global translation. For a
two-person sample, both actors are translated by person 1's root, so their
relative position remains visible. Person 1 is uniformly red and person 2 is
uniformly blue. Joint circles are deliberately larger than the thinner bone
lines so that the VLM can distinguish joints from connections. Joints have no
outline, and the root joint uses the same marker size as every other joint.
The renderer records one actor when no blue skeleton is visible and two actors
when a blue skeleton is visible. The captioner reads this label-free structural
fact from `render_metadata.json` rather than asking the VLM to count repeated
views. If metadata is unavailable, it falls back to detecting blue pixels.

The rendering code can be checked without a dataset:

```bash
python tools/vlm_pilot/render_skeleton_sample.py \
  --demo \
  --output_dir vlm_pilot/demo
```

## 2. Install Qwen3-VL dependencies on Ubuntu

Use the CUDA-compatible PyTorch build already selected for the server, then:

```bash
pip install "transformers>=4.57.0" accelerate qwen-vl-utils==0.0.14 Pillow
```

The default pilot model is `Qwen/Qwen3-VL-2B-Instruct`. It minimizes the cost of
debugging the data path and prompt. Switch to 4B or 8B only after the pipeline
works and the rendered motion is readable.

## 3. Generate the sample-level description

```bash
python tools/vlm_pilot/caption_qwen3vl_sample.py \
  --frames_dir vlm_pilot/sample_000000/frames \
  --output_path vlm_pilot/sample_000000/caption.json
```

For a larger model:

```bash
python tools/vlm_pilot/caption_qwen3vl_sample.py \
  --frames_dir vlm_pilot/sample_000000/frames \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --attn_implementation flash_attention_2 \
  --output_path vlm_pilot/sample_000000/caption_qwen3vl_8b.json
```

The captioner passes an ordered PNG list as one video input, so FFmpeg and video
codec behavior do not affect this first experiment. The current schema returns
one non-empty `persons` entry for each visible skeleton. `person_index: 0` is
red and `person_index: 1` is blue; a single-person sample contains no synthetic
second-person or empty-string target.

The default prompt is `skeleton_motion_prompt_v2.txt`. Each person has one brief
`global` description and six fixed `local` descriptions: `head`, `torso`,
`left_arm`, `right_arm`, `left_leg`, and `right_leg`. The Chinese translation is
`skeleton_motion_prompt_v2_zh.txt`. The v0/v1 prompts remain available explicitly
through `--prompt_path`; their original response schemas are still validated.
The single-sample script saves only `persons` in its output JSON. Model, device,
prompt and settings are saved in a `.metadata.json` sidecar; the raw response and
validation result are saved in `.diagnostics.jsonl`.

## 4. Complete train split as two independent stages

The recommended full pipeline persists every visualization before any model is
loaded. Stage 1 reads only `x_train` and creates one `preview.gif` plus one
`render_metadata.json` under each `train_<index>` directory. Stage 2 reads only
those files; it never reopens the NPZ. This makes rendering independently
inspectable and reusable across prompt or model experiments.

Render all train samples once:

```bash
python tools/vlm_pilot/render_qwen3vl_train.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --num_frames 32 \
  --sample_fps 8 \
  --temporal_smooth savgol \
  --median_window 3 \
  --smooth_window 5 \
  --smooth_polyorder 2 \
  --max_interp_gap 2 \
  --resume
```

Temporal smoothing is applied independently to every person's full joint
trajectories before the 32 output frames are sampled.  A 3-frame median removes
isolated spikes, short internal gaps of at most 2 frames are interpolated, and
a dependency-free 5-frame quadratic Savitzky-Golay filter reduces jitter
without causal lag.  Pass `--temporal_smooth none` for an unsmoothed comparison.
Use a new output root when comparing with renders made before smoothing.

The GIF representation avoids writing 32 separate PNG files per sample. For
NTU60 XSub, allow roughly 10-20 GB depending on the motion content and
filesystem. `--resume` validates both the GIF and matching render configuration
before skipping a sample.

Verify the persisted inputs and expanded prompt without loading Qwen:

```bash
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --output_path vlm_pilot/rendered_dry_run.jsonl \
  --num_shards 2 --shard_id 0 --dry_run
```

The recommended dual-GPU entry point starts both workers, automatically resumes
saved samples, and merges accepted results into one sorted JSON array:

```bash
python tools/vlm_pilot/run_caption_dual_gpu.py --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct --output_dir vlm_pilot/ntu60_xsub_global_local_v2 --gpus 0,1 --expected_samples 40091
```

Use the Qwen inference environment, not the older MacDiff training environment.
The existing GIFs are reused. GPU 0 handles even sample indices and GPU 1 handles
odd indices, with one independent model per GPU. The launcher sets each worker's
`CUDA_VISIBLE_DEVICES` and uses a 1024-token generation ceiling for the new
global/local JSON, including two-person samples.

Run the same command after interruption: it checks model/prompt identity, skips
accepted samples, and retries unsuccessful samples. Only an incomplete final
JSONL line is backed up and removed automatically. Completed malformed lines
and metadata mismatches cause errors. Ctrl+C stops the workers and saves a
sorted partial result; an OS-released lock prevents simultaneous launchers from
writing this output directory. Missing indices are listed in the merged metadata
and produce a nonzero exit code, rather than reporting a complete run.

Outputs under `--output_dir`:
- `captions.json`: text-only JSON array, one sample per line, sorted by `sample_index`.
- `captions.metadata.json`: merged run information and completeness/missing-index summary.
- `shard0.jsonl` / `shard1.jsonl`: resumable per-GPU text results.
- `shard0.metadata.json` / `shard1.metadata.json`: per-worker run metadata.
- `shard0.diagnostics.jsonl` / `shard1.diagnostics.jsonl`: raw responses, errors and retries.
- `shard0.log` / `shard1.log`: worker terminal output.

Keep the directory intact for resume. To test the first four samples, add
`--max_samples 4`; remove that argument to continue the full run in the same
directory. Add `--merge_only` to merge existing results without loading models,
or `--dry_run` to inspect pending counts and worker commands without starting
generation. `--expected_samples 40091` also checks that the rendered input itself
contains the complete NTU60 XSub train index range.

Alternatively, run the workers in two terminals. Each GPU loads one model; GPU 0
handles even indices and GPU 1 handles odd indices:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v2.txt \
  --output_path vlm_pilot/ntu60_xsub_global_local_shard0.jsonl \
  --num_shards 2 --shard_id 0 --resume
```

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v2.txt \
  --output_path vlm_pilot/ntu60_xsub_global_local_shard1.jsonl \
  --num_shards 2 --shard_id 1 --resume
```

Caption resume reads the separate metadata file, checks the model, requested
revision and prompt SHA-256, then skips sample indices in the text-only output.
Invalid and failed samples are retried and recorded in the diagnostic sidecar.
Use a new output path for legacy manifests and the new global/local experiment.
The older `caption_qwen3vl_train_transformers.py` remains available as the
single-stage in-memory path, but it is no longer the recommended full run.

Batch output contains only a sample index and its generated person descriptions.
With `.jsonl`, each sample is a complete JSON object on its own line. With
`.json`, output is a valid JSON array with each sample on its own line. Only
accepted descriptions enter the result file. For example:

```json
{
  "sample_index": 42,
  "persons": [
    {
      "person_index": 0,
      "global": "The person raises the right hand toward the face, then lowers it.",
      "local": {
        "head": "The head remains upright.",
        "torso": "The torso remains upright.",
        "left_arm": "The left arm stays beside the torso.",
        "right_arm": "The right elbow bends as the hand rises toward the face, then lowers.",
        "left_leg": "The left leg remains straight.",
        "right_leg": "The right leg remains straight."
      }
    }
  ]
}
```

For `captions.jsonl` or `captions.json`, `captions.metadata.json` stores the model,
hardware, runtime versions, prompt and run settings, once per run.
`captions.diagnostics.jsonl` stores sample-specific raw responses, retries,
errors and timings. Keep the metadata with the result file for `--resume`.
The inspection utility accepts both formats and still reads legacy manifests.
The existing v2 CLIP token-cache builder remains the legacy word-token pipeline;
use `cache_clip_motion_text.py` for the new global/local sentence cache (v3).
It encodes seven complete sentences per person with fixed body-region slots,
stores unit-RMS FP32 features and is accepted by the existing training reader.
See [TEXT_CACHE_CHAIN_AUDIT.md](TEXT_CACHE_CHAIN_AUDIT.md) for the legacy-chain
audit, separate training configurations, resume behavior and server commands.

Neither stage reads `y_train`. The model prompt contains neither a sample
filename nor an action label. Concatenate the two caption shard files only after
both processes finish; use `sample_index` when ordered records are required.
Only JSONL shards can be concatenated directly; merge JSON arrays as JSON.

## 5. Randomly inspect accepted captions

### Single-GPU AWQ with offline vLLM over existing GIFs

The rendered-GIF entry point also accepts `--engine vllm`. It loads one engine
with tensor parallel size 1 and one active request. It preserves the GIF frames,
FPS metadata, prompt and validation; preprocessing is not repeated by vLLM.
This is sequential full-dataset generation, not a concurrent throughput benchmark.
AWQ is detected from checkpoint configuration. Engine/preprocessing errors are
written to JSONL and then stop the vLLM run; invalid captions receive bounded
validation retries. Use a fresh output file when changing engines or settings.

In an independent, compatible Linux vLLM environment, test train index 2:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --engine vllm \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v1.txt \
  --output_path vlm_pilot/30b_a3b_awq_test_2.jsonl \
  --start_index 2 --max_samples 1 --max_retries 0 \
  --num_shards 1 --shard_id 0 --max_model_len 16384 \
  --gpu_memory_utilization 0.90 --enforce_eager
```

Use `python -m json.tool vlm_pilot/30b_a3b_awq_test_2.jsonl` to inspect the raw
response as well as validation errors. For a full run remove `--start_index`
and `--max_samples`, set `--max_retries 2`, select a new output path, and add
`--resume`. Keep that full-run output path on subsequent restarts. Add
`--dry_run` to check one GIF and its expanded prompt without loading the model.

The 24GB/32-frame AWQ run still requires server verification. CPU-only adapter
checks: `python -m unittest tests.test_rendered_vllm`.

Read both shard files, deduplicate accepted results by `sample_index`, and print
five random samples with all person-level fields:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --num_samples 5
```

Add `--compact` to print only the final `text` fields, or `--seed 42` to make
the random selection reproducible.

If a visualization already exists, do not load the NPZ again. Read its
`render_metadata.json` and look up the matching accepted caption directly:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --visualization_dirs vlm_pilot/sample_000000 \
  --compact
```

Use `--sample_indices 0 42 108` when the desired train indices are already
known. Both lookup modes scan only the caption JSONL files and do not read
`x_train` or `y_train`.

To export labels and skeleton animations into a standalone review folder, add
the aligned NTU60 NPZ and a new output directory:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_dir vlm_pilot/random_caption_review_seed42 \
  --num_samples 10 \
  --seed 42
```

The root `index.html` shows every selected GIF beside its `y_train` class and
person captions. Each sample subdirectory also contains `preview.gif`,
`preview_middle.png`, `summary.txt`, and the complete `review.json`. The output
directory must not already exist, which prevents accidental review overwrites.
