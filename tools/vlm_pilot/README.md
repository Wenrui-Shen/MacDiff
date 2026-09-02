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

Inspect `preview.gif` and `preview_middle.png`. The three panels are per-frame
root-centered X-Y, Z-Y, and X-Z projections. Together they retain all three XYZ
axes while deliberately removing the primary actor's global translation. For a
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

The default prompt is `skeleton_motion_prompt_v1.txt`. It preserves the
person-specific `persons` array while requiring a precise `main_part`, complete
`motion`, separate `beginning`/`middle`/`end`, the strongest anatomical
`interaction`, and a consistent final `text`. The earlier v0 file is retained
only for historical comparison.

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
  --output_root vlm_pilot/ntu60_xsub_train_rendered_v1 \
  --num_frames 32 \
  --sample_fps 8 \
  --resume
```

The GIF representation avoids writing 32 separate PNG files per sample. For
NTU60 XSub, allow roughly 10-20 GB depending on the motion content and
filesystem. `--resume` validates both the GIF and matching render configuration
before skipping a sample.

Verify the persisted inputs and expanded prompt without loading Qwen:

```bash
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v1 \
  --output_path vlm_pilot/rendered_dry_run.jsonl \
  --num_shards 2 --shard_id 0 --dry_run
```

Then run two independent caption processes. Each GPU loads one model; GPU 0
handles even indices and GPU 1 handles odd indices:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v1 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v1.txt \
  --output_path vlm_pilot/ntu60_xsub_train_person_captions_v2_shard0.jsonl \
  --num_shards 2 --shard_id 0 --resume
```

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v1 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v1.txt \
  --output_path vlm_pilot/ntu60_xsub_train_person_captions_v2_shard1.jsonl \
  --num_shards 2 --shard_id 1 --resume
```

Caption resume skips only accepted records generated with the same model and
prompt SHA-256. Invalid and failed samples are retried and appended for audit.
The older `caption_qwen3vl_train_transformers.py` remains available as the
single-stage in-memory path, but it is no longer the recommended full run.

Each JSONL record includes `sample_id` (`train_<index>`), prompt SHA-256,
rendering and inference settings, raw response, parsed caption, retry history,
and validation status. Only an `accepted` record has a non-empty `texts` list:

```json
{
  "sample_id": "train_42",
  "status": "accepted",
  "texts": [
    {"person_index": 0, "color": "red", "text": "..."},
    {"person_index": 1, "color": "blue", "text": "..."}
  ]
}
```

Neither stage reads `y_train`. The model prompt contains neither a sample
filename nor an action label. Concatenate the two caption shard files only after
both processes finish; use `sample_index` when ordered records are required.

## 5. Randomly inspect accepted captions

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
