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

## 4. Extract captions from the complete train split with Transformers

This path uses the same `transformers` + `qwen-vl-utils` inference flow as the
single-sample pilot and does not require vLLM. The model is loaded once and kept
resident while a process works through its shard. On two 24 GB GPUs, run two
independent processes: GPU 0 handles even sample indices and GPU 1 handles odd
sample indices. Each process writes its own JSONL file.

First verify label-free loading, actor counting, 32-frame rendering, prompt
expansion, and sharding without loading the model:

```bash
python tools/vlm_pilot/caption_qwen3vl_train_transformers.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_path vlm_pilot/dry_run.jsonl \
  --num_shards 2 \
  --shard_id 0 \
  --max_samples 8 \
  --dry_run
```

Then run a small two-GPU pilot in two terminals. `--max_samples 8` means the
global range `train_0` through `train_7`; each process receives half:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_train_transformers.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --output_path vlm_pilot/ntu60_xsub_train_captions_shard0.jsonl \
  --num_shards 2 --shard_id 0 --num_frames 32 --sample_fps 8 \
  --max_samples 8
```

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python tools/vlm_pilot/caption_qwen3vl_train_transformers.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --output_path vlm_pilot/ntu60_xsub_train_captions_shard1.jsonl \
  --num_shards 2 --shard_id 1 --num_frames 32 --sample_fps 8 \
  --max_samples 8
```

Remove `--max_samples 8` for the full split. Add `--resume` when restarting;
only records whose status is `accepted` are skipped, while invalid and failed
samples are tried again and appended for auditability. Keep the same
`--num_shards` and `--shard_id` for a resumed output file.

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

The batch script reads only `x_train`; it never opens `y_train`. Frames are
rendered and preprocessed in memory, so the model prompt contains neither a
sample filename nor an action label. Concatenate the two shard files only after
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
