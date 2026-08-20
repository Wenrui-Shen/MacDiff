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

## 4. Extract captions from the complete train split with vLLM

Use a separate modern environment; do not upgrade the repository's legacy
training environment in place. Qwen3-VL requires vLLM 0.11 or newer:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U "vllm>=0.11.0" transformers qwen-vl-utils==0.0.14 Pillow
```

First verify label-free loading, actor counting, the 32-frame renderer, and the
expanded prompt without loading a model:

```bash
python tools/vlm_pilot/caption_qwen3vl_train_vllm.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_path vlm_pilot/ntu60_xsub_train_captions.jsonl \
  --max_samples 1 \
  --dry_run
```

Then run a small GPU pilot. The model remains resident for every selected
sample; `--max_samples 8` limits this validation run to `train_0` through
`train_7`:

```bash
OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_train_vllm.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --output_path vlm_pilot/ntu60_xsub_train_captions.jsonl \
  --num_frames 32 \
  --sample_fps 8 \
  --batch_size 1 \
  --max_samples 8
```

After inspecting those records, continue through all remaining `x_train`
samples. `--resume` skips only records whose status is `accepted`; failed or
invalid records are attempted again and appended for auditability:

```bash
OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_train_vllm.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --output_path vlm_pilot/ntu60_xsub_train_captions.jsonl \
  --num_frames 32 \
  --sample_fps 8 \
  --batch_size 1 \
  --resume
```

For a checkpoint that needs both GPUs, expose both devices and set
`--tensor_parallel_size 2`. Increase `--batch_size` only after measuring memory
on the real server; one 32-frame sample is the safe default for a 24 GB card.

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
sample filename nor an action label.
