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
relative position remains visible.

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
codec behavior do not affect this first experiment.
