#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Single GPU; effective batch 64 matches the original 2 GPUs x 32 invocation.
# These defaults have not been profiled on a 4090. Override batch/accum together.
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/ntu60_xsub_macdiff_text}
TEXT_CACHE=${TEXT_CACHE:-vlm_pilot/ntu60_xsub_clip_cache_v1}
BATCH_SIZE=${BATCH_SIZE:-16}
ACCUM_ITER=${ACCUM_ITER:-4}

python -u main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --text_cache "$TEXT_CACHE" --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR/tensorboard" --batch_size "$BATCH_SIZE" --accum_iter "$ACCUM_ITER" "$@"
