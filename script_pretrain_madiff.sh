#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-10234}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/tensorboard}

# Native MacDiff baseline: OSE is disabled by default and this configuration
# uses feeder.feeder_ntu.Feeder.
python -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    main_pretrain.py \
    --config ./config/ntu60_xsub_joint/pretrain_madiff.yaml \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size 32 \
    --accum_iter 1 \
    --epochs 400 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --mask_ratio 0.9 \
    --model model.transformer_macdiff.Transformer
