#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 EXEMPLAR_JSON [CONFIG] [OUTPUT_ROOT]"
    exit 2
fi

EXEMPLAR_JSON=$1
CONFIG=${2:-./config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml}
OUTPUT_ROOT=${3:-./output_dir/ose_memory_profile_$(date +%Y%m%d_%H%M%S)}
PROFILE_EPOCHS=${PROFILE_EPOCHS:-2}
PROFILE_STEPS=${PROFILE_STEPS:-20}
PROFILE_BATCH_SIZE=${PROFILE_BATCH_SIZE:-32}
PROFILE_NUM_WORKERS=${PROFILE_NUM_WORKERS:-4}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
    --config "$CONFIG"
    --ose_exemplar_indices "$EXEMPLAR_JSON"
    --ose_start_epoch 0
    --epochs "$PROFILE_EPOCHS"
    --max_train_steps "$PROFILE_STEPS"
    --batch_size "$PROFILE_BATCH_SIZE"
    --accum_iter 1
    --num_workers "$PROFILE_NUM_WORKERS"
    --output_dir ""
    --model model.transformer_macdiff.Transformer
)

run_case() {
    local case_name=$1
    shift
    local case_dir="$OUTPUT_ROOT/$case_name"
    mkdir -p "$case_dir"
    echo "Running $case_name"
    set +e
    python main_pretrain.py \
        "${COMMON_ARGS[@]}" \
        --log_dir "$case_dir/tensorboard" \
        "$@" 2>&1 | tee "$case_dir/console.log"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

no_checkpoint_status=0
checkpoint_status=0
run_case no_checkpoint || no_checkpoint_status=$?
run_case checkpoint --ose_exemplar_checkpoint || checkpoint_status=$?

echo "Memory summary (per-epoch global max across ranks):"
grep -H "CUDA peak memory" "$OUTPUT_ROOT"/*/console.log || true
echo "Exit status: no_checkpoint=$no_checkpoint_status checkpoint=$checkpoint_status"

if [[ $no_checkpoint_status -ne 0 || $checkpoint_status -ne 0 ]]; then
    exit 1
fi
