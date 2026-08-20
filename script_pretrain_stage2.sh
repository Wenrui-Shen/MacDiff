#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml}"
STAGE1_CHECKPOINT="${1:-./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff_stage2_seed0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/tensorboard}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-10237}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Missing MacDiff Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi
echo "Running MacDiff Stage2 unit tests"
"${PYTHON_BIN}" -m unittest tests.test_stage2

echo "Starting independent 100-epoch MacDiff RSDG Stage2"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.launch \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        main_pretrain_stage2.py \
        --config "${CONFIG}" \
        --stage1_weights "${STAGE1_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --batch_size "${BATCH_SIZE}"
else
    "${PYTHON_BIN}" main_pretrain_stage2.py \
        --config "${CONFIG}" \
        --stage1_weights "${STAGE1_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --batch_size "${BATCH_SIZE}"
fi
