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
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_ITER="${ACCUM_ITER:-1}"
ENABLE_AMP="${ENABLE_AMP:-false}"
BACKBONE_LR="${BACKBONE_LR:-0.25}"
HEAD_LR="${HEAD_LR:-0.25}"
RESA_WEIGHT="${RESA_WEIGHT:-1.0}"
OSE_LAMBDA="${OSE_LAMBDA:-1.0}"
OSE_MIX_PROTO_WEIGHT="${OSE_MIX_PROTO_WEIGHT:-1.0}"
OSE_MIX_INS_WEIGHT="${OSE_MIX_INS_WEIGHT:-1.0}"
OSE_TAU_S="${OSE_TAU_S:-0.1}"
OSE_TAU_T="${OSE_TAU_T:-0.04}"
MASK_PROTOCOL="${MASK_PROTOCOL:-shared_qk_joint_v1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Missing MacDiff Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi
echo "Running MacDiff Stage2 unit tests"
"${PYTHON_BIN}" -m unittest tests.test_stage2

echo "Stage2 weights: ReSA=${RESA_WEIGHT}, OSE=${OSE_LAMBDA}, mix-proto=${OSE_MIX_PROTO_WEIGHT}, mix-ins=${OSE_MIX_INS_WEIGHT}"
echo "OSE temperatures: student=${OSE_TAU_S}, teacher=${OSE_TAU_T}"
echo "Stage2 mask protocol: ${MASK_PROTOCOL}"
echo "Stage2 micro-batch=${BATCH_SIZE}, accum_iter=${ACCUM_ITER}, AMP=${ENABLE_AMP}"
echo "Starting independent MacDiff Stage2 protocol ${MASK_PROTOCOL}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.launch \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        main_pretrain_stage2.py \
        --config "${CONFIG}" \
        --stage1_weights "${STAGE1_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --accum_iter "${ACCUM_ITER}" \
        --enable_amp "${ENABLE_AMP}" \
        --lr "${BACKBONE_LR}" \
        --head_lr "${HEAD_LR}" \
        --resa_weight "${RESA_WEIGHT}" \
        --ose_lambda "${OSE_LAMBDA}" \
        --ose_mix_proto_weight "${OSE_MIX_PROTO_WEIGHT}" \
        --ose_mix_ins_weight "${OSE_MIX_INS_WEIGHT}" \
        --ose_tau_s "${OSE_TAU_S}" \
        --ose_tau_t "${OSE_TAU_T}" \
        --mask_protocol "${MASK_PROTOCOL}"
else
    "${PYTHON_BIN}" main_pretrain_stage2.py \
        --config "${CONFIG}" \
        --stage1_weights "${STAGE1_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --accum_iter "${ACCUM_ITER}" \
        --enable_amp "${ENABLE_AMP}" \
        --lr "${BACKBONE_LR}" \
        --head_lr "${HEAD_LR}" \
        --resa_weight "${RESA_WEIGHT}" \
        --ose_lambda "${OSE_LAMBDA}" \
        --ose_mix_proto_weight "${OSE_MIX_PROTO_WEIGHT}" \
        --ose_mix_ins_weight "${OSE_MIX_INS_WEIGHT}" \
        --ose_tau_s "${OSE_TAU_S}" \
        --ose_tau_t "${OSE_TAU_T}" \
        --mask_protocol "${MASK_PROTOCOL}"
fi
