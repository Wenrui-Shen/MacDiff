#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Full-token OSE core: one online view, one EMA view, and epoch-cached EMA
# exemplar prototypes. ReSA and both mixed losses are structurally absent.
export CONFIG="${CONFIG:-./config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml}"
export BACKBONE_LR="${BACKBONE_LR:-0.001}"
export HEAD_LR="${HEAD_LR:-0.25}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ACCUM_ITER="${ACCUM_ITER:-16}"
export ENABLE_AMP="${ENABLE_AMP:-true}"
export RESA_WEIGHT="0.0"
export OSE_LAMBDA="1.0"
export OSE_MIX_PROTO_WEIGHT="0.0"
export OSE_MIX_INS_WEIGHT="0.0"
export OSE_TAU_S="0.1"
export OSE_TAU_T="0.04"
export MASK_PROTOCOL="dense_ose_proto_ema_v1"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff_stage2_dense_ose_proto_ema}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/tensorboard}"

exec bash "${SCRIPT_DIR}/script_pretrain_stage2.sh" "$@"
