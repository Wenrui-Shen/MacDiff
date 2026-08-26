#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Single combined control requested after the Stage1/Stage2 geometry audit:
# low-LR ReSA-only with exactly three temporal tokens visible per joint.
export BACKBONE_LR="${BACKBONE_LR:-0.001}"
export HEAD_LR="${HEAD_LR:-0.25}"
export RESA_WEIGHT="1.0"
export OSE_LAMBDA="0.0"
export OSE_MIX_PROTO_WEIGHT="0.0"
export OSE_MIX_INS_WEIGHT="0.0"
export OSE_TAU_S="0.1"
export OSE_TAU_T="0.04"
export MASK_PROTOCOL="shared_qk_per_joint_v1"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff_stage2_noaug_syncbn_lr1e3_resaonly_perjoint3}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/tensorboard}"

exec bash "${SCRIPT_DIR}/script_pretrain_stage2.sh" "$@"
