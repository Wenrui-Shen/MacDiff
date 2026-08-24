#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Low-LR ReSA+OSE temperature control. Relative to the completed low-LR
# baseline, change only the EMA-teacher OSE temperature from 0.04 to 0.06.
export BACKBONE_LR="${BACKBONE_LR:-0.001}"
export HEAD_LR="${HEAD_LR:-0.25}"
export RESA_WEIGHT="1.0"
export OSE_LAMBDA="1.0"
export OSE_MIX_PROTO_WEIGHT="1.0"
export OSE_MIX_INS_WEIGHT="1.0"
export OSE_TAU_S="0.1"
export OSE_TAU_T="0.06"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3_resaose_taut006}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/tensorboard}"

exec bash "${SCRIPT_DIR}/script_pretrain_stage2.sh" "$@"
