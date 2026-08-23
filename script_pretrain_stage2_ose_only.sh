#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# OSE-only control: retain the successful low backbone LR and disable only
# the ReSA gradient. All three OSE objectives remain enabled.
export BACKBONE_LR="${BACKBONE_LR:-0.001}"
export HEAD_LR="${HEAD_LR:-0.25}"
export RESA_WEIGHT="0.0"
export OSE_LAMBDA="1.0"
export OSE_MIX_PROTO_WEIGHT="1.0"
export OSE_MIX_INS_WEIGHT="1.0"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3_oseonly}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/tensorboard}"

exec bash "${SCRIPT_DIR}/script_pretrain_stage2.sh" "$@"
