#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-./config/ntu60_xsub_joint/linprobe_madiff.yaml}"
STAGE2_DIR="${1:-./output_dir/ntu60_xsub_macdiff_stage2_mask90_oseonly_lr5e4_gpu1_bs64}"
LP_ROOT="${LP_ROOT:-${STAGE2_DIR}_lp_early_final}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LP_EPOCHS="${LP_EPOCHS:-100}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Verify all four inputs before starting any LP run.
STAGE2_EPOCHS=(3 5 10 100)
for STAGE2_EPOCH in "${STAGE2_EPOCHS[@]}"; do
    TAG="$(printf '%03d' "${STAGE2_EPOCH}")"
    CHECKPOINT="${STAGE2_DIR}/checkpoint-${TAG}-backbone.pth"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing Stage2 backbone checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
done

mkdir -p "${LP_ROOT}"

SUMMARY_CSV="${LP_ROOT}/best_acc_summary.csv"
if [[ -f "${SUMMARY_CSV}" ]]; then
    SUMMARY_BACKUP="$(mktemp "${LP_ROOT}/best_acc_summary.previous.XXXXXX.csv")"
    cp -- "${SUMMARY_CSV}" "${SUMMARY_BACKUP}"
fi
printf 'stage2_epoch,checkpoint,best_acc\n' > "${SUMMARY_CSV}"

for STAGE2_EPOCH in "${STAGE2_EPOCHS[@]}"; do
    TAG="$(printf '%03d' "${STAGE2_EPOCH}")"
    CHECKPOINT="${STAGE2_DIR}/checkpoint-${TAG}-backbone.pth"
    RUN_DIR="${LP_ROOT}/checkpoint-${TAG}"
    CONSOLE_LOG="${RUN_DIR}/console.log"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing Stage2 backbone checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
    # A partial run also prints Max accuracy; require all epochs and clean exit.
    if [[ -f "${CONSOLE_LOG}" ]] && BEST_ACC="$("${PYTHON_BIN}" -c 'import re, sys; text=open(sys.argv[1], encoding="utf-8").read(); values=[float(x) for x in re.findall(r"Max accuracy: ([0-9.]+)%", text)]; complete=len(values)==int(sys.argv[2]) and "Training time " in text; sys.exit(1) if not complete else print(max(values))' "${CONSOLE_LOG}" "${LP_EPOCHS}")"; then
        echo "Reusing completed LP for checkpoint ${TAG}: ${BEST_ACC}%"
        printf '%s,%s,%s\n' "${STAGE2_EPOCH}" "${CHECKPOINT}" "${BEST_ACC}" >> "${SUMMARY_CSV}"
        continue
    fi
    # Keep incomplete logs and TensorBoard events separate from the fresh run.
    if [[ -d "${RUN_DIR}" ]]; then
        PREVIOUS_DIR="$(mktemp -d "${RUN_DIR}.previous.XXXXXX")"
        mv -- "${RUN_DIR}" "${PREVIOUS_DIR}/run"
    fi
    mkdir -p "${RUN_DIR}"
    echo "Running LP for Stage2 checkpoint ${TAG}"
    # Single GPU; AMP remains disabled (main_linprobe default).
    "${PYTHON_BIN}" main_linprobe.py \
        --config "${CONFIG}" \
        --output_dir "" \
        --log_dir "${RUN_DIR}/tensorboard" \
        --finetune "${CHECKPOINT}" \
        --accum_iter 1 \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${LP_EPOCHS}" \
        --model model.transformer_downstream.Transformer \
        2>&1 | tee "${CONSOLE_LOG}"
    BEST_ACC="$("${PYTHON_BIN}" -c 'import re, sys; text=open(sys.argv[1], encoding="utf-8").read(); values=[float(x) for x in re.findall(r"Max accuracy: ([0-9.]+)%", text)]; print(max(values) if values else (_ for _ in ()).throw(RuntimeError("No Max accuracy entry found")))' "${CONSOLE_LOG}")"
    printf '%s,%s,%s\n' "${STAGE2_EPOCH}" "${CHECKPOINT}" "${BEST_ACC}" >> "${SUMMARY_CSV}"
done

echo
echo "Best LP accuracy for every Stage2 checkpoint:"
while IFS=, read -r STAGE2_EPOCH CHECKPOINT BEST_ACC; do
    [[ "${STAGE2_EPOCH}" == "stage2_epoch" ]] && continue
    printf 'checkpoint-%03d: %s%%\n' "${STAGE2_EPOCH}" "${BEST_ACC}"
done < "${SUMMARY_CSV}"

awk -F, 'NR > 1 && (NR == 2 || ($3 + 0) > best) {best=$3 + 0; epoch=$1} END {printf "Overall best: checkpoint-%03d, %.2f%%\n", epoch, best}' "${SUMMARY_CSV}"
echo "Summary saved to ${SUMMARY_CSV}"
