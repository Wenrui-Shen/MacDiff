#!/usr/bin/env bash
set -euo pipefail

# Run from the MacDiff repository root, in the server's macdiff environment.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" tools/run_macdiff_rotation_ablation.py "$@"
