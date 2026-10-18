#!/usr/bin/env bash
set -euo pipefail

# Proposed LP-only comparison; run from the MacDiff repository root.
# Reuse the original saved LP config (including seed and LR); override only batch.
test -f output_dir/native_rotation_dec3_ep500_20261011_012631_490772/configs/lp.json
test -f output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/pt/checkpoint-499.pth
if [ -e output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/lp_bs64_compare ]; then
  echo "Refusing to reuse existing LP comparison directory" >&2
  exit 1
fi
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10272 main_linprobe.py --config output_dir/native_rotation_dec3_ep500_20261011_012631_490772/configs/lp.json --finetune output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/pt/checkpoint-499.pth --batch_size 64 --accum_iter 1 --dist_eval --output_dir output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/lp_bs64_compare --log_dir output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/lp_bs64_compare/tensorboard
