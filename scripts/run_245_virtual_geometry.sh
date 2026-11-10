#!/usr/bin/env bash
# 已审核160机位的正式联合入口；训练始终由仓库train.py执行。
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY [additional train.py arguments]" >&2
  exit 2
fi
GGGS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GGGS_OUTPUT=$1
shift
cd "$GGGS_ROOT"
python train.py \
  -s /workspace/dataset/245_for_train \
  -m "$GGGS_OUTPUT" \
  --virtual_geometry_config configs/virtual_geometry_245.json \
  --use_decoupled_appearance 3 \
  --depth_prior_dir depth --normal_prior_dir normals \
  --frame_gpu_budget 4 --iterations 20000 --training_seed 0 \
  --test_iterations 7000 15000 20000 \
  --save_iterations 7000 15000 20000 \
  --checkpoint_iterations 7000 15000 --log_interval 100 \
  "$@"
