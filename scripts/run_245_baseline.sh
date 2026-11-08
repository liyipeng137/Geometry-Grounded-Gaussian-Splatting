#!/usr/bin/env bash
# 当前真实相机基线；不自动启动提面，不修改原始数据。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_245_baseline.sh NEW_OUTPUT_DIR [train.py options...]" >&2
  exit 2
fi

gggs_output=$1
shift
gggs_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gggs_python=${GGGS_PYTHON:-/usr/local/miniconda3/envs/py312/bin/python}
gggs_dataset=${GGGS_DATASET:-/workspace/dataset/245_for_train}

cd "$gggs_repo"
exec "$gggs_python" train.py \
  --source_path "$gggs_dataset" \
  --model_path "$gggs_output" \
  --depth_prior_dir global_depth/depth_npy \
  --depth_prior_format npy \
  --frame_gpu_budget 4 \
  --lambda_normal_prior 0 \
  --lambda_multi_view_ncc 0.02 \
  "$@"
