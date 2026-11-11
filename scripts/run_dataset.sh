#!/usr/bin/env bash
# 通用训练入口：虚拟几何由 train.py 按数据集缓存或显式选项决定。
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [additional train.py arguments]" >&2
  exit 2
fi
# 在切换到仓库前解析数据集路径，保留调用者的相对路径语义。
GGGS_DATASET=$(cd -- "$1" && pwd -P)
shift
GGGS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$GGGS_ROOT"
exec python train.py \
  -s "$GGGS_DATASET" \
  --use_decoupled_appearance 3 \
  --depth_prior_dir depth --normal_prior_dir normals \
  --frame_gpu_budget 4 --iterations 20000 --training_seed 0 \
  --test_iterations 7000 15000 20000 \
  --save_iterations 7000 15000 20000 \
  --checkpoint_iterations 7000 15000 --log_interval 100 \
  "$@"
