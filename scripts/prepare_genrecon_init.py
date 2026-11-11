#!/usr/bin/env python3
"""全局 GenRecon 面积采样 20 万点，并仅用实际训练相机投影 RGB。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 保留原脚本的函数导入接口，采样/遮挡实现统一由训练入口与 CLI 共用。
from utils.mesh_initialization import (  # noqa: E402,F401
    POINT_COUNT, MeshScene, bilinear_rgb, load_real_views, prepare, prepare_from_cameras,
    project_colors, project_points, sample_surface, visible_samples, views_from_cameras, write_point_cloud,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train_exclude_images", type=str, default="")
    args = parser.parse_args()
    prepare(args.dataset, args.mesh, args.output, args.seed, args.threads,
            images=args.images, eval_mode=args.eval, train_exclude_images=args.train_exclude_images)


if __name__ == "__main__":
    main()
