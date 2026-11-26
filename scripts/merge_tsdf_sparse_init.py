#!/usr/bin/env python3
"""Append COLMAP points within the TSDF initialization's nearest-camera distance range."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sparse_initialization import merge_tsdf_sparse_ply, sparse_points_path  # noqa: E402


def merge(tsdf_run: Path, output: Path) -> dict:
    split = json.loads((tsdf_run / "camera_split.json").read_text())
    config = json.loads((tsdf_run / "geometry_config.json").read_text())["args"]
    if split["initialization"]["kind"] != "train_only_rgbd":
        raise ValueError("The reference run must have a TSDF-only initialization")
    cameras = json.loads((tsdf_run / "cameras.json").read_text())
    positions = {camera["img_name"]: camera["position"] for camera in cameras}
    centers = [positions[name] for name in split["train_image_names"]]
    tsdf_path = tsdf_run / "input.ply"
    sparse_path = sparse_points_path(Path(split["dataset_root"]) / config.get("colmap_dir", "sparse/0"))
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = merge_tsdf_sparse_ply(tsdf_path, sparse_path, centers, output)
    return {
        "tsdf_ply": str(tsdf_path.resolve()),
        "sparse_model": str(sparse_path.resolve()),
        "output": str(output.resolve()),
        **stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsdf-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Path for the merged XYZ/RGB PLY.")
    args = parser.parse_args()
    print(json.dumps(merge(args.tsdf_run, args.output), indent=2))
