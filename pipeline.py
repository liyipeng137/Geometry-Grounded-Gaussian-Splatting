#!/usr/bin/env python3
"""Reconstruct a real-view mesh from registered photos and aligned camera-Z depth."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from arguments import REAL_GEOMETRY_DEPTH_LOSS_MODES

ROOT = Path(__file__).resolve().parent


def run(
    dataset,
    output=None,
    *,
    init_ply=None,
    iterations=20000,
    real_depth_loss_mode="absolute",
    virtual_geometry_config=None,
):
    if virtual_geometry_config is not None:
        # Future integration point for the registered virtual cameras and fixed D/N teacher
        # prepared by utils.delivery_pipeline.prepare. The real trainer must first consume them.
        raise NotImplementedError(
            "Virtual supervision is reserved for a later validation of this geometry pipeline."
        )
    if iterations < 1:
        raise ValueError("--iterations must be positive")
    if real_depth_loss_mode not in REAL_GEOMETRY_DEPTH_LOSS_MODES:
        raise ValueError(f"Unknown real-view depth loss: {real_depth_loss_mode!r}")
    dataset = Path(dataset).expanduser().resolve(strict=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            dataset / "gggs" / "runs" / f"real_geometry_{stamp}_{uuid.uuid4().hex[:8]}"
        )
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    env = os.environ.copy()
    native_lib = ROOT / "build" / "geometry_rasterizer" / "lib"
    if native_lib.is_dir():
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(native_lib), env.get("PYTHONPATH")))
        )
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    train = [
        sys.executable,
        "-u",
        str(ROOT / "train_real_geometry.py"),
        "-s",
        str(dataset),
        "-m",
        str(output),
        "--iterations",
        str(iterations),
        "--real_depth_loss_mode",
        real_depth_loss_mode,
    ]
    if init_ply is not None:
        train.extend(
            ["--init_ply", str(Path(init_ply).expanduser().resolve(strict=True))]
        )
    print(f"Real-view reconstruction in {output}", flush=True)
    subprocess.run(train, cwd=ROOT, env=env, check=True)

    print(f"Extracting the mesh at iteration {iterations}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "mesh_extract_mini.py"),
            "-m",
            str(output),
            "--iteration",
            str(iterations),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    model = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    mesh = output / "recon_bounded_post_smooth.ply"
    for path in (model, mesh):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing reconstruction output: {path}")
    receipt = {
        "status": "complete",
        "dataset": str(dataset),
        "iterations": iterations,
        "real_depth_loss_mode": real_depth_loss_mode,
        "densification_schedule": "accelerated",
        "virtual_training_cameras": 0,
        "normal_prior": json.loads((output / "depth_normal_prior.json").read_text()),
        "model": str(model),
        "mesh": str(mesh),
    }
    temporary = output / "pipeline_summary.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(output / "pipeline_summary.json")
    print(f"Final mesh: {mesh}", flush=True)
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="New run directory (default DATASET/gggs/runs/real_geometry_...).",
    )
    parser.add_argument(
        "--init-ply",
        "--init_ply",
        type=Path,
        help="Existing aligned XYZ/RGB PLY. Otherwise fuse training RGB-D with TSDF.",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument(
        "--real-depth-loss-mode",
        "--real_depth_loss_mode",
        choices=REAL_GEOMETRY_DEPTH_LOSS_MODES,
        default="absolute",
        help="Depth supervision for real training views (default: absolute camera-Z L1).",
    )
    parser.add_argument(
        "--virtual-geometry-config",
        "--virtual_geometry_config",
        type=Path,
        help="Reserved input for future virtual-view supervision.",
    )
    args = parser.parse_args()
    try:
        run(**vars(args))
    except (
        ValueError,
        FileExistsError,
        FileNotFoundError,
        NotImplementedError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
