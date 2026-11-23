"""Per-camera first-surface comparison of appearance-free geometry checkpoints."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from gaussian_renderer import render
from render_real_geometry import (
    arrays,
    camera,
    geometry_panels,
    interpolate_views,
    stack,
    strip,
    video_writer,
)
from scene.geometry_gaussians import GeometryGaussians
from utils.camera_utils import _load_depth_prior, _load_normal_prior
from utils.training_io import write_json


def priors(data, cfg, frame):
    name = frame["img_name"]
    hint = SimpleNamespace(
        colmap_image_name=name,
        image_name=Path(name).stem,
        width=frame["width"],
        height=frame["height"],
        image_path=str(data / cfg.images / name),
    )
    size = (hint.width, hint.height)
    return (
        _load_depth_prior(cfg, hint, size)[0].numpy(),
        _load_normal_prior(cfg, hint, size).permute(1, 2, 0).numpy(),
    )


def close_encoder(writer):
    if writer is not None:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("Video encoding failed")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[7000, 15000, 20000])
    parser.add_argument("--interpolate", nargs=2)
    args = parser.parse_args()
    if len(args.steps) < 2 or args.steps != sorted(set(args.steps)):
        parser.error("Specify at least two distinct increasing checkpoint steps")
    run, dest = args.model_path, args.output
    cfg = SimpleNamespace(
        **json.loads((run / "geometry_config.json").read_text())["args"]
    )
    data = Path(cfg.source_path)
    frames = sorted(
        json.loads((run / "cameras.json").read_text()), key=lambda f: f["img_name"]
    )
    models = []
    for step in args.steps:
        m = GeometryGaussians()
        m.load_ply(run / f"point_cloud/iteration_{step}/point_cloud.ply")
        models.append(m)
    dest.mkdir(parents=True, exist_ok=False)
    pipe = SimpleNamespace(
        debug=False, convert_SHs_python=False, compute_cov3D_python=False
    )
    bg = torch.empty(0, device="cuda")
    sample = []
    for f in frames[:: max(1, len(frames) // 20)]:
        p = priors(data, cfg, f)[0][::40, ::40]
        sample.append(p[np.isfinite(p) & (p > 0.01)])
    limits = np.percentile(np.concatenate(sample), [1, 99]).tolist()
    limits[1] = max(limits[1], limits[0] + 1e-6)
    selected = (
        np.linspace(0, len(frames) - 1, min(6, len(frames)))
        .round()
        .astype(int)
        .tolist()
    )
    detail = selected[len(selected) // 3]
    rows, sheets = [], []
    writer = None
    try:
        for idx, f in enumerate(frames):
            cam = camera(f, idx)
            small = []
            for model in models:
                pkg = render(
                    cam, model, pipe, bg, cfg.kernel_size, depth_search_expansions=2
                )
                if "render" in pkg:
                    raise RuntimeError("Unexpected RGB render")
                geom = arrays(pkg)
                small.append({k: v[2::5, 2::5].copy() for k, v in geom.items()})
                del pkg
            p, n = priors(data, cfg, f)
            p, n = p[2::5, 2::5], n[2::5, 2::5]
            dom = np.isfinite(p) & (p > 0.01)
            common = dom & np.logical_and.reduce([s["hit"] for s in small])
            norm = np.linalg.norm(n, axis=-1)
            normal_common = common & (norm > 0.9) & (norm < 1.1)
            row = {
                "name": f["img_name"],
                "prior": int(dom.sum()),
                "common": int(common.sum()),
                "normal_common": int(normal_common.sum()),
                "new_missing": int((dom & small[0]["hit"] & ~small[-1]["hit"]).sum()),
                "recovered": int((dom & ~small[0]["hit"] & small[-1]["hit"]).sum()),
            }
            for step, s in zip(args.steps, small):
                hit = s["hit"] & dom
                rel = np.abs(s["z"] / np.maximum(p, 0.01) - 1)
                a = s["n"][normal_common]
                a /= np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)
                dot = (a * n[normal_common] / norm[normal_common, None]).sum(-1)
                row.update(
                    {
                        f"{step}_missing": int((dom & ~hit).sum()),
                        f"{step}_within5": int((hit & (rel <= 0.05)).sum()),
                        f"{step}_rel_sum": float(rel[common].sum(dtype=float)),
                        f"{step}_angle_sum": float(
                            np.rad2deg(np.arccos(dot.clip(-1, 1))).sum(dtype=float)
                        ),
                    }
                )
            rows.append(row)
            with Image.open(data / cfg.images / f["img_name"]) as raw:
                reference = raw.convert("L")
            final_small = geometry_panels(small[-1], limits)
            overview = strip(
                [reference, geometry_panels(small[0], limits)[0], *final_small],
                [
                    f["img_name"],
                    f"Shading / {args.steps[0]}",
                    f"Shading / {args.steps[-1]}",
                    "Camera normal",
                    "Camera-Z",
                    "Alpha",
                ],
            )
            if idx in selected:
                sheets.append(overview)
            if idx == detail:
                full = geometry_panels(geom, limits)
                strip(
                    [reference, full[0], full[1], full[2]],
                    ["Observed gray", "Normal shading", "Camera normal", "Camera-Z"],
                    width=300,
                ).save(dest / "detail.jpg", quality=95)
                np.savez_compressed(
                    dest / "native_geometry.npz",
                    depth=geom["z"],
                    normal=geom["n"],
                    alpha=geom["alpha"],
                    valid=geom["hit"],
                    K=f["K"],
                    rotation_c2w=f["rotation"],
                    position=f["position"],
                )
            if writer is None:
                writer = video_writer(dest / "walkthrough.mp4", overview.size, 18)
            writer.stdin.write(overview.tobytes())
    finally:
        close_encoder(writer)
    stack(sheets).save(dest / "comparison.jpg", quality=94)
    if args.interpolate:
        interpolate_views(
            dest,
            frames,
            args.interpolate,
            models[-1],
            pipe,
            bg,
            cfg.kernel_size,
            limits,
        )
    sums = {k: sum(row[k] for row in rows) for k in rows[0] if k != "name"}
    results = {
        str(it): {
            "missing": sums[f"{it}_missing"],
            "within5_fixed": sums[f"{it}_within5"] / sums["prior"],
            "paired_absrel": sums[f"{it}_rel_sum"] / max(1, sums["common"]),
            "paired_normal_degrees": sums[f"{it}_angle_sum"]
            / max(1, sums["normal_common"]),
        }
        for it in args.steps
    }
    write_json(
        dest / "metrics.json",
        {
            "steps": args.steps,
            "camera_count": len(frames),
            "stride": 5,
            "totals": sums,
            "results": results,
            "depth_visual_limits": limits,
            "detail_frame": frames[detail]["img_name"],
            "interpolated_endpoints": args.interpolate,
            "per_camera": rows,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
