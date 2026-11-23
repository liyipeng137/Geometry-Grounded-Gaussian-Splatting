"""Render a real-geometry run at original resolution and compare its 0/final Gaussians."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.training_io import write_json


def camera(frame, uid):
    w, h = frame["width"], frame["height"]
    k, r, c = (
        np.asarray(frame["K"]),
        np.asarray(frame["rotation"]),
        np.asarray(frame["position"]),
    )
    return Camera(
        uid,
        r,
        -r.T @ c,
        2 * math.atan(w / (2 * k[0, 0])),
        2 * math.atan(h / (2 * k[1, 1])),
        torch.empty(1, h, w, dtype=torch.uint8),
        None,
        frame["img_name"],
        uid,
        K=k,
        lazy_payload=True,
    )


def read_model(out, iteration, cfg):
    m = GaussianModel(cfg["sh_degree"], cfg["sg_degree"])
    m.load_ply(str(out / f"point_cloud/iteration_{iteration}/point_cloud.ply"))
    if torch.count_nonzero(m._features_rest):
        raise ValueError("Unexpected learned higher-order appearance in geometry run")
    m.active_sh_degree = 0
    return m


def arrays(buf):
    def hw(key):
        val = buf[key].detach()
        if val.shape[0] == 1:
            return val[0].cpu().numpy()
        return val.permute(1, 2, 0).contiguous().cpu().numpy()

    return dict(
        **({"rgb": hw("render")} if "render" in buf else {}),
        z=hw("median_depth"),
        n=hw("normal"),
        alpha=hw("mask"),
        hit=hw("depth_status") == 3,
    )


def image(data):
    return Image.fromarray(
        np.ascontiguousarray(np.rint(np.clip(data, 0, 1) * 255).astype(np.uint8))
    )


def geometry_panels(a, limits):
    valid = a["hit"]
    n = a["n"] / np.linalg.norm(a["n"], axis=-1, keepdims=True).clip(1e-8)
    light = np.array([-0.4, -0.6, -1.0], np.float32)
    diffuse = np.maximum(0, n @ (light / np.linalg.norm(light)))
    gray = np.where(valid, 0.78 * (0.3 + 0.7 * diffuse), 0.035)
    norm = np.where(valid[..., None], 0.5 + 0.5 * n, 0)
    inv = (limits[1] - np.where(valid, a["z"], limits[1])) / (limits[1] - limits[0])
    colors = cv2.applyColorMap(
        np.rint(255 * np.clip(inv, 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO
    )[..., ::-1].copy()
    colors[~valid] = 0
    return [
        image(gray).convert("RGB"),
        image(norm),
        Image.fromarray(colors),
        image(a["alpha"]).convert("RGB"),
    ]


def panels(a, limits):
    return [image(a["rgb"]), *geometry_panels(a, limits)]


def strip(images, labels, width=210):
    h = round(width * images[0].height / images[0].width)
    canvas = Image.new("RGB", (len(images) * width, h + 26), "#171c20")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for i, (im, label) in enumerate(zip(images, labels)):
        draw.text((i * width + 5, 4), label, font=font, fill="white")
        canvas.paste(
            im.convert("RGB").resize((width, h), Image.Resampling.LANCZOS),
            (i * width, 26),
        )
    return canvas


def stack(rows):
    w = max(im.width for im in rows)
    canvas = Image.new("RGB", (w, sum(im.height for im in rows)), "#171c20")
    top = 0
    for im in rows:
        canvas.paste(im, (0, top))
        top += im.height
    return canvas


def video_writer(path, size, fps):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    return subprocess.Popen(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size[0]}x{size[1]}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "3",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def interpolate_views(dest, frames, names, model, pipe, bg, kernel, limits):
    from scipy.spatial.transform import Rotation, Slerp

    by_name = {f["img_name"]: f for f in frames}
    a, b = (by_name[name] for name in names)
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        raise ValueError("Choose endpoints of the same image size")
    rotations = Slerp([0, 1], Rotation.from_matrix([a["rotation"], b["rotation"]]))
    writer = None
    try:
        for i, t in enumerate((1 - np.cos(np.linspace(0, np.pi, 73))) / 2):
            f = dict(a)
            for key in ("position", "K"):
                f[key] = (1 - t) * np.asarray(a[key]) + t * np.asarray(b[key])
            f["rotation"] = rotations(t).as_matrix()
            cam = camera(f, i)
            output = render(
                cam,
                model,
                pipe,
                bg,
                kernel,
                require_depth=True,
                depth_search_expansions=2,
            )
            small = {k: v[1::3, 1::3] for k, v in arrays(output).items()}
            clay, normal, depth, alpha = geometry_panels(small, limits)
            if "rgb" in small:
                card = strip([image(small["rgb"]),clay,depth],["Frozen-color RGB","Normal shading","Camera-Z depth"],width=360)
            else:
                card = strip([clay,normal,depth,alpha],["Normal shading","Camera normal","Camera-Z","Alpha"],width=300)
            if writer is None:
                writer = video_writer(dest / "interpolated.mp4", card.size, 24)
            writer.stdin.write(card.tobytes())
            if i == 36:
                card.save(dest / "interpolated_midpoint.jpg", quality=93)
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("Interpolated video encoding failed")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument(
        "--interpolate",
        nargs=2,
        metavar=("FROM_IMAGE", "TO_IMAGE"),
        help="Also render 73 linearly translated, SLERP-rotated poses between these training cameras",
    )
    args = parser.parse_args()
    out, dest = args.model_path, args.output
    cfg = json.loads((out / "geometry_config.json").read_text())["args"]
    stats = json.loads((out / "geometry_stats.json").read_text())
    if stats["status"] != "complete":
        raise ValueError("A completed geometry run is required")
    dest.mkdir(parents=True, exist_ok=False)
    data = Path(cfg["source_path"])
    frames = sorted(
        json.loads((out / "cameras.json").read_text()), key=lambda f: f["img_name"]
    )
    last = stats["optimizer_steps"]
    models = [read_model(out, i, cfg) for i in (0, last)]
    pipe = SimpleNamespace(
        debug=False, convert_SHs_python=False, compute_cov3D_python=False
    )
    bg = torch.full((3,), float(cfg["white_background"]), device="cuda")
    # Read target values through the same loader as training (including file format and scale).
    from utils.camera_utils import _load_depth_prior, _load_normal_prior

    dataset = SimpleNamespace(**cfg)

    def priors(frame):
        hint = SimpleNamespace(
            image_name=frame["img_name"],
            colmap_image_name=frame["img_name"],
            width=frame["width"],
            height=frame["height"],
            image_path=str(data / cfg["images"] / frame["img_name"]),
        )
        size = (frame["width"], frame["height"])
        depth = _load_depth_prior(dataset, hint, size)[0].numpy()
        normal = _load_normal_prior(dataset, hint, size).permute(1, 2, 0).numpy()
        return depth, normal

    vals = []
    for f in frames[:: max(1, len(frames) // 20)]:
        p, _ = priors(f)
        sampled = p[::40, ::40]
        vals.append(sampled[np.isfinite(sampled) & (sampled > 0.01)])
    limits = np.percentile(np.concatenate(vals), [1, 99]).tolist()
    if limits[1] <= limits[0]:
        limits[1] = limits[0] + 1
    preview_ids = [
        int(i) for i in np.linspace(0, len(frames) - 1, min(6, len(frames))).round()
    ]
    detail_idx = preview_ids[len(preview_ids) // 3]
    metrics, rows, evolution = [], [], []
    writer = None
    try:
        for idx, frame in enumerate(frames):
            cam = camera(frame, idx)
            pred = [
                arrays(
                    render(
                        cam,
                        m,
                        pipe,
                        bg,
                        cfg["kernel_size"],
                        require_depth=True,
                        depth_search_expansions=2,
                    )
                )
                for m in models
            ]
            p, pn = priors(frame)
            sl = np.s_[2::5, 2::5]
            ref = p[sl]
            refn = pn[sl]
            length = np.linalg.norm(refn, axis=-1)
            valid = np.isfinite(ref) & (ref > 0.01)
            h0, h1 = [a["hit"][sl] for a in pred]
            both = valid & h0 & h1
            nv = both & (length > 0.9) & (length < 1.1)
            row = {
                "name": frame["img_name"],
                "domain": int(valid.sum()),
                "both": int(both.sum()),
                "normal_both": int(nv.sum()),
                "recovered": int((valid & ~h0 & h1).sum()),
                "new_missing": int((valid & h0 & ~h1).sum()),
            }
            for label, a in zip(("initial", "trained"), pred):
                hit = valid & a["hit"][sl]
                rel = np.abs(a["z"][sl][hit] / ref[hit] - 1)
                row[label + "_hit_absrel_sum"] = float(rel.sum(dtype=float))
                row[label + "_within5"] = int((rel <= 0.05).sum())
                nn = a["n"][sl][nv]
                nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-8)
                angles = np.rad2deg(
                    np.arccos(
                        np.clip((nn * (refn[nv] / length[nv, None])).sum(-1), -1, 1)
                    )
                )
                row[label + "_direction_sum"] = float(angles.sum(dtype=float))
                row[label + "_absrel_sum"] = float(
                    np.abs(a["z"][sl][both] / ref[both] - 1).sum(dtype=float)
                )
                row[label + "_miss"] = int((valid & ~a["hit"][sl]).sum())
            metrics.append(row)
            with Image.open(data / cfg["images"] / frame["img_name"]) as file:
                rgb = file.convert("RGB")
            # Only selected stills need full-resolution visual channels. RGB/D/N were rendered natively above.
            if idx in preview_ids:
                vis = panels(pred[1], limits)
                pair = panels(pred[0], limits)
                rows.append(
                    strip(
                        [rgb, *vis],
                        [
                            frame["img_name"],
                            "Frozen-color RGB",
                            "Normal shading",
                            "Camera normals",
                            "Camera-Z depth",
                            "Alpha",
                        ],
                    )
                )
                evolution.append(
                    strip(
                        [rgb, pair[0], vis[0], pair[1], vis[1]],
                        [
                            frame["img_name"],
                            "RGB / initial",
                            f"RGB / {last}",
                            "Shading / initial",
                            f"Shading / {last}",
                        ],
                    )
                )
                if idx == detail_idx:
                    detail = dest / "full_resolution"
                    detail.mkdir()
                    strip(
                        [rgb, vis[0], vis[1], vis[3]],
                        [
                            "Real photo",
                            "Frozen-color RGB",
                            "Normal shading",
                            "Camera-Z depth",
                        ],
                        width=300,
                    ).save(dest / "detail_preview.jpg", quality=95)
                    for name, im in zip(
                        ("photo", "frozen_rgb", "shading", "normal", "depth", "alpha"),
                        [rgb, *vis],
                    ):
                        im.save(detail / f"{name}.png")
                    np.savez_compressed(
                        detail / "native_geometry.npz",
                        depth=pred[1]["z"],
                        normal=pred[1]["n"],
                        alpha=pred[1]["alpha"],
                        valid=pred[1]["hit"],
                        K=frame["K"],
                        rotation_c2w=frame["rotation"],
                        position=frame["position"],
                    )
            small = {k: v[2::5, 2::5] for k, v in pred[1].items()}
            thumb = panels(small, limits)
            card = strip(
                [rgb, thumb[0], thumb[1], thumb[3], thumb[4]],
                [
                    frame["img_name"],
                    "Frozen-color RGB",
                    "Normal shading",
                    "Camera-Z depth",
                    "Alpha",
                ],
                width=240,
            )
            if writer is None:
                writer = video_writer(dest / "walkthrough.mp4", card.size, args.fps)
            writer.stdin.write(card.tobytes())
    finally:
        if writer:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("ffmpeg encoding failed")
    stack(rows).save(dest / "contact_sheet.jpg", quality=94)
    stack(evolution).save(dest / "initial_vs_trained.jpg", quality=94)
    totals = {k: sum(r[k] for r in metrics) for k in metrics[0] if k != "name"}
    summary = {
        k: {
            "missing_fraction": totals[k + "_miss"] / totals["domain"],
            "depth_absrel_own_hits": totals[k + "_hit_absrel_sum"]
            / max(1, totals["domain"] - totals[k + "_miss"]),
            "within5_fixed_prior_domain": totals[k + "_within5"] / totals["domain"],
            "common_depth_absrel": totals[k + "_absrel_sum"] / max(1, totals["both"]),
            "common_normal_degrees": totals[k + "_direction_sum"]
            / max(1, totals["normal_both"]),
        }
        for k in ("initial", "trained")
    }
    if args.interpolate:
        interpolate_views(
            dest,
            frames,
            args.interpolate,
            models[1],
            pipe,
            bg,
            cfg["kernel_size"],
            limits,
        )
    write_json(
        dest / "metrics.json",
        {
            "iteration": last,
            "views": len(frames),
            "stride": 5,
            "totals": totals,
            "results": summary,
            "depth_visual_range_scene_units": limits,
            "detail_frame": frames[detail_idx]["img_name"],
            "video_fps": args.fps,
            "interpolated_camera_endpoints": args.interpolate,
            "per_view": metrics,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
