#!/usr/bin/env python3
"""Export an offline, inspectable camera preview from a schema-v2 manifest.

This viewer never loads or simplifies the full GenRecon mesh. The interactive
context uses scene_preview.npz samples; per-camera PNGs are the exact-mesh
raycasts prepared by the camera selection pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


GROUP_COLORS = {"real": "#4f9cff", "trajectory": "#ffad49", "surface": "#bd86ff"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def camera_group(camera):
    if camera["kind"] == "real":
        return "real"
    return "surface" if camera["selection_source"] == "surface_lookat" else "trajectory"


def load_cameras(path: Path):
    document = read_json(path)
    if document.get("schema_version") != 2:
        raise ValueError(f"Expected schema_version=2: {path}")
    cameras = document["cameras"]
    for camera in cameras:
        for key, shape in (("K", (3, 3)), ("c2w", (4, 4))):
            matrix = np.asarray(camera[key], dtype=np.float64)
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid {key} for camera {camera.get('id')}")
        if camera["width"] <= 0 or camera["height"] <= 0:
            raise ValueError(f"Invalid image size for camera {camera['id']}")
        if abs(np.linalg.det(camera["K"])) < 1e-12:
            raise ValueError(f"Singular K for camera {camera['id']}")
        camera["group"] = camera_group(camera)
    return cameras


def build_payload(manifest_path: Path, max_points: int):
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    cameras = []
    for key in ("real_cameras_path", "virtual_cameras_path"):
        cameras.extend(load_cameras(resolve_artifact(root, manifest[key])))
    identifiers = [str(camera["id"]) for camera in cameras]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Camera IDs must be unique across real and virtual cameras")
    if not cameras:
        raise ValueError("No cameras in manifest")

    with np.load(resolve_artifact(root, manifest["scene_preview_path"])) as scene:
        xyz = np.asarray(scene["xyz"], dtype=np.float32)
        rgb = np.asarray(scene["rgb"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
            raise ValueError("scene_preview.npz must contain matching xyz/rgb [N,3]")
        valid = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[valid], rgb[valid]
        if len(xyz) > max_points:
            indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
            xyz, rgb = xyz[indices], rgb[indices]
        if rgb.dtype.kind == "f" and len(rgb) and np.nanmax(rgb) <= 1:
            rgb = rgb * 255
        rgb = np.nan_to_num(rgb, nan=140).clip(0, 255).astype(np.uint8)

    centers = np.asarray([np.asarray(c["c2w"])[:3, 3] for c in cameras])
    context = np.concatenate((xyz, centers), axis=0)
    lo, hi = context.min(axis=0), context.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    center = (lo + hi) / 2
    real = [c for c in cameras if c["kind"] == "real"]
    real_centers = np.asarray([np.asarray(c["c2w"])[:3, 3] for c in real])
    position_tolerance = max(float(manifest.get("scene_scale", np.linalg.norm(span))), 1e-12) * 1e-7
    for camera in cameras:
        if camera["kind"] == "virtual":
            camera_center = np.asarray(camera["c2w"])[:3, 3]
            same_position = len(real_centers) > 0 and np.any(
                np.linalg.norm(real_centers - camera_center, axis=1) <= position_tolerance
            )
            camera["position_mode"] = "rotation_only" if same_position else "translated"
    reference = real or cameras
    # COLMAP camera axes are right/down/forward. This changes only the viewer's
    # presentation, never camera poses, mesh samples, or stored world axes.
    up = np.mean([-np.asarray(c["c2w"])[:3, 1] for c in reference], axis=0)
    if np.linalg.norm(up) < 1e-6:
        up = -np.asarray(reference[0]["c2w"])[:3, 1]
    up /= np.linalg.norm(up)
    right = np.asarray(reference[0]["c2w"])[:3, 0]
    right = right - up * np.dot(right, up)
    right /= max(np.linalg.norm(right), 1e-12)
    forward = np.cross(up, right)
    forward /= max(np.linalg.norm(forward), 1e-12)
    links = {"manifest": manifest_path.name}
    for key in ("real_cameras_path", "virtual_cameras_path", "report_path", "candidates_path", "atlas_path"):
        if manifest.get(key):
            links[key] = manifest[key]
    return {
        "title": root.name,
        "manifest": manifest,
        "cameras": cameras,
        "scene": {"xyz": xyz.round(6).tolist(), "rgb": rgb.tolist(), "count": len(xyz)},
        "bounds": {"center": center.tolist(), "span": span.tolist(), "lo": lo.tolist(), "hi": hi.tolist()},
        "basis": {"up": up.tolist(), "right": right.tolist(), "forward": forward.tolist()},
        "links": links,
        "preview_contract": "Interactive scene is a sampled mesh context, not the complete mesh. Per-camera PNGs are exact-mesh raycasts.",
    }


def unknown_text(camera):
    value = camera.get("stats", {}).get("unknown_fraction")
    return f"unknown {float(value):.1%}" if isinstance(value, (int, float)) else "unknown n/a"


def text_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def export_contact_sheet(cameras, root: Path, path: Path, pose_only=False):
    if not cameras:
        return
    columns, cell_width = (6, 260) if pose_only else (5, 312)
    cell_height = 130 if pose_only else 238
    rows = math.ceil(len(cameras) / columns)
    canvas = Image.new("RGB", (columns * cell_width, 66 + rows * cell_height), "#111723")
    draw = ImageDraw.Draw(canvas)
    font, small, title = text_font(15), text_font(12), text_font(21)
    kind = "REAL" if all(c["kind"] == "real" for c in cameras) else "VIRTUAL"
    heading = f"{kind} CAMERAS / " + ("POSE ONLY" if pose_only else "EXACT-MESH RAYCASTS")
    draw.text((16, 12), heading, fill="#f1f5fb", font=title)
    draw.text((16, 40), f"{len(cameras)} cameras. COLMAP world coordinates. Ordering does not imply a traversable path.", fill="#9cacc2", font=small)
    for index, camera in enumerate(cameras):
        x, y = (index % columns) * cell_width, 66 + (index // columns) * cell_height
        color = GROUP_COLORS[camera_group(camera)]
        draw.rectangle((x + 4, y + 3, x + cell_width - 5, y + cell_height - 5), outline=color, width=2)
        identifier = str(camera["id"])
        draw.text((x + 12, y + 10), identifier[:38], fill="#ffffff", font=font)
        if pose_only:
            matrix = np.asarray(camera["c2w"])
            c, f = matrix[:3, 3], matrix[:3, 2]
            draw.text((x + 12, y + 37), "C " + " ".join(f"{v:+.3f}" for v in c), fill="#c3cfdf", font=small)
            draw.text((x + 12, y + 59), "F " + " ".join(f"{v:+.3f}" for v in f), fill="#c3cfdf", font=small)
            draw.text((x + 12, y + 81), f"{camera['width']} x {camera['height']}   K / pose in JSON", fill="#8fa5c1", font=small)
            continue
        image_path = camera.get("preview", {}).get("shading")
        destination = (x + 12, y + 37, x + cell_width - 12, y + 198)
        if image_path and resolve_artifact(root, image_path).is_file():
            with Image.open(resolve_artifact(root, image_path)) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (destination[2] - destination[0], destination[3] - destination[1]))
                canvas.paste(thumb, (destination[0] + (destination[2] - destination[0] - thumb.width) // 2,
                                     destination[1] + (destination[3] - destination[1] - thumb.height) // 2))
        else:
            draw.text((x + 30, y + 110), "Exact-mesh PNG unavailable", fill="#ffb56a", font=small)
        draw.text((x + 12, y + 207), f"{camera_group(camera)} | {unknown_text(camera)}", fill="#c3cfdf", font=small)
    canvas.save(path, quality=90, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="HTML path; defaults to preview.html beside manifest")
    parser.add_argument("--max-scene-points", type=int, default=40000)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    args = parser.parse_args()
    if args.max_scene_points < 1:
        parser.error("--max-scene-points must be positive")
    manifest_path = args.manifest.resolve()
    output = args.output.resolve() if args.output else manifest_path.parent / "preview.html"
    if output.parent != manifest_path.parent:
        parser.error("HTML must be beside manifest so offline artifact links remain valid")
    payload = build_payload(manifest_path, args.max_scene_points)
    from plotly.offline import get_plotlyjs

    template = Path(__file__).parent / "assets" / "virtual_camera_preview.html"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.read_text(encoding="utf-8").replace("__PLOTLY_JS__", get_plotlyjs()).replace("__VIEWER_DATA__", data)
    output.write_text(html, encoding="utf-8")
    if not args.skip_contact_sheets:
        cameras = payload["cameras"]
        root = manifest_path.parent
        virtual = sorted((c for c in cameras if c["kind"] == "virtual"), key=lambda c: (c["group"], str(c["id"])))
        export_contact_sheet(virtual, root, root / "contact_sheet.jpg")
        export_contact_sheet([c for c in virtual if c["group"] == "trajectory"], root, root / "contact_sheet_trajectory.jpg")
        export_contact_sheet([c for c in virtual if c["group"] == "surface"], root, root / "contact_sheet_surface.jpg")
        real = [c for c in cameras if c["kind"] == "real"]
        has_real_previews = any(c.get("preview", {}).get("shading") for c in real)
        export_contact_sheet(real, root, root / "contact_sheet_real.jpg", pose_only=not has_real_previews)
    print(json.dumps({"html": str(output), "cameras": len(payload["cameras"]), "scene_samples": payload["scene"]["count"]}))


if __name__ == "__main__":
    main()
