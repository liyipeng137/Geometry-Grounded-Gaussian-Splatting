"""Prepare the fixed camera-space normal prior used by real-geometry training."""

import fcntl
import hashlib
import io
import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from utils.depth_prior_io import find_depth_file, read_native_depth

NORMAL_DIR = "depth_normal_prior"
METHOD = "camera_z_half_pixel_one_sided_5pct_v1"


@torch.no_grad()
def normals_from_depth(depth, K):
    height, width = depth.shape
    K = np.asarray(K, dtype=np.float64)
    if (
        min(height, width) < 3
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or K[0, 1] != 0
        or K[1, 0] != 0
        or not np.allclose(K[2], [0, 0, 1])
    ):
        raise ValueError("Expected a calibrated, undistorted pinhole depth grid")
    u = (
        torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5 - K[0, 2]
    ) / K[0, 0]
    v = (
        torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5 - K[1, 2]
    ) / K[1, 1]
    valid = torch.isfinite(depth) & (depth > 0.01)
    z = torch.where(valid, depth, 0.0)
    points = torch.stack((u[None, :] * z, v[:, None] * z, z), dim=-1)
    center = points[1:-1, 1:-1]
    center_z = center[..., 2]

    def continuous(point):
        nz = point[..., 2]
        return (nz > 0.01) & (
            (nz - center_z).abs() <= 0.05 * torch.minimum(nz, center_z)
        )

    def difference(left, right):
        a, b = continuous(left), continuous(right)
        tangent = torch.where(b[..., None], right, center) - torch.where(
            a[..., None], left, center
        )
        return tangent, a | b

    dx, valid_x = difference(points[1:-1, :-2], points[1:-1, 2:])
    dy, valid_y = difference(points[:-2, 1:-1], points[2:, 1:-1])
    raw = torch.linalg.cross(dy, dx, dim=-1)
    length = torch.linalg.vector_norm(raw, dim=-1)
    ok = valid[1:-1, 1:-1] & valid_x & valid_y & torch.isfinite(length) & (length > 0)
    unit = raw / length.clamp_min(torch.finfo(depth.dtype).tiny)[..., None]
    unit = torch.where((unit * center).sum(-1, keepdim=True) > 0, -unit, unit)
    normal = torch.zeros_like(points)
    normal[1:-1, 1:-1] = torch.where(ok[..., None], unit, 0.0)
    return normal, int(ok.sum())


@contextmanager
def depth_normal_preparation(args, *, device="cuda"):
    """Keep one normal set stable through generation and loading into camera memory."""
    root = Path(args.source_path).resolve(strict=True)
    depth_root = root / args.depth_prior_dir
    output = root / NORMAL_DIR
    if depth_root.resolve() == output.resolve():
        raise ValueError(
            "The depth input and generated normal directory must be distinct"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield partial(_prepare_depth_normal_priors, args, depth_root, output, device)


def _prepare_depth_normal_priors(args, depth_root, output, device, cameras):
    manifest = output / "manifest.json"
    frames, changed = {}, []
    with TemporaryDirectory(dir=output) as temporary:
        try:
            previous = json.loads(manifest.read_text())
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            previous = {}
        cached = (
            previous.get("frames")
            if isinstance(previous, dict) and previous.get("method") == METHOD
            else None
        )
        cached = cached if isinstance(cached, dict) else {}
        stage = Path(temporary)
        for cam in sorted(cameras, key=lambda c: c.colmap_image_name):
            name = Path(cam.colmap_image_name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Invalid registered image name: {name}")
            key = name.with_suffix(".png").as_posix()
            if key in frames:
                raise ValueError(
                    f"Two registered frames map to the same normal file: {key}"
                )
            path = find_depth_file(
                depth_root, name.with_suffix(".npy"), args.depth_prior_format
            )
            if path is None:
                raise FileNotFoundError(
                    f"Missing aligned depth for {name} under {depth_root}"
                )
            data = path.read_bytes()
            shape = (cam.height, cam.width)
            identity = {
                "depth_sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(shape),
                "K": np.asarray(cam.K).tolist(),
                "format": path.suffix.lower(),
                "scale": (
                    None if path.suffix.lower() == ".npy" else args.depth_prior_scale
                ),
            }
            dest = output / key
            entry = cached.get(key, {})
            if (
                isinstance(entry, dict)
                and entry.get("source") == identity
                and dest.is_file()
                and hashlib.sha256(dest.read_bytes()).hexdigest()
                == entry.get("png_sha256")
            ):
                frames[key] = entry
                continue
            depth = read_native_depth(
                path, shape, args.depth_prior_scale, stream=io.BytesIO(data)
            )
            depth = torch.as_tensor(depth, device=device, dtype=torch.float32)
            if depth.shape != shape:
                # Legacy depth images use the same nearest-exact resampling as the camera loader.
                depth = torch.nn.functional.interpolate(
                    depth[None, None], size=shape, mode="nearest-exact"
                )[0, 0]
            normal, count = normals_from_depth(depth, cam.K)
            if not count:
                raise ValueError(f"No usable depth-derived normals for {name}")
            # The GGGS loader decodes RGB as 1 - 2*rgb/255; neutral gray masks invalid pixels.
            rgb = (
                ((1 - normal).clamp(0, 2) * 127.5).round().to(torch.uint8).cpu().numpy()
            )
            ok, png = cv2.imencode(
                ".png",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 2],
            )
            if not ok:
                raise OSError(f"Failed to encode {dest}")
            staged = stage / key
            staged.parent.mkdir(exist_ok=True, parents=True)
            staged.write_bytes(png.tobytes())
            frames[key] = {
                "source": identity,
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "valid_pixels": count,
            }
            changed.append(key)
        if not frames:
            raise ValueError("No registered cameras for depth-derived normals")
        record = {"method": METHOD, "frames": frames}
        for key in changed:
            dest = output / key
            dest.parent.mkdir(exist_ok=True, parents=True)
            (stage / key).replace(dest)
        if previous != record:
            new_manifest = stage / "manifest.json"
            new_manifest.write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n"
            )
            new_manifest.replace(manifest)
        generated = len(changed)
        receipt = {
            "directory": str(output),
            "method": METHOD,
            "frames": len(frames),
            "generated": generated,
            "reused": len(frames) - generated,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    print(
        f"Depth normals: {generated} generated, {receipt['reused']} reused in {output}",
        flush=True,
    )
    return receipt
