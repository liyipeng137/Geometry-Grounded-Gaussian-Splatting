"""Depth-derived camera-space normal priors for real-geometry training."""

from pathlib import Path

import cv2
import numpy as np
import torch

from utils.depth_prior_io import find_depth_file, read_native_depth

NORMAL_DIR = "depth_normal_prior"


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


@torch.no_grad()
def prepare_depth_normal_priors(dataset, depth_root, cameras, depth_format, depth_scale):
    """Write the PNG of every camera that lacks one; existing PNGs are used as they are."""
    output = Path(dataset) / NORMAL_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated = 0
    for cam in cameras:
        name = Path(cam.colmap_image_name)
        dest = output / name.with_suffix(".png")
        if dest.is_file():
            continue
        path = find_depth_file(depth_root, name.with_suffix(".npy"), depth_format)
        if path is None:
            raise FileNotFoundError(f"Missing aligned depth for {name} under {depth_root}")
        shape = (cam.height, cam.width)
        depth = torch.as_tensor(read_native_depth(path, shape, depth_scale), device=device)
        if depth.shape != shape:
            # Low-resolution depth images use the same nearest-exact resampling as the camera loader.
            depth = torch.nn.functional.interpolate(depth[None, None], size=shape, mode="nearest-exact")[0, 0]
        normal, count = normals_from_depth(depth, cam.K)
        if not count:
            raise ValueError(f"No usable depth-derived normals for {name}")
        # The GGGS loader decodes RGB as 1 - 2*rgb/255; neutral gray masks invalid pixels.
        rgb = ((1 - normal).clamp(0, 2) * 127.5).round().to(torch.uint8).cpu().numpy()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise OSError(f"Failed to write {dest}")
        generated += 1
    print(f"Depth normals: {generated} generated, {len(cameras) - generated} reused in {output}", flush=True)
    return {"directory": str(output), "frames": len(cameras), "generated": generated,
            "reused": len(cameras) - generated}
