"""Locate and decode aligned camera-Z depths on their native pixel grids."""

from pathlib import Path

import numpy as np
from PIL import Image


def find_depth_file(root, npy_name, fallback_format):
    npy = Path(root) / npy_name
    if npy.is_file():
        return npy
    extension = fallback_format.lower().lstrip(".")
    if not extension or extension == "npy":
        return None
    fallback = npy.with_suffix("." + extension)
    return fallback if fallback.is_file() else None


def read_native_depth(path, shape, scale, *, stream=None):
    source = path if stream is None else stream
    if path.suffix.lower() == ".npy":
        depth = np.load(source, allow_pickle=False).astype(np.float32, copy=False)
        if depth.shape != shape:
            raise ValueError(f"Depth must match the original camera grid: {path}")
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    with Image.open(source) as image:
        depth = np.asarray(image).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / float(scale)
