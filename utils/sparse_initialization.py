"""Build a real-geometry initialization from TSDF vertices and registered SfM points."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from utils.colmap_read_model import read_points3D_text, read_points3d_binary


def sparse_points_path(model):
    for name in ("points3D.bin", "points3D.txt"):
        path = Path(model) / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No points3D.bin or points3D.txt in {model}. "
        "Use --no-merge-sparse-init for TSDF-only initialization."
    )


def merge_tsdf_sparse_ply(tsdf_path, sparse_path, centers, output):
    """Write every TSDF row followed by every in-range sparse row, in source order."""
    source = PlyData.read(tsdf_path)["vertex"].data
    if not len(source) or not len(centers):
        raise ValueError("Merging requires TSDF vertices and at least one training camera")
    xyz_fields = ("x", "y", "z")
    rgb_fields = ("red", "green", "blue")
    tsdf_xyz = np.column_stack([source[field] for field in xyz_fields])
    tree = cKDTree(centers)
    cutoff = float(tree.query(tsdf_xyz)[0].max())

    if sparse_path.suffix == ".bin":
        points = read_points3d_binary(sparse_path)
    else:
        points = read_points3D_text(sparse_path)
    sparse_xyz = np.array([p.xyz for p in points.values()], dtype=np.float64).reshape(-1, 3)
    sparse_rgb = np.array([p.rgb for p in points.values()], dtype=np.uint8).reshape(-1, 3)
    keep = tree.query(sparse_xyz)[0] <= cutoff
    dtype = [(key, "f8") for key in xyz_fields] + [(key, "u1") for key in rgb_fields]
    vertices = np.empty(len(source) + int(keep.sum()), dtype=dtype)
    for key in xyz_fields + rgb_fields:
        vertices[key][:len(source)] = source[key]
    for fields, values in ((xyz_fields, sparse_xyz[keep]), (rgb_fields, sparse_rgb[keep])):
        for axis, key in enumerate(fields):
            vertices[key][len(source):] = values[:, axis]
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(output)
    return {
        "tsdf_points": len(source),
        "sparse_points": len(points),
        "sparse_kept": int(keep.sum()),
        "sparse_clipped": int((~keep).sum()),
        "merged_points": len(vertices),
        "max_nearest_camera_distance": cutoff,
        "train_cameras": len(centers),
    }
