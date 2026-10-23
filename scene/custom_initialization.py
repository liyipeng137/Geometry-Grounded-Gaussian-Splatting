"""Validate ordinary, real-supported point clouds for explicit cold starts."""

import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_custom_initialization(ply_path, metadata_path, dataset_root, camera_calibrations, train_names, test_names):
    if not ply_path or not metadata_path:
        raise ValueError("Custom initialization requires both custom_init_ply and custom_init_metadata")
    if not Path(ply_path).is_absolute() or not Path(metadata_path).is_absolute():
        raise ValueError("Custom initialization paths must be absolute")
    ply_path, metadata_path = Path(ply_path).resolve(), Path(metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1 or metadata.get("initialization_kind") != "supported_surface_coldstart":
        raise ValueError("Unsupported custom point-cloud initialization provenance")
    if metadata.get("coordinate_system") != "world" or metadata.get("units") != "scene" or metadata.get("pixel_center_convention") != "colmap_half_integer":
        raise ValueError("Custom initialization must preserve world/scene units and half-integer calibration")
    if Path(metadata["dataset_root"]).resolve() != Path(dataset_root).resolve():
        raise ValueError("Custom initialization belongs to another dataset")
    if metadata.get("camera_calibration_fingerprints") != camera_calibrations or metadata.get("calibration_fingerprint") != _fingerprint(camera_calibrations):
        raise ValueError("Custom initialization camera pose/K proof changed")
    if sorted(metadata.get("train_image_names", [])) != sorted(train_names) or sorted(metadata.get("test_image_names", [])) != sorted(test_names):
        raise ValueError("Custom initialization does not match the requested training/heldout split")
    if set(train_names) & set(test_names):
        raise ValueError("Custom initialization training and heldout identities overlap")
    source_names = metadata.get("source_image_names", [])
    if not source_names or len(source_names) != len(set(source_names)) or not set(source_names) <= set(train_names):
        raise ValueError("Custom initialization real support contains heldout or unknown cameras")
    if Path(metadata["point_cloud_path"]).resolve() != ply_path or _sha256(ply_path) != metadata["point_cloud_sha256"]:
        raise ValueError("Custom initialization PLY does not match its metadata SHA256")
    parent = metadata["parent_model"]
    if _sha256(parent["path"]) != parent["sha256"]:
        raise ValueError("Custom initialization parent model changed")
    if parent.get("camera_split_path"):
        if _sha256(parent["camera_split_path"]) != parent["camera_split_sha256"]:
            raise ValueError("Custom initialization parent camera proof changed")
        parent_proof = json.loads(Path(parent["camera_split_path"]).read_text())
        if not set(parent_proof["train_image_names"]) <= set(train_names):
            raise ValueError("Custom initialization parent model used a heldout camera")
        if Path(parent_proof.get("dataset_root", "")).resolve() != Path(dataset_root).resolve() or parent_proof.get("camera_calibration_fingerprints") != camera_calibrations:
            raise ValueError("Custom initialization parent model dataset or pose/K proof disagrees")
    elif test_names:
        raise ValueError("Heldout custom initialization requires the parent model's training-camera proof")
    document = PlyData.read(str(ply_path))
    if len(document.elements) != 1 or document.elements[0].name != "vertex":
        raise ValueError("Custom initialization must be an ordinary point cloud, not a mesh")
    vertices = document["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z", "red", "green", "blue"} <= names or not names <= {"x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"}:
        raise ValueError("Custom initialization requires ordinary XYZ/RGB, not native Gaussian properties")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.float32)
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 255)) or colors.max() <= 1e-6:
        raise ValueError("Custom initialization points/RGB must be finite, nonempty and valid, without random-color fallback")
    support_path = (metadata_path.parent / metadata["support_path"]).resolve()
    if _sha256(support_path) != metadata["support_sha256"]:
        raise ValueError("Custom initialization support provenance changed")
    with np.load(support_path, allow_pickle=False) as support:
        if support["positions"].shape != points.shape or not np.array_equal(support["positions"], points):
            raise ValueError("Custom initialization per-point support does not match its PLY")
        counts, conflicts = support["real_source_count"], support["free_conflict_count"]
        if counts.shape != (len(points),) or conflicts.shape != (len(points),) or not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(conflicts.dtype, np.integer) or np.any(counts < 1) or np.any(conflicts != 0):
            raise ValueError("Custom initialization includes unobserved or real-free-space points")
        color_sources = support["color_source_indices"]
        if color_sources.ndim != 2 or color_sources.shape[0] != len(points) or not np.issubdtype(color_sources.dtype, np.integer) or np.any((color_sources < -1) | (color_sources >= len(source_names))) or np.any((color_sources >= 0).sum(axis=1) == 0):
            raise ValueError("Custom initialization colors lack explicit real training-camera support")
    colors /= 255
    metadata_sha = _sha256(metadata_path)
    record = {
        "kind": "supported_surface_coldstart", "algorithm": "real_supported_surface_v1",
        "ply_path": str(ply_path), "ply_sha256": metadata["point_cloud_sha256"],
        "metadata_path": str(metadata_path), "metadata_sha256": metadata_sha,
        "source_model_sha256": parent["sha256"], "support_scope": "training_cameras_only",
        "cache_fingerprint": _fingerprint({"ply_sha256": metadata["point_cloud_sha256"], "metadata_sha256": metadata_sha}),
    }
    return BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points)), record
