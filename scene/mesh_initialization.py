"""Content-bound surface point-cloud initialization, independent of CUDA."""

import hashlib
import json
from pathlib import Path


MESH_INITIALIZATION_ALGORITHM = "mesh_surface_xyzrgb_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_asset(value, label):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not Path(value["path"]).is_absolute():
        raise ValueError(f"Mesh initialization {label} requires an absolute source path")
    path = Path(value["path"]).resolve()
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"Mesh initialization {label} is missing or its content changed")
    return path


def validate_mesh_manifest(ply_path, manifest_path, dataset_root, train_image_names):
    """Bind preserved surface samples to identical current RGB/calibration content.

    The producer's dataset root describes where sampling/coloring happened.  A
    new normalized dataset may reuse those samples with another depth prior,
    but must retain every training RGB and both COLMAP calibration files byte
    for byte.  Depth identity belongs to the training camera/data proof.
    """
    for value in (ply_path, manifest_path, dataset_root):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Mesh initialization paths must be explicit and absolute")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "mesh_surface":
        raise ValueError("Mesh initialization requires a schema1 mesh_surface manifest")
    root = manifest.get("dataset_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Mesh initialization manifest requires its absolute sampling dataset root")
    active_root = Path(dataset_root).resolve()
    if not active_root.is_dir():
        raise ValueError("Mesh initialization current dataset root is missing")
    names = manifest.get("train_image_names")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names) or sorted(names) != sorted(train_image_names):
        raise ValueError("Mesh initialization manifest training-camera membership changed")
    source = manifest.get("initialization_ply")
    preserved_ply = _checked_asset(source, "PLY")
    if preserved_ply != Path(ply_path).resolve():
        raise ValueError("Mesh initialization manifest names a different PLY")
    count = source.get("point_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Mesh initialization manifest requires a positive point count")
    _checked_asset(manifest.get("source_mesh"), "source mesh")
    for name in ("sampling", "coloring"):
        if not isinstance(manifest.get(name), dict) or not manifest[name]:
            raise ValueError(f"Mesh initialization must preserve its {name} procedure")
    if "support_npz" in manifest:
        _checked_asset(manifest["support_npz"], "color support")
    calibrations = manifest.get("calibration_sha256")
    if not isinstance(calibrations, dict) or len(calibrations) != 2:
        raise ValueError("Mesh initialization must preserve both COLMAP calibration sources")
    source_calibrations = {}
    for path, checksum in calibrations.items():
        _checked_asset({"path": path, "sha256": checksum}, "calibration")
        source_calibrations[Path(path).name] = checksum
    model = active_root / "sparse/0"
    extension = "bin" if all((model / f"{name}.bin").is_file() for name in ("cameras", "images")) else "txt"
    expected_names = {f"{name}.{extension}" for name in ("cameras", "images")}
    if set(source_calibrations) != expected_names:
        raise ValueError("Mesh initialization calibration format differs from the current training dataset")
    for name, checksum in source_calibrations.items():
        _checked_asset({"path": str(model / name), "sha256": checksum}, "current calibration")
    views = manifest.get("views")
    if not isinstance(views, list) or len(views) != len(names) or {view.get("name") for view in views if isinstance(view, dict)} != set(names):
        raise ValueError("Mesh initialization must preserve RGB identity for every training view")
    for view in views:
        _checked_asset({"path": view.get("image_path"), "sha256": view.get("image_sha256")}, "color RGB")
        name = Path(view["name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Mesh initialization training RGB name must stay within the images directory")
        _checked_asset({"path": str(active_root / "images" / name), "sha256": view.get("image_sha256")}, "current color RGB")
    return manifest, sha256_file(manifest_path)
