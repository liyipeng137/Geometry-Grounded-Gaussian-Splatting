"""Provenance for an unchanged, full-capture FFBA TSDF point cloud."""

import json
from pathlib import Path

from scene.mesh_initialization import sha256_file


FFBA_TSDF_INITIALIZATION_ALGORITHM = "ffba_original_tsdf_xyzrgb_v1"
FFBA_TSDF_EVALUATION_SCOPE = "full_capture_initialization_not_heldout"


def _asset(record, label):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not Path(record["path"]).is_absolute():
        raise ValueError(f"FFBA TSDF {label} needs an absolute source path")
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"FFBA TSDF {label} is missing or its content changed")
    return path


def _names(value, label):
    if (not isinstance(value, list) or not value or
            any(not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts for name in value) or
            len(set(value)) != len(value)):
        raise ValueError(f"FFBA TSDF {label} must list distinct relative COLMAP image names")
    return set(value)


def point_cloud_header(path):
    """Read a PLY vertex count, permitting original TSDF faces to be ignored."""
    fields, count, current_element = set(), None, None
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("FFBA TSDF initializer must be a PLY point cloud")
        for _ in range(4096):
            line = stream.readline(65536)
            if not line:
                break
            parts = line.decode("ascii").strip().split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "element" and len(parts) == 3:
                current_element = parts[1]
                element_count = int(parts[2])
                if element_count < 0:
                    raise ValueError("FFBA TSDF PLY has a negative element count")
                if current_element == "vertex":
                    if count is not None:
                        raise ValueError("FFBA TSDF PLY has duplicate vertex elements")
                    count = element_count
            elif parts[0] == "property" and current_element == "vertex":
                fields.add(parts[-1])
            elif parts[0] == "end_header":
                if not count or not {"x", "y", "z", "red", "green", "blue"}.issubset(fields) or "opacity" in fields:
                    raise ValueError("FFBA TSDF PLY requires nonempty XYZ/RGB points, not Gaussian state")
                return count
    raise ValueError("FFBA TSDF PLY has an incomplete or oversized header")


def validate_ffba_tsdf_manifest(ply_path, manifest_path, dataset_root, train_image_names, source_image_names):
    """Bind full-capture initialization to the current camera model and split.

    This deliberately permits capture-wide initialization. It records that
    provenance instead of claiming the initializer excludes evaluation images.
    RGB/depth supervision content remains bound by the scene's schema3 proof.
    """
    for value in (ply_path, manifest_path, dataset_root):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("FFBA TSDF initialization paths must be explicit and absolute")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "ffba_tsdf" or
            manifest.get("algorithm") != FFBA_TSDF_INITIALIZATION_ALGORITHM):
        raise ValueError("FFBA TSDF initialization requires its schema1 ffba_tsdf manifest and algorithm")
    root = manifest.get("dataset_root")
    if not isinstance(root, str) or not Path(root).is_absolute() or not Path(root).is_dir():
        raise ValueError("FFBA TSDF manifest requires the absolute original dataset root")
    if manifest.get("initialization_scope") != "full_capture" or manifest.get("evaluation_scope") != FFBA_TSDF_EVALUATION_SCOPE:
        raise ValueError("FFBA TSDF initialization must disclose full-capture evaluation scope")
    if manifest.get("geometry_selection") != "vertices_only":
        raise ValueError("FFBA TSDF initialization must explicitly select original vertices only")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("producer") != "ffba":
        raise ValueError("FFBA TSDF manifest must explicitly preserve FFBA producer provenance")
    source_names = _names(manifest.get("source_image_names"), "source cameras")
    train_names = _names(manifest.get("train_image_names"), "training cameras")
    if source_names != set(source_image_names) or train_names != set(train_image_names) or not train_names.issubset(source_names):
        raise ValueError("FFBA TSDF manifest source/training camera membership changed")
    source = manifest.get("initialization_ply")
    source_path = _asset(source, "PLY")
    if source_path != Path(ply_path).resolve():
        raise ValueError("FFBA TSDF manifest names a different initialization PLY")
    count = source.get("point_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count != point_cloud_header(source_path):
        raise ValueError("FFBA TSDF manifest point count disagrees with its PLY")
    calibrations = manifest.get("calibration_sha256")
    if not isinstance(calibrations, dict) or len(calibrations) != 2:
        raise ValueError("FFBA TSDF manifest needs both COLMAP calibration source hashes")
    model = Path(dataset_root).resolve() / "sparse/0"
    extension = "bin" if all((model / f"{name}.bin").is_file() for name in ("cameras", "images")) else "txt"
    expected_names = {f"{name}.{extension}" for name in ("cameras", "images")}
    if {Path(path).name for path in calibrations} != expected_names:
        raise ValueError("FFBA TSDF calibration format differs from the current dataset")
    for path, checksum in calibrations.items():
        _asset({"path": path, "sha256": checksum}, "source calibration")
        _asset({"path": str(model / Path(path).name), "sha256": checksum}, "current calibration")
    reports = manifest.get("source_reports", [])
    if not isinstance(reports, list):
        raise ValueError("FFBA TSDF source_reports must be a list of hashed source assets")
    for report in reports:
        _asset(report, "source report")
    return manifest, sha256_file(manifest_path)
