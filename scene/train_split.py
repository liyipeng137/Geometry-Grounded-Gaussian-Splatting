"""Explicit camera membership and provenance for same-dataset checkpoint resume."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


CHECKPOINT_STATE_KEY = "scene_reconstruction_training_state"
RGBD_INITIALIZATION_ALGORITHM = "rgbd_colmap_half_integer_float32_v3"


def load_excluded_image_names(split_path):
    if not split_path:
        return set(), None
    path = Path(split_path).resolve()
    contents = path.read_bytes()
    if path.suffix.lower() == ".json":
        names = json.loads(contents)
    else:
        names = [line for line in contents.decode("utf-8").splitlines() if line]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("train_exclude_images must list full COLMAP image.name strings")
    if len(set(names)) != len(names):
        raise ValueError("train_exclude_images contains duplicate names")
    return set(names), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def partition_cameras(cam_infos, excluded_names, eval_mode=False, llffhold=8):
    names = [camera.colmap_image_name or camera.image_name for camera in cam_infos]
    if len(set(names)) != len(names):
        raise ValueError("COLMAP image.name values must be unique")
    unknown = excluded_names.difference(names)
    if unknown:
        raise ValueError(f"Unknown excluded camera names: {sorted(unknown)}")
    if excluded_names and eval_mode:
        raise ValueError("Explicit camera exclusion cannot be combined with --eval")
    if eval_mode:
        if isinstance(llffhold, bool) or not isinstance(llffhold, int) or llffhold < 2:
            raise ValueError("llffhold must be an integer >= 2")
        excluded_names = {name for index, name in enumerate(names) if index % llffhold == 0}
    train = [camera for camera, name in zip(cam_infos, names) if name not in excluded_names]
    test = [camera for camera, name in zip(cam_infos, names) if name in excluded_names]
    if not train:
        raise ValueError("Camera split leaves no training cameras")
    return train, test


def training_camera_fingerprint(train_cameras):
    names = sorted(camera.colmap_image_name or camera.image_name for camera in train_cameras)
    return calibration_fingerprint(names)


def calibration_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_calibration_proof(record):
    if not isinstance(record, dict) or record.get("schema_version") != 3:
        raise ValueError("Checkpoint requires a formal schema3 camera/data proof")
    root = record.get("dataset_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Checkpoint lacks an absolute dataset root")
    train, test = record.get("train_image_names"), record.get("test_image_names")
    if not isinstance(train, list) or not train or not isinstance(test, list):
        raise ValueError("Checkpoint lacks explicit training/validation membership")
    if any(not isinstance(name, str) or not name for name in train + test) or len(set(train + test)) != len(train + test):
        raise ValueError("Checkpoint camera membership overlaps or contains duplicates")
    cameras = record.get("camera_calibration_fingerprints")
    if not isinstance(cameras, dict) or set(train + test) != set(cameras) or any(not _is_sha256(value) for value in cameras.values()):
        raise ValueError("Checkpoint calibration proof must cover every input camera")
    if record.get("calibration_fingerprint") != calibration_fingerprint(cameras):
        raise ValueError("Checkpoint calibration fingerprint is inconsistent")
    if not _is_sha256(record.get("dataset_content_fingerprint")):
        raise ValueError("Checkpoint lacks RGB/depth content identity")
    if record.get("pixel_center_convention") != "colmap_half_integer":
        raise ValueError("Checkpoint pixel convention must be COLMAP half-integer")
    expected_train = calibration_fingerprint(sorted(train))
    if record.get("train_camera_fingerprint") != expected_train:
        raise ValueError("Checkpoint training membership fingerprint is inconsistent")
    return cameras


def validate_initialization_proof(record):
    _validate_calibration_proof(record)
    initialization = record.get("initialization", {})
    if initialization.get("kind") != "train_only_rgbd" or initialization.get("algorithm") != RGBD_INITIALIZATION_ALGORITHM:
        raise ValueError("Checkpoint initialization must use current-split float32 RGBD v3")
    path = initialization.get("ply_path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Checkpoint initialization must preserve its absolute PLY source")
    if not _is_sha256(initialization.get("ply_sha256")) or not _is_sha256(initialization.get("cache_fingerprint")):
        raise ValueError("Checkpoint initialization lacks hashed RGBD/cache provenance")
    count = initialization.get("point_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Checkpoint RGBD initialization must contain a positive point count")


def checkpoint_training_state(model_params, iteration):
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 15:
        raise ValueError("Unsupported Gaussian checkpoint layout")
    optimizer = model_params[14]
    state = optimizer.get(CHECKPOINT_STATE_KEY) if isinstance(optimizer, dict) else None
    if state is not None and (not isinstance(state, dict) or state.get("iteration") != iteration):
        raise ValueError("Embedded checkpoint state does not match its saved iteration")
    return state


def read_checkpoint_split(checkpoint_path, embedded_state=None):
    if not isinstance(embedded_state, dict) or not isinstance(embedded_state.get("camera_split"), dict):
        raise ValueError("Formal resume requires embedded camera/data provenance; sidecars are not checkpoint authority")
    record = embedded_state["camera_split"]
    validate_initialization_proof(record)
    return record


def validate_checkpoint_split(checkpoint_path, model_path, previous_record=None):
    if previous_record is None:
        raise ValueError("Same-split resume requires the checkpoint's embedded provenance")
    path = Path(model_path) / "camera_split.json"
    current = json.loads(path.read_text())
    previous = previous_record
    validate_initialization_proof(previous)
    validate_initialization_proof(current)
    for field in ("train_camera_fingerprint", "train_image_names", "test_image_names",
                  "camera_calibration_fingerprints", "calibration_fingerprint", "dataset_content_fingerprint",
                  "pixel_center_convention"):
        if previous[field] != current[field]:
            raise ValueError(f"Checkpoint dataset/calibration/split changed: {field}")
    if Path(previous["dataset_root"]).resolve() != Path(current["dataset_root"]).resolve():
        raise ValueError("Checkpoint resume requires the same normalized dataset root")
    if previous["initialization"]["cache_fingerprint"] != current["initialization"]["cache_fingerprint"]:
        raise ValueError("Checkpoint RGBD initialization operator or training inputs changed")
    current["scene_placeholder_initialization"] = current["initialization"]
    current["initialization"] = previous["initialization"]
    current["checkpoint_restoration"] = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()), "initialization_preserved_from_checkpoint": True,
        "scene_placeholder_used_for_model_state": False,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(current, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"expanded": False, "initialization_preserved": True, "initialization_kind": "train_only_rgbd",
            "previous_train_count": len(previous["train_image_names"]), "current_train_count": len(current["train_image_names"]),
            "remaining_heldout_image_names": current["test_image_names"]}
