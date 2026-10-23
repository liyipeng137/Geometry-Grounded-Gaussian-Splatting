import hashlib
import json
from pathlib import Path


CHECKPOINT_STATE_KEY = "scene_reconstruction_training_state"


def load_excluded_image_names(split_path):
    if not split_path:
        return set(), None
    path = Path(split_path).resolve()
    contents = path.read_bytes()
    if path.suffix.lower() == ".json":
        names = json.loads(contents)
        if isinstance(names, dict):
            names = names.get("exclude_images")
    else:
        names = [line.strip() for line in contents.decode("utf-8").splitlines() if line.strip()]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("train_exclude_images must be a list of full COLMAP image.name strings; an explicit empty list selects all training cameras")
    if len(set(names)) != len(names):
        raise ValueError("train_exclude_images contains duplicate image names")
    return set(names), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def partition_cameras(cam_infos, excluded_names, eval_mode=False, llffhold=8):
    names = [camera.colmap_image_name or Path(camera.image_path).name for camera in cam_infos]
    if len(set(names)) != len(names):
        raise ValueError("COLMAP image.name values must be unique")
    missing = excluded_names.difference(names)
    if missing:
        raise ValueError(f"Unknown excluded COLMAP image.name values: {sorted(missing)}")
    if excluded_names and eval_mode:
        raise ValueError("Explicit train_exclude_images cannot be combined with implicit --eval splitting")
    if excluded_names:
        train_cameras = [camera for camera, name in zip(cam_infos, names) if name not in excluded_names]
        test_cameras = [camera for camera, name in zip(cam_infos, names) if name in excluded_names]
    elif eval_mode:
        train_cameras = [camera for index, camera in enumerate(cam_infos) if index % llffhold != 0]
        test_cameras = [camera for index, camera in enumerate(cam_infos) if index % llffhold == 0]
    else:
        train_cameras, test_cameras = list(cam_infos), []
    if not train_cameras:
        raise ValueError("Camera split leaves no training cameras")
    return train_cameras, test_cameras


def training_camera_fingerprint(train_cameras):
    names = sorted(camera.colmap_image_name or Path(camera.image_path).name for camera in train_cameras)
    return hashlib.sha256(json.dumps(names, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_checkpoint_split(checkpoint_path, embedded_state=None):
    if embedded_state is not None:
        record = embedded_state.get("camera_split")
        if record is None:
            raise ValueError("Checkpoint was trained without an explicit camera exclusion contract")
        return record
    checkpoint_path = Path(checkpoint_path).resolve()
    state_path = checkpoint_path.with_suffix(".state.json")
    if state_path.exists():
        record = json.loads(state_path.read_text()).get("camera_split")
        if record is not None:
            return record
    checkpoint_record_path = checkpoint_path.parent / "camera_split.json"
    if not checkpoint_record_path.exists():
        raise ValueError("Split training cannot resume an unverified checkpoint without camera_split.json")
    return json.loads(checkpoint_record_path.read_text())


def checkpoint_training_state(model_params, iteration):
    optimizer_state = model_params[14]
    state = optimizer_state.get(CHECKPOINT_STATE_KEY) if isinstance(optimizer_state, dict) else None
    if state is not None:
        if not isinstance(state, dict) or state.get("iteration") != iteration:
            raise ValueError("Embedded checkpoint training state does not match its model iteration")
        return state
    return None


def calibration_fingerprint(calibrations):
    return hashlib.sha256(json.dumps(calibrations, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _validate_calibration_proof(record):
    cameras = record.get("camera_calibration_fingerprints")
    if not isinstance(record.get("dataset_root"), str) or not isinstance(cameras, dict) or not cameras:
        raise ValueError("Split expansion requires checkpoint dataset identity and per-camera pose/K proof; legacy checkpoints without this evidence cannot expand")
    names = set(record["train_image_names"]) | set(record["test_image_names"])
    if names != set(cameras) or any(not isinstance(value, str) or len(value) != 64 for value in cameras.values()):
        raise ValueError("Split calibration proof does not cover every source camera")
    if record.get("calibration_fingerprint") != calibration_fingerprint(cameras):
        raise ValueError("Split calibration fingerprint does not match its per-camera evidence")
    return cameras


def _no_appearance(model):
    if getattr(model, "name", None) is not None:
        return model.name == "NO"
    return model == 0 or model == "NO"


def validate_initialization_proof(record):
    initialization = record.get("initialization", {})
    if initialization.get("kind") == "train_only_rgbd":
        return
    if initialization.get("kind") != "supported_surface_coldstart":
        raise ValueError("Split initialization must be verified training-only RGBD or a supported-surface coldstart")
    if initialization.get("support_scope") != "training_cameras_only":
        raise ValueError("Supported-surface initialization must exclude heldout observations")
    for key in ("ply_path", "metadata_path"):
        path = initialization.get(key)
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Supported-surface initialization lacks absolute {key}")
    for key in ("ply_sha256", "metadata_sha256"):
        digest = initialization.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Supported-surface initialization lacks a valid {key}")
    _validate_calibration_proof(record)


def validate_checkpoint_split(checkpoint_path, model_path, previous_record=None,
                              allow_expansion=False, current_appearance=0, checkpoint_appearance=0):
    current_record_path = Path(model_path) / "camera_split.json"
    previous = read_checkpoint_split(checkpoint_path) if previous_record is None else previous_record
    current = json.loads(current_record_path.read_text())
    validate_initialization_proof(previous)
    validate_initialization_proof(current)
    changed = any(previous.get(field) != current.get(field) for field in ("train_camera_fingerprint", "train_image_names", "test_image_names"))
    if changed and not allow_expansion:
        raise ValueError("Checkpoint camera split disagrees with the requested split; expansion requires explicit --allow_train_split_expansion")
    old_calibration = previous.get("camera_calibration_fingerprints")
    new_calibration = current.get("camera_calibration_fingerprints")
    if changed or (old_calibration is not None and new_calibration is not None):
        old_calibration = _validate_calibration_proof(previous)
        new_calibration = _validate_calibration_proof(current)
        if Path(previous["dataset_root"]).resolve() != Path(current["dataset_root"]).resolve():
            raise ValueError("Checkpoint expansion/resume requires the same dataset root")
        if not set(old_calibration).issubset(new_calibration):
            raise ValueError("Requested dataset removed a camera from the checkpoint calibration proof")
        for name, fingerprint in old_calibration.items():
            if new_calibration[name] != fingerprint:
                raise ValueError(f"Checkpoint camera pose/K changed: {name}")
    old_train = set(previous["train_image_names"])
    new_train = set(current["train_image_names"])
    if changed:
        if not old_train < new_train:
            raise ValueError("Explicit split expansion only permits a strict superset of the old training cameras")
        if not _no_appearance(current_appearance) or not _no_appearance(checkpoint_appearance):
            raise ValueError("Split expansion currently requires NO appearance model; embedding/optimizer remapping is not implemented")
        if set(current["test_image_names"]) & new_train:
            raise ValueError("Expanded training and heldout cameras overlap")
    placeholder_initialization = current.get("scene_placeholder_initialization", current["initialization"])
    current["initialization"] = previous["initialization"]
    current["scene_placeholder_initialization"] = placeholder_initialization
    current["checkpoint_restoration"] = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "initialization_preserved_from_checkpoint": True,
        "scene_placeholder_used_for_model_state": False,
        "previous_train_camera_fingerprint": previous["train_camera_fingerprint"],
        "expanded_split": changed,
    }
    temporary_record_path = current_record_path.with_suffix(".json.tmp")
    temporary_record_path.write_text(json.dumps(current, indent=2) + "\n")
    temporary_record_path.replace(current_record_path)
    return {
        "expanded": changed,
        "initialization_preserved": True,
        "initialization_kind": previous["initialization"]["kind"],
        "scene_placeholder_initialization_kind": placeholder_initialization["kind"],
        "previous_train_count": len(old_train),
        "current_train_count": len(new_train),
        "newly_trained_image_names": sorted(new_train - old_train),
        "remaining_heldout_image_names": current["test_image_names"],
        "evaluation_scope": "All source cameras are now training views; no heldout evaluation remains" if not current["test_image_names"] else "Only remaining test cameras are heldout; newly absorbed cameras are training views",
    }
