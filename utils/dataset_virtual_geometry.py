"""Dataset-local virtual-geometry artifacts: discovery only, never preparation."""

import hashlib
import json
from pathlib import Path


ARTIFACT_DIRECTORY = "virtual_geometry"


def _document(path):
    try:
        contents = path.read_bytes()
        document = json.loads(contents)
    except (OSError, ValueError) as error:
        raise ValueError(f"Incomplete or unreadable virtual geometry artifact: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Virtual geometry artifact must be a JSON object: {path}")
    return document, hashlib.sha256(contents).hexdigest()


def validate_dataset_pipeline(dataset, root=None):
    """Only publish/use a completed A pipeline bound to this real dataset.

    The existing VirtualGeometryDataset performs the full source and NPZ checks
    afterward. Here we check completion and the exact config/teacher to load,
    so a partial preparation cannot silently turn into real-only training.
    """
    dataset = Path(dataset).expanduser().resolve()
    root = Path(root).expanduser().resolve() if root is not None else dataset / ARTIFACT_DIRECTORY
    marker, _ = _document(root / "pipeline.json")
    if marker.get("schema_version") != 1 or marker.get("status") != "complete":
        raise ValueError(f"Virtual geometry preparation is not complete: {root}. Finish pipeline A first.")
    recorded_dataset = marker.get("dataset")
    if (not isinstance(recorded_dataset, str) or not Path(recorded_dataset).is_absolute()
            or Path(recorded_dataset).resolve() != dataset):
        raise ValueError(f"Virtual geometry pipeline belongs to a different dataset: {root}")
    mesh = marker.get("mesh")
    if not isinstance(mesh, str) or not Path(mesh).is_absolute():
        raise ValueError(f"Virtual geometry pipeline lacks an absolute mesh source: {root}")
    config_path = root / "train_config.json"
    config, config_sha = _document(config_path)
    if config_sha != marker.get("config_sha256"):
        raise ValueError(f"Dataset virtual geometry training config changed: {config_path}")
    target = config.get("teacher_manifest")
    if config.get("schema_version") != 1 or not isinstance(target, str) or not target:
        raise ValueError(f"Invalid dataset virtual geometry training config: {config_path}")
    target = Path(target)
    teacher_path = (target if target.is_absolute() else root / target).resolve()
    if teacher_path != (root / "teacher/manifest.json").resolve():
        raise ValueError("Dataset training config must reference its own teacher/manifest.json")
    teacher, teacher_sha = _document(teacher_path)
    if teacher_sha != marker.get("teacher_manifest_sha256"):
        raise ValueError(f"Dataset virtual geometry teacher manifest changed: {teacher_path}")
    if (teacher.get("schema_version") != 3 or teacher.get("status") != "complete"
            or teacher.get("virtual_rgb") is not False):
        raise ValueError(f"Completed geometry-only teacher required: {teacher_path}")
    teacher_mesh = teacher.get("genrecon_mesh")
    if (not isinstance(teacher_mesh, str) or not Path(teacher_mesh).is_absolute()
            or Path(teacher_mesh).resolve() != Path(mesh).resolve()):
        raise ValueError(f"Pipeline and teacher mesh sources differ: {teacher_path}")
    teacher_dataset = teacher.get("dataset")
    if (not isinstance(teacher_dataset, str) or not Path(teacher_dataset).is_absolute()
            or Path(teacher_dataset).resolve() != dataset):
        raise ValueError(f"Teacher belongs to a different dataset: {teacher_path}")
    return str(config_path.resolve())


def resolve_virtual_geometry_config(dataset, explicit_config=None, disabled=False):
    """Explicit config > dataset-local discovery; absence preserves the baseline."""
    if disabled and explicit_config:
        raise ValueError("--disable_virtual_geometry and --virtual_geometry_config are mutually exclusive")
    if disabled:
        return None
    if explicit_config:
        # Keep legacy external configs usable even if dataset artifacts are absent
        # or incomplete. load_joint_config still validates this explicit choice.
        return str(Path(explicit_config).expanduser().resolve())
    root = Path(dataset).expanduser().resolve() / ARTIFACT_DIRECTORY
    if not root.exists() and not root.is_symlink():
        return None
    return validate_dataset_pipeline(dataset, root)
