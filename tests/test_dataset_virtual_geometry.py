"""Dataset discovery validates real artifact bytes without preparing anything."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from utils.dataset_virtual_geometry import (
    resolve_virtual_geometry_config,
    validate_dataset_pipeline,
)


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pipeline(tmp_path):
    dataset = tmp_path / "dataset"
    root = dataset / "virtual_geometry"
    teacher_dir = root / "teacher"
    teacher_dir.mkdir(parents=True)
    mesh = dataset / "genrecon_mesh.ply"
    mesh.write_bytes(b"mesh input; never decoded by discovery")
    teacher = teacher_dir / "manifest.json"
    write_json(teacher, {"schema_version": 3, "status": "complete", "virtual_rgb": False,
                         "dataset": str(dataset), "genrecon_mesh": str(mesh)})
    config = root / "train_config.json"
    write_json(config, {"schema_version": 1, "teacher_manifest": "teacher/manifest.json"})
    marker = root / "pipeline.json"

    def refresh_marker(**changes):
        document = {"schema_version": 1, "status": "complete", "dataset": str(dataset),
                    "mesh": str(mesh), "config_sha256": digest(config),
                    "teacher_manifest_sha256": digest(teacher)}
        document.update(changes)
        write_json(marker, document)

    refresh_marker()
    return SimpleNamespace(dataset=dataset, root=root, teacher=teacher, config=config,
                           marker=marker, mesh=mesh, refresh=refresh_marker, tmp=tmp_path)


def test_complete_default_pipeline_resolves_exact_config(pipeline):
    expected = str(pipeline.config.resolve())
    assert validate_dataset_pipeline(pipeline.dataset) == expected
    assert resolve_virtual_geometry_config(str(pipeline.dataset)) == expected


def test_validation_can_check_staged_root_before_publication(pipeline):
    staged = pipeline.tmp / "prepared_staging"
    shutil.copytree(pipeline.root, staged)
    pipeline.marker.write_text("broken default marker")
    assert validate_dataset_pipeline(pipeline.dataset, root=staged) == str(staged / "train_config.json")


def test_absent_default_preserves_baseline_without_creating_artifacts(tmp_path):
    dataset = tmp_path / "new_dataset"
    dataset.mkdir()
    assert resolve_virtual_geometry_config(dataset) is None
    assert not (dataset / "virtual_geometry").exists()


@pytest.mark.parametrize("kind", ["empty_directory", "plain_file", "dangling_symlink"])
def test_existing_incomplete_default_never_silently_falls_back(tmp_path, kind):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    root = dataset / "virtual_geometry"
    if kind == "empty_directory":
        root.mkdir()
    elif kind == "plain_file":
        root.write_text("not an artifact directory")
    else:
        root.symlink_to(tmp_path / "removed_artifacts", target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_virtual_geometry_config(dataset)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("status", "running"),
                                         ("dataset", "/different/dataset"), ("dataset", None),
                                         ("mesh", None), ("mesh", "relative_mesh.ply")])
def test_bad_marker_is_rejected(pipeline, field, value):
    marker = json.loads(pipeline.marker.read_text())
    marker[field] = value
    write_json(pipeline.marker, marker)
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


def test_marker_requires_absolute_dataset_identity(pipeline, monkeypatch):
    monkeypatch.chdir(pipeline.tmp)
    pipeline.refresh(dataset="dataset")
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


def test_marker_mesh_must_be_the_teacher_mesh_source(pipeline):
    other_mesh = pipeline.dataset / "different_mesh.ply"
    other_mesh.write_bytes(b"another valid source path")
    pipeline.refresh(mesh=str(other_mesh))
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("contents", ["{truncated", "[]", "null"])
def test_malformed_marker_document_is_rejected(pipeline, contents):
    pipeline.marker.write_text(contents)
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("artifact", ["marker", "config", "teacher"])
def test_missing_required_artifact_is_a_clear_failure(pipeline, artifact):
    getattr(pipeline, artifact).unlink()
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("artifact", ["config", "teacher"])
def test_artifact_byte_change_is_detected_even_with_identical_json_meaning(pipeline, artifact):
    with getattr(pipeline, artifact).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("teacher_manifest", ""),
                                         ("teacher_manifest", None), ("teacher_manifest", 42)])
def test_invalid_config_is_rejected_after_its_hash_is_updated(pipeline, field, value):
    config = json.loads(pipeline.config.read_text())
    config[field] = value
    write_json(pipeline.config, config)
    pipeline.refresh()
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("absolute", [False, True])
def test_config_cannot_validate_one_teacher_but_train_with_another(pipeline, absolute):
    outside = pipeline.dataset / "other_teacher.json"
    outside.write_bytes(pipeline.teacher.read_bytes())  # Same content, wrong target.
    config = json.loads(pipeline.config.read_text())
    config["teacher_manifest"] = str(outside) if absolute else "../other_teacher.json"
    write_json(pipeline.config, config)
    pipeline.refresh()
    with pytest.raises(ValueError, match="own teacher"):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("status", "running"),
                                         ("virtual_rgb", True), ("virtual_rgb", None),
                                         ("dataset", "/other/dataset"), ("dataset", None)])
def test_incomplete_or_wrong_teacher_is_rejected_with_matching_hash(pipeline, field, value):
    teacher = json.loads(pipeline.teacher.read_text())
    teacher[field] = value
    write_json(pipeline.teacher, teacher)
    pipeline.refresh()
    with pytest.raises(ValueError):
        validate_dataset_pipeline(pipeline.dataset)


@pytest.mark.parametrize("default_state", ["complete", "broken", "absent"])
def test_explicit_legacy_config_always_takes_precedence(pipeline, default_state):
    if default_state == "broken":
        pipeline.marker.write_text("half-written pipeline")
    elif default_state == "absent":
        shutil.rmtree(pipeline.root)
    explicit = pipeline.tmp / "legacy_config.json"
    # Old explicit configs do not need dataset-local marker layout.
    write_json(explicit, {"schema_version": 1, "teacher_manifest": "/old/output/teacher/manifest.json"})
    assert resolve_virtual_geometry_config(pipeline.dataset, explicit_config=explicit) == str(explicit)


def test_explicit_relative_path_remains_relative_to_cwd_not_dataset(pipeline, monkeypatch):
    config_dir = pipeline.tmp / "working_directory/configs"
    config_dir.mkdir(parents=True)
    explicit = config_dir / "legacy.json"
    write_json(explicit, {"teacher_manifest": "/legacy/teacher.json"})
    monkeypatch.chdir(config_dir.parent)
    assert resolve_virtual_geometry_config(pipeline.dataset, explicit_config="configs/legacy.json") == str(explicit)


@pytest.mark.parametrize("break_default", [False, True])
def test_disabled_bypasses_complete_or_broken_default(pipeline, break_default):
    if break_default:
        pipeline.marker.write_text("partial pipeline")
    assert resolve_virtual_geometry_config(pipeline.dataset, disabled=True) is None


def test_disable_and_explicit_config_are_mutually_exclusive(pipeline):
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_virtual_geometry_config(pipeline.dataset, explicit_config=pipeline.config, disabled=True)


def test_discovery_is_read_only_for_existing_artifacts(pipeline):
    paths = (pipeline.marker, pipeline.config, pipeline.teacher, pipeline.mesh)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    resolve_virtual_geometry_config(pipeline.dataset)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before
