"""A 入口只做工程编排；mock 大 mesh 子阶段，验证发布/失败/复用契约。"""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_virtual_geometry.py"
SPEC = importlib.util.spec_from_file_location("virtual_geometry_pipeline_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def stages(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    mesh = dataset / "genrecon_mesh.ply"
    mesh.write_text("fixture mesh")
    calls = []

    def selection(source, mesh_path, output, **kwargs):
        assert source == dataset and mesh_path == mesh
        calls.append(("selection", kwargs))
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"dataset": str(source), "genrecon_mesh": str(mesh_path)}))
        cameras = [
            {"id": "real_0", "selection_source": "real"},
            {"id": "virtual_0", "selection_source": "trajectory_offset"},
            {"id": "virtual_1", "selection_source": "surface_lookat"}]
        (output / "previews").mkdir()
        for camera in cameras:
            camera["preview"] = {}
            for kind in ("shading", "normal", "depth"):
                relative = f"previews/{camera['id']}_{kind}.png"
                (output / relative).write_bytes(b"fixture png")
                camera["preview"][kind] = relative
        (output / "real_cameras.json").write_text(json.dumps({"schema_version": 2, "cameras": cameras[:1]}))
        (output / "virtual_cameras.json").write_text(json.dumps({"schema_version": 2, "cameras": cameras[1:]}))

    def preview(command, check):
        calls.append(("preview", command))
        assert check and command[0] == MODULE.sys.executable
        root = Path(command[-1]).parent
        for name in MODULE.PREVIEW_FILES:
            (root / name).write_text("fixture preview")

    def teacher(approved, output, **kwargs):
        calls.append(("teacher", kwargs))
        assert approved.name == "manifest.json" and approved.parent.name == "cameras"
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"schema_version": 3, "status": "complete", "virtual_rgb": False,
                                                         "dataset": str(dataset), "genrecon_mesh": str(mesh)}))

    def verify(manifest_path, device):
        calls.append(("verify", device))
        assert device == "cpu"
        marker = json.loads((manifest_path.parent.parent / "pipeline.json").read_text())
        assert marker["status"] in ("running", "complete")
        return SimpleNamespace(manifest=json.loads(manifest_path.read_text()))

    def validate(source, root):
        calls.append(("reuse_validate", str(root)))
        marker = json.loads((root / "pipeline.json").read_text())
        assert marker["status"] == "complete" and marker["dataset"] == str(source)
        assert marker["config_sha256"] == MODULE.sha256(root / "train_config.json")
        assert marker["teacher_manifest_sha256"] == MODULE.sha256(root / "teacher/manifest.json")
        return root / "train_config.json"

    monkeypatch.setattr(MODULE, "run_selection", selection)
    monkeypatch.setattr(MODULE.subprocess, "run", preview)
    monkeypatch.setattr(MODULE, "prepare_teacher", teacher)
    monkeypatch.setattr(MODULE, "VirtualGeometryDataset", verify)
    monkeypatch.setattr(MODULE, "validate_dataset_pipeline", validate)
    return dataset, calls


def test_default_pipeline_preserves_original_algorithm_parameters_and_publishes_last(stages):
    dataset, calls = stages
    config = MODULE.prepare_pipeline(dataset)
    root = dataset / "virtual_geometry"
    assert config == root / "train_config.json"
    assert [entry[0] for entry in calls] == ["selection", "preview", "teacher", "verify"]
    assert calls[0][1] == {"target_count": 160, "candidate_limit": 1200, "long_edge": 256,
                           "surface_samples": 120000, "threads": 8, "seed": 0, "preview_long_edge": 512}
    assert calls[2][1] == {"dataset": dataset, "long_edge": 512, "threads": 8}
    assert json.loads(config.read_text()) == {"schema_version": 1, "teacher_manifest": "teacher/manifest.json"}
    marker = json.loads((root / "pipeline.json").read_text())
    assert marker["status"] == marker["stage"] == "complete"
    assert marker["config_sha256"] == MODULE.sha256(config)
    assert marker["teacher_manifest_sha256"] == MODULE.sha256(root / "teacher/manifest.json")


def test_complete_output_is_strongly_verified_and_reused_without_mutation(stages):
    dataset, calls = stages
    config = MODULE.prepare_pipeline(dataset)
    root = config.parent
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls.clear()
    assert MODULE.prepare_pipeline(dataset) == config
    assert [entry[0] for entry in calls] == ["reuse_validate", "verify"]
    after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("failed_stage", ["selection", "preview", "teacher", "verify"])
def test_failure_records_stage_and_never_publishes_complete(stages, monkeypatch, failed_stage):
    dataset, _ = stages

    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")

    if failed_stage == "preview":
        monkeypatch.setattr(MODULE.subprocess, "run", fail)
    else:
        monkeypatch.setattr(MODULE, {"selection": "run_selection", "teacher": "prepare_teacher", "verify": "VirtualGeometryDataset"}[failed_stage], fail)
    with pytest.raises(RuntimeError, match="controlled failure"):
        MODULE.prepare_pipeline(dataset)
    marker_path = dataset / "virtual_geometry/pipeline.json"
    marker = json.loads(marker_path.read_text())
    assert marker["status"] == "failed"
    assert marker["stage"] == {"selection": "camera_selection", "preview": "camera_preview",
                               "teacher": "teacher_preparation", "verify": "teacher_validation"}[failed_stage]
    assert "controlled failure" in marker["error"]
    assert not (marker_path.parent / "train_config.json").exists()
    before = marker_path.read_bytes()
    with pytest.raises(FileExistsError, match="incomplete"):
        MODULE.prepare_pipeline(dataset)
    assert marker_path.read_bytes() == before


def test_unowned_nonempty_output_is_preserved(stages):
    dataset, calls = stages
    root = dataset / "virtual_geometry"
    root.mkdir()
    (root / "user.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="incomplete"):
        MODULE.prepare_pipeline(dataset)
    assert not calls and (root / "user.txt").read_text() == "keep me"
    assert not (root / "pipeline.json").exists()


def test_successful_preview_process_without_artifacts_is_not_complete(stages, monkeypatch):
    dataset, _ = stages
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Incomplete virtual-camera preview"):
        MODULE.prepare_pipeline(dataset)
    marker = json.loads((dataset / "virtual_geometry/pipeline.json").read_text())
    assert marker["status"] == "failed" and marker["stage"] == "camera_preview"


def test_complete_output_rejects_mesh_mismatch(stages):
    dataset, calls = stages
    MODULE.prepare_pipeline(dataset)
    other = dataset / "other.ply"
    other.write_text("another mesh")
    calls.clear()
    with pytest.raises(ValueError, match="another dataset or mesh"):
        MODULE.prepare_pipeline(dataset, mesh=other)
    assert not calls


def test_teacher_binding_mismatch_prevents_publication(stages, monkeypatch):
    dataset, _ = stages
    monkeypatch.setattr(MODULE, "VirtualGeometryDataset", lambda *args, **kwargs: SimpleNamespace(
        manifest={"dataset": str(dataset / "other"), "genrecon_mesh": str(dataset / "genrecon_mesh.ply")}))
    with pytest.raises(ValueError, match="another dataset"):
        MODULE.prepare_pipeline(dataset)
    marker = json.loads((dataset / "virtual_geometry/pipeline.json").read_text())
    assert marker["status"] == "failed"


def test_preview_subprocess_failure_propagates_as_failed_stage(stages, monkeypatch):
    dataset, _ = stages

    def failed_process(command, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(MODULE.subprocess, "run", failed_process)
    with pytest.raises(subprocess.CalledProcessError):
        MODULE.prepare_pipeline(dataset)
    marker = json.loads((dataset / "virtual_geometry/pipeline.json").read_text())
    assert marker["status"] == "failed" and "CalledProcessError" in marker["error"]


def test_published_pipeline_is_accepted_by_shared_training_validator(stages, monkeypatch):
    from utils.dataset_virtual_geometry import validate_dataset_pipeline
    dataset, _ = stages
    config = MODULE.prepare_pipeline(dataset)
    assert validate_dataset_pipeline(dataset) == str(config)
    monkeypatch.setattr(MODULE, "validate_dataset_pipeline", validate_dataset_pipeline)
    assert MODULE.prepare_pipeline(dataset) == config


def test_optional_output_seed_threads_are_passed_without_new_algorithm_settings(stages):
    dataset, calls = stages
    output = dataset.parent / "custom_output"
    config = MODULE.prepare_pipeline(dataset, output=output, threads=3, seed=17)
    assert config == output / "train_config.json"
    assert not (dataset / "virtual_geometry").exists()
    assert calls[0][1]["seed"] == 17 and calls[0][1]["threads"] == 3
    assert calls[2][1]["threads"] == 3


def test_reuse_rejects_different_seed_without_mutating_existing_output(stages):
    dataset, calls = stages
    config = MODULE.prepare_pipeline(dataset, seed=7)
    marker = config.parent / "pipeline.json"
    before = marker.read_bytes()
    calls.clear()
    with pytest.raises(ValueError, match="different seed"):
        MODULE.prepare_pipeline(dataset, seed=8)
    assert not calls and marker.read_bytes() == before


def test_reuse_allows_different_thread_count_without_rerunning_algorithms(stages):
    dataset, calls = stages
    config = MODULE.prepare_pipeline(dataset, threads=8)
    calls.clear()
    assert MODULE.prepare_pipeline(dataset, threads=2) == config
    assert [call[0] for call in calls] == ["reuse_validate", "verify"]


@pytest.mark.parametrize("source,absent_sheet", [("surface_lookat", "contact_sheet_trajectory.jpg"),
                                                ("trajectory_offset", "contact_sheet_surface.jpg")])
def test_empty_camera_family_does_not_require_a_family_contact_sheet(stages, monkeypatch, source, absent_sheet):
    dataset, _ = stages
    original_selection, original_preview = MODULE.run_selection, MODULE.subprocess.run

    def one_family(*args, **kwargs):
        original_selection(*args, **kwargs)
        output = args[2]
        document = json.loads((output / "virtual_cameras.json").read_text())
        document["cameras"] = document["cameras"][:1]
        document["cameras"][0]["selection_source"] = source
        (output / "virtual_cameras.json").write_text(json.dumps(document))

    def family_preview(command, check):
        original_preview(command, check)
        (Path(command[-1]).parent / absent_sheet).unlink()

    monkeypatch.setattr(MODULE, "run_selection", one_family)
    monkeypatch.setattr(MODULE.subprocess, "run", family_preview)
    config = MODULE.prepare_pipeline(dataset)
    assert json.loads((config.parent / "pipeline.json").read_text())["status"] == "complete"
    assert MODULE.prepare_pipeline(dataset) == config


def test_nonempty_camera_family_still_requires_its_contact_sheet(stages, monkeypatch):
    dataset, _ = stages
    original_preview = MODULE.subprocess.run

    def missing_sheet(command, check):
        original_preview(command, check)
        (Path(command[-1]).parent / "contact_sheet_surface.jpg").unlink()

    monkeypatch.setattr(MODULE.subprocess, "run", missing_sheet)
    with pytest.raises(ValueError, match="contact_sheet_surface"):
        MODULE.prepare_pipeline(dataset)
    marker = json.loads((dataset / "virtual_geometry/pipeline.json").read_text())
    assert marker["status"] == "failed" and marker["stage"] == "camera_preview"


@pytest.mark.parametrize("camera_file", ["real_cameras.json", "virtual_cameras.json"])
@pytest.mark.parametrize("kind", ["shading", "normal", "depth"])
@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_complete_reuse_rejects_missing_or_empty_referenced_camera_png(stages, camera_file, kind, damage):
    dataset, _ = stages
    config = MODULE.prepare_pipeline(dataset)
    root = config.parent
    marker_before = (root / "pipeline.json").read_bytes()
    camera = json.loads((root / "cameras" / camera_file).read_text())["cameras"][0]
    image = root / "cameras" / camera["preview"][kind]
    if damage == "missing":
        image.unlink()
    else:
        image.write_bytes(b"")
    with pytest.raises(ValueError, match="Incomplete camera preview"):
        MODULE.prepare_pipeline(dataset)
    assert (root / "pipeline.json").read_bytes() == marker_before
