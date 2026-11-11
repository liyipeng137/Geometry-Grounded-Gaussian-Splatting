"""Selecting or resuming an initial cloud must never silently replace its source."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("initialization_options", ROOT / "utils/initialization.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def options(mesh="", ply=""):
    return SimpleNamespace(init_mesh=str(mesh), init_ply=str(ply))


@pytest.fixture
def saved(tmp_path):
    run = tmp_path / "saved"
    run.mkdir()
    snapshot = run / "input.ply"
    snapshot.write_bytes(b"the exact initial XYZ/RGB cloud")
    mesh = tmp_path / "already_removed_mesh.ply"
    initial = {"requested_source": "mesh", "mesh_path": str(mesh),
               "ply_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               "ply_path": str(tmp_path / "removed_original_cloud.ply")}
    return run / "chkpnt32.pth", {"initialization": initial}, snapshot, mesh


def test_mesh_and_prepared_ply_cannot_both_be_selected():
    for args in (options(), options(mesh="source.ply"), options(ply="points.ply")):
        module.validate_initialization_options(args)
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.validate_initialization_options(options(mesh="source.ply", ply="points.ply"))


def test_resume_uses_saved_snapshot_without_source_mesh(saved):
    checkpoint, split, snapshot, mesh = saved
    assert not mesh.exists()
    for args in (options(), options(mesh=mesh)):
        assert module.resolve_checkpoint_initialization(args, checkpoint, split) == str(snapshot)


def test_resume_accepts_relocated_identical_cloud(saved, tmp_path):
    checkpoint, split, snapshot, _ = saved
    replacement = tmp_path / "copy.ply"
    replacement.write_bytes(snapshot.read_bytes())
    assert module.resolve_checkpoint_initialization(options(ply=replacement), checkpoint, split) == str(replacement)


def test_explicit_replacement_cannot_fall_back_to_valid_snapshot(saved, tmp_path):
    checkpoint, split, _, _ = saved
    replacement = tmp_path / "new_points.ply"
    replacement.write_bytes(b"different initialization")
    with pytest.raises(ValueError, match="content changed"):
        module.resolve_checkpoint_initialization(options(ply=replacement), checkpoint, split)
    with pytest.raises(FileNotFoundError, match="missing"):
        module.resolve_checkpoint_initialization(options(ply=tmp_path / "absent.ply"), checkpoint, split)


def test_changed_mesh_selection_is_rejected_without_resampling(saved, tmp_path):
    checkpoint, split, _, _ = saved
    with pytest.raises(ValueError, match="cannot change initialization mesh"):
        module.resolve_checkpoint_initialization(options(mesh=tmp_path / "different_mesh.ply"), checkpoint, split)


def test_old_initialization_can_be_reused_without_mesh_metadata(saved):
    checkpoint, split, snapshot, mesh = saved
    split["initialization"].pop("requested_source")
    assert module.resolve_checkpoint_initialization(options(), checkpoint, split) == str(snapshot)
    with pytest.raises(ValueError, match="cannot change initialization mesh"):
        module.resolve_checkpoint_initialization(options(mesh=mesh), checkpoint, split)


def test_missing_snapshot_uses_original_cloud_but_corruption_is_not_hidden(saved):
    checkpoint, split, snapshot, _ = saved
    original = Path(split["initialization"]["ply_path"])
    original.write_bytes(snapshot.read_bytes())
    snapshot.write_bytes(b"corrupted saved input")
    with pytest.raises(ValueError, match="content changed"):
        module.resolve_checkpoint_initialization(options(), checkpoint, split)
    snapshot.unlink()
    assert module.resolve_checkpoint_initialization(options(), checkpoint, split) == str(original)
    original.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        module.resolve_checkpoint_initialization(options(), checkpoint, split)
