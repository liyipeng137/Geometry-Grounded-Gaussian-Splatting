"""Checkpoint continuation must not silently mix recipes or source geometry.

These CPU tests execute the production validation functions without importing
the CUDA training stack. Records represent two output directories for one real
dataset; the current Scene initialization is only a placeholder before restore.
"""

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def execute_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)


@pytest.fixture
def checkpoint_writer(tmp_path):
    output = tmp_path / "new-run"
    output.mkdir()
    (output / "camera_split.json").write_text("{}")
    namespace = {
        "Path": Path, "json": json, "hashlib": hashlib,
        "RECIPE": "test", "RESUME_POLICY": "test", "CHECKPOINT_STATE_KEY": "state",
        "require_baseline_checkpoint": lambda state, iteration: state,
        "torch": SimpleNamespace(save=lambda payload, path: Path(path).write_bytes(b"new checkpoint")),
    }
    execute_functions("utils/training_io.py", {"write_json", "save_checkpoint"}, namespace)
    return SimpleNamespace(output=output, namespace=namespace,
                           scene=SimpleNamespace(model_path=str(output)),
                           model=SimpleNamespace(capture=lambda: tuple([None] * 14 + [{}])))


@pytest.mark.parametrize("retain_latest", [False, True])
def test_checkpoint_retention_keeps_current_pair_and_never_deletes_resume_source(checkpoint_writer, tmp_path, retain_latest):
    writer = checkpoint_writer
    source = tmp_path / "resume-source"
    source.mkdir()
    original, original_receipt = source / "chkpnt40.pth", source / "chkpnt40.state.json"
    original.write_bytes(b"resume original")
    original_receipt.write_text("original receipt")
    old, old_receipt = writer.output / "chkpnt40.pth", writer.output / "chkpnt40.state.json"
    old.write_bytes(b"old checkpoint")
    old_receipt.write_text("old receipt")
    alias, alias_receipt = writer.output / "chkpnt7.pth", writer.output / "chkpnt7.state.json"
    alias.symlink_to(original)
    alias_receipt.symlink_to(original_receipt)
    unrelated = writer.output / "appearance.pth"
    unrelated.write_bytes(b"keep appearance")
    args = SimpleNamespace(training_seed=0)
    if retain_latest:
        args.keep_last_checkpoint = True
    writer.namespace["save_checkpoint"](writer.model, writer.scene, 80, args)
    current = writer.output / "chkpnt80.pth"
    receipt = json.loads((writer.output / "chkpnt80.state.json").read_text())
    assert current.read_bytes() == b"new checkpoint"
    assert receipt["checkpoint_sha256"] == hashlib.sha256(current.read_bytes()).hexdigest()
    assert old.exists() is (not retain_latest)
    assert old_receipt.exists() is (not retain_latest)
    assert alias.is_symlink() is (not retain_latest)
    assert alias_receipt.is_symlink() is (not retain_latest)
    assert original.read_bytes() == b"resume original"
    assert original_receipt.read_text() == "original receipt"
    assert unrelated.read_bytes() == b"keep appearance"


def test_checkpoint_receipt_failure_does_not_prune_previous_pair(checkpoint_writer):
    writer = checkpoint_writer
    old, old_receipt = writer.output / "chkpnt40.pth", writer.output / "chkpnt40.state.json"
    old.write_bytes(b"old checkpoint")
    old_receipt.write_text("old receipt")

    def fail_receipt(*args):
        raise OSError("receipt unavailable")

    writer.namespace["write_json"] = fail_receipt
    with pytest.raises(OSError, match="receipt unavailable"):
        writer.namespace["save_checkpoint"](writer.model, writer.scene, 80,
                                            SimpleNamespace(training_seed=0, keep_last_checkpoint=True))
    assert (writer.output / "chkpnt80.pth").is_file()
    assert not (writer.output / "chkpnt80.state.json").exists()
    assert old.read_bytes() == b"old checkpoint"
    assert old_receipt.read_text() == "old receipt"


@pytest.fixture
def contract():
    namespace = {
        "Path": Path, "hashlib": hashlib, "json": json, "os": os, "tempfile": tempfile,
        "RECIPE": "gggs_real_baseline",
        "RGBD_INITIALIZATION_ALGORITHM": "rgbd_colmap_half_integer_float32_v3",
    }
    execute_functions(
        "scene/train_split.py",
        {
            "calibration_fingerprint", "_is_sha256", "_validate_calibration_proof",
            "validate_initialization_proof", "validate_checkpoint_split",
        },
        namespace,
    )
    execute_functions("utils/training_io.py", {"require_baseline_checkpoint"}, namespace)
    return namespace


@pytest.fixture
def records(tmp_path, contract):
    fingerprint = contract["calibration_fingerprint"]
    calibration = {
        "width": 1440, "height": 1920,
        "K": [[1000.0, 0.0, 719.5], [0.0, 1000.0, 959.5], [0.0, 0.0, 1.0]],
        "world_to_camera_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "world_to_camera_translation": [0.0, 0.0, 0.0],
        "pixel_center_convention": "colmap_half_integer",
    }
    camera_proof = {"nested/frame.png": fingerprint(calibration)}
    previous = {
        "schema_version": 3, "dataset_root": str(tmp_path / "dataset"),
        "train_image_names": ["nested/frame.png"], "test_image_names": [],
        "camera_calibration_fingerprints": camera_proof,
        "calibration_fingerprint": fingerprint(camera_proof),
        "dataset_content_fingerprint": fingerprint({"rgb": "same photo", "depth": "same depth"}),
        "train_camera_fingerprint": fingerprint(["nested/frame.png"]),
        "pixel_center_convention": "colmap_half_integer",
        "initialization": {
            "kind": "train_only_rgbd", "algorithm": "rgbd_colmap_half_integer_float32_v3",
            "ply_path": str(tmp_path / "first_run/rgbd_init.ply"),
            "proof_path": str(tmp_path / "first_run/rgbd_init.ply.json"),
            "ply_sha256": fingerprint("same finite point cloud"),
            "cache_fingerprint": fingerprint("same RGBD inputs and operator"), "point_count": 200000,
        },
    }
    output = tmp_path / "continued_run"
    output.mkdir()
    current = deepcopy(previous)
    current["initialization"]["ply_path"] = str(output / "rgbd_init.ply")
    current["initialization"]["proof_path"] = str(output / "rgbd_init.ply.json")
    return previous, current, calibration, output


def test_recipe_and_recorded_iteration_must_match_baseline(contract, records):
    previous, _, _, _ = records
    valid = {
        "schema_version": 1, "recipe": "gggs_real_baseline", "iteration": 1000,
        "camera_split": previous,
    }
    require = contract["require_baseline_checkpoint"]
    require(valid, 1000)
    with pytest.raises(ValueError, match="real-baseline checkpoint"):
        require({**valid, "recipe": "r2c"}, 1000)
    with pytest.raises(ValueError, match="real-baseline checkpoint"):
        require(valid, 1001)


def test_changed_camera_intrinsics_or_pose_cannot_resume(contract, records):
    previous, current, calibration, output = records
    fingerprint = contract["calibration_fingerprint"]
    for field in ("K", "world_to_camera_translation"):
        changed_calibration = deepcopy(calibration)
        if field == "K":
            changed_calibration["K"][0][0] += 10.0
        else:
            changed_calibration["world_to_camera_translation"][0] += 0.1
        changed = deepcopy(current)
        # The new proof is internally valid, but represents different geometry.
        cameras = {"nested/frame.png": fingerprint(changed_calibration)}
        changed["camera_calibration_fingerprints"] = cameras
        changed["calibration_fingerprint"] = fingerprint(cameras)
        (output / "camera_split.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="dataset/calibration/split changed"):
            contract["validate_checkpoint_split"]("first_run/chkpnt1000.pth", output, previous)


def test_changed_rgbd_cache_identity_cannot_resume(contract, records):
    previous, current, _, output = records
    current["initialization"]["cache_fingerprint"] = contract["calibration_fingerprint"](
        "different depth directory or fusion operator"
    )
    (output / "camera_split.json").write_text(json.dumps(current))
    with pytest.raises(ValueError, match="initialization operator or training inputs changed"):
        contract["validate_checkpoint_split"]("first_run/chkpnt1000.pth", output, previous)


def test_same_dataset_restore_preserves_original_initialization(contract, records, tmp_path):
    previous, current, _, output = records
    placeholder = deepcopy(current["initialization"])
    (output / "camera_split.json").write_text(json.dumps(current))
    checkpoint = tmp_path / "first_run/chkpnt1000.pth"
    result = contract["validate_checkpoint_split"](checkpoint, output, previous)
    saved = json.loads((output / "camera_split.json").read_text())
    assert result["initialization_preserved"] is True
    assert result["expanded"] is False
    assert saved["initialization"] == previous["initialization"]
    assert saved["scene_placeholder_initialization"] == placeholder
    assert saved["checkpoint_restoration"]["scene_placeholder_used_for_model_state"] is False
    assert saved["checkpoint_restoration"]["checkpoint_path"] == str(checkpoint)
    assert saved["train_image_names"] == previous["train_image_names"]
