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

import pytest


ROOT = Path(__file__).resolve().parents[1]


def execute_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)


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
