"""CPU checks for RGBD source selection without importing Open3D or CUDA.

The real dataset-reader functions run with a tiny fusion boundary stub. This
exercises the on-disk cache, content fingerprints, and source rechecks while
leaving actual TSDF/GPU validation to the remote smoke run.
"""

import ast
from collections import namedtuple
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def depth_helpers():
    return load_functions(
        "utils/pd_utils.py",
        {"resolve_rgbd_depth_directory", "_load_tsdf_depth"},
        {"Path": Path, "np": np},
    )


def write_depth(directory, value):
    path = directory / "nested/frame.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((2, 3), value, dtype=np.float32))
    return path


def test_depth_directory_default_relative_and_absolute(tmp_path, depth_helpers):
    resolve = depth_helpers["resolve_rgbd_depth_directory"]
    assert resolve(tmp_path) == tmp_path / "depth"
    custom = tmp_path / "global_depth/depth_npy"
    assert resolve(tmp_path, "global_depth/depth_npy") == custom
    assert resolve(tmp_path / "another_dataset", str(custom)) == custom
    with pytest.raises(ValueError, match="nonempty"):
        resolve(tmp_path, "")


def test_depth_loading_keeps_full_name_and_float_camera_z(tmp_path, depth_helpers):
    load = depth_helpers["_load_tsdf_depth"]
    write_depth(tmp_path / "depth", 2.0)
    custom = tmp_path / "global_depth/depth_npy"
    path = write_depth(custom, 4.0)
    assert np.all(load(tmp_path, "nested/frame.png") == 2.0)
    assert np.all(load(tmp_path, "nested/frame.png", "global_depth/depth_npy") == 4.0)
    assert np.all(load(tmp_path, "nested/frame.jpg", str(custom)) == 4.0)
    assert load(tmp_path, "missing/frame.png", str(custom)) is None
    np.save(path, np.ones((2, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="floating camera-Z"):
        load(tmp_path, "nested/frame.png", str(custom))


@pytest.fixture
def reader_fixture(tmp_path, depth_helpers):
    dataset = tmp_path / "dataset"
    rgb = dataset / "images/nested/frame.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"rgb fixture; no image decoding in this test")
    camera = SimpleNamespace(
        colmap_image_name="nested/frame.png",
        image_name="nested/frame.png",
        image_path=str(rgb),
        width=3,
        height=2,
        K=np.eye(3),
        R=np.eye(3),
        T=np.zeros(3),
    )
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "tempfile": tempfile,
    }
    load_functions(
        "scene/train_split.py",
        {
            "calibration_fingerprint",
            "load_excluded_image_names",
            "partition_cameras",
            "training_camera_fingerprint",
        },
        namespace,
    )
    fusion_calls = []

    def fusion_stub(cameras, **kwargs):
        fusion_calls.append(kwargs)
        # Exercise the actual depth-loader boundary with the forwarded source.
        assert np.all(
            depth_helpers["_load_tsdf_depth"](
                kwargs["source_path"],
                cameras[0].colmap_image_name,
                kwargs["depth_prior_dir"],
            )
            > 0
        )
        Path(kwargs["ply_path"]).write_bytes(
            b"finite colored point cloud boundary stub"
        )

    namespace.update(
        {
            "np": np,
            "fcntl": fcntl,
            "read_extrinsics_binary": lambda path: {},
            "read_intrinsics_binary": lambda path: {},
            "read_extrinsics_text": lambda path: {},
            "read_intrinsics_text": lambda path: {},
            "readColmapCameras": lambda *args: [camera],
            "generate_ply_from_rgbd": fusion_stub,
            "fetchOpen3DPly": lambda path: SimpleNamespace(
                points=np.ones((3, 3)), colors=np.zeros((3, 3))
            ),
            "getNerfppNorm": lambda cameras: {"radius": 1.0},
            "resolve_rgbd_depth_directory": depth_helpers[
                "resolve_rgbd_depth_directory"
            ],
            "SceneInfo": namedtuple(
                "SceneInfo",
                "point_cloud train_cameras test_cameras nerf_normalization ply_path",
            ),
        }
    )
    load_functions(
        "scene/dataset_readers.py",
        {"_sha256_file", "_write_record", "readColmapSceneInfo"},
        namespace,
    )
    output = tmp_path / "output"

    def read(depth_dir="depth"):
        result = namespace["readColmapSceneInfo"](
            str(dataset),
            "images",
            False,
            init_cache_dir=str(output),
            depth_prior_dir=depth_dir,
        )
        return result, json.loads((output / "camera_split.json").read_text())

    return dataset, rgb, fusion_calls, namespace, read


def test_reader_hashes_configured_depth_and_ignores_shadow_default(reader_fixture):
    dataset, rgb, calls, namespace, read = reader_fixture
    legacy = write_depth(dataset / "depth", 2.0)
    custom = write_depth(dataset / "global_depth/depth_npy", 4.0)
    result, record = read("global_depth/depth_npy")
    expected = {
        "nested/frame.png": {
            "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(custom.read_bytes()).hexdigest(),
        }
    }
    assert record["dataset_content_fingerprint"] == namespace[
        "calibration_fingerprint"
    ](expected)
    assert calls[0]["depth_prior_dir"] == str(custom.parent.parent)
    assert calls[0]["num_points"] == 200000
    assert calls[0]["pixel_center_convention"] == "colmap_half_integer"
    metadata = json.loads(Path(result.ply_path).with_suffix(".ply.json").read_text())
    assert metadata["identity"]["parameters"] == {
        "num_points": 200000,
        "voxel_length": 0.02,
        "sdf_trunc": 0.06,
        "depth_min": 0.05,
        "depth_max": 50.0,
        "largest_mesh_components": 3,
        "component_rank": "triangle_count",
        "point_sampling": "final_filtered_mesh_vertices",
    }

    legacy.write_bytes(b"changing the unused default must not invalidate this cache")
    same_result, same_record = read(str(custom.parent.parent))
    assert same_result.ply_path == result.ply_path
    assert (
        same_record["dataset_content_fingerprint"]
        == record["dataset_content_fingerprint"]
    )
    assert len(calls) == 1

    write_depth(custom.parent.parent, 5.0)
    updated_result, updated_record = read("global_depth/depth_npy")
    assert updated_result.ply_path != result.ply_path
    assert (
        updated_record["dataset_content_fingerprint"]
        != record["dataset_content_fingerprint"]
    )
    assert len(calls) == 2


def test_equal_contents_in_different_depth_roots_do_not_share_cache(reader_fixture):
    dataset, _, calls, _, read = reader_fixture
    write_depth(dataset / "depth", 4.0)
    write_depth(dataset / "global_depth/depth_npy", 4.0)
    default_result, default_record = read()
    custom_result, custom_record = read("global_depth/depth_npy")
    # File contents coincide, but the declared initialization source is distinct.
    assert (
        default_record["dataset_content_fingerprint"]
        == custom_record["dataset_content_fingerprint"]
    )
    assert default_result.ply_path != custom_result.ply_path
    assert len(calls) == 2
