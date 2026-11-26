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
import struct
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
        {"_sha256_file", "_write_record", "_prepare_real_geometry_input", "readColmapSceneInfo"},
        namespace,
    )
    output = tmp_path / "output"

    def read(depth_dir="depth", *, output_dir=output, **options):
        result = namespace["readColmapSceneInfo"](
            str(dataset),
            "images",
            False,
            init_cache_dir=str(output_dir),
            depth_prior_dir=depth_dir,
            **options,
        )
        return result, json.loads((output_dir / "camera_split.json").read_text())

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
    assert calls[0]["num_points"] == 250000
    assert calls[0]["pixel_center_convention"] == "colmap_half_integer"
    metadata = json.loads(Path(result.ply_path).with_suffix(".ply.json").read_text())
    assert metadata["identity"]["parameters"] == {
        "num_points": 250000,
        "voxel_length": 0.02,
        "sdf_trunc": 0.06,
        "depth_min": 0.05,
        "depth_max": 50.0,
        "largest_mesh_components": 1,
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


def write_sparse_points(path, points):
    if path.suffix == ".bin":
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(points)))
            for point_id, record in enumerate(points, 1):
                stream.write(struct.pack("<QdddBBBdQ", point_id, *record, 0.0, 0))
    else:
        path.write_text(
            "".join(
                f"{point_id} {' '.join(map(str, record))} 0\n"
                for point_id, record in enumerate(points, 1)
            )
        )


@pytest.mark.parametrize("extension", ["txt", "bin"])
def test_real_geometry_writes_run_owned_initialization_without_hashes(
    reader_fixture, extension
):
    from plyfile import PlyData, PlyElement

    dataset, _, calls, ns, read = reader_fixture
    write_depth(dataset / "depth", 1.0)
    (camera,) = ns["readColmapCameras"]()
    # C = -R @ T in this reader. The training center is (2, 3, 0).
    camera.R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    camera.T = np.array([-3., 2., 0.])
    heldout = SimpleNamespace(**vars(camera))
    heldout.colmap_image_name = heldout.image_name = "other.png"
    heldout.image_path = str(dataset / "images/other.png")
    Path(heldout.image_path).write_bytes(b"other")
    heldout.R, heldout.T = np.eye(3), np.array([-50., 0., 0.])
    excluded = dataset / "exclude.json"
    excluded.write_text(json.dumps([heldout.colmap_image_name]))
    ns["readColmapCameras"] = lambda *args: [camera, heldout]

    xyz = ("x", "y", "z")
    rgb = ("red", "green", "blue")
    dtype = [(field, "f8") for field in xyz] + [(field, "u1") for field in rgb]
    base = np.array([(2, 3, 1, 1, 2, 3), (3, 3, 0, 4, 5, 6)], dtype=dtype)

    def fusion(cameras, ply_path, **options):
        assert cameras == [camera]
        calls.append(ply_path)
        PlyData([PlyElement.describe(base.copy(), "vertex")]).write(ply_path)

    def load_ply(path):
        vertices = PlyData.read(path)["vertex"].data
        return SimpleNamespace(
            points=np.column_stack([vertices[k] for k in xyz]),
            colors=np.column_stack([vertices[k] for k in rgb]) / 255,
        )

    ns.update(generate_ply_from_rgbd=fusion, fetchOpen3DPly=load_ply)
    model = dataset / "sparse/0"
    model.mkdir(parents=True)
    points_file = model / f"points3D.{extension}"
    (model / f"cameras.{extension}").touch()
    (model / f"images.{extension}").touch()
    points = [
        (2, 3, 1, 80, 81, 82),  # Keep the duplicate TSDF position and its own RGB.
        (2.25, 3, 0, 90, 91, 92),
        (1, 3, 0, 10, 11, 12),  # Inclusive boundary, one unit from the training camera.
        (2, 4.5, 0, 20, 21, 22),
        (50, 0, 0, 30, 31, 32),  # A held-out center cannot expand the cutoff.
    ]
    def forbidden_hash(*args, **kwargs):
        pytest.fail("Fresh geometry initialization should not hash inputs")

    ns.update(_sha256_file=forbidden_hash, calibration_fingerprint=forbidden_hash)
    options = dict(train_exclude_images=str(excluded), geometry_only=True)
    write_sparse_points(points_file, points)
    normals = dataset / "depth_normal_prior"
    (normals / "nested").mkdir(parents=True)
    (normals / "nested/frame.png").touch()
    with pytest.raises(FileNotFoundError, match="Missing aligned depth for other.png"):
        read(**options, merge_sparse_init=True)
    assert not calls
    (normals / "other.png").touch()
    merged, split = read(**options, merge_sparse_init=True)
    expected = np.concatenate((base, np.array(points[:3], dtype=dtype)))
    assert Path(merged.ply_path).name == "input.ply"
    assert (split["normal_prior"]["generated"], split["normal_prior"]["reused"]) == (0, 2)
    assert "ply_sha256" not in split["initialization"]
    init = split["initialization"]
    assert init["kind"] == "tsdf_sparse" and init["point_count"] == 5
    proof = init["sparse_merge"]
    assert (proof["tsdf_points"], proof["sparse_points"]) == (2, 5)
    assert (proof["sparse_kept"], proof["sparse_clipped"]) == (3, 2)
    assert proof["max_nearest_camera_distance"] == pytest.approx(1.)
    assert split["train_image_names"] == [camera.colmap_image_name]
    assert split["test_image_names"] == [heldout.colmap_image_name]
    np.testing.assert_array_equal(PlyData.read(merged.ply_path)["vertex"].data, expected)
    np.testing.assert_array_equal(
        merged.point_cloud.colors * 255, np.column_stack([expected[k] for k in rgb])
    )
    np.testing.assert_array_equal(PlyData.read(proof["tsdf_ply"])["vertex"].data, base)

    write_sparse_points(points_file, [])
    empty, _ = read(**options, merge_sparse_init=True, output_dir=dataset.parent / "empty")
    np.testing.assert_array_equal(PlyData.read(empty.ply_path)["vertex"].data, base)
    points_file.unlink()
    with pytest.raises(FileNotFoundError, match="points3D"):
        read(**options, merge_sparse_init=True, output_dir=dataset.parent / "missing")
    assert len(calls) == 2
    disabled, record = read(**options, merge_sparse_init=False, output_dir=dataset.parent / "tsdf_only")
    assert record["initialization"]["kind"] == "train_only_rgbd"
    np.testing.assert_array_equal(PlyData.read(disabled.ply_path)["vertex"].data, base)
    assert len(calls) == 3
