"""External XYZ/RGB initialization preserves real-data and resume contracts.

Run the production reader/proof functions with a point-cloud decoding boundary
stub; invoking RGBD fusion is an error. These CPU tests need no Open3D/CUDA.
"""

import ast
from collections import namedtuple
from copy import deepcopy
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
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def external_reader(tmp_path):
    dataset = tmp_path / "dataset"
    cameras = []
    for index, name in enumerate(("nested/a.png", "nested/b.png")):
        image = dataset / "images" / name
        depth = dataset / "depth" / Path(name).with_suffix(".npy")
        image.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"RGB {index}".encode())
        np.save(depth, np.full((2, 3), 1 + index, dtype=np.float32))
        cameras.append(SimpleNamespace(
            colmap_image_name=name, source_image_name=name, image_name=Path(name).stem,
            image_path=str(image), width=3, height=2, K=np.eye(3), R=np.eye(3),
            T=np.array([index, 0., 0.]),
        ))
    source = tmp_path / "generated_colored.ply"
    source.write_bytes(b"frozen colored GenRecon point cloud")
    cloud = SimpleNamespace(points=np.array([[0., 0., 1.], [1., 0., 2.]]),
                            colors=np.array([[.2, .3, .4], [.5, .6, .7]]),
                            normals=np.zeros((2, 3)))
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "os": os,
                 "tempfile": tempfile, "np": np, "fcntl": fcntl,
                 "RGBD_INITIALIZATION_ALGORITHM": "rgbd_colmap_half_integer_float32_v3",
                 "RECIPE": "gggs_real_baseline", "JOINT_RECIPE": "gggs_virtual_geometry_joint_v1"}
    load_functions("scene/train_split.py", {
        "calibration_fingerprint", "training_camera_fingerprint", "load_excluded_image_names",
        "partition_cameras", "_is_sha256", "_validate_calibration_proof",
        "validate_initialization_proof", "validate_checkpoint_split",
    }, namespace)
    load_functions("utils/pd_utils.py", {"resolve_rgbd_depth_directory"}, namespace)

    def forbidden_fusion(*args, **kwargs):
        raise AssertionError("Explicit init_ply must never construct a TSDF")

    namespace.update(
        read_extrinsics_text=lambda path: {}, read_intrinsics_text=lambda path: {},
        read_extrinsics_binary=lambda path: {}, read_intrinsics_binary=lambda path: {},
        readColmapCameras=lambda *args: cameras.copy(), generate_ply_from_rgbd=forbidden_fusion,
        fetchOpen3DPly=lambda path: cloud,
        getNerfppNorm=lambda selected: {"radius": 2.5, "real_camera_count": len(selected)},
        SceneInfo=namedtuple("SceneInfo", "point_cloud train_cameras test_cameras nerf_normalization ply_path"),
    )
    load_functions("scene/dataset_readers.py", {"_sha256_file", "_write_record", "readColmapSceneInfo"}, namespace)
    load_functions("utils/training_io.py", {"require_baseline_checkpoint", "require_joint_checkpoint"}, namespace)

    def read(output_name="first_run", init_ply=None, excluded=""):
        output = tmp_path / output_name
        result = namespace["readColmapSceneInfo"](
            str(dataset), "images", False, init_cache_dir=str(output),
            init_ply=str(source if init_ply is None else init_ply), train_exclude_images=excluded,
        )
        return result, json.loads((output / "camera_split.json").read_text()), output

    return SimpleNamespace(namespace=namespace, dataset=dataset, source=source, cloud=cloud,
                           cameras=cameras, read=read, tmp=tmp_path)


def test_external_init_skips_tsdf_and_preserves_real_cameras(external_reader):
    f = external_reader
    result, proof, _ = f.read()
    assert result.point_cloud is f.cloud
    assert result.train_cameras == f.cameras and result.test_cameras == []
    assert result.nerf_normalization == {"radius": 2.5, "real_camera_count": 2}
    assert result.ply_path == str(f.source.resolve())
    assert proof["train_image_names"] == [c.colmap_image_name for c in f.cameras]
    assert proof["initialization"]["kind"] == "external_ply"
    assert proof["initialization"]["algorithm"] == "external_colored_ply_v1"
    assert proof["initialization"]["point_count"] == 2
    assert proof["initialization"]["ply_sha256"] == hashlib.sha256(f.source.read_bytes()).hexdigest()
    assert proof["dataset_content_fingerprint"] and proof["calibration_fingerprint"]
    f.namespace["validate_initialization_proof"](proof)


@pytest.mark.parametrize("invalid", ["empty", "xyz_rank", "xyz_width", "xyz_nan", "xyz_inf",
                                      "missing_rgb", "rgb_rows", "rgb_nan", "rgb_inf"])
def test_external_xyz_rgb_invalid_input_is_rejected(external_reader, invalid):
    f = external_reader
    if invalid == "empty":
        f.cloud.points = f.cloud.colors = np.empty((0, 3))
    elif invalid == "xyz_rank":
        f.cloud.points = np.zeros(3)
    elif invalid == "xyz_width":
        f.cloud.points = np.zeros((2, 4))
    elif invalid == "xyz_nan":
        f.cloud.points[0, 0] = np.nan
    elif invalid == "xyz_inf":
        f.cloud.points[0, 0] = np.inf
    elif invalid == "missing_rgb":
        f.cloud.colors = np.empty((0, 3))
    elif invalid == "rgb_rows":
        f.cloud.colors = np.zeros((1, 3))
    elif invalid == "rgb_nan":
        f.cloud.colors[0, 0] = np.nan
    elif invalid == "rgb_inf":
        f.cloud.colors[0, 0] = np.inf
    with pytest.raises(ValueError, match="External initialization"):
        f.read()


def test_external_source_missing_is_rejected(external_reader):
    with pytest.raises(FileNotFoundError):
        external_reader.read(init_ply=external_reader.tmp / "missing.ply")


def test_external_keeps_explicit_real_split(external_reader):
    f = external_reader
    exclude = f.tmp / "exclude.json"
    exclude.write_text(json.dumps([f.cameras[1].colmap_image_name]))
    result, proof, _ = f.read(excluded=str(exclude))
    assert result.train_cameras == f.cameras[:1]
    assert result.test_cameras == f.cameras[1:]
    assert proof["initialization"]["kind"] == "external_ply"
    assert proof["test_image_names"] == [f.cameras[1].colmap_image_name]


def test_external_does_not_bypass_real_asset_recheck(external_reader):
    f = external_reader

    def changed_rgb(path):
        Path(f.cameras[0].image_path).write_bytes(b"modified during external point load")
        return f.cloud

    f.namespace["fetchOpen3DPly"] = changed_rgb
    with pytest.raises(RuntimeError, match="source changed during initialization"):
        f.read()


def test_external_source_change_during_read_is_rejected(external_reader):
    f = external_reader

    def changed_ply(path):
        Path(path).write_bytes(b"point-cloud bytes changed while decoder was reading")
        return f.cloud

    f.namespace["fetchOpen3DPly"] = changed_ply
    with pytest.raises((RuntimeError, ValueError), match="changed"):
        f.read()


@pytest.mark.parametrize("relocate_source", [False, True])
def test_same_external_bytes_resume_in_new_output(external_reader, relocate_source):
    f = external_reader
    _, previous, _ = f.read()
    source = f.source
    if relocate_source:
        source = f.tmp / "identical_copy.ply"
        source.write_bytes(f.source.read_bytes())
    _, current, output = f.read("resume_run", init_ply=source)
    placeholder = deepcopy(current["initialization"])
    result = f.namespace["validate_checkpoint_split"](f.tmp / "first_run/chkpnt1000.pth", output, previous)
    saved = json.loads((output / "camera_split.json").read_text())
    assert result["initialization_kind"] == "external_ply"
    assert saved["initialization"] == previous["initialization"]
    assert saved["scene_placeholder_initialization"] == placeholder
    assert saved["checkpoint_restoration"]["scene_placeholder_used_for_model_state"] is False


def test_changed_external_ply_cannot_resume(external_reader):
    f = external_reader
    _, previous, _ = f.read()
    f.source.write_bytes(b"different positions or colors")
    _, _, output = f.read("changed_run")
    with pytest.raises(ValueError, match="initialization operator or training inputs changed"):
        f.namespace["validate_checkpoint_split"](f.tmp / "first_run/chkpnt1000.pth", output, previous)


def test_external_and_rgbd_initializations_cannot_be_swapped_on_resume(external_reader):
    f = external_reader
    _, previous, output = f.read()
    changed = deepcopy(previous)
    changed["initialization"].update(kind="train_only_rgbd", algorithm="rgbd_colmap_half_integer_float32_v3")
    (output / "camera_split.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="initialization operator or training inputs changed"):
        f.namespace["validate_checkpoint_split"](f.tmp / "checkpoint.pth", output, previous)


def test_legacy_rgbd_resume_still_uses_cache_identity_not_placeholder_bytes(external_reader):
    f = external_reader
    _, previous, output = f.read()
    previous["initialization"].update(kind="train_only_rgbd", algorithm="rgbd_colmap_half_integer_float32_v3")
    current = deepcopy(previous)
    current["initialization"].update(ply_sha256="e" * 64, point_count=3)
    (output / "camera_split.json").write_text(json.dumps(current))
    result = f.namespace["validate_checkpoint_split"](f.tmp / "checkpoint.pth", output, previous)
    assert result["initialization_kind"] == "train_only_rgbd"


@pytest.mark.parametrize("kind,algorithm", [("invented", "external_colored_ply_v1"),
                                            ("external_ply", "rgbd_colmap_half_integer_float32_v3")])
def test_invalid_initialization_kind_or_algorithm_rejected(external_reader, kind, algorithm):
    f = external_reader
    _, record, _ = f.read()
    record["initialization"].update(kind=kind, algorithm=algorithm)
    with pytest.raises(ValueError, match="Checkpoint initialization"):
        f.namespace["validate_initialization_proof"](record)


def test_external_proof_is_accepted_by_baseline_and_joint_checkpoint_contract(external_reader):
    f = external_reader
    _, record, _ = f.read()
    base = {"schema_version": 1, "recipe": "gggs_real_baseline", "iteration": 20, "camera_split": record}
    f.namespace["require_baseline_checkpoint"](base, 20)
    joint = {**base, "schema_version": 2, "recipe": "gggs_virtual_geometry_joint_v1", "joint_identity": {"teacher": "unchanged"}}
    f.namespace["require_joint_checkpoint"](joint, 20)
