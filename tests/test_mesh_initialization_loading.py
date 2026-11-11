"""Mesh initialization routing and read-only model-load behavior on CPU fakes."""

import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

from test_external_initialization import external_reader as _external_reader_fixture


ROOT = Path(__file__).resolve().parents[1]
external_reader = _external_reader_fixture


@pytest.fixture
def mesh_preparer(external_reader, monkeypatch):
    fixture = external_reader
    mesh = fixture.tmp / "source_mesh.ply"
    mesh.write_bytes(b"mesh in unchanged COLMAP world")
    calls = []
    module = ModuleType("utils.mesh_initialization")

    def prepare(dataset, mesh_path, output, cameras, seed=0, num_points=200000):
        calls.append((dataset, mesh_path, output, cameras.copy(), seed))
        output.mkdir(parents=True, exist_ok=True)
        path = output / "point_cloud.ply"
        path.write_bytes(b"cached global samples with real projected RGB")
        metadata = {"status": "complete", "genrecon_mesh": str(mesh_path),
                    "source_files_sha256": {str(mesh_path): hashlib.sha256(mesh_path.read_bytes()).hexdigest()},
                    "point_cloud_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (output / "metadata.json").write_text(json.dumps(metadata))
        return path

    module.prepare_from_cameras = prepare
    monkeypatch.setitem(sys.modules, "utils.mesh_initialization", module)
    return mesh, calls, module


def read_mesh(fixture, mesh, output_name="mesh_run", **options):
    output = fixture.tmp / output_name
    result = fixture.namespace["readColmapSceneInfo"](
        str(fixture.dataset), options.pop("images", "images"), False,
        init_cache_dir=str(output), init_mesh=str(mesh), **options,
    )
    proof_path = output / "camera_split.json"
    return result, json.loads(proof_path.read_text()) if proof_path.exists() else None, output


def test_mesh_preparation_uses_actual_training_cameras_image_paths_and_seed(external_reader, mesh_preparer):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    images = fixture.dataset / "custom_images"
    for camera in fixture.cameras:
        path = images / camera.colmap_image_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"custom training RGB")
        camera.image_path = str(path)
    excluded = fixture.tmp / "heldout.json"
    excluded.write_text(json.dumps([fixture.cameras[1].colmap_image_name]))
    result, proof, output = read_mesh(fixture, mesh, images="custom_images", init_seed=27,
                                     train_exclude_images=str(excluded))
    assert len(calls) == 1
    dataset, actual_mesh, prepared_output, cameras, seed = calls[0]
    assert dataset == str(fixture.dataset) and actual_mesh == mesh
    assert prepared_output == output / "initialization" and seed == 27
    assert cameras == fixture.cameras[:1]
    assert cameras[0].image_path.startswith(str(images))
    assert result.train_cameras == fixture.cameras[:1]
    assert result.test_cameras == fixture.cameras[1:]
    initialization = proof["initialization"]
    assert initialization["kind"] == "external_ply"
    assert initialization["algorithm"] == "external_colored_ply_v1"
    assert initialization["requested_source"] == "mesh"
    assert initialization["mesh_path"] == str(mesh)
    assert initialization["mesh_sha256"] == hashlib.sha256(mesh.read_bytes()).hexdigest()
    assert initialization["metadata_path"] == str(output / "initialization/metadata.json")
    assert initialization["ply_path"] == result.ply_path


def test_fresh_mesh_and_ply_are_rejected_before_preparation(external_reader, mesh_preparer):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    with pytest.raises(ValueError, match="mutually exclusive"):
        read_mesh(fixture, mesh, init_ply=str(fixture.source))
    assert calls == []


def test_failed_mesh_preparation_does_not_fall_back_to_tsdf(external_reader, mesh_preparer):
    fixture, (mesh, _, module) = external_reader, mesh_preparer

    def failure(*args, **kwargs):
        raise RuntimeError("preparation failed")

    module.prepare_from_cameras = failure
    with pytest.raises(RuntimeError, match="preparation failed"):
        read_mesh(fixture, mesh)


@pytest.mark.parametrize("field,value", [("status", "running"), ("genrecon_mesh", "/another/mesh.ply"),
                                         ("point_cloud_sha256", "0" * 64)])
def test_mesh_preparation_proof_must_match_returned_artifact(external_reader, mesh_preparer, field, value):
    fixture, (mesh, _, module) = external_reader, mesh_preparer
    original = module.prepare_from_cameras

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = result.parent / "metadata.json"
        record = json.loads(path.read_text())
        record[field] = value
        path.write_text(json.dumps(record))
        return result

    module.prepare_from_cameras = changed
    with pytest.raises(ValueError, match="provenance|generation metadata"):
        read_mesh(fixture, mesh)


def test_resume_reuses_verified_cache_without_source_mesh_or_old_ply(external_reader, mesh_preparer):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    first, proof, _ = read_mesh(fixture, mesh)
    original = deepcopy(proof["initialization"])
    cached = fixture.tmp / "recovered_input.ply"
    cached.write_bytes(Path(first.ply_path).read_bytes())
    Path(first.ply_path).unlink()
    Path(original["metadata_path"]).unlink()
    mesh.unlink()
    resumed, resumed_proof, _ = read_mesh(fixture, mesh, "resumed", init_ply=str(cached),
                                         initialization_record=original)
    assert len(calls) == 1
    assert resumed.ply_path == str(cached)
    assert resumed_proof["initialization"] == original
    assert original == proof["initialization"]


@pytest.mark.parametrize("corruption", ["sha", "count", "missing_resolved_ply"])
def test_resume_cache_validation_fails_without_regeneration(external_reader, mesh_preparer, corruption):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    initial, proof, _ = read_mesh(fixture, mesh)
    record = deepcopy(proof["initialization"])
    path = initial.ply_path
    if corruption == "sha":
        record["ply_sha256"] = "0" * 64
    elif corruption == "count":
        record["point_count"] += 1
    else:
        path = ""
    with pytest.raises(ValueError, match="cached PLY|checkpoint SHA|point count"):
        read_mesh(fixture, mesh, "failed_resume", init_ply=path, initialization_record=record)
    assert len(calls) == 1


def test_reader_model_load_ignores_initialization_paths_depth_and_proof(external_reader, monkeypatch):
    fixture = external_reader
    output = fixture.tmp / "existing_model"
    output.mkdir()
    original = b"existing provenance must stay byte-identical"
    (output / "camera_split.json").write_bytes(original)

    def forbidden(*args, **kwargs):
        raise AssertionError("Model loading must not access initialization or depth provenance")

    for name in ("_sha256_file", "fetchOpen3DPly", "generate_ply_from_rgbd", "resolve_rgbd_depth_directory", "_write_record"):
        monkeypatch.setitem(fixture.namespace, name, forbidden)
    result = fixture.namespace["readColmapSceneInfo"](
        str(fixture.dataset), "images", False, init_cache_dir=str(output),
        init_mesh="/removed/source_mesh.ply", init_ply="/removed/initial.ply", load_initialization=False,
    )
    assert result.point_cloud is None and result.ply_path is None
    assert result.train_cameras == fixture.cameras
    assert (output / "camera_split.json").read_bytes() == original


class Tensor(np.ndarray):
    def float(self):
        return self.astype(np.float32).view(Tensor)

    def cuda(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def clamp(self, minimum, maximum):
        return np.clip(self, minimum, maximum).view(Tensor)

    def repeat(self, *shape):
        return np.tile(np.asarray(self), shape).view(Tensor)

    def permute(self, *axes):
        return self.transpose(axes).view(Tensor)


def tensor(value):
    return np.asarray(value).view(Tensor)


class FakeTorch:
    Tensor = Tensor
    float32 = np.float32
    no_grad = staticmethod(nullcontext)
    nn = SimpleNamespace(functional=SimpleNamespace(
        normalize=lambda value, dim: tensor(value / np.maximum(np.linalg.norm(value, axis=dim, keepdims=True), 1e-12))))
    tensor = staticmethod(lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype).view(Tensor))
    stack = staticmethod(lambda values, dim=0: np.stack(values, axis=dim).view(Tensor))
    norm = staticmethod(lambda value, dim: np.linalg.norm(value, axis=dim).view(Tensor))
    sum = staticmethod(lambda value, dim: np.sum(value, axis=dim).view(Tensor))
    arccos = staticmethod(lambda value: np.arccos(value).view(Tensor))
    log = staticmethod(lambda value: np.log(value).view(Tensor))
    clamp_max = staticmethod(lambda value, maximum: np.minimum(value, maximum).view(Tensor))


def scene_harness(fixture):
    (fixture.dataset / "sparse/0").mkdir(parents=True, exist_ok=True)

    def camera_loader(infos, scale, args):
        return [SimpleNamespace(camera_center=tensor([float(i), 0., 0.]), R=tensor(np.eye(3)),
                                image_name=info.image_name, nearest_id=[])
                for i, info in enumerate(infos)]

    namespace = {"os": os, "json": json, "np": np, "torch": FakeTorch,
                 "nullcontext": nullcontext, "ModelParams": SimpleNamespace, "GaussianModel": object,
                 "Sequence": Sequence, "Camera": object,
                 "sceneLoadTypeCallbacks": {"Colmap": fixture.namespace["readColmapSceneInfo"]},
                 "cameraList_from_camInfos": camera_loader,
                 "camera_to_JSON": lambda index, camera: {"id": index, "name": camera.colmap_image_name},
                 "searchForMaxIteration": lambda path: 20000}
    path = ROOT / "scene/__init__.py"
    definition = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Scene")
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    # Force evaluation on Python 3.14 too, so its lazy annotations do not hide
    # names needed when the same production class is compiled on Python 3.12.
    assert namespace["Scene"].getTrainCameras.__annotations__["return"] == Sequence[object]
    assert namespace["Scene"].getTestCameras.__annotations__["return"] == Sequence[object]

    class Model:
        ppisp = None

        def create_app_model(self, *args, **kwargs):
            pass

        def create_from_pcd(self, pcd, extent):
            self.initialized_from = pcd
            self.get_xyz = tensor(pcd.points)
            self._scaling = tensor(np.zeros_like(pcd.points))

        def load_ply(self, path):
            self.loaded_path = path

    output = fixture.tmp / "scene_model"
    output.mkdir()
    args = SimpleNamespace(model_path=str(output), source_path=str(fixture.dataset), images="images", eval=False,
                           frame_gpu_budget=-1, init_ply="", init_mesh="", use_decoupled_appearance=3,
                           multi_view_max_angle=180., multi_view_min_dis=.01, multi_view_max_dis=10., multi_view_num=8)
    return namespace["Scene"], Model(), args, output


def test_scene_fresh_mesh_keeps_public_selectors_and_copies_generated_input(external_reader, mesh_preparer):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    Scene, model, args, output = scene_harness(fixture)
    args.init_mesh = str(mesh)
    before = vars(args).copy()
    Scene(args, model, initialization_seed=13)
    assert vars(args) == before and args.init_ply == ""
    assert calls[0][-1] == 13 and calls[0][3] == fixture.cameras
    assert (output / "input.ply").read_bytes() == (output / "initialization/point_cloud.ply").read_bytes()
    assert model.initialized_from is fixture.cloud


def test_scene_resume_uses_resolved_cache_and_preserves_mesh_source(external_reader, mesh_preparer):
    fixture, (mesh, calls, _) = external_reader, mesh_preparer
    initial, proof, _ = read_mesh(fixture, mesh)
    cached = fixture.tmp / "resume_cached_input.ply"
    cached.write_bytes(Path(initial.ply_path).read_bytes())
    Path(initial.ply_path).unlink()
    mesh.unlink()
    Scene, model, args, output = scene_harness(fixture)
    args.init_mesh = str(mesh)
    Scene(args, model, initialization_seed=99, initialization_record=proof["initialization"],
          resolved_init_ply=str(cached))
    assert len(calls) == 1
    assert args.init_ply == "" and args.init_mesh == str(mesh)
    assert model.initialized_from is fixture.cloud
    assert (output / "input.ply").read_bytes() == cached.read_bytes()
    restored = json.loads((output / "camera_split.json").read_text())
    assert restored["initialization"] == proof["initialization"]


def test_scene_does_not_truncate_input_ply_when_it_is_already_the_source(external_reader):
    Scene, model, args, output = scene_harness(external_reader)
    path = output / "input.ply"
    original = b"precomputed source is already the owned input copy"
    path.write_bytes(original)
    args.init_ply = str(path)
    Scene(args, model)
    assert path.read_bytes() == original
    assert model.initialized_from is external_reader.cloud


@pytest.mark.parametrize("iteration", [20000, -1])
def test_scene_load_iteration_preserves_all_existing_provenance(external_reader, monkeypatch, iteration):
    fixture = external_reader
    Scene, model, args, output = scene_harness(fixture)
    args.init_mesh = "/removed/source_mesh.ply"
    expected = {}
    for name in ("camera_split.json", "input.ply", "cameras.json", "multi_view.json"):
        expected[name] = f"original {name}".encode()
        (output / name).write_bytes(expected[name])

    def forbidden(*args, **kwargs):
        raise AssertionError("render/load_iteration cannot recreate initialization")

    monkeypatch.setitem(fixture.namespace, "fetchOpen3DPly", forbidden)
    monkeypatch.setitem(fixture.namespace, "generate_ply_from_rgbd", forbidden)
    Scene(args, model, load_iteration=iteration)
    assert model.loaded_path.endswith("point_cloud/iteration_20000/point_cloud.ply")
    assert not hasattr(model, "initialized_from")
    for name, contents in expected.items():
        assert (output / name).read_bytes() == contents


def test_reader_and_scene_keep_crlf():
    for name in ("scene/dataset_readers.py", "scene/__init__.py"):
        data = (ROOT / name).read_bytes()
        assert data.count(b"\n") > 0 and data.count(b"\n") == data.count(b"\r\n")
