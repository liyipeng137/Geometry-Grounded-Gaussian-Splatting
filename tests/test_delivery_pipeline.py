from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import utils.delivery_pipeline as delivery
from utils.delivery_pipeline import load_config, cache_keys, _publish, _ready


def test_production_defaults_and_selective_cache_invalidation():
    cfg = load_config()
    assert cfg["initialization"] == {"method": "gen_mesh", "num_points": 200000}
    assert cfg["virtual_views"]["trajectory_offset_fraction"] == 0.16
    assert cfg["outputs"]["camera_preview"] is False
    paths = {k: Path("/data") / v for k, v in cfg["inputs"].items()}
    groups = {
        k: {k: "digest"} for k in ("mesh", "calibration", "images", "depth", "normals")
    }
    original = cache_keys(cfg, paths, groups)
    changed = deepcopy(cfg)
    changed["initialization"]["num_points"] = 100000
    keys = cache_keys(changed, paths, groups)
    assert keys["initialization"] != original["initialization"]
    assert (
        keys["cameras"] == original["cameras"]
        and keys["teacher"] == original["teacher"]
    )
    changed = deepcopy(cfg)
    changed["virtual_views"]["trajectory_offset_fraction"] = 0.08
    keys = cache_keys(changed, paths, groups)
    assert (
        keys["cameras"] != original["cameras"]
        and keys["teacher"] != original["teacher"]
    )
    assert keys["initialization"] == original["initialization"]
    changed = deepcopy(cfg)
    changed["geometry"]["loss_weights"]["normal_depth"] = 0.1
    assert cache_keys(changed, paths, groups) == original


@pytest.mark.parametrize(
    "override",
    [
        {"initialization": {"method": "bad"}},
        {"initialization": {"num_points": 0}},
        {"initialization": {"num_points": 3000000}},
        {"training": {"iterations": 10}},
        {"virtual_views": {"trajectory_offset_fraction": 0}},
        {"mispelled": True},
        {"runtime": {"real_gpu_cache": False}},
        {"outputs": {"save_iterations": [999999]}},
    ],
)
def test_invalid_config_fails_early(tmp_path, override):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(override))
    with pytest.raises(ValueError):
        load_config(path)


def test_cache_requires_completed_receipt_and_unchanged_files(tmp_path):
    cache = tmp_path / "stage"
    assert not _ready(cache, "key")
    cache.mkdir()
    (cache / "payload").write_bytes(b"complete bytes")
    with pytest.raises(ValueError, match="Incomplete"):
        _ready(cache, "key")
    _publish(cache, "key")
    assert _ready(cache, "key")
    before = (cache / "delivery_cache.json").read_bytes()
    assert (
        _ready(cache, "key") and (cache / "delivery_cache.json").read_bytes() == before
    )
    with pytest.raises(ValueError, match="identity"):
        _ready(cache, "other")
    (cache / "payload").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        _ready(cache, "key")


def test_tsdf_preparation_forwards_requested_count_and_depth_directory(tmp_path, monkeypatch):
    import plyfile
    import utils.mesh_initialization as mesh_initialization
    import utils.mesh_resources as mesh_resources
    import utils.virtual_camera_selection as selection
    import utils.virtual_geometry_data as geometry_data

    cfg = load_config()
    cfg["initialization"] = {"method": "tsdf", "num_points": 12345}
    cfg["seed"] = 23
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    paths = {"images": dataset / "rgb_custom", "depth": dataset / "depth_custom",
             "normals": dataset / "normal_custom", "colmap": dataset / "colmap_custom",
             "mesh": dataset / "mesh_custom.ply"}
    cameras, intrinsics = [SimpleNamespace(name="frame")], {1: "intrinsic"}
    groups = {key: {key: "digest"} for key in ("mesh", "calibration", "images", "depth", "normals")}
    monkeypatch.setattr(delivery, "read_inputs", lambda *args: (paths, cameras, intrinsics, [], groups))
    monkeypatch.setattr(delivery, "_ready", lambda path, key, only_files=None: path.parent.name != "initialization")

    def unexpected(*args, **kwargs):
        pytest.fail("Complete camera/teacher caches and TSDF initialization must not rebuild mesh resources")

    monkeypatch.setattr(mesh_resources, "MeshResources", unexpected)
    monkeypatch.setattr(selection, "run_selection", unexpected)
    monkeypatch.setattr(geometry_data, "prepare_teacher", unexpected)
    monkeypatch.setattr(mesh_initialization, "prepare_from_cameras", unexpected)
    calls = {}

    def generate(actual_cameras, num_points, ply_path, **kwargs):
        calls.update(cameras=actual_cameras, num_points=num_points, ply_path=ply_path, kwargs=kwargs)
        np.random.random(10)
        Path(ply_path).write_bytes(b"small valid-boundary PLY fixture")

    pd_utils = ModuleType("utils.pd_utils")
    pd_utils.generate_ply_from_rgbd = generate
    monkeypatch.setitem(sys.modules, "utils.pd_utils", pd_utils)

    def load_teacher(path, **kwargs):
        calls["teacher_kwargs"] = kwargs
        return SimpleNamespace(path=Path(path))

    monkeypatch.setattr(geometry_data, "VirtualGeometryDataset", load_teacher)
    monkeypatch.setattr(plyfile.PlyData, "read", lambda path: {"vertex": SimpleNamespace(data=np.zeros(7))})
    previous = np.random.get_state()
    summary = {}
    actual_paths, dirs, teacher = delivery.prepare(dataset, cfg, dataset / "gggs", summary)
    after = np.random.get_state()
    assert previous[0] == after[0] and np.array_equal(previous[1], after[1]) and previous[2:] == after[2:]
    assert calls["cameras"] is cameras and calls["num_points"] == 12345
    assert calls["kwargs"]["depth_prior_dir"] == str(paths["depth"])
    assert calls["kwargs"]["cam_intrinsics"] is intrinsics
    assert calls["kwargs"]["pixel_center_convention"] == "colmap_half_integer"
    assert actual_paths == paths and teacher.path == dirs["teacher"] / "manifest.json"
    assert calls["teacher_kwargs"]["preloaded_arrays"] is None
    assert summary["initialization"] == {"method": "tsdf", "requested_points": 12345, "actual_points": 7}


def test_run_wires_input_directories_init_count_and_prepared_teacher_identity(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["initialization"]["num_points"] = 100000
    cfg["runtime"]["cpu_threads"] = 3
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    dataset.mkdir()
    paths = {"images": dataset / "custom_rgb", "depth": dataset / "custom_depth",
             "normals": dataset / "custom_normal", "colmap": dataset / "custom_colmap",
             "mesh": dataset / "custom_mesh.ply"}
    dirs = {"teacher": dataset / "gggs/cache/teacher/key",
            "initialization": dataset / "gggs/cache/initialization/key"}
    prepared = SimpleNamespace(path=dirs["teacher"] / "manifest.json")
    monkeypatch.setattr(delivery, "load_config", lambda path=None: deepcopy(cfg))
    monkeypatch.setattr(delivery, "prepare", lambda *args: (paths, dirs, prepared))
    received = {}
    general = ModuleType("utils.general_utils")
    general.safe_state = lambda silent, seed: received.update(seed=seed)
    monkeypatch.setitem(sys.modules, "utils.general_utils", general)
    torch = ModuleType("torch")
    torch.set_num_threads = lambda count: received.update(cpu_threads=count)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    train = ModuleType("train")

    def training(model, optimization, pipeline, args, *, prepared_teacher):
        received.update(model=model, optimization=optimization, args=args, teacher=prepared_teacher)
        resolved = json.loads(Path(args.virtual_geometry_config).read_text())
        assert Path(resolved["teacher_manifest"]).resolve() == prepared_teacher.path.resolve()

    train.training = training
    monkeypatch.setitem(sys.modules, "train", train)
    assert delivery.run(dataset, output=output) == output
    model, args = received["model"], received["args"]
    assert received["teacher"] is prepared
    assert model.images == str(paths["images"])
    assert model.depth_prior_dir == str(paths["depth"])
    assert model.normal_prior_dir == str(paths["normals"])
    assert model.colmap_dir == str(paths["colmap"])
    assert model.init_ply == str(dirs["initialization"] / "point_cloud.ply")
    assert model.init_mesh == "" and model.init_num_points == 100000
    assert received["optimization"].iterations == cfg["training"]["iterations"]
    assert received["seed"] == args.training_seed == cfg["seed"]
    assert received["cpu_threads"] == 3 and os.environ["OMP_NUM_THREADS"] == "3"
    assert args.save_iterations == [cfg["training"]["iterations"]]
    assert args.checkpoint_iterations == [] and args.test_iterations == []
    assert json.loads((output / "summary.json").read_text())["status"] == "completed"
