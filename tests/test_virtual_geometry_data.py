import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_geometry_data import (
    ShuffledViewSampler, VirtualGeometryDataset, exposure_budget, teacher_arrays, validate_arrays, validate_camera,
)
from utils.virtual_camera_selection import sha256


def camera():
    return {"id": "virtual_0", "width": 7, "height": 7, "K": [[10., 0., 3.5], [0., 10., 3.5], [0., 0., 1.]],
            "c2w": np.eye(4).tolist()}


class FrontPlane:
    threads = 1

    def cast(self, rays):
        depth = (2. - rays[..., 2]) / rays[..., 5]
        return {"t_hit": depth.astype(np.float32),
                "primitive_normals": np.broadcast_to([0., 0., -1.], rays.shape[:-1] + (3,)).astype(np.float32),
                "primitive_ids": np.zeros(rays.shape[:-1], np.uint32)}


def test_pool_budget_is_invariant_to_duplicate_exposures_of_same_patch():
    q = exposure_budget(np.array([100, 200]), np.array([1., 1.]))
    np.testing.assert_allclose(q[0] * 100, q[1] * 200)
    assert q[0] > q[1]


def test_sampler_rounds_do_not_repeat_and_resume_preserves_future_order():
    a = ShuffledViewSampler(5, seed=17)
    first = [a.next() for _ in range(5)]
    assert sorted(first) == list(range(5))
    for _ in range(3):
        a.next()
    b = ShuffledViewSampler(5, seed=888)
    b.load_state_dict(a.state_dict())
    assert [a.next() for _ in range(22)] == [b.next() for _ in range(22)]


def test_sampler_rejects_other_pool_or_duplicate_pending_views():
    sampler = ShuffledViewSampler(5)
    state = sampler.state_dict()
    state["remaining"] = [1, 1]
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    state["remaining"], state["count"] = [], 6
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


def test_camera_contract_rejects_reflection_and_skew():
    record = camera()
    validate_camera(record)
    record["c2w"][0][0] = -1
    with pytest.raises(ValueError):
        validate_camera(record)
    record = camera()
    record["K"][0][1] = .1
    with pytest.raises(ValueError):
        validate_camera(record)


@pytest.mark.parametrize("behind", [False, True])
def test_teacher_exact_surface_normal_and_thin_back_layer_evidence(behind):
    atlas = {"xyz": np.array([[0., 0., 2.025 if behind else 2.]], np.float32),
             "normal_world": np.array([[0., 0., -1.]], np.float32), "face_id": np.array([1 if behind else 0]),
             "evidence_class": np.array([3], np.uint8)}
    arrays, _ = teacher_arrays(FrontPlane(), camera(), atlas, cKDTree(atlas["xyz"]), {}, 10.)
    np.testing.assert_allclose(arrays["depth"], 2.)
    np.testing.assert_allclose(arrays["normal"][3, 3], [0, 0, -1])
    assert arrays["evidence_class"][3, 3] == (0 if behind else 3)
    assert arrays["weight_depth"][3, 3] == (.25 if behind else 0)
    assert arrays["valid_depth"].all()
    assert not arrays["valid_normal"][0].any()
    validate_arrays(arrays, camera())


def test_nonunit_teacher_normal_is_rejected():
    atlas = {"xyz": np.array([[0., 0., 2.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
             "face_id": np.array([0]), "evidence_class": np.array([0], np.uint8)}
    arrays, _ = teacher_arrays(FrontPlane(), camera(), atlas, cKDTree(atlas["xyz"]), {}, 10.)
    altered = copy.deepcopy(arrays)
    altered["normal"][3, 3] *= .5
    with pytest.raises(ValueError, match="unit"):
        validate_arrays(altered, camera())


def test_teacher_normal_stencil_excludes_neighbors_of_a_hole_without_erasing_depth():
    class PlaneWithHole(FrontPlane):
        def cast(self, rays):
            result = super().cast(rays)
            if rays.ndim == 3:
                result["t_hit"][3, 3] = np.inf
            return result

    atlas = {"xyz": np.array([[0., 0., 2.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
             "face_id": np.array([0]), "evidence_class": np.array([0], np.uint8)}
    arrays, _ = teacher_arrays(PlaneWithHole(), camera(), atlas, cKDTree(atlas["xyz"]), {}, 10.)
    assert not arrays["valid_normal"][2:5, 2:5].any()
    assert arrays["valid_depth"][2:5, 2:5].sum() == 8
    assert arrays["valid_normal"][1, 1]


def cached_fixture(tmp_path):
    atlas = {"xyz": np.array([[0., 0., 2.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
             "face_id": np.array([0]), "evidence_class": np.array([0], np.uint8)}
    arrays, _ = teacher_arrays(FrontPlane(), camera(), atlas, cKDTree(atlas["xyz"]), {}, 10.)
    arrays["q"][:] = 1
    np.savez(tmp_path / "atlas.npz", **atlas)
    (tmp_path / "real.json").write_text(json.dumps({"cameras": []}))
    (tmp_path / "source.txt").write_text("frozen source")
    records = []
    for index in range(3):
        relative = f"view{index}.npz"
        np.savez(tmp_path / relative, **arrays)
        records.append({**camera(), "id": f"virtual_{index}", "arrays_path": relative,
                        "arrays_sha256": sha256(tmp_path / relative)})
    manifest = {"schema_version": 3, "status": "complete", "virtual_rgb": False, "dataset": str(tmp_path),
                "views": records, "view_count": len(records), "depth_convention": "camera_z",
                "pixel_center_convention": "colmap_half_integer", "normal_convention": "camera_space_unit_face_normal_toward_camera",
                "renderer_near": .2, "teacher_geometry_transform": "identity", "scene_scale": 10.,
                "normalizers": {"depth": 49., "normal": 25., "coverage": 49.},
                "approved_camera_sha256": "approved-pose", "approved_manifest_sha256": "approved-manifest",
                "source_files_sha256": {str(tmp_path / "source.txt"): sha256(tmp_path / "source.txt")},
                "real_cameras_path": "real.json", "real_cameras_sha256": sha256(tmp_path / "real.json"),
                "atlas_path": "atlas.npz", "atlas_sha256": sha256(tmp_path / "atlas.npz")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "manifest.json"


def test_teacher_source_or_array_mutation_is_rejected(tmp_path):
    path = cached_fixture(tmp_path)
    dataset = VirtualGeometryDataset(path, device="cpu")
    # 文件在构造后的读取前变化也不能绕过 immutable 校验。
    with (tmp_path / "view0.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        dataset._arrays(0)
    (tmp_path / "source.txt").write_text("different source")
    with pytest.raises(ValueError, match="source changed"):
        VirtualGeometryDataset(path, device="cpu")


def test_cpu_cache_is_bounded_and_audit_reads_do_not_advance_sampler(tmp_path):
    dataset = VirtualGeometryDataset(cached_fixture(tmp_path), device="cpu", cpu_cache_size=1)
    before = dataset.state_dict()
    for index in range(3):
        dataset._arrays(index)
    assert len(dataset._cpu) == 1
    assert dataset.state_dict() == before


def test_checkpoint_rejects_different_approved_pose_identity(tmp_path):
    dataset = VirtualGeometryDataset(cached_fixture(tmp_path), device="cpu")
    state = dataset.state_dict()
    state["identity"]["approved_camera_sha256"] = "another-pose-pool"
    with pytest.raises(ValueError, match="another immutable"):
        dataset.load_state_dict(state)


def test_producer_provenance_is_not_runtime_data_identity(tmp_path):
    path = cached_fixture(tmp_path)
    producer = tmp_path / "producer.py"
    producer.write_text("original producer")
    manifest = json.loads(path.read_text())
    manifest["producer_files_sha256"] = {str(producer): sha256(producer)}
    path.write_text(json.dumps(manifest))
    producer.write_text("loader logging changed after frozen output")
    dataset = VirtualGeometryDataset(path, device="cpu")
    assert len(dataset) == 3
    (tmp_path / "source.txt").write_text("actual data changed")
    with pytest.raises(ValueError, match="source changed"):
        VirtualGeometryDataset(path, device="cpu")


def test_real_training_split_must_equal_teacher_source_split(tmp_path):
    dataset = VirtualGeometryDataset(cached_fixture(tmp_path), device="cpu")
    dataset.real_cameras = [{"name": "a.jpg"}, {"name": "b.jpg"}]
    assert dataset.validate_real_cameras([{"source_image_name": "b.jpg"}, {"source_image_name": "a.jpg"}], tmp_path)
    with pytest.raises(ValueError, match="split differs"):
        dataset.validate_real_cameras([{"source_image_name": "a.jpg"}])
    with pytest.raises(ValueError, match="path differs"):
        dataset.validate_real_cameras([{"name": "a.jpg"}, {"name": "b.jpg"}], tmp_path / "other")


def test_virtual_camera_get_is_rgb_free_bounded_and_sampler_independent(tmp_path):
    pytest.importorskip("torch")
    dataset = VirtualGeometryDataset(cached_fixture(tmp_path), device="cpu", gpu_cache_size=1)
    state = dataset.state_dict()
    for index in range(3):
        view = dataset.get_view(index)
        assert view.virtual_index == index and not hasattr(view, "original_image")
    assert len(dataset._device) == 1
    assert dataset.state_dict() == state
