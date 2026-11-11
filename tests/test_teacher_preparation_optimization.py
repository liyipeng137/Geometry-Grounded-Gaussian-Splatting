"""Teacher RAM/shared-mesh path matches the legacy two-write numerical path."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

import utils.virtual_geometry_data as teacher_module
from test_virtual_geometry_data import FrontPlane, camera
from utils.virtual_camera_selection import resized_camera


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def prepared_inputs(tmp_path):
    dataset, root = tmp_path / "dataset", tmp_path / "approved"
    root.mkdir()
    for folder in ("images", "normals", "depth"):
        (dataset / folder).mkdir(parents=True)
    rgb, normal, depth = (dataset / "images/frame.png", dataset / "normals/frame.png", dataset / "depth/frame.npy")
    rgb.write_bytes(b"RGB source is only hashed here")
    normal.write_bytes(b"normal source is only hashed here")
    np.save(depth, np.full((7, 7), 2., np.float32))
    mesh = dataset / "genrecon_mesh.ply"
    mesh.write_bytes(b"shared resources already own this exact mesh")
    atlas = {
        "xyz": np.array([[-.2, -.2, 2.], [.2, -.2, 2.], [-.2, .2, 2.], [.2, .2, 2.]], np.float32),
        "normal_world": np.tile([0., 0., -1.], (4, 1)).astype(np.float32),
        "face_id": np.zeros(4, np.int64), "patch_id": np.arange(4, dtype=np.int32),
        "evidence_class": np.arange(4, dtype=np.uint8), "area_weight": np.array([1., 2., 3., 4.], np.float32),
    }
    np.savez_compressed(root / "sample_atlas.npz", **atlas)
    views = []
    for index, x in enumerate((0., .2, -.3)):
        row = copy.deepcopy(camera())
        row.update(id=f"virtual_{index}", selection_source="surface_lookat", anchor_id="real_0")
        row["c2w"][0][3] = x
        views.append(row)
    document(root / "virtual_cameras.json", {"cameras": views})
    document(root / "real_cameras.json", {"cameras": [{"name": "frame.png"}]})
    sources = {str(path): digest(path) for path in (mesh, depth)}
    document(root / "source_fingerprints.json", sources)
    approved = {"schema_version": 2, "status": "complete", "dataset": str(dataset), "genrecon_mesh": str(mesh),
                "scene_scale": 10., "virtual_cameras_path": "virtual_cameras.json", "real_cameras_path": "real_cameras.json",
                "atlas_path": "sample_atlas.npz", "source_fingerprints_path": "source_fingerprints.json"}
    manifest = root / "manifest.json"

    def refresh():
        document(manifest, approved)
        paths = (manifest, root / "virtual_cameras.json", root / "real_cameras.json",
                 root / "sample_atlas.npz", root / "source_fingerprints.json")
        document(root / "receipt.json", {"files": {str(path.relative_to(root)): digest(path) for path in paths}})

    refresh()
    resources = SimpleNamespace(mesh_path=mesh, scene=FrontPlane(), vertices=np.zeros((4, 3)), faces=np.zeros((2, 3), np.int32))
    return SimpleNamespace(dataset=dataset, root=root, mesh=mesh, atlas=atlas, views=views, manifest=manifest,
                           approved=approved, refresh=refresh, resources=resources, rgb=rgb, normal=normal, depth=depth)


def legacy_disk_reference(inputs, output):
    """Old prepare_teacher sequence: write q=0, reload, then fill q and rewrite."""
    output.mkdir()
    atlas, scene, scale = copy.deepcopy(inputs.atlas), FrontPlane(), 10.
    unique, atlas_regions = np.unique(teacher_module.region_keys(atlas["xyz"], atlas["normal_world"], scale), axis=0, return_inverse=True)
    region_ids = {tuple(int(v) for v in key): index for index, key in enumerate(unique)}
    areas = np.bincount(atlas_regions, weights=atlas["area_weight"])
    fallback = float(np.median(atlas["area_weight"]))
    tree = cKDTree(atlas["xyz"])
    pixel_counts = np.zeros(len(region_ids), np.int64)
    statistics = []
    for index, approved in enumerate(inputs.views):
        view = resized_camera(approved, 7)
        arrays, stats = teacher_module.teacher_arrays(scene, view, atlas, tree, region_ids, scale)
        pixel_counts = np.pad(pixel_counts, (0, len(region_ids) - len(pixel_counts)))
        pixel_counts += np.bincount(arrays["patch_id"][arrays["valid_depth"]], minlength=len(pixel_counts))
        np.savez_compressed(output / f"{index}.npz", **arrays)
        statistics.append(stats)
    areas = np.pad(areas, (0, len(region_ids) - len(areas)), constant_values=fallback)
    weights = teacher_module.exposure_budget(pixel_counts, areas)
    normalization = {"depth": 0., "normal": 0., "coverage": 0.}
    result = []
    for index in range(len(inputs.views)):
        path = output / f"{index}.npz"
        arrays = dict(np.load(path, allow_pickle=False))
        valid = arrays["valid_depth"]
        arrays["q"][valid] = weights[arrays["patch_id"][valid]]
        normalization["depth"] += float(arrays["q"][valid].sum(dtype=np.float64)) / len(inputs.views)
        normalization["coverage"] += float(arrays["q"][valid].sum(dtype=np.float64)) / len(inputs.views)
        normalization["normal"] += float(arrays["q"][arrays["valid_normal"]].sum(dtype=np.float64)) / len(inputs.views)
        np.savez_compressed(path, **arrays)
        result.append(arrays)
    return result, statistics, normalization, pixel_counts, areas, weights


def test_shared_mesh_ram_arrays_equal_old_disk_path_and_write_each_view_once(prepared_inputs, tmp_path, monkeypatch):
    f = prepared_inputs
    expected, stats, normalization, counts, areas, weights = legacy_disk_reference(f, tmp_path / "legacy")
    output = tmp_path / "new"
    writes, view_reads = [], []
    save, load = np.savez_compressed, np.load

    def observed_save(path, **arrays):
        writes.append(Path(path))
        return save(path, **arrays)

    def observed_load(path, *args, **kwargs):
        if Path(path).parent == output / "views":
            view_reads.append(Path(path))
        return load(path, *args, **kwargs)

    ply = ModuleType("plyfile")
    ply.PlyData = SimpleNamespace(read=lambda *args, **kwargs: pytest.fail("shared teacher must not reread full mesh"))
    monkeypatch.setitem(sys.modules, "plyfile", ply)
    monkeypatch.setattr(np, "savez_compressed", observed_save)
    monkeypatch.setattr(np, "load", observed_load)
    handed = {}
    manifest_path = teacher_module.prepare_teacher(f.manifest, output, long_edge=7,
                                                    mesh_resources=f.resources, prepared_arrays=handed)
    assert view_reads == []
    assert Counter(path.name for path in writes if path.parent == output / "views") == {
        "0000.npz": 1, "0001.npz": 1, "0002.npz": 1,
    }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["normalizers"] == normalization
    assert [row["stats"] for row in manifest["views"]] == stats
    for index, arrays in enumerate(expected):
        actual = dict(load(output / manifest["views"][index]["arrays_path"], allow_pickle=False))
        assert set(actual) == teacher_module.ARRAY_KEYS
        for name in arrays:
            assert actual[name].dtype == arrays[name].dtype
            np.testing.assert_array_equal(actual[name], arrays[name])
            np.testing.assert_array_equal(handed[index]["arrays"][name], arrays[name])
        assert handed[index]["arrays_sha256"] == manifest["views"][index]["arrays_sha256"]
        assert digest(output / manifest["views"][index]["arrays_path"]) == digest(tmp_path / f"legacy/{index}.npz")
    budget = load(output / "patch_budget.npz", allow_pickle=False)
    for name, value in (("pixel_counts", counts), ("area_estimate", areas), ("q_per_pixel", weights)):
        np.testing.assert_array_equal(budget[name], value)
    # Production selection needs no viewer-only files for this stage.
    assert not (f.root / "previews").exists() and not (f.root / "scene_preview.npz").exists()


@pytest.mark.parametrize("source", ["approved", "explicit"])
def test_teacher_hashes_resolved_rgb_normal_directories(prepared_inputs, tmp_path, source):
    f = prepared_inputs
    images, normals = f.dataset / "custom_rgb", f.dataset / "custom_normal"
    images.mkdir()
    normals.mkdir()
    f.rgb.rename(images / f.rgb.name)
    f.normal.rename(normals / f.normal.name)
    paths = {"images": str(images), "normals": str(normals), "mesh": str(f.mesh)}
    if source == "approved":
        f.approved["input_paths"] = paths
        f.refresh()
    result = teacher_module.prepare_teacher(f.manifest, tmp_path / "out", long_edge=7,
                                            mesh_resources=f.resources, input_paths=paths if source == "explicit" else None)
    fingerprints = json.loads(result.read_text())["source_files_sha256"]
    assert fingerprints[str(images / f.rgb.name)] == digest(images / f.rgb.name)
    assert fingerprints[str(normals / f.normal.name)] == digest(normals / f.normal.name)
    assert str(f.rgb) not in fingerprints and str(f.normal) not in fingerprints


def test_preverified_sources_skip_only_initial_hash_and_still_recheck(prepared_inputs, tmp_path, monkeypatch):
    f = prepared_inputs
    preverified = {str(path): digest(path) for path in (f.mesh, f.depth, f.rgb, f.normal)}
    calls = Counter()
    original = teacher_module.sha256

    def observed(path):
        calls[str(Path(path).resolve())] += 1
        return original(path)

    monkeypatch.setattr(teacher_module, "sha256", observed)
    teacher_module.prepare_teacher(f.manifest, tmp_path / "out", long_edge=7,
                                   mesh_resources=f.resources, preverified_sources=preverified)
    assert all(calls[str(path)] == 1 for path in (f.mesh, f.depth, f.rgb, f.normal))


def test_preverified_sources_cannot_hide_input_change_during_preparation(prepared_inputs, tmp_path):
    f = prepared_inputs
    preverified = {str(path): digest(path) for path in (f.mesh, f.depth, f.rgb, f.normal)}

    class MutatingScene(FrontPlane):
        def cast(self, rays):
            f.mesh.write_bytes(b"changed after input was preverified")
            return super().cast(rays)

    f.resources.scene = MutatingScene()
    handed, output = {}, tmp_path / "out"
    with pytest.raises(ValueError, match="Source changed while preparing"):
        teacher_module.prepare_teacher(f.manifest, output, long_edge=7, mesh_resources=f.resources,
                                       preverified_sources=preverified, prepared_arrays=handed)
    assert handed == {} and not (output / "manifest.json").exists()


def test_public_default_rejects_changed_source_before_render(prepared_inputs, tmp_path):
    f = prepared_inputs
    f.mesh.write_bytes(b"changed since approved selection")
    with pytest.raises(ValueError, match="Approved source changed"):
        teacher_module.prepare_teacher(f.manifest, tmp_path / "out", long_edge=7, mesh_resources=f.resources)
    assert not (tmp_path / "out").exists()


def test_shared_mesh_identity_is_checked(prepared_inputs, tmp_path):
    f = prepared_inputs
    f.resources.mesh_path = tmp_path / "another_mesh.ply"
    with pytest.raises(ValueError, match="Shared mesh resources differ"):
        teacher_module.prepare_teacher(f.manifest, tmp_path / "out", long_edge=7, mesh_resources=f.resources)


def test_preloaded_dataset_never_unzips_view_files_and_keeps_sampler_state(prepared_inputs, tmp_path, monkeypatch):
    f, handed = prepared_inputs, {}
    manifest = teacher_module.prepare_teacher(f.manifest, tmp_path / "out", long_edge=7,
                                              mesh_resources=f.resources, prepared_arrays=handed)
    original = np.load

    def no_view_unzip(path, *args, **kwargs):
        if Path(path).parent == manifest.parent / "views":
            pytest.fail("verified preloaded arrays should not be unzipped again")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", no_view_unzip)
    dataset = teacher_module.VirtualGeometryDataset(manifest, device="cpu", cpu_cache_size=3, preloaded_arrays=handed)
    state = dataset.state_dict()
    for index in handed:
        assert dataset._arrays(index) is handed[index]["arrays"]
    assert dataset.state_dict() == state
    handed.clear()
    assert len(dataset._cpu) == 3 and dataset._arrays(0)["q"].sum() > 0


def test_preloaded_dataset_respects_cpu_cache_bound(prepared_inputs, tmp_path):
    handed = {}
    manifest = teacher_module.prepare_teacher(prepared_inputs.manifest, tmp_path / "out", long_edge=7,
                                              mesh_resources=prepared_inputs.resources, prepared_arrays=handed)
    dataset = teacher_module.VirtualGeometryDataset(manifest, device="cpu", cpu_cache_size=1, preloaded_arrays=handed)
    assert len(dataset._cpu) == 1
    dataset._arrays(0)
    assert len(dataset._cpu) == 1


@pytest.mark.parametrize("bad", ["memory_value", "file_sha", "file_bytes", "shape", "verify_false"])
def test_preloaded_dataset_rejects_mutated_or_unverified_handoff(prepared_inputs, tmp_path, bad):
    handed = {}
    manifest = teacher_module.prepare_teacher(prepared_inputs.manifest, tmp_path / "out", long_edge=7,
                                              mesh_resources=prepared_inputs.resources, prepared_arrays=handed)
    if bad == "memory_value":
        handed[0]["arrays"]["depth"][3, 3] += .1
    elif bad == "file_sha":
        handed[0]["arrays_sha256"] = "0" * 64
    elif bad == "file_bytes":
        with (manifest.parent / "views/0000.npz").open("ab") as stream:
            stream.write(b"different bytes")
    elif bad == "shape":
        handed[0]["arrays"]["normal"] = handed[0]["arrays"]["normal"][:2]
    with pytest.raises(ValueError):
        teacher_module.VirtualGeometryDataset(manifest, device="cpu", preloaded_arrays=handed, verify=bad != "verify_false")
