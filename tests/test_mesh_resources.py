"""Shared mesh preserves legacy sampling and avoids duplicate geometry/BVH loads."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import mesh_resources as resources  # noqa: E402
from utils import mesh_initialization as initialization  # noqa: E402
from utils import virtual_camera_selection as selection  # noqa: E402


def mesh_fixture(path):
    vertices = np.array([[0., 0., 2.], [1.25, 0., 2.], [0., .7, 2.], [.15, -.2, 1.1],
                         [.15, .8, 1.1], [.15, -.2, 2.2], [2., 0., 1.], [2.3, .3, 1.3], [1.9, .6, 1.2]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    rows = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    rows["x"], rows["y"], rows["z"] = vertices.T
    triangles = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    triangles["vertex_indices"] = faces
    PlyData([PlyElement.describe(rows, "vertex"), PlyElement.describe(triangles, "face")]).write(path)


def array_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.tobytes())
    return digest.hexdigest()


def test_cached_initialization_and_atlas_match_legacy_golden_bytes(tmp_path):
    path = tmp_path / "mesh.ply"
    mesh_fixture(path)
    shared = resources.MeshResources(path)
    initial = shared.sample_surface(257, 19, kind="initialization")
    # Captured from the two pre-refactor algorithms on this exact physical fixture.
    assert array_digest(initial[:3]) == "3e8101e36dde846dddc35eb3600d030c5c6dd946e0eeb60a1f9640c5554bc81d"
    _, _, atlas, _ = selection.load_mesh_and_samples(path, 257, 19, 1.31, mesh_resources=shared)
    assert array_digest([atlas[k] for k in ("xyz", "normal_world", "face_id", "patch_id", "area_weight")]) == "e7f11c597cf4da885e4692dd35d412ed57fe653014a5cbf28e64bba3486b7da8"


def test_sampling_counts_are_independent_and_reuse_one_cdf_and_scene(tmp_path, monkeypatch):
    path = tmp_path / "mesh.ply"
    mesh_fixture(path)
    constructions = []

    class FakeScene:
        def __init__(self, vertices, faces, threads):
            constructions.append((id(vertices), id(faces), threads))

    monkeypatch.setattr(resources, "MeshScene", FakeScene)
    shared = resources.MeshResources(path, threads=3)
    first = shared.sample_surface(120, 0, kind="atlas")
    cdf = shared.area_cdf
    initial = shared.sample_surface(200, 0, kind="initialization")
    repeated = shared.sample_surface(120, 0, kind="atlas")
    assert len(initial[0]) == 200 and len(first[0]) == 120
    for a, b in zip(first[:3], repeated[:3]):
        np.testing.assert_array_equal(a, b)
    assert shared.area_cdf is cdf
    assert shared.scene is shared.scene and len(constructions) == 1
    assert constructions[0] == (id(shared.vertices), id(shared.faces), 3)


def test_batched_rgb_raycast_uses_all_origins_once_and_respects_cache_byte_limit():
    views = []
    for index, x in enumerate((-.5, .5)):
        pose = np.eye(4)
        pose[0, 3] = x
        views.append({"name": str(index), "index": index, "width": 9, "height": 9,
                      "K": [[4., 0., 4.5], [0., 4., 4.5], [0., 0., 1.]], "c2w": pose})

    class Plane:
        def __init__(self):
            self.calls = []

        def cast(self, rays):
            self.calls.append(rays.copy())
            return {"t_hit": (2. - rays[:, 2]) / rays[:, 5], "primitive_ids": np.zeros(len(rays), np.uint32)}

    points = np.array([[-.4, 0., 2.], [.4, 0., 2.]], np.float32)
    normals = np.array([[0., 0., -1.], [0., 0., -1.]], np.float32)
    scene = Plane()

    def reader(view):
        return np.full((9, 9, 3), 20 + 40 * view["index"], np.uint8)

    rgb, cameras, stats = initialization.project_colors(points, normals, np.zeros(2, int), views, scene,
                                                        image_reader=reader, rgb_cache_bytes=9 * 9 * 3)
    assert len(scene.calls) == 1
    np.testing.assert_array_equal(scene.calls[0][:, 0], [-.5, .5])
    np.testing.assert_array_equal(rgb, [[20, 20, 20], [60, 60, 60]])
    np.testing.assert_array_equal(cameras, [0, 1])
    assert stats["rgb_cache_peak_bytes"] <= 9 * 9 * 3
    separate = np.concatenate([initialization.visible_samples(Plane(), points[i:i + 1], np.zeros(1, int),
                                                              np.asarray(views[i]["c2w"])[:3, 3]) for i in range(2)])
    np.testing.assert_array_equal(separate, [True, True])


def test_custom_input_paths_are_used_without_copying_dataset(tmp_path):
    colmap, depth, images = (tmp_path / name for name in ("calibration", "measured_z", "photos"))
    for path in (colmap, depth, images):
        path.mkdir()
    (colmap / "cameras.txt").write_text("1 PINHOLE 9 9 4 4 4.5 4.5\n")
    (colmap / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    np.save(depth / "frame.npy", np.full((9, 9), 2., np.float32))
    cameras, depths, _, sources, binary = selection.load_cameras(tmp_path, 18,
        {"colmap": colmap, "depth": depth, "images": images})
    assert not binary and len(cameras) == 1
    assert cameras[0]["image_path"] == str(images / "frame.png")
    assert sources == [depth / "frame.npy"]
    assert depths[0].shape == (18, 18) and (depths[0] == 2).all()


def test_production_selection_skips_preview_renders_and_detailed_artifacts(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "sparse/0").mkdir(parents=True)
    for name in ("cameras.txt", "images.txt"):
        (dataset / "sparse/0" / name).write_text("fixture")
    mesh = dataset / "genrecon_mesh.ply"
    mesh.write_text("fixture")
    depth_path = dataset / "depth.npy"
    np.save(depth_path, np.ones((32, 32), np.float32))
    camera = {"id": "real_0", "name": "frame.png", "kind": "real", "selection_source": "real",
              "width": 32, "height": 32, "K": [[16., 0., 16.], [0., 16., 16.], [0., 0., 1.]],
              "c2w": np.eye(4).tolist(), "stats": {"median_real_depth": 2.}}
    atlas = {"xyz": np.array([[0., 0., 2.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
             "face_id": np.array([0]), "patch_id": np.array([0]), "area_weight": np.ones(1, np.float32)}
    calls = []

    class Scene:
        def cast(self, rays):
            calls.append(rays.shape)
            return {"t_hit": np.full(rays.shape[:-1], 2., np.float32)}

    shared = SimpleNamespace(mesh_path=mesh, mesh_sha256=selection.sha256(mesh), scene=Scene())
    monkeypatch.setattr(selection, "load_cameras", lambda *args: ([camera], [np.ones((32, 32))], 2., [depth_path], False))
    monkeypatch.setattr(selection, "load_mesh_and_samples", lambda *args: (np.empty((3, 3)), np.empty((1, 3)), atlas, {}))
    monkeypatch.setattr(selection, "aggregate_evidence", lambda *args: {"evidence_class": np.zeros(1, np.uint8)})
    candidate = {**camera, "id": "virtual_0", "kind": "virtual", "selection_source": "surface_lookat", "anchor_id": "real_0", "stats": {}}
    monkeypatch.setattr(selection, "propose_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(selection, "path_safety", lambda *args: {"passed": True, "reason": None})
    monkeypatch.setattr(selection, "render_coverage", lambda *args: (
        {"indices": np.array([0]), "directions": np.array([0], np.uint8)}, {"visible_fraction": 1.}, {}))

    def forbidden_preview(*args):
        raise AssertionError("Production must not render preview assets")

    monkeypatch.setattr(selection, "save_preview", forbidden_preview)
    output = tmp_path / "output"
    manifest = selection.run_selection(dataset, mesh, output, target_count=1, candidate_limit=1, long_edge=32,
        surface_samples=100, mesh_resources=shared, export_previews=False, detailed_audit=False)
    assert calls == [(32, 32, 6)]  # The necessary real anchor depth, no 512 preview.
    for name in ("previews", "scene_preview.npz", "candidates.json", "candidate_audit.jsonl", "coverage.npz"):
        assert not (output / name).exists()
    for name in ("manifest.json", "receipt.json", "sample_atlas.npz", "real_cameras.json", "virtual_cameras.json", "report.json"):
        assert (output / name).is_file()
    assert "scene_preview_path" not in manifest and "candidates_path" not in manifest
    assert json.loads((output / "report.json").read_text())["selected_count"] == 1
