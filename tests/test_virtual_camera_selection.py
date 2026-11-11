"""Physical contracts for virtual camera selection, independent of training/CUDA.

The fixtures use measured planes and explicit camera frames.  They exercise
coordinate, visibility and sampling decisions rather than reproducing selector
formulae.  The exact-triangle test is optional locally and runs with Open3D on
the remote validation host.
"""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import virtual_camera_selection as vcs  # noqa: E402


def camera(name="camera", center=(0.0, 0.0, 0.0), rotation=None, width=9, height=9):
    pose = np.eye(4)
    pose[:3, 3] = center
    if rotation is not None:
        pose[:3, :3] = rotation
    return {
        "id": name,
        "name": name,
        "kind": "real",
        "selection_source": "real",
        "width": width,
        "height": height,
        "K": [[4.0, 0.0, width / 2], [0.0, 4.0, height / 2], [0.0, 0.0, 1.0]],
        "c2w": pose.tolist(),
        "target": (pose[:3, 3] + pose[:3, 2]).tolist(),
        "anchor_id": name,
        "stats": {},
    }


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def test_camera_rays_use_colmap_pixel_centers_and_full_intrinsics():
    cam = camera(center=(2.0, -3.0, 4.0), width=2, height=2)
    cam["K"] = [[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]
    rays = vcs.camera_rays(cam)
    assert rays.shape == (2, 2, 6)
    np.testing.assert_allclose(rays[..., :3], np.broadcast_to([2.0, -3.0, 4.0], (2, 2, 3)))
    # Pixel (0, 0) is centered exactly on this deliberately off-center principal point.
    np.testing.assert_allclose(rays[0, 0, 3:] / rays[0, 0, 5], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rays[1, 1, 3:] / rays[1, 1, 5], [0.5, 0.25, 1.0])


def test_camera_rays_rotate_into_world_without_translating_directions():
    reference = camera(width=2, height=2)
    turned = camera(center=(8.0, -5.0, 3.0), rotation=rotation_z(np.pi / 2), width=2, height=2)
    local = vcs.camera_rays(reference)[..., 3:]
    world = vcs.camera_rays(turned)[..., 3:]
    np.testing.assert_allclose(world, local @ rotation_z(np.pi / 2).T, atol=1e-7)


def test_depth_evidence_compares_camera_z_not_euclidean_range():
    cam = camera(width=7, height=7)
    cam["K"] = [[2.0, 0.0, 2.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]]
    # Interior pixel (3, 3) sees (1, .5, 2): range is sqrt(5.25), camera-Z is 2.
    # A plane supplies the complete neighboring stencil needed to trust this depth.
    depth = np.full((7, 7), 2.0)
    evidence = vcs.depth_evidence(np.array([[1.0, 0.5, 2.0]]), cam, depth, scene_scale=10.0)
    assert evidence["measured"].tolist() == [True]
    assert evidence["support"].tolist() == [True]
    assert evidence["free"].tolist() == [False]
    assert evidence["behind"].tolist() == [False]


def test_measured_front_surface_occludes_farther_points_without_contradicting_them():
    cam = camera()
    depth = np.full((9, 9), 3.0)
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 5.0]])
    evidence = vcs.depth_evidence(points, cam, depth, scene_scale=10.0)
    assert evidence["free"].tolist() == [True, False, False]
    assert evidence["support"].tolist() == [False, True, False]
    assert evidence["behind"].tolist() == [False, False, True]


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_depth_is_unknown_not_free_space(invalid):
    evidence = vcs.depth_evidence(
        np.array([[0.0, 0.0, 2.0]]), camera(), np.full((9, 9), invalid), scene_scale=10.0
    )
    for key in ("measured", "support", "free", "behind"):
        assert not bool(evidence[key][0])


def test_out_of_frame_and_behind_camera_points_have_no_depth_vote():
    points = np.array([[100.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    evidence = vcs.depth_evidence(points, camera(), np.full((9, 9), 3.0), scene_scale=10.0)
    for key in ("measured", "support", "free", "behind"):
        assert not np.any(evidence[key])


def test_look_at_is_proper_camera_rotation_with_correct_forward():
    reference = np.asarray(camera(rotation=rotation_z(0.8))["c2w"])
    center = np.array([2.0, -1.0, 0.5])
    target = np.array([-1.0, 3.0, 6.0])
    pose = vcs.look_at_pose(center, target, reference)
    np.testing.assert_allclose(pose[:3, 3], center)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(pose[:3, 2], (target - center) / np.linalg.norm(target - center))
    np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])


def test_look_at_preserves_roll_when_target_is_already_forward():
    reference = np.asarray(camera(rotation=rotation_z(1.1))["c2w"])
    pose = vcs.look_at_pose(np.zeros(3), np.array([0.0, 0.0, 3.0]), reference)
    np.testing.assert_allclose(pose[:3, :3], reference[:3, :3], atol=1e-7)


def test_look_at_is_equivariant_to_scene_rotation_not_a_fixed_world_up():
    reference = np.asarray(camera(rotation=rotation_z(0.4))["c2w"])
    center = np.array([0.2, 0.1, -0.3])
    target = np.array([1.0, -0.7, 2.0])
    # Rotate the entire experiment so that a formerly horizontal direction is vertical.
    scene_rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transformed_reference = reference.copy()
    transformed_reference[:3, :3] = scene_rotation @ reference[:3, :3]
    transformed_reference[:3, 3] = scene_rotation @ reference[:3, 3]
    original = vcs.look_at_pose(center, target, reference)
    transformed = vcs.look_at_pose(scene_rotation @ center, scene_rotation @ target, transformed_reference)
    np.testing.assert_allclose(transformed[:3, :3], scene_rotation @ original[:3, :3], atol=1e-7)


def test_evidence_class_requires_independent_conflict_and_preserves_mixed_case():
    support = np.array([0, 1, 1, 0, 0])
    conflict = np.array([0, 0, 2, 1, 2])
    # One free-space witness is suspicious/mixed, but cannot become hard conflict.
    np.testing.assert_array_equal(vcs.classify_evidence(support, conflict), [0, 1, 2, 2, 3])


def test_camera_groups_distinguish_positions_not_view_names_or_rotation():
    cams = [
        camera("a"),
        camera("b", rotation=rotation_z(1.0)),
        camera("c", center=(0.01, 0.0, 0.0)),
        camera("d", center=(2.0, 0.0, 0.0)),
    ]
    labels = vcs.camera_position_groups(cams, distance=0.1)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_duplicate_camera_position_does_not_create_multiple_free_space_votes():
    cams = [camera(f"copy_{index}") for index in range(5)]
    result = vcs.aggregate_evidence(
        np.array([[0.0, 0.0, 1.0]]), cams, [np.full((9, 9), 3.0) for _ in cams], scene_scale=10.0
    )
    assert int(result["conflict_groups"][0]) == 1
    assert int(result["evidence_class"][0]) == 2


def test_distinct_positions_can_confirm_free_space_conflict():
    cams = [camera("left", center=(-0.8, 0.0, 0.0)), camera("right", center=(0.8, 0.0, 0.0))]
    result = vcs.aggregate_evidence(
        np.array([[0.0, 0.0, 1.0]]), cams, [np.full((9, 9), 3.0) for _ in cams], scene_scale=10.0
    )
    assert int(result["conflict_groups"][0]) == 2
    assert int(result["evidence_class"][0]) == 3


class PlaneScene:
    """Analytic exact plane oracle: paths cannot jump over a zero-thickness wall."""

    def __init__(self, z):
        self.z = float(z)
        self.threads = 1

    def distance(self, points):
        return np.abs(np.asarray(points)[..., 2] - self.z)

    def first_hit(self, origins, directions):
        origins, directions = np.asarray(origins), np.asarray(directions)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (self.z - origins[..., 2]) / directions[..., 2]
        return np.where((distance >= 0) & np.isfinite(distance), distance, np.inf)

    def cast(self, rays):
        rays = np.asarray(rays)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (self.z - rays[..., 2]) / rays[..., 5]
        distance = np.where((distance >= 0) & np.isfinite(distance), distance, np.inf)
        normals = np.zeros(rays.shape[:-1] + (3,))
        normals[..., 2] = -1.0
        return {"t_hit": distance, "primitive_normals": normals, "primitive_ids": np.zeros(distance.shape, np.uint32)}


def test_safe_endpoints_do_not_allow_path_to_cross_thin_wall():
    scene = PlaneScene(2.12345)
    start, end = np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 4.0])
    assert min(scene.distance(np.stack([start, end]))) > 1.0
    result = vcs.path_safety(start, end, scene, [camera()], [np.full((9, 9), 10.0)], 10.0)
    assert not result["passed"]


def test_wall_beyond_camera_path_endpoint_does_not_block_path():
    result = vcs.path_safety(
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 1.5]),
        PlaneScene(6.0),
        [camera()],
        [np.full((9, 9), 10.0)],
        10.0,
    )
    assert result["passed"], result


def test_nearby_parallel_but_occluded_atlas_surface_is_not_visible_coverage():
    # The complete mesh front plane is z=2. The only atlas sample happens to lie
    # on a parallel back layer only .05 away, closer than the coarse atlas radius.
    # A nearest-neighbor hit association must not turn that occluded point visible.
    xyz = np.array([[0.0, 0.0, 2.05]])
    atlas = {
        "xyz": xyz,
        "normal_world": np.array([[0.0, 0.0, -1.0]]),
        "face_id": np.array([1]),
        "evidence_class": np.array([1], dtype=np.uint8),
        "area_weight": np.ones(1),
        "patch_id": np.zeros(1, dtype=np.int32),
    }
    coverage, _, _ = vcs.render_coverage(PlaneScene(2.0), camera(), atlas, cKDTree(xyz), scene_scale=10.0)
    assert len(coverage["indices"]) == 0


def test_exact_front_surface_still_receives_visible_coverage():
    xyz = np.array([[0.0, 0.0, 2.0]])
    atlas = {
        "xyz": xyz,
        "normal_world": np.array([[0.0, 0.0, -1.0]]),
        "face_id": np.array([0]),
        "evidence_class": np.array([1], dtype=np.uint8),
        "area_weight": np.ones(1),
        "patch_id": np.zeros(1, dtype=np.int32),
    }
    coverage, stats, _ = vcs.render_coverage(PlaneScene(2.0), camera(), atlas, cKDTree(xyz), scene_scale=10.0)
    np.testing.assert_array_equal(coverage["indices"], [0])
    assert stats["covered_surface_samples"] == 1


def test_actual_triangle_mesh_keeps_camera_z_and_camera_facing_face_normal(tmp_path):
    pytest.importorskip("open3d", reason="Exact triangle backend is validated on the remote host")
    from PIL import Image

    # Camera looks along world +X; the plane's actual triangle normal is world +X.
    vertices = np.array([[2.0, -3.0, -3.0], [2.0, 3.0, -3.0], [2.0, 3.0, 3.0], [2.0, -3.0, 3.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    cam = camera(rotation=rotation)
    scene = vcs.MeshScene(vertices, faces, threads=1)
    rendered = scene.cast(vcs.camera_rays(cam))
    assert rendered["t_hit"][2, 3] == pytest.approx(2.0, abs=1e-6)
    np.testing.assert_allclose(rendered["primitive_normals"][4, 4], [1.0, 0.0, 0.0], atol=1e-6)
    (tmp_path / "previews").mkdir()
    vcs.save_preview(tmp_path, cam, rendered, scene_scale=10.0)
    normal = np.asarray(Image.open(tmp_path / cam["preview"]["normal"]))[4, 4]
    # Camera-space normal is -Z after facing the viewer, independent of world axes.
    np.testing.assert_allclose(normal, [128, 128, 0], atol=1)


def test_actual_triangle_mesh_first_hit_rejects_back_surface_visibility():
    pytest.importorskip("open3d", reason="Exact triangle backend is validated on the remote host")
    vertices = np.array([
        [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0],
        [-1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [1.0, 1.0, 3.0], [-1.0, 1.0, 3.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    scene = vcs.MeshScene(vertices, faces, threads=1)
    hit = scene.first_hit(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]))[0]
    assert hit == pytest.approx(2.0)
    assert hit < 3.0  # The query point on the rear surface is occluded.


def selection_inputs(sets):
    size = 1 + max((index for indices in sets for index in indices), default=-1)
    atlas = {
        "xyz": np.column_stack((np.arange(size), np.zeros(size), np.full(size, 3.0))),
        "area_weight": np.ones(size),
        "patch_id": np.arange(size),
        "evidence_class": np.ones(size, dtype=np.uint8),
    }
    candidates, coverage = [], []
    for index, indices in enumerate(sets):
        cam = camera(f"virtual_{index}", center=(0.05 * index, 0.0, 0.0))
        cam.update(kind="virtual", selection_source="trajectory_offset", anchor_id="real_0")
        candidates.append(cam)
        coverage.append({"indices": np.asarray(indices, dtype=np.int64), "directions": np.zeros(len(indices), dtype=np.uint8)})
    return candidates, coverage, atlas


def test_greedy_selection_adds_unseen_surface_instead_of_redundant_view():
    candidates, coverage, atlas = selection_inputs([[0, 1, 2], [0], [3]])
    selected, _ = vcs.greedy_select(candidates, coverage, atlas, target_count=2)
    assert list(selected) == [0, 2]


def test_target_160_is_upper_bound_not_permission_to_invent_or_relax_candidates():
    candidates, coverage, atlas = selection_inputs([[0], [1], [2]])
    original = deepcopy(candidates)
    selected, _ = vcs.greedy_select(candidates, coverage, atlas, target_count=160)
    assert len(selected) <= 3
    assert len(set(selected)) == len(selected)
    # Recording selection diagnostics is allowed; camera geometry/identity is fixed.
    assert [{key: value for key, value in cam.items() if key != "stats"} for cam in candidates] == [
        {key: value for key, value in cam.items() if key != "stats"} for cam in original
    ]


def test_selection_is_deterministic_and_enforces_hard_count_limit():
    candidates, coverage, atlas = selection_inputs([[index] for index in range(180)])
    first, _ = vcs.greedy_select(candidates, coverage, atlas, target_count=160)
    second, _ = vcs.greedy_select(candidates, coverage, atlas, target_count=160)
    assert list(first) == list(second)
    assert len(first) <= 160
    assert len(set(first)) == len(first)


def test_empty_candidate_pool_remains_empty():
    candidates, coverage, atlas = selection_inputs([])
    selected, _ = vcs.greedy_select(candidates, coverage, atlas, target_count=160)
    assert len(selected) == 0


def test_unknown_surface_is_eligible_but_hard_conflict_cannot_fill_target():
    candidates, coverage, atlas = selection_inputs([[0], [1]])
    atlas["evidence_class"] = np.array([0, 3], dtype=np.uint8)
    selected, report = vcs.greedy_select(candidates, coverage, atlas, target_count=160)
    assert list(selected) == [0]
    assert report["selected_count"] == 1


def test_small_candidate_budget_is_hard_bound_even_when_trajectory_pool_is_larger():
    cams = [camera(f"real_{index}", center=(0.01 * index, 0.0, 0.0)) for index in range(8)]
    for cam in cams:
        cam["stats"]["median_real_depth"] = 3.0
    atlas = {
        "xyz": np.array([[0.0, 0.0, 3.0]]),
        "normal_world": np.array([[0.0, 0.0, -1.0]]),
        "patch_id": np.zeros(1, dtype=np.int32),
        "evidence_class": np.ones(1, dtype=np.uint8),
    }
    candidates = vcs.propose_candidates(cams, atlas, scene_scale=3.0, candidate_limit=5, seed=7)
    assert len(candidates) <= 5
    repeated = vcs.propose_candidates(cams, atlas, scene_scale=3.0, candidate_limit=5, seed=7)
    assert candidates == repeated
    pose_keys = [tuple(np.round(np.asarray(cam["c2w"]).ravel(), 7)) for cam in candidates]
    assert len(set(pose_keys)) == len(pose_keys)


@pytest.mark.parametrize("teacher_depth, expected_radius", [(2.0, 0.16), (None, 0.08)])
def test_offset_radius_prefers_teacher_depth_without_relabeling_real_statistics(teacher_depth, expected_radius):
    cam = camera("real_anchor")
    cam["stats"].update(median_real_depth=1.0, median_teacher_depth=teacher_depth)
    original = deepcopy(cam)
    _, _, atlas = selection_inputs([[0]])
    atlas["normal_world"] = np.array([[0.0, 0.0, -1.0]])
    candidates = vcs.propose_candidates([cam], atlas, scene_scale=10.0, candidate_limit=32, seed=0,
                                        trajectory_offset_fraction=.08)
    offsets = [candidate for candidate in candidates if candidate["selection_source"] == "trajectory_offset"]
    assert len(offsets) == 4
    origin = np.asarray(cam["c2w"])[:3, 3]
    distances = [np.linalg.norm(np.asarray(candidate["c2w"])[:3, 3] - origin) for candidate in offsets]
    np.testing.assert_allclose(distances, expected_radius, rtol=0, atol=1e-12)
    assert cam == original


def test_larger_offset_changes_only_trajectory_centers_and_defaults_to_sixteen_percent():
    cams = [camera("a", center=(.2, -.3, 0), rotation=rotation_z(.4)),
            camera("b", center=(.7, -.2, 0), rotation=rotation_z(.6))]
    for cam, depth in zip(cams, (1.0, 4.0)):
        cam["stats"].update(median_real_depth=2., median_teacher_depth=depth)
    _, _, atlas = selection_inputs([[0]])
    atlas["normal_world"] = np.array([[0., 0., -1.]])
    default = vcs.propose_candidates(cams, atlas, 3., 1000, 0)
    old = vcs.propose_candidates(cams, atlas, 3., 1000, 0, trajectory_offset_fraction=.08)
    new = vcs.propose_candidates(cams, atlas, 3., 1000, 0, trajectory_offset_fraction=.16)
    assert default == new
    assert [row["id"] for row in old] == [row["id"] for row in new]
    anchors = {cam["id"]: np.asarray(cam["c2w"])[:3, 3] for cam in cams}
    assert {row["selection_source"] for row in old} == {
        "trajectory_offset", "trajectory_interpolation", "surface_lookat"}
    for before, after in zip(old, new):
        if before["selection_source"] != "trajectory_offset":
            assert before == after
            continue
        a, b = np.asarray(before["c2w"]), np.asarray(after["c2w"])
        center = anchors[before["anchor_id"]]
        np.testing.assert_allclose(b[:3, 3] - center, 2 * (a[:3, 3] - center), atol=1e-12)
        np.testing.assert_array_equal(a[:3, :3], b[:3, :3])
        assert before["K"] == after["K"]


@pytest.mark.parametrize("value", [0, -.1, float("nan"), float("inf"), 1.1])
def test_invalid_offset_fraction_fails_before_writing_selection(value, tmp_path):
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="trajectory_offset_fraction"):
        vcs.run_selection(tmp_path, tmp_path / "mesh.ply", output, trajectory_offset_fraction=value)
    assert not output.exists()
