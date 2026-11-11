"""Global surface sampling and real-image visibility contracts, without training."""

import importlib.util
from pathlib import Path

import numpy as np
from plyfile import PlyData
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_genrecon_init.py"
SPEC = importlib.util.spec_from_file_location("prepare_genrecon_init_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def view(index=0, x=0.):
    pose = np.eye(4)
    pose[0, 3] = x
    return {"name": f"frame_{index}.png", "width": 9, "height": 9,
            "K": np.array([[4., 0., 4.5], [0., 4., 4.5], [0., 0., 1.]]),
            "c2w": pose, "image_path": f"unused_{index}.png", "index": index}


def two_triangles():
    vertices = np.array([[0., 0., 2.], [1., 0., 2.], [0., 1., 2.],
                         [10., 0., 2.], [13., 0., 2.], [10., 3., 2.]], np.float32)
    return vertices, np.array([[0, 1, 2], [3, 4, 5]], np.int32)


def test_sampling_is_area_uniform_not_face_or_vertex_density_and_is_repeatable():
    vertices, faces = two_triangles()
    first = MODULE.sample_surface(vertices, faces, 20000, seed=123, block_size=1)
    second = MODULE.sample_surface(vertices, faces, 20000, seed=123, block_size=2)
    points, normals, ids, stats = first
    assert len(points) == 20000
    assert np.mean(ids == 1) == pytest.approx(.9, abs=.01)
    assert stats["total_area"] == pytest.approx(5.)
    for a, b in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(normals, np.broadcast_to([0, 0, 1], normals.shape))
    local_x = np.where(ids == 0, points[:, 0], points[:, 0] - 10)
    limit = np.where(ids == 0, 1., 3.)
    assert (local_x >= 0).all() and (points[:, 1] >= 0).all()
    assert (local_x + points[:, 1] <= limit + 1e-6).all()
    assert (points[:, 2] == 2).all()


def test_zero_area_faces_never_receive_samples():
    vertices, faces = two_triangles()
    faces = np.vstack(([0, 0, 1], faces))
    _, _, ids, stats = MODULE.sample_surface(vertices, faces, 5000, seed=0)
    assert stats["degenerate_faces"] == 1
    assert (ids > 0).all()


def test_colmap_half_integer_centers_map_to_array_indices():
    camera = view()
    points = np.array([[0., 0., 2.], [.5, 0., 2.], [0., .5, 2.], [0., 0., -1.]])
    pixels, valid = MODULE.project_points(points, camera)
    np.testing.assert_allclose(pixels[:3], [[4, 4], [5, 4], [4, 5]])
    np.testing.assert_array_equal(valid, [True, True, True, False])


def test_bilinear_color_uses_all_four_pixels_without_integer_bias():
    image = np.array([[[0, 10, 20], [100, 110, 120]], [[20, 30, 40], [120, 130, 140]]], np.uint8)
    color = MODULE.bilinear_rgb(image, np.array([[.25, .5], [1., 1.]]))
    np.testing.assert_array_equal(color, [[35, 45, 55], [120, 130, 140]])


class PlaneScene:
    def cast(self, rays):
        distance = (2. - rays[..., 2]) / rays[..., 5]
        return {"t_hit": distance, "primitive_ids": np.zeros(rays.shape[:-1], np.uint32)}


def test_occluded_back_surface_cannot_take_foreground_rgb():
    points = np.array([[0., 0., 2.], [0., 0., 3.], [.5, 0., 2.]])
    visible = MODULE.visible_samples(PlaneScene(), points, np.array([0, 1, 0]), np.zeros(3))
    np.testing.assert_array_equal(visible, [True, False, True])


def test_exact_face_identity_rejects_a_very_close_wrong_layer():
    visible = MODULE.visible_samples(PlaneScene(), np.array([[0., 0., 2.00001]]), np.array([1]), np.zeros(3))
    assert not visible[0]


def test_projection_exhausts_more_than_three_candidates_and_keeps_unseen_points():
    class FourthCameraVisible:
        def cast(self, rays):
            result = PlaneScene().cast(rays)
            result["t_hit"] = np.where(np.isclose(rays[..., 0], .3), result["t_hit"], 1.)
            return result

    points = np.array([[0., 0., 2.], [100., 0., 2.]], np.float32)
    normals = np.array([[0, 0, -1.], [0, 0, 1.]], np.float32)
    views = [view(i, i * .1) for i in range(4)]
    def reader(camera):
        return np.full((9, 9, 3), camera["index"] * 40, np.uint8)
    colors, sources, stats = MODULE.project_colors(points, normals, np.array([0, 0]), views, FourthCameraVisible(), reader)
    np.testing.assert_array_equal(colors, [[120, 120, 120], [128, 128, 128]])
    np.testing.assert_array_equal(sources, [3, -1])
    assert stats["checked_rays"] == 4
    assert stats["no_image_candidate_points"] == 1
    assert len(colors) == len(points) == 2


def test_normal_sign_does_not_change_camera_color_priority():
    points = np.array([[0., 0., 2.]], np.float32)
    def reader(camera):
        return np.full((9, 9, 3), 20 + camera["index"] * 40, np.uint8)
    a = MODULE.project_colors(points, np.array([[0, 0, -1.]]), np.array([0]), [view(0), view(1, .3)], PlaneScene(), reader)
    b = MODULE.project_colors(points, np.array([[0, 0, 1.]]), np.array([0]), [view(0), view(1, .3)], PlaneScene(), reader)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_standard_xyz_rgb_normals_ply_round_trip(tmp_path):
    points = np.array([[1., 2., 3.], [-1., 0., .5]], np.float32)
    normals = np.array([[0, 0, 1.], [1., 0, 0]], np.float32)
    colors = np.array([[255, 0, 128], [10, 20, 30]], np.uint8)
    path = tmp_path / "points.ply"
    MODULE.write_point_cloud(path, points, normals, colors)
    loaded = PlyData.read(path)["vertex"].data
    assert loaded.dtype.names == ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")
    np.testing.assert_array_equal(np.column_stack([loaded[k] for k in ("x", "y", "z")]), points)
    np.testing.assert_array_equal(np.column_stack([loaded[k] for k in ("red", "green", "blue")]), colors)


def test_existing_output_is_rejected_before_reading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "keep.txt"
    protected.write_text("user output")
    with pytest.raises(FileExistsError):
        MODULE.prepare(tmp_path / "missing", tmp_path / "missing.ply", output)
    assert protected.read_text() == "user output"


def test_real_image_grid_mismatch_is_rejected():
    points = np.array([[0., 0., 2.]])
    with pytest.raises(ValueError, match="dimensions"):
        MODULE.project_colors(points, np.array([[0., 0., 1.]]), np.array([0]), [view()], PlaneScene(),
                              lambda camera: np.zeros((8, 9, 3), np.uint8))


def test_open3d_exact_mesh_first_hit_occlusion():
    pytest.importorskip("open3d")
    vertices = np.array([[-2., -2., 2.], [2., -2., 2.], [0., 2., 2.],
                         [-2., -2., 3.], [2., -2., 3.], [0., 2., 3.]], np.float32)
    scene = MODULE.MeshScene(vertices, np.array([[0, 1, 2], [3, 4, 5]], np.uint32), 1)
    points = np.array([[0., 0., 2.], [0., 0., 3.]], np.float32)
    np.testing.assert_array_equal(MODULE.visible_samples(scene, points, np.array([0, 1]), np.zeros(3)), [True, False])
