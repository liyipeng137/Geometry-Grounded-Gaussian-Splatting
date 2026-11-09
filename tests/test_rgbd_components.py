"""RGBD 初始化的真实 Open3D 连通分量与融合/抽点顺序回归。"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_functions(**overrides):
    """隔离相机/CUDA 导入，连通分量操作仍使用真实 Open3D。"""
    path = ROOT / "utils/pd_utils.py"
    names = {"_keep_largest_three_components", "generate_ply_from_rgbd"}
    nodes = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "o3d": o3d, "os": os}
    namespace.update(overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def disconnected_fans(counts, orphan=False):
    """每个扇形内部共享边，不同扇形之间没有拓扑连接；颜色编码原组件编号。"""
    vertices, triangles, colors = [], [], []
    for component, count in enumerate(counts):
        offset = len(vertices)
        center = np.array([10.0 * component + 2.0, 2.0, 0.0])
        angles = np.linspace(0, 1.5 * np.pi, count + 1)
        ring = center + np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        vertices.extend([center, *ring])
        triangles.extend([offset, offset + index + 1, offset + index + 2] for index in range(count))
        colors.extend([[component / 5.0, 0.3, 0.7]] * (count + 2))
    if orphan:
        vertices.append([999.0, 999.0, 999.0])
        colors.append([1.0, 0.0, 0.0])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices).reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32).reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors).reshape(-1, 3))
    return mesh


def component_ids(mesh):
    return set(np.rint(np.asarray(mesh.vertex_colors)[:, 0] * 5).astype(int))


def test_keeps_three_largest_by_triangle_count_not_area():
    mesh = disconnected_fans([2, 7, 4, 1, 5], orphan=True)
    # 最少面的组件反而具有最大的面积，防止实现误按面积排序。
    vertices = np.asarray(mesh.vertices)
    vertices[:4] *= 100
    result = load_functions()["_keep_largest_three_components"](mesh)
    assert result is mesh
    assert component_ids(mesh) == {1, 2, 4}
    assert sorted(mesh.cluster_connected_triangles()[1], reverse=True) == [7, 5, 4]
    assert len(mesh.vertices) == 22
    assert len(mesh.vertex_colors) == len(mesh.vertices)
    # remove_unreferenced_vertices 必须同时压缩坐标和颜色，不能遗留已删除碎片的点。
    np.testing.assert_array_equal(np.unique(np.asarray(mesh.triangles)), np.arange(22))
    assert np.asarray(mesh.vertices).max() < 100


@pytest.mark.parametrize("counts", [(3,), (3, 2), (3, 2, 1)])
def test_at_most_three_components_keep_all_and_preserve_vertex_colors(counts):
    mesh = disconnected_fans(counts, orphan=True)
    expected_vertices = np.asarray(mesh.vertices)[:-1].copy()
    expected_colors = np.asarray(mesh.vertex_colors)[:-1].copy()
    load_functions()["_keep_largest_three_components"](mesh)
    assert len(mesh.triangles) == sum(counts)
    np.testing.assert_array_equal(np.asarray(mesh.vertices), expected_vertices)
    np.testing.assert_array_equal(np.asarray(mesh.vertex_colors), expected_colors)


def test_equal_sizes_keep_first_three_components_stably():
    for _ in range(2):
        mesh = disconnected_fans([2, 2, 2, 2])
        load_functions()["_keep_largest_three_components"](mesh)
        assert component_ids(mesh) == {0, 1, 2}
        assert len(mesh.triangles) == 6


def test_empty_surface_does_not_fall_back_to_unfiltered_points():
    mesh = disconnected_fans([], orphan=True)
    with pytest.raises(RuntimeError):
        load_functions()["_keep_largest_three_components"](mesh)


@pytest.mark.parametrize("budget", [5, 100])
def test_fuses_all_frames_then_filters_and_samples_retained_vertices(tmp_path, budget):
    calls = []
    mesh = disconnected_fans([2, 7, 4, 1, 5], orphan=True)
    original_vertices = np.asarray(mesh.vertices).copy()
    original_colors = np.asarray(mesh.vertex_colors).copy()
    retained = np.isin(np.rint(original_colors[:, 0] * 5).astype(int), [1, 2, 4])
    expected_vertices, expected_colors = original_vertices[retained], original_colors[retained]

    class Volume:
        def integrate(self, *args):
            calls.append("integrate")

        def extract_triangle_mesh(self):
            assert calls == ["integrate", "integrate"]
            calls.append("extract_mesh")
            return mesh

        def extract_point_cloud(self):
            raise AssertionError("Intermediate-frame point clouds must not enter initialization")

    volume = Volume()
    fake_open3d = SimpleNamespace(
        geometry=o3d.geometry, utility=o3d.utility, io=o3d.io,
        pipelines=SimpleNamespace(integration=SimpleNamespace(
            ScalableTSDFVolume=lambda **kwargs: volume,
            TSDFVolumeColorType=o3d.pipelines.integration.TSDFVolumeColorType,
        )),
    )
    fake_cv2 = SimpleNamespace(
        IMREAD_COLOR=1, COLOR_BGR2RGB=2,
        imread=lambda *args: np.zeros((2, 3, 3), dtype=np.uint8),
        cvtColor=lambda image, code: image,
    )
    cameras = [SimpleNamespace(
        uid=index, image_path=f"frame_{index}.png", image_name=f"frame_{index}.png",
        colmap_image_name=f"frame_{index}.png", R=np.eye(3), T=np.zeros(3),
    ) for index in range(2)]
    functions = load_functions(
        o3d=fake_open3d, cv2=fake_cv2,
        _load_tsdf_depth=lambda *args: np.ones((2, 3), dtype=np.float32),
        _frame_intrinsic=lambda *args, **kwargs: (
            o3d.camera.PinholeCameraIntrinsic(3, 2, 2.0, 2.0, 1.0, 0.5),
            (2.0, 2.0, 1.0, 0.5),
        ),
        getWorld2View2=lambda *args: np.eye(4),
    )
    output = tmp_path / "initialization.ply"
    functions["generate_ply_from_rgbd"](cameras, budget, str(output), source_path=str(tmp_path))
    assert calls == ["integrate", "integrate", "extract_mesh"]
    cloud = o3d.io.read_point_cloud(str(output))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    assert len(points) == min(budget, len(expected_vertices))
    assert len(np.unique(points, axis=0)) == len(points)
    for point, color in zip(points, colors):
        distances = np.linalg.norm(expected_vertices - point, axis=1)
        nearest = distances.argmin()
        assert distances[nearest] < 1e-12
        # PLY 颜色量化到 uint8，几何和颜色仍必须取自同一个保留顶点。
        np.testing.assert_allclose(color, expected_colors[nearest], atol=1 / 255)
