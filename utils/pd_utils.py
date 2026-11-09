import os
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2

from utils.graphics_utils import fov2focal, getWorld2View2


def _intrinsic_from_sparse(intr):
    if intr.model == "SIMPLE_PINHOLE":
        fx = fy = float(intr.params[0])
        cx, cy = float(intr.params[1]), float(intr.params[2])
    elif intr.model in ("PINHOLE", "OPENCV"):
        fx, fy = float(intr.params[0]), float(intr.params[1])
        cx, cy = float(intr.params[2]), float(intr.params[3])
    else:
        raise ValueError(f"Unsupported camera model for RGBD init ply: {intr.model}")
    return fx, fy, cx, cy, int(intr.width), int(intr.height)


def _frame_intrinsic(cam, cam_intrinsics, color_hw, pixel_center_convention="colmap_half_integer"):
    intr = None if not cam_intrinsics else cam_intrinsics.get(cam.uid)
    if intr is None and cam_intrinsics and len(cam_intrinsics) == 1:
        intr = next(iter(cam_intrinsics.values()))
    if intr is not None:
        fx, fy, cx, cy, width, height = _intrinsic_from_sparse(intr)
    else:
        fx = float(fov2focal(cam.FovX, cam.width))
        fy = float(fov2focal(cam.FovY, cam.height))
        cx, cy = cam.width * 0.5, cam.height * 0.5
        width, height = int(cam.width), int(cam.height)
    image_h, image_w = color_hw
    if image_w != width or image_h != height:
        sx = image_w / float(width)
        sy = image_h / float(height)
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy
    if pixel_center_convention == "colmap_half_integer":
        cx -= 0.5
        cy -= 0.5
    elif pixel_center_convention != "array_integer":
        raise ValueError(f"Unsupported RGBD pixel center convention: {pixel_center_convention}")
    return o3d.camera.PinholeCameraIntrinsic(image_w, image_h, fx, fy, cx, cy), (fx, fy, cx, cy)


def resolve_rgbd_depth_directory(source_path, depth_prior_dir="depth"):
    """解析初始化深度目录；相对路径以数据集为根，绝对路径直接使用。"""
    if not depth_prior_dir or not str(depth_prior_dir).strip():
        raise ValueError("RGBD initialization requires a nonempty depth_prior_dir")
    directory = Path(depth_prior_dir)
    if not directory.is_absolute():
        directory = Path(source_path) / directory
    return directory.resolve()


def _load_tsdf_depth(source_path, image_name, depth_prior_dir="depth"):
    # 保留完整 COLMAP image.name 的子目录，只替换扩展名；不按 basename 配对。
    directory = resolve_rgbd_depth_directory(source_path, depth_prior_dir)
    path = directory / Path(image_name).with_suffix(".npy")
    if not path.is_file():
        return None
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2 or depth.dtype.kind != "f":
        raise ValueError(f"Aligned depth must be an HxW floating camera-Z NPY: {path}")
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)


def _keep_largest_three_components(mesh):
    """按共享边连通的三角面数量保留前三个分量，原地清理其余面和孤立顶点。"""
    if not mesh.has_triangles():
        raise RuntimeError("RGBD initialization produced no TSDF surface triangles")
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    # 同面数时保留原组件顺序，避免初始化因排序不稳定而变化。
    keep = np.argsort(-counts, kind="stable")[:3]
    before_vertices = len(mesh.vertices)
    mesh.remove_triangles_by_mask(~np.isin(np.asarray(labels), keep))
    mesh.remove_unreferenced_vertices()
    print(
        f"RGBD components: {len(counts)} -> {len(keep)}, "
        f"triangles: {int(counts.sum())} -> {len(mesh.triangles)}, "
        f"vertices: {before_vertices} -> {len(mesh.vertices)}, "
        f"kept triangle counts: {counts[keep].tolist()}"
    )
    return mesh


def generate_ply_from_rgbd(
    train_cam_infos,
    num_points,
    ply_path,
    cam_intrinsics=None,
    voxel_length=0.02,
    sdf_trunc=0.06,
    depth_min=0.05,
    depth_max=50.0,
    conf_threshold=200.0 / 255.0,
    source_path=None,
    pixel_center_convention="array_integer",
    depth_prior_dir="depth",
):
    """从真实 RGB-D 融合初始化点云，使用与训练监督相同的深度目录。

    ``depth_prior_dir`` 支持数据集内相对目录或绝对目录；默认 ``depth``
    保持原有入口。全部融合后先保留最大三个 mesh 连通分量，再抽取初始化点。
    """
    print("Generating dense init ply from RGBD ...")
    if len(train_cam_infos) == 0:
        raise ValueError("No cameras provided for RGBD point cloud generation.")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    printed_cameras = set()
    dropped_cameras = 0

    source_path = source_path or os.path.dirname(ply_path)
    for train_cam in train_cam_infos:
        image_path = train_cam.image_path
        image_name = train_cam.colmap_image_name or train_cam.image_name
        # confidence_path = os.path.join(source_path, "confidence", f"{image_name}.png")

        depth_m = _load_tsdf_depth(source_path, image_name, depth_prior_dir)
        if depth_m is None:
            dropped_cameras += 1
            continue

        color_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_np is None:
            dropped_cameras += 1
            continue
        color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        if depth_m.shape[:2] != color_np.shape[:2]:
            raise ValueError(f"RGB/depth must share the original camera grid: {image_name}")
        valid = np.isfinite(depth_m) & (depth_m > depth_min) & (depth_m < depth_max)
        if not np.any(valid):
            dropped_cameras += 1
            continue
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
        color = o3d.geometry.Image(color_np)
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_m))

        camera_intrinsics, (fx, fy, cx, cy) = _frame_intrinsic(
            train_cam, cam_intrinsics, color_np.shape[:2], pixel_center_convention=pixel_center_convention
        )
        if train_cam.uid not in printed_cameras:
            print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, size: {color_np.shape[1]}x{color_np.shape[0]}")
            printed_cameras.add(train_cam.uid)

        # confidence_raw = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)
        # if confidence_raw is not None:
        #     if confidence_raw.ndim == 3:
        #         confidence_raw = confidence_raw[..., 0]
        #     confidence = confidence_raw.astype(np.float32)
        #     if confidence.max() > 1.0:
        #         confidence = confidence / 255.0
        #     conf_mask = confidence > conf_threshold
        #     depth_u16 = np.where(conf_mask, depth_u16, 0).astype(np.uint16)
        # depth cut 0~10.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=float(depth_max),
            convert_rgb_to_intensity=False,
        )

        w2c = getWorld2View2(train_cam.R, train_cam.T)
        c2w = np.linalg.inv(w2c)
        volume.integrate(rgbd, camera_intrinsics, np.linalg.inv(c2w))

    # 连通性必须依据最终融合结果；不再累积各中间 TSDF 快照中的浮点。
    mesh = _keep_largest_three_components(volume.extract_triangle_mesh())
    pick_num = min(num_points, len(mesh.vertices))
    idx = np.random.choice(len(mesh.vertices), size=pick_num, replace=False)
    points = np.asarray(mesh.vertices)[idx]
    colors = np.asarray(mesh.vertex_colors)[idx]

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(ply_path, out_pcd):
        raise RuntimeError(f"Failed to write RGBD initialization: {ply_path}")
    print(f"Dropped cameras without usable depth: {dropped_cameras}")
    print(f"RGBD init ply saved to {ply_path}, points={points.shape[0]}")
