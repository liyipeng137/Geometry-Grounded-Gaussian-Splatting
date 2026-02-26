import os
import numpy as np
import open3d as o3d
import cv2

from utils.graphics_utils import getWorld2View2


def _parse_colmap_intrinsics(cam_intrinsics):
    intr = next(iter(cam_intrinsics.values()))
    if intr.model == "SIMPLE_PINHOLE":
        fx = intr.params[0]
        fy = intr.params[0]
        cx = intr.params[1]
        cy = intr.params[2]
    elif intr.model in ("PINHOLE", "OPENCV"):
        fx = intr.params[0]
        fy = intr.params[1]
        cx = intr.params[2]
        cy = intr.params[3]
    else:
        raise ValueError(f"Unsupported camera model for RGBD init ply: {intr.model}")
    return fx, fy, cx, cy


def generate_ply_from_rgbd(
    train_cam_infos,
    num_points,
    ply_path,
    cam_intrinsics=None,
    depth_trunc=4.0,
    conf_threshold=200.0 / 255.0,
):
    print("Generating dense init ply from RGBD ...")
    if len(train_cam_infos) == 0:
        raise ValueError("No cameras provided for RGBD point cloud generation.")
    if cam_intrinsics is None:
        raise ValueError("cam_intrinsics is required for RGBD point cloud generation.")

    train_example = train_cam_infos[0]
    w, h = train_example.width, train_example.height
    fx, fy, cx, cy = _parse_colmap_intrinsics(cam_intrinsics)
    samples_per_frame = max((num_points + len(train_cam_infos) - 1) // len(train_cam_infos), 1)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    points_list = []
    colors_list = []
    depth_acc = 1000.0
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    source_path = os.path.dirname(ply_path)
    for train_cam in train_cam_infos:
        image_path = train_cam.image_path
        image_name = train_cam.image_name
        depth_path = os.path.join(source_path, "depths", f"{image_name}.png")
        confidence_path = os.path.join(source_path, "confidence", f"{image_name}.png")

        color_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_np is None:
            continue
        color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(color_np)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        if depth_raw.dtype == np.uint16:
            depth_u16 = depth_raw
        else:
            depth_u16 = np.clip(depth_raw.astype(np.float32), 0.0, 65535.0).astype(np.uint16)

        confidence_raw = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)
        if confidence_raw is not None:
            if confidence_raw.ndim == 3:
                confidence_raw = confidence_raw[..., 0]
            confidence = confidence_raw.astype(np.float32)
            if confidence.max() > 1.0:
                confidence = confidence / 255.0
            conf_mask = confidence > conf_threshold
            depth_u16 = np.where(conf_mask, depth_u16, 0).astype(np.uint16)

        depth = o3d.geometry.Image(depth_u16)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=depth_acc,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        w2c = getWorld2View2(train_cam.R, train_cam.T)
        c2w = np.linalg.inv(w2c)
        volume.integrate(rgbd, camera_intrinsics, np.linalg.inv(c2w))

        pcd = volume.extract_point_cloud()
        if len(pcd.points) == 0:
            continue

        pick_num = min(samples_per_frame, len(pcd.points))
        idx = np.random.choice(len(pcd.points), size=pick_num, replace=False)
        points_list.append(np.asarray(pcd.points)[idx])
        colors_list.append(np.asarray(pcd.colors)[idx])

    if not points_list:
        raise RuntimeError("Failed to generate RGBD init ply: no valid RGBD samples were integrated.")

    points = np.concatenate(points_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_path, out_pcd)
    print(f"RGBD init ply saved to {ply_path}, points={points.shape[0]}")
