import os
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


def _frame_intrinsic(cam, cam_intrinsics, color_hw, pixel_center_convention="array_integer"):
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


def _load_tsdf_depth(source_path, image_name):
    depth_dir = os.path.join(source_path, "depth")
    npy_path = os.path.join(depth_dir, f"{image_name}.npy")
    if os.path.exists(npy_path):
        depth_np = np.squeeze(np.load(npy_path)).astype(np.float32, copy=False)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        return np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    png_path = os.path.join(depth_dir, f"{image_name}.png")
    if not os.path.exists(png_path):
        return None
    depth_raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    return depth_raw.astype(np.float32) / 1000.0


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
):
    print("Generating dense init ply from RGBD ...")
    if len(train_cam_infos) == 0:
        raise ValueError("No cameras provided for RGBD point cloud generation.")
    samples_per_frame = max((num_points + len(train_cam_infos) - 1) // len(train_cam_infos), 1)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    points_list = []
    colors_list = []
    printed_cameras = set()
    dropped_cameras = 0

    source_path = source_path or os.path.dirname(ply_path)
    for train_cam in train_cam_infos:
        image_path = train_cam.image_path
        image_name = train_cam.image_name
        # confidence_path = os.path.join(source_path, "confidence", f"{image_name}.png")

        depth_m = _load_tsdf_depth(source_path, image_name)
        if depth_m is None:
            dropped_cameras += 1
            continue

        color_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_np is None:
            dropped_cameras += 1
            continue
        color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        if depth_m.shape[:2] != color_np.shape[:2]:
            depth_m = cv2.resize(depth_m, (color_np.shape[1], color_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth_m) & (depth_m > depth_min) & (depth_m < depth_max)
        if not np.any(valid):
            dropped_cameras += 1
            continue
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
        depth_u16 = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        color = o3d.geometry.Image(color_np)
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_u16))

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
            depth_scale=1000.0,
            depth_trunc=float(depth_max),
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
    print(f"Dropped cameras without usable depth: {dropped_cameras}")
    print(f"RGBD init ply saved to {ply_path}, points={points.shape[0]}")
