#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from scene.custom_initialization import load_custom_initialization
from scene.train_split import calibration_fingerprint, load_excluded_image_names, partition_cameras, training_camera_fingerprint
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.pd_utils import generate_ply_from_rgbd
from utils.sh_utils import SH2RGB

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array = None
    colmap_image_name: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_pfm(filename: str):
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf": # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ("PINHOLE", "OPENCV"):
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              colmap_image_name=extr.name)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    names = vertices.data.dtype.names or ()
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if {'red', 'green', 'blue'}.issubset(names):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
    if {'nx', 'ny', 'nz'}.issubset(names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchOpen3DPly(path):
    plydata = o3d.io.read_point_cloud(path)
    positions = np.asarray(plydata.points)
    colors = np.asarray(plydata.colors)
    if plydata.has_normals():
        normals = np.asarray(plydata.normals)
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, use_rgbd_init_ply=False,
                       train_exclude_images="", init_cache_dir=None,
                       allow_train_split_expansion=False, custom_init_ply="",
                       custom_init_metadata=""):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    excluded_names, split_source = load_excluded_image_names(train_exclude_images)
    custom_initialization = bool(custom_init_ply or custom_init_metadata)
    if custom_initialization and not (custom_init_ply and custom_init_metadata):
        raise ValueError("Custom initialization requires both point cloud and metadata")
    strict_split = bool(train_exclude_images) or allow_train_split_expansion or custom_initialization
    if strict_split and eval:
        raise ValueError("Explicit training split contracts cannot be combined with implicit --eval splitting")
    train_cam_infos, test_cam_infos = partition_cameras(cam_infos, excluded_names, eval, llffhold)
    train_fingerprint = training_camera_fingerprint(train_cam_infos)
    camera_calibrations = {}
    if strict_split:
        for extrinsic in cam_extrinsics.values():
            intrinsic = cam_intrinsics[extrinsic.camera_id]
            definition = {
                "width": int(intrinsic.width), "height": int(intrinsic.height),
                "camera_model": intrinsic.model, "camera_params": np.asarray(intrinsic.params).tolist(),
                "world_to_camera_rotation": qvec2rotmat(extrinsic.qvec).tolist(),
                "world_to_camera_translation": np.asarray(extrinsic.tvec).tolist(),
                "pixel_center_convention": "colmap_half_integer",
            }
            camera_calibrations[extrinsic.name] = calibration_fingerprint(definition)
    init_algorithm = "rgbd_colmap_half_integer_v2" if strict_split else "legacy_array_integer"
    init_identity = train_fingerprint + init_algorithm
    if strict_split:
        training_calibrations = {camera.colmap_image_name: camera_calibrations[camera.colmap_image_name] for camera in train_cam_infos}
        init_identity += calibration_fingerprint(training_calibrations)
    init_fingerprint = hashlib.sha256(init_identity.encode("utf-8")).hexdigest()
    if strict_split and not use_rgbd_init_ply and not custom_initialization:
        raise ValueError("Explicit training exclusion requires train-only RGBD initialization; disable_rgbd_init_ply is incompatible")
    if strict_split and init_cache_dir is None:
        raise ValueError("Explicit training exclusion requires an initialization cache directory")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(f'cameras extent: {nerf_normalization["radius"]}')

    pcd = None
    ply_path = None
    initialization_record = None
    if custom_initialization:
        pcd, initialization_record = load_custom_initialization(
            custom_init_ply, custom_init_metadata, path, camera_calibrations,
            [camera.colmap_image_name for camera in train_cam_infos],
            [camera.colmap_image_name for camera in test_cam_infos],
        )
        ply_path = str(Path(custom_init_ply).resolve())
        print(f"Loaded explicit supported-surface cold-start point cloud: {ply_path}, points={len(pcd.points)}")
    if use_rgbd_init_ply and not custom_initialization:
        if strict_split:
            os.makedirs(init_cache_dir, exist_ok=True)
            rgbd_ply_path = os.path.join(init_cache_dir, f"rgbd_init_train_{init_fingerprint[:16]}.ply")
        else:
            rgbd_ply_path = os.path.join(path, "rgbd_init_points3D.ply")
        if not os.path.exists(rgbd_ply_path):
            try:
                generate_ply_from_rgbd(
                    train_cam_infos=train_cam_infos,
                    num_points=200_000,
                    ply_path=rgbd_ply_path,
                    cam_intrinsics=cam_intrinsics,
                    source_path=path,
                    pixel_center_convention="colmap_half_integer" if strict_split else "array_integer",
                )
            except Exception as e:
                if strict_split:
                    raise RuntimeError("Train-only RGBD initialization failed; refusing full-data sparse fallback") from e
                print(f"[Warn] RGBD init ply generation failed, fallback to sparse point cloud. Reason: {e}")
        if os.path.exists(rgbd_ply_path):
            try:
                ply_o3d = o3d.io.read_point_cloud(rgbd_ply_path)
                if not ply_o3d.has_normals():
                    ply_o3d.estimate_normals()
                positions = np.asarray(ply_o3d.points)
                if positions.size == 0:
                    raise ValueError("RGBD initialization point cloud is empty")
                colors = np.asarray(ply_o3d.colors)
                normals = np.asarray(ply_o3d.normals) if ply_o3d.has_normals() else np.zeros_like(positions)
                pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
                ply_path = rgbd_ply_path
                print(f"Loaded RGBD init point cloud: {ply_path}, points={positions.shape[0]}")
            except Exception as e:
                if strict_split:
                    raise RuntimeError("Train-only RGBD cache could not be loaded; refusing full-data sparse fallback") from e
                print(f"[Warn] Failed to load RGBD init point cloud, fallback to sparse point cloud. Reason: {e}")

    if pcd is None:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            try:
                pcd = trimesh.load(ply_path)
                point_id = np.random.choice(np.arange(len(pcd.vertices)), 1200000)
                pcd = BasicPointCloud(points=pcd.vertices[point_id], colors=pcd.colors[point_id][:,:3].astype(np.float32)/255, normals=None)
            except:
                pcd = None

    if strict_split:
        split_record = {
            "schema_version": 2,
            "dataset_root": str(Path(path).resolve()),
            "camera_calibration_fingerprints": camera_calibrations,
            "calibration_fingerprint": calibration_fingerprint(camera_calibrations),
            "source": split_source,
            "train_camera_fingerprint": train_fingerprint,
            "train_image_names": [camera.colmap_image_name for camera in train_cam_infos],
            "test_image_names": [camera.colmap_image_name for camera in test_cam_infos],
            "initialization": initialization_record or {
                "kind": "train_only_rgbd", "ply_path": os.path.abspath(ply_path),
                "algorithm": init_algorithm, "cache_fingerprint": init_fingerprint,
            },
            "scope": "All source cameras are training views; no heldout cameras remain" if not test_cam_infos else "RGB/depth supervision, RGBD initialization, neighbors, pruning and 3D filter exclude test cameras; FFBA poses/K were estimated upstream using the full capture",
        }
        with open(os.path.join(init_cache_dir, "camera_split.json"), "w") as split_file:
            json.dump(split_record, split_file, indent=2)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        fl_x_global = contents.get("fl_x", None)
        fl_y_global = contents.get("fl_y", None)
        w_global = contents.get("w", None)
        h_global = contents.get("h", None)
        cx_global = contents.get("cx", None)
        cy_global = contents.get("cy", None)
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"])
            cam_name = frame["file_path"] 

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, "images", cam_name)
            print(f"image_path: {image_path}")
            image_name = Path(cam_name).stem

            fl_x = frame.get("fl_x", fl_x_global)
            fl_y = frame.get("fl_y", fl_y_global)
            w = frame.get("w", w_global)
            h = frame.get("h", h_global)
            cx, cy = frame.get("cx", cx_global), frame.get("cy", cy_global)

            print(f"fl_x: {fl_x}, fl_y: {fl_y}, w: {w}, h: {h}, cx: {cx}, cy: {cy}")

            fovx = focal2fov(fl_x, w)
            fovy = focal2fov(fl_y, h)
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                            image_path=image_path, image_name=image_name, width=w, height=h, mask=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    train_cam_infos = train_cam_infos
    test_cam_infos = train_cam_infos
    print("train num:", len(train_cam_infos))
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    cam_intrinsics = None
    if True:
        rgbd_ply_path = os.path.join(path, "rgbd_init_points3D.ply")
        if not os.path.exists(rgbd_ply_path):
            try:
                generate_ply_from_rgbd(
                    train_cam_infos=train_cam_infos,
                    num_points=200_000,
                    ply_path=rgbd_ply_path,
                    cam_intrinsics=cam_intrinsics,
                )
            except Exception as e:
                print(f"[Warn] RGBD init ply generation failed, fallback to sparse point cloud. Reason: {e}")
        if os.path.exists(rgbd_ply_path):
            try:
                print(f"Loading RGBD init point cloud: {rgbd_ply_path}")
                ply_o3d = o3d.io.read_point_cloud(rgbd_ply_path)
                if not ply_o3d.has_normals():
                    ply_o3d.estimate_normals()
                positions = np.asarray(ply_o3d.points)
                colors = np.asarray(ply_o3d.colors)
                normals = np.asarray(ply_o3d.normals) if ply_o3d.has_normals() else np.zeros_like(positions)
                pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
                ply_path = rgbd_ply_path
                print(f"Loaded RGBD init point cloud: {ply_path}, points={positions.shape[0]}")
            except Exception as e:
                print(f"[Warn] Failed to load RGBD init point cloud, fallback to sparse point cloud. Reason: {e}")


    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           point_cloud=pcd)
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
