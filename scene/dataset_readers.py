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
import fcntl
import tempfile
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
from scene.train_split import calibration_fingerprint, load_excluded_image_names, partition_cameras, training_camera_fingerprint
from scene.mesh_initialization import MESH_INITIALIZATION_ALGORITHM, validate_mesh_manifest
from scene.ffba_tsdf_initialization import FFBA_TSDF_INITIALIZATION_ALGORITHM, FFBA_TSDF_EVALUATION_SCOPE, validate_ffba_tsdf_manifest
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
    K: np.array = None

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
    cameras = []
    for extr in cam_extrinsics.values():
        name = extr.name
        if Path(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/")) or "\\" in name:
            raise ValueError(f"Unsafe full COLMAP image.name: {name}")
        intr = cam_intrinsics[extr.camera_id]
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = float(intr.params[0])
            cx, cy = map(float, intr.params[1:3])
        elif intr.model == "PINHOLE":
            fx, fy, cx, cy = map(float, intr.params)
        else:
            raise ValueError("Only undistorted SIMPLE_PINHOLE/PINHOLE cameras are supported")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        quaternion = np.asarray(extr.qvec, dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(K).all() or min(fx, fy, intr.width, intr.height) <= 0 or not np.isfinite(quaternion).all() or norm <= 0:
            raise ValueError(f"Invalid camera geometry: {name}")
        R = qvec2rotmat(quaternion / norm).T
        T = np.asarray(extr.tvec, dtype=np.float64)
        if not np.isfinite(T).all():
            raise ValueError(f"Invalid camera translation: {name}")
        image_path = str(Path(images_folder) / name)
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)
        cameras.append(CameraInfo(uid=intr.id, R=R, T=T,
                                  FovY=focal2fov(fy, intr.height), FovX=focal2fov(fx, intr.width),
                                  image=None, image_path=image_path, image_name=name,
                                  width=intr.width, height=intr.height, colmap_image_name=name, K=K))
    return cameras


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

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_record(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def readColmapSceneInfo(path, images, eval, llffhold=8, use_rgbd_init_ply=True,
                       train_exclude_images="", init_cache_dir=None, initialization_kind="rgbd",
                       initialization_ply="", initialization_manifest=""):
    """Use explicit current-split RGBD or a content-bound mesh surface point cloud."""
    if initialization_kind not in ("rgbd", "mesh_surface", "ffba_tsdf"):
        raise ValueError("initialization_kind must be rgbd, mesh_surface or ffba_tsdf")
    if initialization_kind == "rgbd" and (initialization_ply or initialization_manifest):
        raise ValueError("RGBD initialization does not accept alternate PLY/manifest inputs")
    if not use_rgbd_init_ply:
        raise ValueError("Formal COLMAP input requires an explicit initialization contract")
    if init_cache_dir is None:
        raise ValueError("COLMAP initialization requires an owned output cache directory")
    if (Path(path) / "mixed_dataset.json").exists():
        raise ValueError("Mixed/generated ordinary-camera datasets are outside the formal FFBA input contract")
    model = Path(path) / "sparse/0"
    if (model / "cameras.bin").is_file() and (model / "images.bin").is_file():
        cam_extrinsics = read_extrinsics_binary(str(model / "images.bin"))
        cam_intrinsics = read_intrinsics_binary(str(model / "cameras.bin"))
    else:
        cam_extrinsics = read_extrinsics_text(str(model / "images.txt"))
        cam_intrinsics = read_intrinsics_text(str(model / "cameras.txt"))
    infos = sorted(readColmapCameras(cam_extrinsics, cam_intrinsics, str(Path(path) / (images or "images"))),
                   key=lambda camera: camera.colmap_image_name)
    excluded, split_source = load_excluded_image_names(train_exclude_images)
    if train_exclude_images and eval:
        raise ValueError("Explicit train_exclude_images cannot be combined with implicit --eval")
    train, test = partition_cameras(infos, excluded, eval, llffhold)
    calibrations = {}
    assets = {}
    for camera in infos:
        definition = {"width": camera.width, "height": camera.height, "K": camera.K.tolist(),
                      "world_to_camera_rotation": camera.R.T.tolist(), "world_to_camera_translation": camera.T.tolist(),
                      "pixel_center_convention": "colmap_half_integer"}
        calibrations[camera.colmap_image_name] = calibration_fingerprint(definition)
        depth = Path(path) / "depth" / Path(camera.colmap_image_name).with_suffix(".npy")
        assets[camera.colmap_image_name] = {"rgb_sha256": _sha256_file(camera.image_path),
                                          "depth_sha256": _sha256_file(depth) if depth.is_file() else None}
    train_assets = {camera.colmap_image_name: assets[camera.colmap_image_name] for camera in train}
    cache = Path(init_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if initialization_kind == "ffba_tsdf":
        manifest, manifest_sha256 = validate_ffba_tsdf_manifest(
            initialization_ply, initialization_manifest, str(Path(path).resolve()),
            [camera.colmap_image_name for camera in train],
            [camera.colmap_image_name for camera in infos],
        )
        ply_path = Path(initialization_ply).resolve()
        pcd = fetchOpen3DPly(str(ply_path))
        if (not len(pcd.points) or not np.isfinite(pcd.points).all() or
                not np.isfinite(pcd.colors).all() or np.any(pcd.colors < 0) or np.any(pcd.colors > 1)):
            raise ValueError("FFBA TSDF initialization requires finite nonempty XYZ and RGB in [0,1]")
        if len(pcd.points) != manifest["initialization_ply"]["point_count"]:
            raise ValueError("FFBA TSDF manifest point count disagrees with its loaded PLY")
        identity = {
            "algorithm": FFBA_TSDF_INITIALIZATION_ALGORITHM, "manifest_sha256": manifest_sha256,
            "train_camera_fingerprint": training_camera_fingerprint(train),
            "calibration_fingerprint": calibration_fingerprint(calibrations),
            "dataset_content_fingerprint": calibration_fingerprint(assets),
        }
        initialization = {
            "kind": "ffba_tsdf", "algorithm": FFBA_TSDF_INITIALIZATION_ALGORITHM,
            "ply_path": str(ply_path), "ply_sha256": manifest["initialization_ply"]["sha256"],
            "point_count": len(pcd.points), "cache_fingerprint": calibration_fingerprint(identity),
            "proof_path": str(Path(initialization_manifest).resolve()), "proof_sha256": manifest_sha256,
            "initialization_scope": "full_capture", "evaluation_scope": FFBA_TSDF_EVALUATION_SCOPE,
            "geometry_selection": "vertices_only",
            "source_dataset_root": manifest["dataset_root"], "identity": identity,
        }
        if validate_ffba_tsdf_manifest(
            initialization_ply, initialization_manifest, str(Path(path).resolve()),
            [camera.colmap_image_name for camera in train],
            [camera.colmap_image_name for camera in infos],
        )[1] != manifest_sha256:
            raise RuntimeError("FFBA TSDF initialization source changed while loading")
    elif initialization_kind == "mesh_surface":
        manifest, manifest_sha256 = validate_mesh_manifest(
            initialization_ply, initialization_manifest, str(Path(path).resolve()),
            [camera.colmap_image_name for camera in train],
        )
        extension = "bin" if (model / "cameras.bin").is_file() and (model / "images.bin").is_file() else "txt"
        expected_calibration_hashes = {f"{name}.{extension}": _sha256_file(model / f"{name}.{extension}")
                                       for name in ("cameras", "images")}
        preserved_calibration_hashes = {Path(name).name: checksum
                                        for name, checksum in manifest["calibration_sha256"].items()}
        if preserved_calibration_hashes != expected_calibration_hashes:
            raise ValueError("Mesh coloring calibration content differs from the training dataset")
        color_views = {view["name"]: view for view in manifest["views"]}
        for camera in train:
            view = color_views[camera.colmap_image_name]
            if view["image_sha256"] != train_assets[camera.colmap_image_name]["rgb_sha256"]:
                raise ValueError("Mesh coloring RGB differs from the current training camera")
        ply_path = Path(initialization_ply).resolve()
        # A Gaussian checkpoint or triangle mesh is never a point-cloud initializer.
        ply_document = PlyData.read(str(ply_path))
        if ("face" in ply_document and len(ply_document["face"])) or "vertex" not in ply_document:
            raise ValueError("mesh_surface initialization must be a sampled point PLY, not a mesh")
        fields = set(ply_document["vertex"].data.dtype.names or ())
        if not {"x", "y", "z", "red", "green", "blue"}.issubset(fields) or "opacity" in fields:
            raise ValueError("mesh_surface PLY requires explicit XYZ/RGB and cannot be a Gaussian checkpoint")
        pcd = fetchPly(str(ply_path))
        if not len(pcd.points) or not np.isfinite(pcd.points).all() or not np.isfinite(pcd.colors).all():
            raise ValueError("mesh_surface initialization requires finite, nonempty XYZ/RGB")
        if np.any(pcd.colors < 0) or np.any(pcd.colors > 1) or not np.any(pcd.colors > 0):
            raise ValueError("mesh_surface RGB must be in range and cannot trigger random-color fallback")
        if len(pcd.points) != manifest["initialization_ply"]["point_count"]:
            raise ValueError("Mesh initialization manifest point count disagrees with its PLY")
        identity = {
            "algorithm": MESH_INITIALIZATION_ALGORITHM, "manifest_sha256": manifest_sha256,
            "train_camera_fingerprint": training_camera_fingerprint(train),
            "calibration_fingerprint": calibration_fingerprint(calibrations),
            "dataset_content_fingerprint": calibration_fingerprint(assets),
        }
        initialization = {
            "kind": "mesh_surface", "algorithm": MESH_INITIALIZATION_ALGORITHM,
            "ply_path": str(ply_path), "ply_sha256": manifest["initialization_ply"]["sha256"],
            "point_count": len(pcd.points), "cache_fingerprint": calibration_fingerprint(identity),
            "sampling_dataset_root": manifest["dataset_root"],
            "proof_path": str(Path(initialization_manifest).resolve()), "proof_sha256": manifest_sha256,
            "identity": identity,
        }
        # The preserved producer proof is the source of sampling/color provenance.
        if validate_mesh_manifest(initialization_ply, initialization_manifest, str(Path(path).resolve()),
                                  [camera.colmap_image_name for camera in train])[1] != manifest_sha256:
            raise RuntimeError("Mesh initialization source changed while loading")
    else:
        if not any(record["depth_sha256"] for record in train_assets.values()):
            raise ValueError("No training camera has aligned NPY depth; refusing sparse/full-capture initialization fallback")
        algorithm = "rgbd_colmap_half_integer_float32_v3"
        operator_sha256 = _sha256_file(generate_ply_from_rgbd.__code__.co_filename)
        identity = {"algorithm": algorithm, "operator_sha256": operator_sha256,
                    "train_assets": train_assets,
                    "calibrations": {name: calibrations[name] for name in train_assets},
                    "parameters": {"num_points": 200000, "voxel_length": 0.02, "sdf_trunc": 0.06, "depth_min": 0.05, "depth_max": 50.0}}
        init_fingerprint = calibration_fingerprint(identity)
        cache = Path(init_cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        ply_path = cache / f"rgbd_init_train_{init_fingerprint[:16]}.ply"
        metadata_path = ply_path.with_suffix(".ply.json")
        with ply_path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if ply_path.exists() != metadata_path.exists():
                raise ValueError(f"Incomplete RGBD cache/proof pair: {ply_path}")
            if not ply_path.exists():
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=cache, suffix=".ply", delete=False) as stream:
                        temporary = Path(stream.name)
                    generate_ply_from_rgbd(train, num_points=200000, ply_path=str(temporary), cam_intrinsics=cam_intrinsics,
                                          source_path=path, pixel_center_convention="colmap_half_integer")
                    pcd = fetchOpen3DPly(str(temporary))
                    if not len(pcd.points) or not np.isfinite(pcd.points).all() or not np.isfinite(pcd.colors).all():
                        raise ValueError("RGBD initialization produced no finite colored points")
                    for camera in infos:
                        depth = Path(path) / "depth" / Path(camera.colmap_image_name).with_suffix(".npy")
                        current = {"rgb_sha256": _sha256_file(camera.image_path), "depth_sha256": _sha256_file(depth) if depth.is_file() else None}
                        if current != assets[camera.colmap_image_name]:
                            raise RuntimeError("RGB/depth source changed during RGBD fusion")
                    temporary.replace(ply_path)
                    _write_record(metadata_path, {"schema_version": 1, "cache_fingerprint": init_fingerprint,
                                                  "identity": identity, "ply_sha256": _sha256_file(ply_path),
                                                  "point_count": len(pcd.points)})
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("cache_fingerprint") != init_fingerprint or metadata.get("identity") != identity or metadata.get("ply_sha256") != _sha256_file(ply_path):
                raise ValueError("RGBD cache/proof differs from the current training data/operator")
            pcd = fetchOpen3DPly(str(ply_path))
            if not len(pcd.points) or not np.isfinite(pcd.points).all() or not np.isfinite(pcd.colors).all():
                raise ValueError("RGBD cache must contain finite, nonempty XYZ/RGB")
            if metadata.get("point_count") != len(pcd.points):
                raise ValueError("RGBD proof point count disagrees with its PLY")
        initialization = {"kind": "train_only_rgbd", "algorithm": algorithm, "ply_path": str(ply_path.resolve()),
                          "ply_sha256": metadata["ply_sha256"], "point_count": metadata["point_count"],
                          "cache_fingerprint": init_fingerprint, "proof_path": str(metadata_path.resolve())}
    # Recheck content after fusion: mutable source files cannot earn a proof.
    for camera in infos:
        depth = Path(path) / "depth" / Path(camera.colmap_image_name).with_suffix(".npy")
        current = {"rgb_sha256": _sha256_file(camera.image_path), "depth_sha256": _sha256_file(depth) if depth.is_file() else None}
        if current != assets[camera.colmap_image_name]:
            raise RuntimeError("RGB/depth source changed during initialization")
    split_record = {
        "schema_version": 3, "dataset_root": str(Path(path).resolve()), "source": split_source,
        "camera_calibration_fingerprints": calibrations, "calibration_fingerprint": calibration_fingerprint(calibrations),
        "dataset_content_fingerprint": calibration_fingerprint(assets),
        "train_camera_fingerprint": training_camera_fingerprint(train),
        "train_image_names": [camera.colmap_image_name for camera in train], "test_image_names": [camera.colmap_image_name for camera in test],
        "initialization": initialization,
        "pixel_center_convention": "colmap_half_integer",
        "scope": "All input cameras are training views" if not test else "Initialization and supervision membership are explicit; neighbors exclude validation cameras",
    }
    if initialization_kind == "ffba_tsdf":
        split_record["evaluation_scope"] = FFBA_TSDF_EVALUATION_SCOPE
        split_record["scope"] = ("Unchanged FFBA full-capture TSDF initialization; source capture may include "
                                 "evaluation images. Metrics are fitting diagnostics, not independent heldout accuracy.")
    _write_record(cache / "camera_split.json", split_record)
    return SceneInfo(point_cloud=pcd, train_cameras=train, test_cameras=test,
                     nerf_normalization=getNerfppNorm(train), ply_path=str(ply_path))


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
