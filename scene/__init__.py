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

import json
import os
from typing import Sequence

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_bg_model import GaussianBackgroundModel
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p, searchForMaxIteration

UNIT_SPHERE_PATH = os.path.join(os.path.dirname(__file__), "unit_sphere.ply")


class Scene:
    gaussians: GaussianModel
    bg_gaussians: GaussianBackgroundModel | None
    should_train_with_bg: bool

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        bg_gaussians: GaussianBackgroundModel | None = None,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.bg_gaussians = bg_gaussians
        self.should_train_with_bg = bg_gaussians is not None
        self.scene_scale = None
        self.low_resolution = float(args.low_resolution)
        if self.low_resolution < 1.0:
            raise ValueError(f"low_resolution must be >= 1.0, got {self.low_resolution}")

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise AssertionError("Could not recognize scene type!")

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        camera_centers_list: list[torch.Tensor] = []
        for resolution_scale in resolution_scales:
            load_aux_data = len(resolution_scales) == 1 or np.isclose(resolution_scale, self.low_resolution)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
                load_mask=load_aux_data,
                load_normal=load_aux_data,
            )
            print(f"Loading Training Cameras: {len(self.train_cameras[resolution_scale])} .")

            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
                load_mask=load_aux_data,
                load_normal=load_aux_data,
            )
            print(f"Loading Test Cameras: {len(self.test_cameras[resolution_scale])} .")

            print("computing nearest_id")
            current_centers: list[torch.Tensor] = []
            center_rays_list: list[torch.Tensor] = []
            with torch.no_grad():
                for cur_cam in self.train_cameras[resolution_scale]:
                    current_centers.append(cur_cam.camera_center)
                    center_ray = torch.tensor([0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
                    center_rays_list.append(center_ray @ cur_cam.R.transpose(-1, -2))

                if current_centers:
                    camera_centers = torch.stack(current_centers, dim=0)
                    center_rays = torch.nn.functional.normalize(torch.stack(center_rays_list, dim=0), dim=-1)
                    diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
                    tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
                    angles_np = (torch.arccos(tmp) * 180 / 3.14159).detach().cpu().numpy()
                    with open(os.path.join(self.model_path, "multi_view.json"), "w") as file:
                        for idx, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                            sorted_indices = np.lexsort((angles_np[idx], diss[idx]))
                            mask = (
                                (angles_np[idx][sorted_indices] < args.multi_view_max_angle)
                                & (diss[idx][sorted_indices] > args.multi_view_min_dis)
                                & (diss[idx][sorted_indices] < args.multi_view_max_dis)
                            )
                            sorted_indices = sorted_indices[mask]
                            multi_view_num = min(args.multi_view_num, len(sorted_indices))
                            json_d = {"ref_name": cur_cam.image_name, "nearest_name": []}
                            for index in sorted_indices[:multi_view_num]:
                                cur_cam.nearest_id.append(index)
                                json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                            file.write(json.dumps(json_d, separators=(",", ":")))
                            file.write("\n")
                    camera_centers_list = current_centers

        self.gaussians.create_app_model(len(scene_info.train_cameras), args.use_decoupled_appearance)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            with torch.no_grad():
                for camera_center in camera_centers_list:
                    dists_cam_gauss = torch.norm(self.gaussians.get_xyz - camera_center[None, :], dim=1)
                    max_scale = 0.05 * dists_cam_gauss.flatten()
                    log_max_scale = torch.log(max_scale).repeat(3, 1).permute(1, 0)
                    self.gaussians._scaling[:] = torch.clamp_max(self.gaussians._scaling, log_max_scale)

        if self.bg_gaussians is not None:
            self._prepare_background(scene_info.point_cloud)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.app_model == GaussianModel.App_model.PGSR:
            self.gaussians.save_3dgsviewer_ply(os.path.join(point_cloud_path, "point_cloud_3dgsviewer.ply"))

    def save_bg(self, iteration):
        if self.bg_gaussians is None:
            return

        point_cloud_path = os.path.join(self.model_path, "point_cloud_bg/iteration_{}".format(iteration))
        mkdir_p(point_cloud_path)
        raw_path = os.path.join(point_cloud_path, "point_cloud.ply")
        filled_path = os.path.join(point_cloud_path, "point_cloud.filled.ply")
        self.bg_gaussians.save_ply(raw_path)
        self.fill_bg_empty(raw_path, filled_path)

    def fill_bg_empty(self, input_ply, output_ply):
        with open(input_ply, "rb") as file:
            ply = PlyData.read(file)

        empty_color = 1.77245378

        def fill(element, condition):
            is_empty_color = (
                (element["f_dc_0"] == empty_color)
                & (element["f_dc_1"] == empty_color)
                & (element["f_dc_2"] == empty_color)
            )
            empty_indices = np.argwhere(condition & is_empty_color)
            non_empty_indices = np.argwhere(condition & (~is_empty_color)).flatten()
            if non_empty_indices.size == 0:
                return
            non_empty = np.take(element, non_empty_indices)
            non_empty_f_dc = np.column_stack((non_empty["f_dc_0"], non_empty["f_dc_1"], non_empty["f_dc_2"]))
            filling = np.mean(non_empty_f_dc, axis=0)
            for idx in empty_indices:
                element["f_dc_0"][idx] = filling[0]
                element["f_dc_1"][idx] = filling[1]
                element["f_dc_2"][idx] = filling[2]

        for element in ply.elements:
            is_empty_color = (
                (element["f_dc_0"] == empty_color)
                & (element["f_dc_1"] == empty_color)
                & (element["f_dc_2"] == empty_color)
            )
            if len(np.argwhere(is_empty_color)) > 50000 * 0.6:
                self.should_train_with_bg = False
            fill(element, element["y"] > 0)
            fill(element, element["y"] < 0)

        with open(output_ply, "wb") as file:
            ply.write(file)

    def _prepare_background(self, point_cloud):
        latest_bg = self._latest_bg_path()
        if latest_bg is not None:
            self.bg_gaussians.load_ply(latest_bg)
            self._update_background_scale()
            return

        bg_path = os.path.join(self.model_path, "bg.ply")
        if not os.path.exists(bg_path):
            if point_cloud is None or point_cloud.points.size == 0:
                print("No point cloud available to build a background sphere. Skipping background training.")
                self.should_train_with_bg = False
                return
            bg_xyz, bg_rgb, bg_scale, scene_scale = generate_background_sphere(point_cloud.points, 0.75)
            if bg_xyz is None:
                self.should_train_with_bg = False
                return
            store_bg_ply(bg_path, bg_xyz, bg_rgb, bg_scale)
            self.scene_scale = scene_scale

        self.bg_gaussians.load_ply(bg_path)
        self._update_background_scale()

    def _latest_bg_path(self):
        bg_dir = os.path.join(self.model_path, "point_cloud_bg")
        if not os.path.exists(bg_dir):
            return None
        latest_iter = searchForMaxIteration(bg_dir)
        if latest_iter is None:
            return None
        base = os.path.join(bg_dir, f"iteration_{latest_iter}")
        filled = os.path.join(base, "point_cloud.filled.ply")
        raw = os.path.join(base, "point_cloud.ply")
        if os.path.exists(filled):
            return filled
        if os.path.exists(raw):
            return raw
        return None

    def _update_background_scale(self):
        if self.bg_gaussians is None or self.bg_gaussians.get_xyz.numel() == 0:
            return
        with torch.no_grad():
            self.scene_scale = float(torch.linalg.norm(self.bg_gaussians.get_xyz, dim=1).max().item())

    def getTrainCameras(self, scale=1.0) -> Sequence[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> Sequence[Camera]:
        return self.test_cameras[scale]


def generate_background_sphere(xyz, distance):
    if xyz is None or len(xyz) == 0:
        return None, None, None, None

    length = np.linalg.norm(xyz, axis=1)
    sorted_indices = np.argsort(length)
    xyz_without_outliers = xyz[sorted_indices[: int(len(xyz) * 0.995)]]
    point_max_coordinate = np.max(xyz_without_outliers, axis=0)
    point_min_coordinate = np.min(xyz_without_outliers, axis=0)
    scene_size = np.max(point_max_coordinate - point_min_coordinate)
    scene_scale = max(50, min(10_000, scene_size * distance))

    with open(UNIT_SPHERE_PATH, "rb") as file:
        ply = PlyData.read(file)
        unit_sphere = ply.elements[0].data

    unit_sphere_points = np.stack((unit_sphere["x"], unit_sphere["y"], unit_sphere["z"]), axis=1)
    bg_sphere_xyz = unit_sphere_points * scene_scale
    bg_sphere_rgb = np.full((unit_sphere.shape[0], 3), [255, 255, 255], dtype=np.uint8)
    bg_sphere_scale_c = np.log(np.exp(unit_sphere["scale"]) * scene_scale)
    bg_sphere_scale = np.stack((bg_sphere_scale_c, bg_sphere_scale_c, bg_sphere_scale_c), axis=1)
    return bg_sphere_xyz, bg_sphere_rgb, bg_sphere_scale, scene_scale


def store_bg_ply(path, xyz, rgb, scales):
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs.extend([f"f_dc_{i}" for i in range(3)])
    attrs.extend([f"f_rest_{i}" for i in range(45)])
    attrs.append("opacity")
    attrs.extend([f"scale_{i}" for i in range(3)])
    attrs.extend([f"rot_{i}" for i in range(4)])
    dtype = [(attribute, "f4") for attribute in attrs]

    sh_0 = 0.28209479177387814
    normals = np.zeros_like(xyz)
    dcs = (rgb.astype(np.float32) / 255.0 - 0.5) / sh_0
    opacity = np.full((xyz.shape[0], 1), 4.595121, dtype=np.float32)
    shs = np.zeros((xyz.shape[0], 45), dtype=np.float32)
    rots = np.full((xyz.shape[0], 4), [1, 0, 0, 0], dtype=np.float32)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    packed = np.concatenate((xyz, normals, dcs, shs, opacity, scales, rots), axis=1)
    elements[:] = list(map(tuple, packed))
    vertex_element = PlyElement.describe(elements, "vertex")
    mkdir_p(os.path.dirname(path))
    PlyData([vertex_element]).write(path)
