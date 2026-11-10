"""无 RGB、无外观参数的几何相机；不依赖 scene 包的训练初始化。"""

import math

import numpy as np
import torch


class VirtualCamera:
    def __init__(self, record, teacher, normalizers, device="cuda", virtual_index=0):
        self.virtual_index = int(virtual_index)
        self.uid = self.virtual_index
        self.image_name = record["id"]
        self.kind = "virtual_geometry"
        self.image_width, self.image_height = int(record["width"]), int(record["height"])
        self.source_K_colmap = np.asarray(record["K"], np.float64)
        self.c2w = np.asarray(record["c2w"], np.float64)
        intrinsic, pose = self.source_K_colmap, self.c2w
        self.pixel_center_convention = "colmap_half_integer"
        self.Fx, self.Fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        self.Cx, self.Cy = float(intrinsic[0, 2]) - .5, float(intrinsic[1, 2]) - .5
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.Fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.Fy))
        self.znear, self.zfar = .01, 100.
        self.data_device = torch.device(device)
        w2c = np.linalg.inv(pose)
        self.R = torch.tensor(pose[:3, :3], dtype=torch.float32, device=device)
        self.T = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device)
        self.world_view_transform = torch.tensor(w2c.T.copy(), dtype=torch.float32, device=device)
        projection = np.zeros((4, 4), np.float32)
        projection[0, 0], projection[1, 1] = 2 * self.Fx / self.image_width, 2 * self.Fy / self.image_height
        projection[0, 2] = 2 * intrinsic[0, 2] / self.image_width - 1
        projection[1, 2] = 2 * intrinsic[1, 2] / self.image_height - 1
        projection[3, 2] = 1
        projection[2, 2] = self.zfar / (self.zfar - self.znear)
        projection[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection_matrix = torch.tensor(projection.T.copy(), device=device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.tensor(pose[:3, 3], dtype=torch.float32, device=device)
        self.teacher = {key: value.to(device) for key, value in teacher.items()}
        self.normalizers = dict(normalizers)
        self.depth_z = self.teacher["depth"]
        self.normal_cam = self.teacher["normal"]
        self.primitive_id = self.teacher["face_id"]
        self.patch_id = self.teacher["patch_id"]

    def camera_z_depth(self, depth):
        """与真实 Camera 保持相同的非中心主点射线深度转换。"""
        if tuple(depth.shape[-2:]) != (self.image_height, self.image_width):
            raise ValueError("Virtual render dimensions differ from its frozen camera")
        center_x, center_y = (self.image_width - 1) / 2, (self.image_height - 1) / 2
        if self.Cx == center_x and self.Cy == center_y:
            return depth
        y = torch.arange(self.image_height, device=depth.device, dtype=depth.dtype)[:, None]
        x = torch.arange(self.image_width, device=depth.device, dtype=depth.dtype)[None, :]
        raster_radius = torch.sqrt(1 + ((x - center_x) / self.Fx) ** 2 + ((y - center_y) / self.Fy) ** 2)
        camera_radius = torch.sqrt(1 + ((x - self.Cx) / self.Fx) ** 2 + ((y - self.Cy) / self.Fy) ** 2)
        return depth * raster_radius / camera_radius
