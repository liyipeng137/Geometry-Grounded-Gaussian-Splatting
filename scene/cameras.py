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

import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix, getWorld2View2

_PAYLOAD_FIELDS = (
    "original_image",
    "gt_mask",
    "normal_prior",
    "normal_prior_mask",
    "depth_prior",
    "depth_confidence",
)


class FramePayloadCache:
    def __init__(self, budget):
        self.budget = budget
        self.enabled = budget != -1
        if self.enabled and budget < 1:
            raise ValueError("frame_gpu_budget must be -1 or a positive integer")
        self._cache = OrderedDict()

    def ensure(self, camera, keep=()):
        if not self.enabled:
            return camera

        key = id(camera)
        if key in self._cache:
            self._cache.move_to_end(key)
            return camera

        keep_ids = {id(item) for item in keep}
        while len(self._cache) >= self.budget:
            evicted = False
            for old_key in list(self._cache.keys()):
                if old_key == key or old_key in keep_ids:
                    continue
                old_camera = self._cache.pop(old_key)
                old_camera.release_payload_from_device()
                evicted = True
                break
            if not evicted:
                break

        camera.load_payload_to_device()
        self._cache[key] = camera
        return camera

    def clear(self):
        if not self.enabled:
            return
        for camera in self._cache.values():
            camera.release_payload_from_device()
        self._cache.clear()


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 normal_prior=None,
                 depth_prior=None,
                 depth_confidence=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 lazy_payload=False,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.nearest_id = []
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.R = torch.tensor(R, dtype=torch.float32).cuda()
        self.T = torch.tensor(T, dtype=torch.float32).cuda()
        self.lazy_payload = lazy_payload

        if image.dtype != torch.uint8:
            image = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        if gt_alpha_mask is not None:
            if gt_alpha_mask.dtype != torch.uint8:
                mask_f = gt_alpha_mask.clamp(0.0, 1.0).contiguous()
                mask_u8 = (mask_f * 255.0).round().to(torch.uint8)
            else:
                mask_u8 = gt_alpha_mask
                mask_f = gt_alpha_mask.float().mul(1.0 / 255.0).contiguous()
            image = ((image.to(torch.int32) * mask_u8.to(torch.int32) + 127) // 255).to(torch.uint8)
        else:
            mask_f = None
        image = image.contiguous()
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        self.Fx = self.image_width / (2 * math.tan(self.FoVx / 2.))
        self.Fy = self.image_height / (2 * math.tan(self.FoVy / 2.))
        self.Cx = float(self.image_width - 1) / 2
        self.Cy = float(self.image_height - 1) / 2

        self.has_gt_mask = mask_f is not None
        if normal_prior is not None:
            normal_norm = torch.norm(normal_prior, dim=0, keepdim=True)
            normal_prior_mask = (normal_norm > 0.9) & (normal_norm < 1.1)
            normal_prior = normal_prior / torch.clamp(normal_norm, min=1e-6)
        else:
            normal_prior_mask = None
        self.has_normal_prior = normal_prior is not None
        self.has_depth_prior = depth_prior is not None
        self.has_depth_confidence = depth_confidence is not None

        self._store_payload("original_image", image)
        self._store_payload("gt_mask", mask_f)
        self._store_payload("normal_prior", normal_prior)
        self._store_payload("normal_prior_mask", normal_prior_mask)
        self._store_payload("depth_prior", depth_prior)
        self._store_payload("depth_confidence", depth_confidence)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _store_payload(self, name, tensor):
        if tensor is not None:
            tensor = tensor.contiguous()
        if self.lazy_payload:
            setattr(self, name + "_cpu", tensor)
            setattr(self, name, None)
        else:
            setattr(self, name + "_cpu", None)
            setattr(self, name, None if tensor is None else tensor.to(self.data_device))

    def load_payload_to_device(self):
        if not self.lazy_payload:
            return
        for name in _PAYLOAD_FIELDS:
            cpu_tensor = getattr(self, name + "_cpu")
            if cpu_tensor is not None and getattr(self, name) is None:
                setattr(self, name, cpu_tensor.to(self.data_device, non_blocking=True))

    def release_payload_from_device(self):
        if not self.lazy_payload:
            return
        for name in _PAYLOAD_FIELDS:
            setattr(self, name, None)

    def original_image_float(self):
        image = self.original_image
        if image.dtype == torch.uint8:
            return image.float().mul(1.0 / 255.0)
        return image

    @property
    def gray_image(self):
        image = self.original_image_float()
        return (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])[None]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

