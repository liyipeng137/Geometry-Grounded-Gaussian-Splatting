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

import os

from PIL import Image

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch

WARNED = False


def _load_gt_mask(args, cam_info, resolution):
    mask_dir = args.mask_dir.strip()
    if not mask_dir:
        return None

    if os.path.isabs(mask_dir):
        mask_root = mask_dir
    else:
        mask_root = os.path.join(args.source_path, mask_dir)

    mask_format = args.mask_format.lower().lstrip(".")
    mask_path = os.path.join(mask_root, f"{cam_info.image_name}.{mask_format}")
    if not os.path.exists(mask_path):
        return None

    mask_img = Image.open(mask_path)
    loaded_mask = PILtoTorch(mask_img, resolution)[:1]
    # Keep a binary supervision target for alpha loss.
    loaded_mask = (loaded_mask > 0.5).float()
    return loaded_mask


def _load_normal_prior(args, cam_info, resolution):
    normal_prior_dir = args.normal_prior_dir.strip()
    if normal_prior_dir:
        if os.path.isabs(normal_prior_dir):
            normal_root = normal_prior_dir
        else:
            normal_root = os.path.join(args.source_path, normal_prior_dir)
    else:
        normal_root = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "normals")

    normal_format = args.normal_prior_format.lower().lstrip(".")
    normal_path = os.path.join(normal_root, f"{cam_info.image_name}.{normal_format}")
    if not os.path.exists(normal_path):
        return None

    normal_img = Image.open(normal_path)
    resized_normal = PILtoTorch(normal_img, resolution)[:3]
    normal_prior = resized_normal * 2.0 - 1.0

    # default is stable normal
    normal_prior = -normal_prior

    return normal_prior


# def _load_delight_image(args, cam_info, resolution):
#     delight_dir = args.delight_dir.strip()
#     if not delight_dir:
#         return None

#     if os.path.isabs(delight_dir):
#         delight_root = delight_dir
#     else:
#         delight_root = os.path.join(args.source_path, delight_dir)

#     delight_format = args.delight_format.lower().lstrip(".")
#     delight_path = os.path.join(delight_root, f"{cam_info.image_name}.{delight_format}")
#     if not os.path.exists(delight_path):
#         return None
#     print(f"Loading delight image from {delight_path}")
#     delight_img = Image.open(delight_path)
#     return PILtoTorch(delight_img, resolution)[:3]

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = _load_gt_mask(args, cam_info, resolution)
        gt_image = resized_image_rgb

    normal_prior = _load_normal_prior(args, cam_info, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  normal_prior=normal_prior, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
