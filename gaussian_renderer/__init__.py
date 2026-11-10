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
from typing import Optional
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


DEPTH_NO_CONTRIBUTOR = 0
DEPTH_LOW_COVERAGE = 1
DEPTH_ROOT_FAILED = 2
DEPTH_VALID = 3


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size,
    scaling_modifier=1.0,
    require_depth: bool = True,
    get_flag: bool = False,
    metric_map: Optional[torch.Tensor] = None,
    record_transmittance: bool = False,
    depth_search_expansions: int = 0,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if metric_map is None:
        metric_map = torch.empty(0, dtype=torch.int32, device="cuda")
    else:
        metric_map = metric_map.reshape(-1).to(device="cuda", dtype=torch.int32).contiguous()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=pc.active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_depth = require_depth,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map,
        record_transmittance=record_transmittance,
        depth_search_expansions=depth_search_expansions,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None
    
    sg_axis = pc.get_sg_axis
    sg_sharpness = pc.get_sg_sharpness
    sg_color = pc.get_sg_color

    raster_output = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sg_axis = sg_axis,
        sg_sharpness = sg_sharpness,
        sg_color = sg_color,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    if record_transmittance:
        (
            rendered_image,
            radii,
            rendered_median_depth,
            rendered_alpha,
            rendered_normal,
            accum_metric_counts,
            transmittance_avg,
            num_covered_pixels,
        ) = raster_output
    else:
        rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal, accum_metric_counts = raster_output
        transmittance_avg = None
        num_covered_pixels = None



    if require_depth and hasattr(viewpoint_camera, "camera_z_depth"):
        rendered_median_depth = viewpoint_camera.camera_z_depth(rendered_median_depth)

    depth_status = None
    if require_depth:
        # no_contributor 指没有通过 rasterizer alpha 截断的实际贡献；并非
        # 声称像素邻域中完全没有 Gaussian 中心或 footprint。
        with torch.no_grad():
            depth_status = torch.full_like(rendered_alpha, DEPTH_ROOT_FAILED, dtype=torch.int8)
            depth_status[rendered_alpha < 0.55] = DEPTH_LOW_COVERAGE
            depth_status[rendered_alpha == 0] = DEPTH_NO_CONTRIBUTOR
            depth_status[torch.isfinite(rendered_median_depth) & (rendered_median_depth > 0)] = DEPTH_VALID

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "accum_metric_counts": accum_metric_counts,
            "transmittance_avg": transmittance_avg,
            "num_covered_pixels": num_covered_pixels,
            "depth_status": depth_status,
            }


@torch.no_grad()
def geometry_score(camera, pc, pipe, bg, kernel_size, target_depth, depth_tolerance,
                   error_map, support_map, weight_map):
    """几何归属统计，不反传，也不复用 RGB 的整数 metricCount。

    输入 D/tolerance 为 camera-Z；每条有效 teacher 射线按 W*alpha*T 累计。
    weight_sum 覆盖所有实际贡献，near_weight_sum/error_sum/support_sum 仅
    覆盖 ray-plane 峰值落在目标深度带内的贡献，off_surface_sum 是带外贡献。
    完全空射线不会产生逐点机会；它们由像素 coverage/repair 单独处理。
    """
    device = pc.get_xyz.device
    height, width = camera.image_height, camera.image_width

    def image_map(value, name):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.numel() == 1:
            tensor = tensor.expand(height, width)
        if tensor.numel() != height * width:
            raise ValueError(f"{name} must match the camera pixel grid")
        return tensor.reshape(height, width).contiguous()

    depth = image_map(target_depth, "target_depth")
    tolerance = image_map(depth_tolerance, "depth_tolerance")
    error = image_map(error_map, "error_map")
    support = image_map(support_map, "support_map")
    weight = image_map(weight_map, "weight_map")
    for name, tensor in (("error_map", error), ("support_map", support), ("weight_map", weight)):
        if not torch.isfinite(tensor).all() or (tensor < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(tolerance) & (tolerance > 0)
    weight = torch.where(valid, weight, 0).contiguous()
    # CUDA ray_plane 的 t 是射线距离。不能乘图像中心的旧 rln 因子：
    # 非中心主点必须使用真实 K，与 camera_z_depth 的逆变换一致。
    x = (torch.arange(width, device=device, dtype=torch.float32) - camera.Cx) / camera.Fx
    y = (torch.arange(height, device=device, dtype=torch.float32) - camera.Cy) / camera.Fy
    ray_norm = torch.sqrt(1 + x[None, :].square() + y[:, None].square())
    target_range = torch.where(valid, depth * ray_norm, 0).contiguous()
    tolerance_range = torch.where(valid, tolerance * ray_norm, 0).contiguous()
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(camera.FoVx * 0.5), tanfovy=math.tan(camera.FoVy * 0.5),
        kernel_size=kernel_size, bg=bg, scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=0, sg_degree=0, campos=camera.camera_center,
        prefiltered=False, require_depth=False, debug=pipe.debug,
        get_flag=False, metric_map=torch.empty(0, dtype=torch.int32, device=device),
    )
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    outputs = GaussianRasterizer(settings).geometry_score(
        pc.get_xyz, opacity, scales, pc.get_rotation, target_range,
        tolerance_range, error, support, weight,
    )
    names = ("error_sum", "support_sum", "near_weight_sum", "weight_sum", "off_surface_sum")
    return dict(zip(names, outputs))

# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate(points3D, viewpoint_camera, pc : GaussianModel, pipe, kernel_size : float, scaling_modifier = 1.0):
    """
    integrate Gaussians to the points
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=pc.active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
        get_flag=False,
        metric_map=torch.empty(0, dtype=torch.int32, device="cuda"),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    alpha_integrated, inside = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"alpha_integrated": alpha_integrated,
            "inside": inside}
    
def evaluate_sdf(points3D, viewpoint_camera, pc : GaussianModel, pipe, kernel_size : float, scaling_modifier = 1.0):
    """
    integrate Gaussians to the points
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=pc.active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
        get_flag=False,
        metric_map=torch.empty(0, dtype=torch.int32, device="cuda"),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth, sdf, inside = rasterizer.evaluate_sdf(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"depth": depth,
            "sdf": sdf,
            "inside": inside}


def _sampled_points_to_camera_rays(points_world, sampled_points, camera):
    """Retain sampled range while replacing the CUDA sampler's centered ray.

    The native sampler returns XYZ on its centered pixel ray. Exact-K projection
    selects the correct pixel, but an off-center camera needs its actual ray to
    reconstruct camera coordinates for multiview geometry losses.
    """
    center_x, center_y = (camera.image_width - 1) / 2, (camera.image_height - 1) / 2
    if getattr(camera, "Cx", center_x) == center_x and getattr(camera, "Cy", center_y) == center_y:
        return sampled_points
    query_camera = points_world @ camera.R + camera.T
    query_length = torch.linalg.vector_norm(query_camera, dim=-1, keepdim=True)
    sample_range = torch.linalg.vector_norm(sampled_points, dim=-1, keepdim=True)
    denominator = query_length.clamp_min(torch.finfo(query_camera.dtype).tiny)
    return query_camera / denominator * sample_range


def sample_depth(points3D, viewpoint_camera, pc : GaussianModel, pipe : torch.Tensor, kernel_size : float, scaling_modifier = 1.0):

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=pc.active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
        get_flag=False,
        metric_map=torch.empty(0, dtype=torch.int32, device="cuda"),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth, inside = rasterizer.sample_depth(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    depth = _sampled_points_to_camera_rays(points3D, depth, viewpoint_camera)
    return {"sampled_depth": depth,
            "inside": inside}
