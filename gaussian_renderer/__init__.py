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
from scene.gaussian_bg_model import GaussianBackgroundModel
from scene.gaussian_model import GaussianModel


def render(
    viewpoint_camera,
    pc: GaussianModel | GaussianBackgroundModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size,
    scaling_modifier=1.0,
    require_depth: bool = True,
    get_flag: bool = False,
    metric_map: Optional[torch.Tensor] = None,
    bg_splats: Optional[GaussianBackgroundModel] = None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    means3D = pc.get_xyz
    if bg_splats is not None:
        means3D = torch.cat([means3D, bg_splats.get_xyz], dim=0)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if metric_map is None:
        metric_map = torch.empty(0, dtype=torch.int32, device="cuda")
    else:
        metric_map = metric_map.reshape(-1).to(device="cuda", dtype=torch.int32).contiguous()

    sg_degree = getattr(pc, "active_sg_degree", 0)

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
        sg_degree=sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_depth = require_depth,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if isinstance(pc, GaussianBackgroundModel):
        scales = pc.get_scaling
        opacity = pc.get_opacity
        rotations = pc.get_rotation
        shs = pc.get_features
        sg_axis = torch.zeros((pc.get_xyz.shape[0], 0, 3), dtype=means3D.dtype, device=means3D.device)
        sg_sharpness = torch.zeros((pc.get_xyz.shape[0], 0), dtype=means3D.dtype, device=means3D.device)
        sg_color = torch.zeros((pc.get_xyz.shape[0], 0, 3), dtype=means3D.dtype, device=means3D.device)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
        rotations = pc.get_rotation
        shs = pc.get_features
        sg_axis = pc.get_sg_axis
        sg_sharpness = pc.get_sg_sharpness
        sg_color = pc.get_sg_color

    if bg_splats is not None:
        bg_count = bg_splats.get_xyz.shape[0]
        scales = torch.cat([scales, bg_splats.get_scaling], dim=0)
        opacity = torch.cat([opacity, bg_splats.get_opacity], dim=0)
        rotations = torch.cat([rotations, bg_splats.get_rotation], dim=0)
        shs = torch.cat([shs, bg_splats.get_features], dim=0)
        zero_axis = torch.zeros((bg_count, pc.max_sg_degree, 3), dtype=sg_axis.dtype, device=sg_axis.device)
        zero_sharpness = torch.zeros((bg_count, pc.max_sg_degree), dtype=sg_sharpness.dtype, device=sg_sharpness.device)
        zero_color = torch.zeros((bg_count, pc.max_sg_degree, 3), dtype=sg_color.dtype, device=sg_color.device)
        sg_axis = torch.cat([sg_axis, zero_axis], dim=0)
        sg_sharpness = torch.cat([sg_sharpness, zero_sharpness], dim=0)
        sg_color = torch.cat([sg_color, zero_color], dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None

    rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal, accum_metric_counts = rasterizer(
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
        cov3D_precomp = cov3D_precomp,)



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
            }

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
    return {"sampled_depth": depth,
            "inside": inside}
