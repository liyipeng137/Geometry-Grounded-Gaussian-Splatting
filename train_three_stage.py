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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from random import randint, sample
from typing import Any, Sequence, TypedDict, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianBackgroundModel, GaussianModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.graphics_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import L1_loss_appearance, PatchMatch, l1_loss, ssim
from utils.vcd_utils import compute_vcd_vcp_scores, sample_vcd_cameras


@dataclass(frozen=True)
class StageConfig:
    name: str
    start_iter: int                 # 开始迭代次数
    end_iter: int                   # 结束迭代次数
    resolution_scale: float         # 分辨率缩放因子
    use_background_rgb: bool        # 是否使用背景RGB
    enable_depth_normal_loss: bool  # 是否启用深度法线损失
    enable_normal_prior: bool       # 是否启用法线先验
    enable_ncc_geo: bool            # 是否启用NCC几何损失


def build_stage_configs(
    low_resolution: float,
    stage1_end_iter: int,
    stage2_end_iter: int,
    opt_iterations: int,
    stage3_use_depth_normal: bool,
) -> tuple[StageConfig, ...]:
    stage3_start_iter = stage2_end_iter + 1
    stage3_end_iter = max(stage3_start_iter, opt_iterations)
    return (
        StageConfig(
            name="init",
            start_iter=1,
            end_iter=stage1_end_iter,
            resolution_scale=low_resolution,
            use_background_rgb=True,
            enable_depth_normal_loss=False,
            enable_normal_prior=False,
            enable_ncc_geo=False,
        ),
        StageConfig(
            name="geometry",
            start_iter=stage1_end_iter + 1,
            end_iter=stage2_end_iter,
            resolution_scale=low_resolution,
            use_background_rgb=False,
            enable_depth_normal_loss=True,
            enable_normal_prior=True,
            enable_ncc_geo=True,
        ),
        StageConfig(
            name="render",
            start_iter=stage3_start_iter,
            end_iter=stage3_end_iter,
            resolution_scale=1.0,
            use_background_rgb=False,
            enable_depth_normal_loss=stage3_use_depth_normal,
            enable_normal_prior=False,
            enable_ncc_geo=False,
        ),
    )


def get_stage_config(iteration: int, stage_configs: tuple[StageConfig, ...]) -> StageConfig:
    for stage in stage_configs:
        if stage.start_iter <= iteration <= stage.end_iter:
            return stage
    return stage_configs[-1]


def get_low_resolution(dataset) -> float:
    low_resolution = float(dataset.low_resolution)
    if low_resolution < 1.0:
        raise ValueError(f"low_resolution must be >= 1.0, got {low_resolution}")
    return low_resolution


def get_training_resolution_scales(dataset) -> list[float]:
    low_resolution = get_low_resolution(dataset)
    if np.isclose(low_resolution, 1.0):
        return [1.0]
    return [1.0, low_resolution]


def edge_aware_normal_smoothness_loss(
    normals: torch.Tensor,
    guidance_prior_normal: torch.Tensor,
    guidance_image: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    edge_scale: float = 10.0,
    image_grad_mix_ratio: float = 0.2,
) -> torch.Tensor:
    # normals: [3, H, W], guidance_prior_normal: [3, H, W], guidance_image: [3, H, W] or None
    n_dx = torch.linalg.norm(normals[:, :, 1:] - normals[:, :, :-1], dim=0)
    n_dy = torch.linalg.norm(normals[:, 1:, :] - normals[:, :-1, :], dim=0)

    prior = guidance_prior_normal.to(normals.dtype)
    prior_dx = torch.linalg.norm(prior[:, :, 1:] - prior[:, :, :-1], dim=0)
    prior_dy = torch.linalg.norm(prior[:, 1:, :] - prior[:, :-1, :], dim=0)

    if guidance_image is not None:
        gray = 0.299 * guidance_image[0] + 0.587 * guidance_image[1] + 0.114 * guidance_image[2]
        i_dx = torch.abs(gray[:, 1:] - gray[:, :-1])
        i_dy = torch.abs(gray[1:, :] - gray[:-1, :])
        mixed_dx = torch.maximum(prior_dx, image_grad_mix_ratio * i_dx)
        mixed_dy = torch.maximum(prior_dy, image_grad_mix_ratio * i_dy)
    else:
        mixed_dx = prior_dx
        mixed_dy = prior_dy

    w_dx = torch.exp(-edge_scale * mixed_dx)
    w_dy = torch.exp(-edge_scale * mixed_dy)

    if valid_mask is None:
        return (w_dx * n_dx).mean() + (w_dy * n_dy).mean()

    mask_dx = valid_mask[:, 1:] & valid_mask[:, :-1]
    mask_dy = valid_mask[1:, :] & valid_mask[:-1, :]
    mask_dx_f = mask_dx.float()
    mask_dy_f = mask_dy.float()

    denom_x = mask_dx_f.sum().clamp_min(1.0)
    denom_y = mask_dy_f.sum().clamp_min(1.0)
    loss_x = (w_dx * n_dx * mask_dx_f).sum() / denom_x
    loss_y = (w_dy * n_dy * mask_dy_f).sum() / denom_y
    return loss_x + loss_y


def training_bg(dataset, opt, pipe, scene: Scene, background: torch.Tensor, kernel_size: float) -> None:
    gaussians = scene.bg_gaussians
    if gaussians is None or not scene.should_train_with_bg or opt.bg_iterations <= 0:
        return

    gaussians.training_setup(opt)
    progress_bar = tqdm(range(0, opt.bg_iterations), desc="Background training")
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    for iteration in range(1, opt.bg_iterations + 1):
        gaussians.update_learning_rate(iteration)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size,
            require_depth=False,
        )
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        rgb_loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (
            1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )
        rgb_loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}"})
                progress_bar.update(10)
            if iteration == opt.bg_iterations:
                progress_bar.close()
                print(f"\n[BG ITER {iteration}] Saving background Gaussians")
                scene.save_bg(iteration)

            if gaussians.optimizer is not None:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    stage_schedule,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    bg_gaussians = GaussianBackgroundModel(dataset.sh_degree) if dataset.enable_background_sphere else None
    scene = Scene(dataset, gaussians, bg_gaussians, resolution_scales=get_training_resolution_scales(dataset))
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    if scene.should_train_with_bg:
        training_bg(dataset, opt, pipe, scene, background, kernel_size)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    low_resolution = get_low_resolution(dataset)
    stage1_end_iter = max(1, min(int(stage_schedule["stage1_end_iter"]), int(opt.iterations)))
    stage2_end_iter = max(stage1_end_iter, min(int(stage_schedule["stage2_end_iter"]), int(opt.iterations)))
    stage3_start_iter = stage2_end_iter + 1
    stage3_use_depth_normal = bool(stage_schedule["stage3_use_depth_normal"])

    # set stage2 edge aware normal smooth params
    stage2_edge_aware_normal_smooth_lambda = float(stage_schedule.get("stage2_edge_aware_normal_smooth_lambda", 0.0))

    # set stage configs
    stage_configs = build_stage_configs(
        low_resolution=low_resolution,
        stage1_end_iter=stage1_end_iter,
        stage2_end_iter=stage2_end_iter,
        opt_iterations=int(opt.iterations),
        stage3_use_depth_normal=stage3_use_depth_normal,
    )

    # set final prune iterations
    final_prune_iterations = sorted(
        {
            int(it)
            for it in stage_schedule["final_prune_iterations"]
            if stage3_start_iter <= int(it) < int(opt.iterations)
        }
    )

    low_res_train_cameras = scene.getTrainCameras(scale=low_resolution).copy()
    full_res_train_cameras = scene.getTrainCameras(scale=1.0).copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=full_res_train_cameras)

    # 多视角几何损失
    patchmatch = PatchMatch(
        opt.multi_view_patch_size,
        opt.multi_view_pixel_noise_th,
        kernel_size=kernel_size,
        pipe=pipe,
        debug=True,
        model_path=dataset.model_path,
    )

    has_normal_dir = os.path.exists(os.path.join(dataset.source_path, dataset.normal_prior_dir))
    has_mask_dir = os.path.exists(os.path.join(dataset.source_path, dataset.mask_dir))
    has_loaded_normal_prior = any(cam.normal_prior is not None for cam in low_res_train_cameras)
    has_loaded_mask = any(cam.gt_mask is not None for cam in low_res_train_cameras)

    #----------------------------------打印训练信息----------------------------------
    if has_normal_dir and not has_loaded_normal_prior:
        print("[Pipeline][Warn] normals directory exists but no valid normal priors were loaded.")
    if has_mask_dir and not has_loaded_mask:
        print("[Pipeline][Warn] masks directory exists but no valid mask priors were loaded.")
    print(f"[Pipeline] has_normal_dir={has_normal_dir} (loaded_priors={has_loaded_normal_prior})")
    print(f"[Pipeline] has_mask_dir={has_mask_dir} (loaded_masks={has_loaded_mask}), lambda_mask={opt.lambda_mask}")
    print(f"[Pipeline] low_resolution={low_resolution}")
    for stage in stage_configs:
        if stage.start_iter <= stage.end_iter and stage.start_iter <= int(opt.iterations):
            print(
                f"[Pipeline] Stage {stage.name}: [{stage.start_iter}, {stage.end_iter}], "
                f"scale={stage.resolution_scale}, bg_rgb={stage.use_background_rgb}, "
                f"depth_normal={stage.enable_depth_normal_loss}, normal_prior={stage.enable_normal_prior}, "
                f"ncc_geo={stage.enable_ncc_geo}"
            )
    print(f"[Pipeline] Densify follows original schedule: from>{opt.densify_from_iter} until<{opt.densify_until_iter}, interval={opt.densification_interval}")
    print(f"[Pipeline] NCC/GEO enabled only when iteration <= densify_until_iter ({opt.densify_until_iter})")
    print(
        "[Pipeline] Edge-aware normal smoothness in Stage-2 post-densify: "
        f"lambda={stage2_edge_aware_normal_smooth_lambda}"
    )
    print(f"[Pipeline] Fixed final_prune_fastgs iters={final_prune_iterations} (full-res all cameras)")
    if final_prune_iterations and not opt.vcp_enable:
        print("[Pipeline][Warn] vcp_enable=False, final_prune_fastgs will be skipped.")
    #----------------------------------打印训练信息----------------------------------



    viewpoint_stacks: dict[float, list[Camera] | None] = {scale: None for scale in get_training_resolution_scales(dataset)}
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_normal_prior_loss_for_log = 0.0
    ema_ncc_loss_for_log = 0.0
    ema_normal_smooth_loss_for_log = 0.0
    ema_mask_loss_for_log = 0.0
    os.makedirs(os.path.join(dataset.model_path, "debug"), exist_ok=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         (
        #             custom_cam,
        #             do_training,
        #             pipe.convert_SHs_python,
        #             pipe.compute_cov3D_python,
        #             keep_alive,
        #             scaling_modifer,
        #         ) = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(
        #                 custom_cam,
        #                 gaussians,
        #                 pipe,
        #                 background,
        #                 kernel_size,
        #                 scaling_modifer,
        #             )["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if gaussians.max_sh_degree == gaussians.max_sh_degree:
                gaussians.unlockSGdegree(100)
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # set normal prior loss lambda and ncc geo loss lambda
        # ------------------------------SET UP----------------------------------
        stage_cfg = get_stage_config(iteration, stage_configs)
        in_stage_geometry = stage_cfg.name == "geometry"
        ncc_geo_kick_on = stage_cfg.enable_ncc_geo and iteration <= int(opt.densify_until_iter)
        if in_stage_geometry:
            if has_loaded_normal_prior:
                ramp_denom = max(stage_cfg.end_iter - stage_cfg.start_iter, 1)
                stage_progress = (iteration - stage_cfg.start_iter) / ramp_denom
                # ncc_cur from 0.4 to 0.2
                # normal_prior_cur from 0.2 to 0.4
                lambda_multi_view_ncc_cur = 0.4 + (0.2 - 0.4) * stage_progress if ncc_geo_kick_on else 0.0
                lambda_normal_prior_cur = 0.2 + (0.4 - 0.2) * stage_progress
                # lambda_multi_view_ncc_cur = 0.2 if ncc_geo_kick_on else 0.0
                # lambda_normal_prior_cur = 0.2 + (0.4 - 0.2) * stage_progress
            else:
                lambda_multi_view_ncc_cur = 0.6 if ncc_geo_kick_on else 0.0
                lambda_normal_prior_cur = 0.0
        else:
            lambda_multi_view_ncc_cur = 0.0
            lambda_normal_prior_cur = 0.0

        depth_normal_loss_kick_on = stage_cfg.enable_depth_normal_loss
        normal_prior_kick_on = in_stage_geometry and stage_cfg.enable_normal_prior and has_loaded_normal_prior and lambda_normal_prior_cur > 0
        normal_smooth_kick_on = (
            normal_prior_kick_on
            and iteration > int(opt.densify_until_iter)
            and stage2_edge_aware_normal_smooth_lambda > 0
        )
        mask_kick_on = opt.lambda_mask > 0 and iteration >= opt.mask_from_iter and has_loaded_mask  # if load mask, then use mask loss

        depth_render_on = depth_normal_loss_kick_on or normal_prior_kick_on or ncc_geo_kick_on
        active_scale = stage_cfg.resolution_scale

        if viewpoint_stacks[active_scale] is None or len(viewpoint_stacks[active_scale]) == 0:
            viewpoint_stacks[active_scale] = scene.getTrainCameras(scale=active_scale).copy()

        viewpoint_cam = viewpoint_stacks[active_scale].pop(
            randint(0, len(viewpoint_stacks[active_scale]) - 1)
        )
        use_background_rgb = (
            stage_cfg.use_background_rgb
            and dataset.train_with_background_rgb
            and scene.should_train_with_bg
            and not has_loaded_mask
        )
        bg_model = scene.bg_gaussians if use_background_rgb else None
        # ------------------------------SET UP----------------------------------

        # render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size,
            require_depth=depth_render_on,
            bg_splats=bg_model,
        )
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # normal consistency / depth-derived normal
        if depth_render_on:
            depth_map: torch.Tensor = render_pkg["median_depth"]
            rendered_normal: torch.Tensor = render_pkg["normal"]
            depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
            if depth_normal_loss_kick_on and opt.lambda_depth_normal > 0:
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
                depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
            else:
                depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        else:
            depth_normal = None
            valid_points = None
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Normal prior loss
        if normal_prior_kick_on:
            rend_alpha = render_pkg["mask"]
            rendered_normal_prior = render_pkg["normal"]
            prior_normal = viewpoint_cam.normal_prior
            prior_normal_eff = prior_normal * rend_alpha.detach()
            prior_mask = viewpoint_cam.normal_prior_mask.squeeze(0)
            if valid_points is not None:
                prior_mask = prior_mask & valid_points.squeeze()

            if prior_mask.any().item():
                prior_cos = F.cosine_similarity(rendered_normal_prior, prior_normal_eff, dim=0)
                prior_error_render = 1.0 - torch.abs(prior_cos)

                if depth_normal is not None:
                    prior_cos_depth = F.cosine_similarity(depth_normal, prior_normal_eff, dim=0)
                    prior_error_depth = 1.0 - torch.abs(prior_cos_depth)
                else:
                    prior_error_depth = torch.zeros_like(prior_error_render)

                normal_prior_error = prior_error_render + prior_error_depth
                normal_prior_loss = normal_prior_error[prior_mask].mean()
            else:
                normal_prior_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        else:
            normal_prior_loss = torch.tensor([0], dtype=torch.float32, device="cuda")


        # patch match loss
        if patchmatch is not None and ncc_geo_kick_on and (lambda_multi_view_ncc_cur > 0 or opt.lambda_multi_view_geo > 0):
            nearest_cam = (
                None
                if len(viewpoint_cam.nearest_id) == 0
                else scene.getTrainCameras(scale=active_scale)[sample(viewpoint_cam.nearest_id, 1)[0]]
            )
            ncc_loss, geo_loss = patchmatch(gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # rgb loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0)))

        # normal smoothness loss
        if normal_smooth_kick_on:
            smooth_valid_mask = render_pkg["mask"].squeeze(0) > 1e-4
            if valid_points is not None:
                smooth_valid_mask = smooth_valid_mask & valid_points.squeeze()
            prior_normal_for_smooth = viewpoint_cam.normal_prior
            prior_mask_for_smooth = viewpoint_cam.normal_prior_mask
            if prior_normal_for_smooth is not None and prior_mask_for_smooth is not None:
                smooth_valid_mask = smooth_valid_mask & prior_mask_for_smooth.squeeze(0)
                normal_smooth_loss = edge_aware_normal_smoothness_loss(
                    normals=render_pkg["normal"],
                    guidance_prior_normal=prior_normal_for_smooth,
                    guidance_image=None,
                    valid_mask=smooth_valid_mask,
                    edge_scale=8.0,
                    image_grad_mix_ratio=0.0,
                )
            else:
                normal_smooth_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        else:
            normal_smooth_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # mask loss
        if mask_kick_on:
            opacity = 1.0 - render_pkg["mask"].clamp(1e-6, 1.0 - 1e-6)
            bg = 1.0 - viewpoint_cam.gt_mask
            mask_loss = (-bg * torch.log(opacity)).mean()
        else:
            mask_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        loss = (
            rgb_loss
            + opt.lambda_mask * mask_loss
            + opt.lambda_depth_normal * depth_normal_loss
            + lambda_normal_prior_cur * normal_prior_loss
            + stage2_edge_aware_normal_smooth_lambda * normal_smooth_loss
            + lambda_multi_view_ncc_cur * ncc_loss
            + opt.lambda_multi_view_geo * geo_loss
        )
        loss.backward()

        if bg_model is not None:
            foreground_count = gaussians.get_xyz.shape[0]
            radii = radii[:foreground_count]
            foreground_viewspace = viewspace_point_tensor[:foreground_count]
            if viewspace_point_tensor.grad is not None:
                foreground_viewspace.grad = viewspace_point_tensor.grad[:foreground_count]
            viewspace_point_tensor = foreground_viewspace
            visibility_filter = visibility_filter[:foreground_count]

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_normal_prior_loss_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_normal_prior_loss_for_log
            ema_normal_smooth_loss_for_log = 0.4 * normal_smooth_loss.item() + 0.6 * ema_normal_smooth_loss_for_log
            ema_ncc_loss_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_loss_for_log
            ema_mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "loss_mask": f"{ema_mask_loss_for_log:.{4}f}",
                        "loss_normal": f"{ema_normal_loss_for_log:.{4}f}",
                        "loss_normal_prior": f"{ema_normal_prior_loss_for_log:.{4}f}",
                        "loss_normal_smooth": f"{ema_normal_smooth_loss_for_log:.{4}f}",
                        "loss_ncc": f"{ema_ncc_loss_for_log:.{4}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1_render,
                loss,
                depth_normal_loss,
                normal_prior_loss,
                normal_smooth_loss,
                ncc_loss,
                mask_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, kernel_size),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # VCD/VCP
                importance_score = None
                pruning_score = None
                need_vcd = opt.vcd_enable and iteration >= opt.vcd_from_iter
                need_vcp = opt.vcp_enable and iteration >= opt.vcp_from_iter
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    scaling_finite = torch.isfinite(gaussians.get_scaling)
                    if not scaling_finite.all():
                        invalid_count = int((~scaling_finite).sum().item())
                        total_count = int(scaling_finite.numel())
                        print(f"[Warn][iter {iteration}] Non-finite scaling detected before densify: {invalid_count}/{total_count}")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if need_vcd or need_vcp:
                        camlist = sample_vcd_cameras(scene.getTrainCameras().copy(), opt.vcd_num_cams)
                        importance_score, pruning_score = compute_vcd_vcp_scores(
                            camlist=camlist,
                            gaussians=gaussians,
                            pipe=pipe,
                            background=background,
                            kernel_size=kernel_size,
                            loss_thresh=opt.vcd_loss_thresh,
                            need_vcd=need_vcd,
                            need_vcp=need_vcp,
                        )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.05,
                        scene.cameras_extent,
                        size_threshold,
                        importance_score=importance_score,
                        importance_threshold=opt.vcd_importance_thresh,
                        pruning_score=pruning_score,
                        vcp_remove_ratio=opt.vcp_remove_ratio,
                        outside_prune_radius=None,
                    )
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=full_res_train_cameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Fixed final pruning for render-oriented point count control.
            if opt.vcp_enable and iteration in final_prune_iterations:
                camlist = scene.getTrainCameras(scale=1.0).copy()
                _, final_pruning_score = compute_vcd_vcp_scores(
                    camlist=camlist,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    kernel_size=kernel_size,
                    loss_thresh=opt.vcd_loss_thresh,
                    need_vcd=False,
                    need_vcp=True,
                )
                gaussians.final_prune_fastgs(
                    min_opacity=0.1,
                    pruning_score=final_pruning_score,
                    score_threshold=0.9,
                    outside_prune_radius=(scene.scene_scale * 1.5) if scene.scene_scale is not None else None,
                )
                if dataset.disable_filter3D:
                    gaussians.reset_3D_filter()
                else:
                    gaussians.compute_3D_filter(cameras=full_res_train_cameras)

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=full_res_train_cameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer_step(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    normal_loss,
    normal_prior_loss,
    normal_smooth_loss,
    ncc_loss,
    mask_loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_prior_loss", normal_prior_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_smooth_loss", normal_smooth_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ncc_loss", ncc_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/mask_loss", mask_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    class ValidationConfig(TypedDict):
        name: str
        cameras: Sequence[Camera]

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs: tuple[ValidationConfig, ...] = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                        depth = render_result["median_depth"].squeeze().detach().cpu().numpy()
                        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                            depth_color.transpose(2, 0, 1)[None],
                            global_step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print(
                            "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test),
                            file=file_object,
                        )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15000, 25000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    # 三阶段
    # 第一阶段：初始化   0 - 5000
    # 第二阶段：几何优化 5001 - 15000
    # 第三阶段：渲染优化 15001 - 25000
    # 致密化 500 - 12000
    # 最终修剪 12001/21000
    stage1_end_iter = 5000
    stage2_end_iter = 15000
    stage3_start_iter = stage2_end_iter + 1
    stage3_use_depth_normal = True  # Manual toggle for Stage-3 depth-normal loss.
    stage2_edge_aware_normal_smooth_lambda = 0.1  # Active only in Stage-2 post-densify window.
    # stage2_edge_aware_normal_smooth_img_mix_ratio = 0.1
    fixed_final_prune_iterations = [stage3_start_iter, 21000]

    milestones = [stage1_end_iter, stage2_end_iter, stage3_start_iter, args.iterations]
    milestones = [it for it in milestones if 1 <= it <= args.iterations]
    args.test_iterations = sorted(set(args.test_iterations + milestones))
    # args.save_iterations = sorted(set(args.save_iterations + milestones))
    # args.checkpoint_iterations = sorted(set(args.checkpoint_iterations + milestones))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    stage_schedule = {
        "stage1_end_iter": stage1_end_iter,
        "stage2_end_iter": stage2_end_iter,
        "stage3_use_depth_normal": stage3_use_depth_normal,
        "stage2_edge_aware_normal_smooth_lambda": stage2_edge_aware_normal_smooth_lambda,
        "final_prune_iterations": fixed_final_prune_iterations,
    }

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        stage_schedule=stage_schedule,
    )

    # All done
    print("\nTraining complete.")
