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
import math
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import Random, randint, sample
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
from scene import GaussianModel, Scene
from scene.cameras import Camera, FramePayloadCache
from scene.pseudo_views import accumulate_pseudo_densification, load_pseudo_manifest, make_pseudo_cameras, pseudo_depth_l1, pseudo_rgb_l1, ramp_weight, screen_gradient_statistics
from scene.pseudo_seeds import admit_pseudo_seeds
from scene.train_split import CHECKPOINT_STATE_KEY, checkpoint_training_state, read_checkpoint_split, validate_checkpoint_split
from utils.general_utils import safe_state
from utils.graphics_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import PatchMatch, l1_loss, photometric_rgb_loss
from utils.vcd_utils import compute_vcd_vcp_scores, sample_vcd_cameras


def masked_l1_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        return pred_depth.new_tensor(0.0)
    return torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask]).mean()


def record_source_losses(training_stats, source, rgb_loss, depth_loss, rgb_weight, depth_weight, total_loss):
    record = training_stats["loss_by_source"].setdefault(source, {
        "samples": 0,
        "rgb_loss_sum": 0.0,
        "weighted_rgb_loss_sum": 0.0,
        "depth_loss_sum": 0.0,
        "weighted_depth_loss_sum": 0.0,
        "total_loss_sum": 0.0,
    })
    record["samples"] += 1
    rgb_value = float(rgb_loss.detach().item())
    depth_value = 0.0 if depth_loss is None else float(depth_loss.detach().item())
    record["rgb_loss_sum"] += rgb_value
    record["weighted_rgb_loss_sum"] += rgb_value * rgb_weight
    record["depth_loss_sum"] += depth_value
    record["weighted_depth_loss_sum"] += depth_value * depth_weight
    record["total_loss_sum"] += float(total_loss.detach().item())
    for field in ("rgb_loss", "weighted_rgb_loss", "depth_loss", "weighted_depth_loss", "total_loss"):
        record[field + "_mean"] = record[field + "_sum"] / record["samples"]


def weighted_charbonnier_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        return pred_depth.new_tensor(0.0)

    depth_err = pred_depth - gt_depth
    depth_charb = torch.sqrt(depth_err * depth_err + eps * eps)
    if weight_map is None:
        return depth_charb[valid_mask].mean()

    weights = weight_map[valid_mask]
    if weights.numel() == 0:
        return pred_depth.new_tensor(0.0)
    return (depth_charb[valid_mask] * weights).sum() / (weights.sum() + 1e-6)


def relative_charbonnier_depth_loss(pred_depth, gt_depth, valid_mask, eps=0.01, denominator_floor=0.05):
    valid_mask = valid_mask.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
    if not valid_mask.any():
        return pred_depth.new_tensor(0.0)
    scale = gt_depth[valid_mask].detach().clamp_min(denominator_floor)
    prediction, target = pred_depth[valid_mask] / scale, gt_depth[valid_mask] / scale
    return weighted_charbonnier_depth_loss(
        prediction, target, torch.ones_like(target, dtype=torch.bool), eps=eps,
    ) - eps


def masked_pearson_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = valid_mask.bool()
    if valid_mask.sum().item() < 2:
        return pred_depth.new_tensor(0.0)

    src = pred_depth[valid_mask]
    tgt = gt_depth[valid_mask]
    src = src - src.mean()
    tgt = tgt - tgt.mean()

    src_std = src.std(unbiased=False)
    tgt_std = tgt.std(unbiased=False)
    if src_std.item() < 1e-6 or tgt_std.item() < 1e-6:
        return pred_depth.new_tensor(0.0)

    src = src / (src_std + 1e-6)
    tgt = tgt / (tgt_std + 1e-6)
    corr = (src * tgt).mean()
    return 1.0 - corr


def masked_local_pearson_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence_map: torch.Tensor | None = None,
    box_p: int = 128,
    p_corr: float = 0.5,
    min_valid_ratio: float = 0.1,
) -> torch.Tensor:
    _, h, w = pred_depth.shape
    if h < box_p or w < box_p:
        return masked_pearson_depth_loss(pred_depth, gt_depth, valid_mask)

    num_box_h = max(h // box_p, 1)
    num_box_w = max(w // box_p, 1)
    n_corr = max(int(p_corr * num_box_h * num_box_w), 1)
    max_h = h - box_p + 1
    max_w = w - box_p + 1
    if confidence_map is not None:
        conf2d = confidence_map.squeeze(0).clamp(0.0, 1.0)
        valid2d = valid_mask.squeeze(0).float()
        conf_patch = F.avg_pool2d(conf2d[None, None], kernel_size=box_p, stride=1)[0, 0]
        valid_ratio_patch = F.avg_pool2d(valid2d[None, None], kernel_size=box_p, stride=1)[0, 0]
        sample_scores = conf_patch * (valid_ratio_patch >= min_valid_ratio).float()
        flat_scores = sample_scores.reshape(-1)
        if flat_scores.sum().item() > 0:
            sampled = torch.multinomial(flat_scores, n_corr, replacement=True)
            x_0 = torch.div(sampled, max_w, rounding_mode="floor")
            y_0 = sampled % max_w
        else:
            x_0 = torch.randint(0, max_h, size=(n_corr,), device=pred_depth.device)
            y_0 = torch.randint(0, max_w, size=(n_corr,), device=pred_depth.device)
    else:
        x_0 = torch.randint(0, max_h, size=(n_corr,), device=pred_depth.device)
        y_0 = torch.randint(0, max_w, size=(n_corr,), device=pred_depth.device)
    min_valid_pixels = max(int(box_p * box_p * min_valid_ratio), 1)

    loss_sum = pred_depth.new_tensor(0.0)
    valid_patch_count = 0
    for i in range(n_corr):
        x_start, y_start = int(x_0[i].item()), int(y_0[i].item())
        x_end, y_end = x_start + box_p, y_start + box_p
        patch_mask = valid_mask[:, x_start:x_end, y_start:y_end]
        if patch_mask.sum().item() < min_valid_pixels:
            continue
        patch_pred = pred_depth[:, x_start:x_end, y_start:y_end]
        patch_gt = gt_depth[:, x_start:x_end, y_start:y_end]
        loss_sum = loss_sum + masked_pearson_depth_loss(patch_pred, patch_gt, patch_mask)
        valid_patch_count += 1

    if valid_patch_count == 0:
        return masked_pearson_depth_loss(pred_depth, gt_depth, valid_mask)
    return loss_sum / valid_patch_count


@torch.no_grad()
def prune_low_contribution_gaussians(
    gaussians: GaussianModel,
    cameras: Sequence[Camera],
    pipe,
    bg: torch.Tensor,
    kernel_size: float,
    K: int = 5,
    prune_ratio: float = 0.1,
) -> None:
    if len(cameras) == 0 or K <= 0:
        return

    contributions = []
    for cam in cameras:
        transmittance_pkg = render(
            cam,
            gaussians,
            pipe,
            bg,
            kernel_size,
            require_depth=False,
            record_transmittance=True,
        )
        trans = transmittance_pkg["transmittance_avg"]
        if trans is not None:
            contributions.append(trans.to(torch.float32))

    if len(contributions) == 0:
        return

    k = min(K, len(contributions))
    contribution_stack = torch.stack(contributions, dim=0)
    contribution_topk = torch.topk(contribution_stack, k=k, dim=0, largest=True, sorted=False).values

    prune_ratio = max(0.0, min(float(prune_ratio), 1.0))
    contribution_score = contribution_topk.mean(dim=0)
    threshold = torch.quantile(contribution_score, prune_ratio)
    prune_mask = contribution_score < threshold
    if prune_mask.any().item():
        gaussians.prune_points(prune_mask)
        torch.cuda.empty_cache()


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    execution_args=None,
):
    if opt.opacity_reset_min_remaining < 0:
        raise ValueError("opacity_reset_min_remaining must be nonnegative")
    if opt.real_depth_loss_mode not in ("absolute", "relative_charbonnier"):
        raise ValueError("real_depth_loss_mode must be absolute or relative_charbonnier")
    if not math.isfinite(opt.real_depth_relative_epsilon) or opt.real_depth_relative_epsilon <= 0:
        raise ValueError("real_depth_relative_epsilon must be finite and positive")
    if not math.isfinite(opt.real_depth_relative_floor) or opt.real_depth_relative_floor <= 0:
        raise ValueError("real_depth_relative_floor must be finite and positive")
    if checkpoint and (getattr(dataset, "custom_init_ply", "") or getattr(dataset, "custom_init_metadata", "")):
        raise ValueError("Custom point-cloud initialization is a coldstart and cannot be combined with start_checkpoint")
    if not math.isfinite(opt.pseudo_densification_weight) or opt.pseudo_densification_weight < 0:
        raise ValueError("pseudo_densification_weight must be finite and nonnegative")
    if not math.isfinite(opt.pseudo_densification_depth_tolerance) or not 0 < opt.pseudo_densification_depth_tolerance < 1:
        raise ValueError("pseudo_densification_depth_tolerance must lie in (0,1)")
    if not math.isfinite(opt.pseudo_densification_min_confidence) or not 0 <= opt.pseudo_densification_min_confidence <= 1:
        raise ValueError("pseudo_densification_min_confidence must lie in [0,1]")
    if opt.pseudo_gradient_diagnostics_interval < 0:
        raise ValueError("pseudo_gradient_diagnostics_interval must be nonnegative")
    if opt.pseudo_densification_weight > 0 and not dataset.pseudo_manifest:
        raise ValueError("Pseudo densification requires a pseudo manifest")
    first_iter = 0
    model_params = None
    embedded_checkpoint_state = None
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint, weights_only=False)
        embedded_checkpoint_state = checkpoint_training_state(model_params, first_iter)
    checkpoint_split_record = None
    checkpoint_state_path = Path(checkpoint).with_suffix(".state.json") if checkpoint else None
    checkpoint_state = embedded_checkpoint_state
    if checkpoint_state is None:
        checkpoint_state = json.loads(checkpoint_state_path.read_text()) if checkpoint_state_path is not None and checkpoint_state_path.exists() else {}
        if checkpoint_state.get("seed_admissions"):
            raise ValueError("Legacy seeded checkpoint lacks embedded provenance; refusing an unverifiable seed resume")
    seed_admissions = list(checkpoint_state.get("seed_admissions", []))
    allow_split_expansion = bool(getattr(dataset, "allow_train_split_expansion", False))
    explicit_split = bool(getattr(dataset, "train_exclude_images", "")) or allow_split_expansion
    if allow_split_expansion and not checkpoint:
        raise ValueError("--allow_train_split_expansion requires an existing verified checkpoint")
    if checkpoint and (explicit_split or checkpoint_state.get("camera_split") is not None or (Path(checkpoint).parent / "camera_split.json").exists()):
        checkpoint_split_record = read_checkpoint_split(checkpoint, embedded_state=embedded_checkpoint_state)
    if checkpoint_split_record is not None and not explicit_split:
        raise ValueError("Resuming an explicit-split checkpoint requires an explicit exclusion list; use an empty list and --allow_train_split_expansion to absorb former heldout cameras")
    split_expansion_report = checkpoint_state.get("split_expansion")
    tb_writer = prepare_output_and_logger(dataset, opt, pipe, execution_args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians)
    frame_gpu_budget = getattr(dataset, "frame_gpu_budget", -1)
    if frame_gpu_budget is None:
        frame_gpu_budget = -1
    frame_cache = None if frame_gpu_budget == -1 else FramePayloadCache(frame_gpu_budget)
    gaussians.training_setup(opt)
    if checkpoint:
        if checkpoint_split_record is not None:
            previous_appearance = model_params[16] if len(model_params) == 19 else model_params[17]
            expansion = validate_checkpoint_split(
                checkpoint, dataset.model_path, previous_record=checkpoint_split_record,
                allow_expansion=allow_split_expansion,
                current_appearance=dataset.use_decoupled_appearance,
                checkpoint_appearance=previous_appearance,
            )
            if expansion["expanded"]:
                split_expansion_report = {**expansion, "checkpoint": str(Path(checkpoint).resolve())}
                (Path(dataset.model_path) / "train_split_expansion.json").write_text(json.dumps(split_expansion_report, indent=2))
                print(f"[Split] Explicitly expanded training cameras {expansion['previous_train_count']} -> {expansion['current_train_count']}; former heldout images are now training views")
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = scene.getTrainCameras().copy()
    seed_report = admit_pseudo_seeds(
        dataset.pseudo_seed_manifest, gaussians, trainCameras, opt, dataset.model_path, first_iter,
        previous_admissions=seed_admissions,
    )
    if seed_report is not None:
        if seed_report["appended_gaussians"] > 0:
            seed_admissions.append(seed_report["source_sha256"])
        print(f"[Seeds] Appended {seed_report['appended_gaussians']} verified real-depth Gaussians")
    pseudo_cameras = []
    if dataset.pseudo_manifest:
        if opt.pseudo_real_updates < 1 or opt.pseudo_from_iter < 0 or opt.pseudo_ramp_iters < 0:
            raise ValueError("Pseudo sampling requires positive pseudo_real_updates and nonnegative start/ramp iterations")
        if not math.isfinite(opt.pseudo_rgb_weight) or opt.pseudo_rgb_weight <= 0:
            raise ValueError("pseudo_rgb_weight must be finite and positive when pseudo_manifest is supplied")
        if not math.isfinite(opt.pseudo_depth_weight) or opt.pseudo_depth_weight < 0:
            raise ValueError("pseudo_depth_weight must be finite and nonnegative")
        if opt.pseudo_depth_weight > 0 and opt.pseudo_depth_weight >= opt.lambda_depth_prior:
            raise ValueError("pseudo_depth_weight must remain below the trusted real depth weight")
        pseudo_views, manifest_hash = load_pseudo_manifest(dataset.pseudo_manifest, trainCameras)
        if opt.pseudo_depth_weight > 0 and not any(view["depth"] is not None for view in pseudo_views):
            raise ValueError("Positive pseudo_depth_weight requires at least one depth-supervised pseudo view")
        if opt.pseudo_densification_weight > 0 and not any(view["densification_mask"] is not None and view["densification_mask"].any().item() for view in pseudo_views):
            raise ValueError("Pseudo densification requires actual real-depth-supported geometry pixels")
        pseudo_cameras = make_pseudo_cameras(pseudo_views, dataset.data_device, frame_cache is not None)
        manifest_snapshot = []
        for view in pseudo_views:
            manifest_snapshot.append({
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in view.items() if key not in (
                    "image", "valid_mask", "confidence", "depth", "depth_mask", "depth_confidence", "densification_mask"
                )
            })
        with open(os.path.join(dataset.model_path, "pseudo_manifest.json"), "w") as manifest_file:
            json.dump({"schema_version": 1, "source_sha256": manifest_hash, "views": manifest_snapshot}, manifest_file, indent=2)
        pseudo_mode = "additive" if opt.pseudo_additive else "replacement"
        print(f"[Pseudo] Loaded {len(pseudo_cameras)} isolated RGB cameras; {opt.pseudo_real_updates} real : 1 pseudo ({pseudo_mode})")
    if opt.depth_prior_ramp_iters < 0:
        raise ValueError("depth_prior_ramp_iters must be nonnegative")
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    if opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0:
        patchmatch = PatchMatch(
            opt.multi_view_patch_size,
            opt.multi_view_pixel_noise_th,
            kernel_size=kernel_size,
            pipe=pipe,
            debug=True,
            model_path=dataset.model_path,
        )

    if os.path.isabs(dataset.normal_prior_dir):
        normal_root = dataset.normal_prior_dir
    else:
        normal_root = os.path.join(dataset.source_path, dataset.normal_prior_dir)
    has_normal_dir = os.path.isdir(normal_root)
    has_loaded_normal_prior = any(cam.has_normal_prior for cam in scene.getTrainCameras())
    if has_normal_dir and not has_loaded_normal_prior:
        print("[Pipeline][Warn] normals directory exists but no valid normal priors were loaded.")
    print(f"[Pipeline] has_normal_dir={has_normal_dir} (normal_dir={normal_root}, loaded_priors={has_loaded_normal_prior})")

    if dataset.depth_prior_dir.strip():
        if os.path.isabs(dataset.depth_prior_dir):
            depth_root = dataset.depth_prior_dir
        else:
            depth_root = os.path.join(dataset.source_path, dataset.depth_prior_dir)
    else:
        depth_root = None
    has_depth_dir = depth_root is not None and os.path.isdir(depth_root)
    has_loaded_depth_prior = any(cam.has_depth_prior for cam in scene.getTrainCameras())
    if has_depth_dir and not has_loaded_depth_prior:
        print("[Pipeline][Warn] depth directory exists but no valid depth priors were loaded.")
    print(f"[Pipeline] depth_prior_ready={has_loaded_depth_prior} (depth_dir={depth_root})")

    scene_case = has_normal_dir and has_loaded_depth_prior
    reflective_case = has_normal_dir and not has_loaded_depth_prior
    print(f"[Pipeline] scene_case={scene_case}, reflective_case={reflective_case}")

    viewpoint_stack = None
    pseudo_stack = []
    pseudo_rng = Random(0)
    training_stats = {
        "start_iteration": first_iter,
        "training_seed": getattr(execution_args, "training_seed", 0),
        "pseudo_sampling_seed": 0,
        "rng_resume_policy": "Process seed reset; checkpoint RNG and camera/pseudo stack states are not restored",
        "real_samples": 0,
        "pseudo_samples": 0,
        "actual_main_optimizer_steps": 0,
        "actual_sh_optimizer_steps": 0,
        "opacity_reset_min_remaining": opt.opacity_reset_min_remaining,
        "opacity_reset_iterations": [],
        "opacity_reset_skipped_iterations": [],
        "opacity_restore": getattr(gaussians, "opacity_restore_report", None),
        "loss_by_source": {},
        "real_depth_loss_mode": opt.real_depth_loss_mode,
        "real_depth_relative_epsilon": opt.real_depth_relative_epsilon,
        "real_depth_relative_floor": opt.real_depth_relative_floor,
        "pseudo_densification_weight": opt.pseudo_densification_weight,
        "pseudo_densification_observations": 0,
        "pseudo_densification_eligible_gaussians_sum": 0,
        "pseudo_densification_screen_gradient_norm_sum": 0.0,
        "pseudo_densification_by_mode": {"additive": 0, "replacement": 0},
        "gradient_diagnostics": [],
        "densify_until_iter": opt.densify_until_iter,
        "late_prune_after_iteration": max(15000, opt.densify_until_iter),
        "densification_count": 0,
        "densification_iterations": [],
        "densification_gaussian_counts": [],
        "late_prune_count": 0,
        "late_prune_iterations": [],
        "late_prune_gaussian_counts": [],
    }

    def count_main_step(optimizer, step_args, step_kwargs):
        training_stats["actual_main_optimizer_steps"] += 1

    def count_sh_step(optimizer, step_args, step_kwargs):
        training_stats["actual_sh_optimizer_steps"] += 1

    optimizer_hooks = []
    if gaussians.optimizer is not None:
        optimizer_hooks.append(gaussians.optimizer.register_step_post_hook(count_main_step))
    if gaussians.shoptimizer is not None:
        optimizer_hooks.append(gaussians.shoptimizer.register_step_post_hook(count_sh_step))
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_normal_prior_loss_for_log = 0.0
    ema_depth_prior_loss_for_log = 0.0
    ema_ncc_loss_for_log = 0.0
    ema_mask_loss_for_log = 0.0
    os.makedirs(os.path.join(dataset.model_path, "debug"), exist_ok=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        kernel_size,
                        scaling_modifer,
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if gaussians.max_sh_degree == gaussians.max_sh_degree:
                gaussians.unlockSGdegree(100)
            gaussians.oneupSHdegree()

        # Pick a random Camera
        pseudo_due = bool(pseudo_cameras) and iteration >= opt.pseudo_from_iter and (
            (iteration - opt.pseudo_from_iter + 1) % (opt.pseudo_real_updates + int(not opt.pseudo_additive)) == 0
        )
        is_pseudo = pseudo_due and not opt.pseudo_additive
        if is_pseudo:
            if not pseudo_stack:
                pseudo_stack = pseudo_cameras.copy()
            viewpoint_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        training_stats["pseudo_samples" if is_pseudo else "real_samples"] += 1
        if frame_cache is not None:
            viewpoint_cam = frame_cache.ensure(viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        lambda_multi_view_ncc_cur = 0.02
        lambda_normal_prior_cur = 0.0

        reg_kick_on = not is_pseudo and iteration >= opt.regularization_from_iter
        normal_prior_kick_on = (
            not is_pseudo and (reflective_case or scene_case)
            and
            lambda_normal_prior_cur > 0
            and viewpoint_cam.normal_prior is not None
        )
        depth_prior_kick_on = (
            not is_pseudo and opt.lambda_depth_prior > 0
            and iteration >= opt.depth_prior_from_iter
            and viewpoint_cam.depth_prior is not None
        )
        pseudo_depth_kick_on = is_pseudo and opt.pseudo_depth_weight > 0 and viewpoint_cam.pseudo_has_depth
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size,
            require_depth=reg_kick_on or normal_prior_kick_on or depth_prior_kick_on or pseudo_depth_kick_on,
        )
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()

        pseudo_weight = 0.0
        pseudo_depth_weight = 0.0
        pseudo_l1_loss = None
        pseudo_depth_loss = None
        pseudo_source = None
        if is_pseudo:
            Ll1_render = pseudo_rgb_l1(
                rendered_image, gt_image, viewpoint_cam.pseudo_valid_mask_cpu, gaussians, viewpoint_cam.uid,
                viewpoint_cam.pseudo_confidence_cpu,
            )
            pseudo_l1_loss = Ll1_render
            pseudo_weight = opt.pseudo_rgb_weight * ramp_weight(iteration, opt.pseudo_from_iter, opt.pseudo_ramp_iters)
            rgb_loss = pseudo_weight * Ll1_render
            pseudo_source = viewpoint_cam.pseudo_source
            if pseudo_depth_kick_on:
                pseudo_depth_loss = pseudo_depth_l1(render_pkg["median_depth"], viewpoint_cam)
                pseudo_depth_weight = (
                    opt.pseudo_depth_weight * ramp_weight(iteration, opt.pseudo_from_iter, opt.pseudo_ramp_iters)
                    * ramp_weight(iteration, opt.depth_prior_from_iter, opt.depth_prior_ramp_iters)
                )
        else:
            rgb_loss, Ll1_render = photometric_rgb_loss(
                rendered_image, gt_image, gaussians, viewpoint_cam.uid, opt.lambda_dssim
            )
        # normal consistency / depth-derived normal
        if reg_kick_on or normal_prior_kick_on or depth_prior_kick_on:
            depth_map: torch.Tensor = render_pkg["median_depth"]
        else:
            depth_map = None

        # depth normal loss
        if reg_kick_on or normal_prior_kick_on:
            rendered_normal: torch.Tensor = render_pkg["normal"]
            depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
            if reg_kick_on and opt.lambda_depth_normal > 0:
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
                depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
            else:
                depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        else:
            depth_normal = None
            valid_points = None
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # normal prior loss
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

        # depth prior loss
        if depth_prior_kick_on and depth_map is not None:
            gt_depth_prior = viewpoint_cam.depth_prior
            valid_depth_mask = gt_depth_prior > 0.01
            if valid_depth_mask.any().item():
                if opt.real_depth_loss_mode == "relative_charbonnier":
                    depth_prior_loss = relative_charbonnier_depth_loss(
                        depth_map, gt_depth_prior, valid_depth_mask,
                        eps=opt.real_depth_relative_epsilon, denominator_floor=opt.real_depth_relative_floor,
                    )
                else:
                    depth_prior_loss = masked_l1_depth_loss(
                        depth_map, gt_depth_prior, valid_depth_mask
                    )
            else:
                print(f"[ITER {iteration}] valid_depth_mask none for {viewpoint_cam.image_name}")
                depth_prior_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        else:
            depth_prior_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # patch match loss
        if reg_kick_on and (lambda_multi_view_ncc_cur > 0 or opt.lambda_multi_view_geo):
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[sample(viewpoint_cam.nearest_id, 1)[0]]
            if nearest_cam is not None and frame_cache is not None:
                nearest_cam = frame_cache.ensure(nearest_cam, keep=(viewpoint_cam,))
            ncc_loss, geo_loss = patchmatch(gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # mask loss
        if not is_pseudo and opt.lambda_mask > 0 and viewpoint_cam.gt_mask is not None:
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
            + opt.lambda_depth_prior * ramp_weight(iteration, opt.depth_prior_from_iter, opt.depth_prior_ramp_iters) * depth_prior_loss
            + lambda_multi_view_ncc_cur * ncc_loss
            + opt.lambda_multi_view_geo * geo_loss
        )
        if pseudo_depth_loss is not None:
            loss = loss + pseudo_depth_weight * pseudo_depth_loss
        real_loss = None if is_pseudo else loss
        pseudo_render_pkg = None
        if opt.pseudo_additive and pseudo_due:
            if not pseudo_stack:
                pseudo_stack = pseudo_cameras.copy()
            pseudo_cam = pseudo_stack.pop(pseudo_rng.randint(0, len(pseudo_stack) - 1))
            if frame_cache is not None:
                pseudo_cam = frame_cache.ensure(pseudo_cam, keep=(viewpoint_cam,))
            pseudo_depth_kick_on = opt.pseudo_depth_weight > 0 and pseudo_cam.pseudo_has_depth
            pseudo_render_pkg = render(pseudo_cam, gaussians, pipe, background, kernel_size, require_depth=pseudo_depth_kick_on)
            pseudo_l1_loss = pseudo_rgb_l1(
                pseudo_render_pkg["render"], pseudo_cam.original_image.cuda(),
                pseudo_cam.pseudo_valid_mask_cpu, gaussians, pseudo_cam.uid,
                pseudo_cam.pseudo_confidence_cpu,
            )
            pseudo_weight = opt.pseudo_rgb_weight * ramp_weight(iteration, opt.pseudo_from_iter, opt.pseudo_ramp_iters)
            loss = loss + pseudo_weight * pseudo_l1_loss
            pseudo_source = pseudo_cam.pseudo_source
            if pseudo_depth_kick_on:
                pseudo_depth_loss = pseudo_depth_l1(pseudo_render_pkg["median_depth"], pseudo_cam)
                pseudo_depth_weight = (
                    opt.pseudo_depth_weight * ramp_weight(iteration, opt.pseudo_from_iter, opt.pseudo_ramp_iters)
                    * ramp_weight(iteration, opt.depth_prior_from_iter, opt.depth_prior_ramp_iters)
                )
                loss = loss + pseudo_depth_weight * pseudo_depth_loss
            training_stats["pseudo_samples"] += 1
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if opt.pseudo_gradient_diagnostics_interval and (
                iteration % opt.pseudo_gradient_diagnostics_interval == 0
                or (pseudo_render_pkg is not None and training_stats["pseudo_samples"] % opt.pseudo_gradient_diagnostics_interval == 1)
            ):
                diagnostic = {"iteration": iteration, "real_screen": screen_gradient_statistics(render_pkg) if not is_pseudo else None}
                if pseudo_render_pkg is not None:
                    diagnostic["pseudo_screen"] = screen_gradient_statistics(pseudo_render_pkg)
                elif is_pseudo:
                    diagnostic["pseudo_screen"] = screen_gradient_statistics(render_pkg)
                diagnostic["merged_parameter_gradient_norms"] = {
                    name: float(parameter.grad.norm().item()) if parameter.grad is not None else None
                    for name, parameter in (("xyz", gaussians._xyz), ("scaling", gaussians._scaling), ("opacity", gaussians._opacity), ("sh_dc", gaussians._features_dc), ("sh_rest", gaussians._features_rest))
                }
                diagnostic["scope"] = "Screen-space gradients are separate render branches; parameter gradients are merged real+pseudo, not causal source attribution."
                training_stats["gradient_diagnostics"].append(diagnostic)
            if real_loss is not None:
                record_source_losses(
                    training_stats, "real", rgb_loss, depth_prior_loss, 1.0,
                    opt.lambda_depth_prior * ramp_weight(iteration, opt.depth_prior_from_iter, opt.depth_prior_ramp_iters),
                    real_loss,
                )
            if pseudo_source is not None:
                pseudo_total_loss = pseudo_weight * pseudo_l1_loss
                if pseudo_depth_loss is not None:
                    pseudo_total_loss = pseudo_total_loss + pseudo_depth_weight * pseudo_depth_loss
                record_source_losses(
                    training_stats, "pseudo:" + pseudo_source, pseudo_l1_loss, pseudo_depth_loss,
                    pseudo_weight, pseudo_depth_weight, pseudo_total_loss,
                )
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_normal_prior_loss_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_normal_prior_loss_for_log
            ema_depth_prior_loss_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_depth_prior_loss_for_log
            ema_ncc_loss_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_loss_for_log
            ema_mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "loss_depth_prior": f"{ema_depth_prior_loss_for_log:.{4}f}",
                        "loss_mask": f"{ema_mask_loss_for_log:.{4}f}",
                        "loss_normal": f"{ema_normal_loss_for_log:.{4}f}",
                        "loss_normal_prior": f"{ema_normal_prior_loss_for_log:.{4}f}",
                        "loss_ncc": f"{ema_ncc_loss_for_log:.{4}f}",
                    }
                )
                progress_bar.update(10)
            if iteration % 100 == 0:
                print(f"\n[ITER {iteration}] depth_prior_loss {ema_depth_prior_loss_for_log:.4f}")
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
                depth_prior_loss,
                ncc_loss,
                mask_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, kernel_size),
                frame_cache,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if not is_pseudo:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if opt.pseudo_densification_weight > 0 and (is_pseudo or pseudo_render_pkg is not None):
                    densification_render = render_pkg if is_pseudo else pseudo_render_pkg
                    densification_camera = viewpoint_cam if is_pseudo else pseudo_cam
                    pseudo_densification = accumulate_pseudo_densification(
                        gaussians, densification_render, densification_camera, opt.pseudo_densification_weight,
                        opt.pseudo_densification_depth_tolerance, opt.pseudo_densification_min_confidence,
                    )
                    training_stats["pseudo_densification_observations"] += 1
                    training_stats["pseudo_densification_by_mode"]["replacement" if is_pseudo else "additive"] += 1
                    training_stats["pseudo_densification_eligible_gaussians_sum"] += pseudo_densification["eligible_gaussians"]
                    training_stats["pseudo_densification_screen_gradient_norm_sum"] += pseudo_densification["screen_gradient_norm_sum"]

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    scaling_finite = torch.isfinite(gaussians.get_scaling)
                    if not scaling_finite.all():
                        invalid_count = int((~scaling_finite).sum().item())
                        total_count = int(scaling_finite.numel())
                        print(f"[Warn][iter {iteration}] Non-finite scaling detected before densify: {invalid_count}/{total_count}")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    importance_score = None
                    pruning_score = None
                    need_vcd = opt.vcd_enable and iteration >= opt.vcd_from_iter
                    need_vcp = opt.vcp_enable and iteration >= opt.vcp_from_iter
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
                            frame_cache=frame_cache,
                        )
                    before_densification = len(gaussians.get_xyz)
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.05,
                        scene.cameras_extent,
                        size_threshold,
                        importance_score=importance_score,
                        importance_threshold=opt.vcd_importance_thresh,
                        pruning_score=pruning_score,
                        vcp_remove_ratio=opt.vcp_remove_ratio,
                    )
                    training_stats["densification_count"] += 1
                    training_stats["densification_iterations"].append(iteration)
                    training_stats["densification_gaussian_counts"].append({
                        "iteration": iteration, "before": before_densification, "after": len(gaussians.get_xyz),
                    })
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                # if iteration > opt.contribution_prune_from_iter and iteration % opt.contribution_prune_interval == 0:
                #     if iteration % opt.opacity_reset_interval == opt.contribution_prune_interval:
                #         print(f"[Iter {iteration}] Skipped contribution pruning near opacity reset.")
                #     else:
                #         prune_low_contribution_gaussians(
                #             gaussians,
                #             trainCameras[::2],
                #             pipe,
                #             background,
                #             kernel_size,
                #             K=1,
                #             prune_ratio=opt.contribution_prune_ratio,
                #         )
                #         if dataset.disable_filter3D:
                #             gaussians.reset_3D_filter()
                #         else:
                #             gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if opt.iterations - iteration >= opt.opacity_reset_min_remaining:
                        gaussians.reset_opacity()
                        training_stats["opacity_reset_iterations"].append(iteration)
                    else:
                        training_stats["opacity_reset_skipped_iterations"].append(iteration)
                        print(f"[Opacity][ITER {iteration}] Skipped reset: remaining={opt.iterations - iteration} < minimum={opt.opacity_reset_min_remaining}")

            # FastGS-style final-stage pruning: every 3k iterations after 15k.
            if opt.vcp_enable and iteration % 3000 == 0 and iteration > max(15000, opt.densify_until_iter) and iteration < 30_000:
                camlist = sample_vcd_cameras(scene.getTrainCameras().copy(), opt.vcd_num_cams)
                _, final_pruning_score = compute_vcd_vcp_scores(
                    camlist=camlist,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    kernel_size=kernel_size,
                    loss_thresh=opt.vcd_loss_thresh,
                    need_vcd=False,
                    need_vcp=True,
                    frame_cache=frame_cache,
                )
                before_late_prune = len(gaussians.get_xyz)
                gaussians.final_prune_fastgs(
                    min_opacity=0.1,
                    pruning_score=final_pruning_score,
                    score_threshold=0.9,
                )
                training_stats["late_prune_count"] += 1
                training_stats["late_prune_iterations"].append(iteration)
                training_stats["late_prune_gaussian_counts"].append({
                    "iteration": iteration, "before": before_late_prune, "after": len(gaussians.get_xyz),
                })
                if dataset.disable_filter3D:
                    gaussians.reset_3D_filter()
                else:
                    gaussians.compute_3D_filter(cameras=trainCameras)

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer_step(iteration)

            if tb_writer and pseudo_cameras:
                tb_writer.add_scalar("pseudo/rgb_weight", pseudo_weight, iteration)
                tb_writer.add_scalar("pseudo/depth_weight", pseudo_depth_weight, iteration)
                for stat_name in ("real_samples", "pseudo_samples", "actual_main_optimizer_steps", "actual_sh_optimizer_steps"):
                    tb_writer.add_scalar("pseudo/" + stat_name, training_stats[stat_name], iteration)
                if pseudo_l1_loss is not None:
                    tb_writer.add_scalar("pseudo/masked_l1", pseudo_l1_loss.item(), iteration)
                if pseudo_depth_loss is not None:
                    tb_writer.add_scalar("pseudo/masked_depth_l1", pseudo_depth_loss.item(), iteration)

            if iteration % 100 == 0 or iteration in checkpoint_iterations or iteration == opt.iterations:
                training_stats["last_iteration"] = iteration
                with open(os.path.join(dataset.model_path, "training_stats.json"), "w") as stats_file:
                    json.dump(training_stats, stats_file, indent=2)
                if pseudo_cameras:
                    print(f"[Pseudo][ITER {iteration}] real_samples={training_stats['real_samples']} pseudo_samples={training_stats['pseudo_samples']} main_steps={training_stats['actual_main_optimizer_steps']} sh_steps={training_stats['actual_sh_optimizer_steps']}")

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                split_path = Path(scene.model_path) / "camera_split.json"
                checkpoint_state = {
                    "schema_version": 1,
                    "iteration": iteration,
                    "training_seed": getattr(execution_args, "training_seed", 0),
                    "rng_resume_policy": "Process seed reset; checkpoint RNG and camera/pseudo stack states are not restored",
                    "camera_split": json.loads(split_path.read_text()) if split_path.exists() else None,
                    "seed_admissions": seed_admissions,
                    "split_expansion": split_expansion_report,
                }
                captured = gaussians.capture()
                captured[14][CHECKPOINT_STATE_KEY] = checkpoint_state
                checkpoint_path = Path(scene.model_path) / f"chkpnt{iteration}.pth"
                temporary_checkpoint_path = checkpoint_path.with_suffix(".pth.tmp")
                torch.save((captured, iteration), temporary_checkpoint_path)
                os.replace(temporary_checkpoint_path, checkpoint_path)
                checkpoint_hash = hashlib.sha256()
                with checkpoint_path.open("rb") as checkpoint_file:
                    for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
                        checkpoint_hash.update(chunk)
                state_path = Path(scene.model_path) / f"chkpnt{iteration}.state.json"
                temporary_state_path = state_path.with_suffix(".json.tmp")
                temporary_state_path.write_text(json.dumps({
                    **checkpoint_state, "checkpoint_sha256": checkpoint_hash.hexdigest(),
                }, indent=2))
                os.replace(temporary_state_path, state_path)
    for optimizer_hook in optimizer_hooks:
        optimizer_hook.remove()


def prepare_output_and_logger(args, opt=None, pipe=None, execution_args=None):
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
    config_groups = {
        "execution": vars(execution_args) if execution_args is not None else {},
        "model": vars(args),
        "optimization": vars(opt) if opt is not None else {},
        "pipeline": vars(pipe) if pipe is not None else {},
    }
    resolved_args = {key: value for group in config_groups.values() for key, value in group.items()}
    effective_loss_weights = {} if opt is None else {
        "lambda_dssim": opt.lambda_dssim,
        "lambda_mask": opt.lambda_mask,
        "lambda_depth_normal": opt.lambda_depth_normal,
        "lambda_normal_prior": 0.0,
        "lambda_depth_prior_max": opt.lambda_depth_prior,
        "lambda_multi_view_ncc": 0.02,
        "lambda_multi_view_geo": opt.lambda_multi_view_geo,
        "pseudo_rgb_weight_max": opt.pseudo_rgb_weight,
        "pseudo_depth_weight_max": opt.pseudo_depth_weight,
        "pseudo_densification_weight": opt.pseudo_densification_weight,
    }
    with open(os.path.join(args.model_path, "training_config.json"), "w") as config_file:
        json.dump({
            **config_groups, "resolved_args": resolved_args, "effective_loss_weights": effective_loss_weights,
            "effective_real_depth_loss": {
                "mode": opt.real_depth_loss_mode,
                "relative_epsilon": opt.real_depth_relative_epsilon,
                "denominator_floor_scene_units": opt.real_depth_relative_floor,
                "loss_units": "scene" if opt.real_depth_loss_mode == "absolute" else "dimensionless_relative",
            } if opt is not None else {},
            "effective_optimizer_schedule": "dense_main_sh_every_16" if opt is not None and opt.dense_optimizer_updates else "legacy_fastgs",
            "argv": sys.argv,
        }, config_file, indent=2)

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
    depth_prior_loss,
    ncc_loss,
    mask_loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    frame_cache=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_prior_loss", normal_prior_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/depth_prior_loss", depth_prior_loss.item(), iteration)
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
                    if frame_cache is not None:
                        viewpoint = frame_cache.ensure(viewpoint)
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image_float()
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000, 20000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,7000, 20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--training_seed", type=int, default=0)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if not 0 <= args.training_seed < 2**32:
        parser.error("training_seed must lie in [0, 2**32)")
    if args.opacity_reset_min_remaining < 0:
        parser.error("opacity_reset_min_remaining must be nonnegative")
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.training_seed)

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
        execution_args=args,
    )

    # All done
    print("\nTraining complete.")
