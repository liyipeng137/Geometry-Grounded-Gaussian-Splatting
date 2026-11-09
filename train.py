#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr
#
"""GGGS 真实相机基线训练。

阅读顺序：main → training → update_topology。
损失细节在 utils/training_losses.py；配置、日志、评估和 checkpoint 在
utils/training_io.py。这里保留训练阶段的先后关系，不隐藏在 Trainer 框架中。

此入口不再包含伪视角 / R2C 实验。初始化仍是当前训练相机的 RGBD→TSDF
点云，并非直接加载 tsdf_mesh.ply。normal prior 与 NCC 默认采用 train_demo 日程；
训练帧必须同时有 normal/depth prior，详见 docs/training_baseline.md。
"""

import math
from argparse import ArgumentParser
from random import randint, sample

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.cameras import FramePayloadCache
from utils.general_utils import safe_state
from utils.loss_utils import PatchMatch
from utils.training_io import (
    TrainingLogger,
    evaluate_views,
    load_checkpoint,
    prepare_output,
    restore_checkpoint,
    save_checkpoint,
)
from utils.training_losses import compute_training_losses, loss_schedule
from utils.vcd_utils import compute_vcd_vcp_scores, sample_vcd_cameras


def validate_options(opt, args):
    """启动前检查会影响训练阶段的参数，不把错误留到数千步后才暴露。"""
    if not 0 <= args.training_seed < 2**32:
        raise ValueError("training_seed must lie in [0, 2**32)")
    for name in ("iterations", "densification_interval", "opacity_reset_interval"):
        if getattr(opt, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    for name in (
        "opacity_reset_min_remaining",
        "depth_prior_ramp_iters",
        "regularization_from_iter",
        "normal_prior_from_iter",
        "depth_prior_from_iter",
    ):
        if getattr(opt, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "lambda_depth_normal",
        "lambda_normal_prior",
        "lambda_depth_prior",
        "lambda_multi_view_ncc",
        "lambda_multi_view_geo",
        "lambda_mask",
    ):
        value = getattr(opt, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= opt.lambda_dssim <= 1:
        raise ValueError("lambda_dssim must lie in [0, 1]")
    if opt.real_depth_loss_mode not in ("absolute", "relative_charbonnier"):
        raise ValueError(
            "real_depth_loss_mode must be absolute or relative_charbonnier"
        )
    for name in ("real_depth_relative_epsilon", "real_depth_relative_floor"):
        value = getattr(opt, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def validate_training_priors(cameras):
    """逐训练帧检查两种 prior；即使损失权重为零，输入契约也不降级。

    检查加载标记，不能检查 GPU 属性：合法的 lazy payload 此时可能仅驻留 CPU。
    """
    for camera in cameras:
        missing = [name for name in ("normal", "depth") if not getattr(camera, f"has_{name}_prior")]
        if missing:
            raise ValueError(f"Training frame {camera.image_name} is missing required prior(s): {', '.join(missing)}")


def refresh_filter(gaussians, dataset, cameras):
    """点数/尺度变化后重建 Mip 3D filter；显式禁用时保持零 filter。"""
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=cameras)


@torch.no_grad()
def update_topology(
    iteration,
    gaussians,
    scene,
    dataset,
    opt,
    pipe,
    background,
    buffers,
    stats,
    frame_cache,
):
    """沿用已有 VCD/VCP、opacity reset 与 filter 日程。

    必须在 backward 后、optimizer_step 前执行。这里会替换参数及对应 Adam
    状态，改动顺序会改变原 FastGS 配方。250 万点门槛仍在 GaussianModel 内，
    它是增密前门控，并非严格的增密后硬上限。
    """
    cameras = scene.getTrainCameras()
    if iteration < opt.densify_until_iter:
        visible = buffers["visibility_filter"]
        gaussians.max_radii2D[visible] = torch.maximum(
            gaussians.max_radii2D[visible], buffers["radii"][visible]
        )
        gaussians.add_densification_stats(buffers["viewspace_points"], visible)

        if (
            iteration > opt.densify_from_iter
            and iteration % opt.densification_interval == 0
        ):
            if not torch.isfinite(gaussians.get_scaling).all():
                print(
                    f"[Warn][ITER {iteration}] Non-finite scaling before densification"
                )
            importance, pruning = None, None
            need_vcd = opt.vcd_enable and iteration >= opt.vcd_from_iter
            need_vcp = opt.vcp_enable and iteration >= opt.vcp_from_iter
            if need_vcd or need_vcp:
                importance, pruning = compute_vcd_vcp_scores(
                    camlist=sample_vcd_cameras(cameras.copy(), opt.vcd_num_cams),
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    kernel_size=dataset.kernel_size,
                    loss_thresh=opt.vcd_loss_thresh,
                    need_vcd=need_vcd,
                    need_vcp=need_vcp,
                    frame_cache=frame_cache,
                )
            before = len(gaussians.get_xyz)
            gaussians.densify_and_prune(
                opt.densify_grad_threshold,
                0.05,
                scene.cameras_extent,
                20 if iteration > opt.opacity_reset_interval else None,
                importance_score=importance,
                importance_threshold=opt.vcd_importance_thresh,
                pruning_score=pruning,
                vcp_remove_ratio=opt.vcp_remove_ratio,
            )
            stats["densification_count"] += 1
            stats["densification_iterations"].append(iteration)
            stats["densification_gaussian_counts"].append(
                {
                    "iteration": iteration,
                    "before": before,
                    "after": len(gaussians.get_xyz),
                }
            )
            refresh_filter(gaussians, dataset, cameras)

        reset_due = iteration % opt.opacity_reset_interval == 0 or (
            dataset.white_background and iteration == opt.densify_from_iter
        )
        if reset_due:
            if opt.iterations - iteration >= opt.opacity_reset_min_remaining:
                gaussians.reset_opacity()
                stats["opacity_reset_iterations"].append(iteration)
            else:
                stats["opacity_reset_skipped_iterations"].append(iteration)

    # 保留旧后期剪枝：增密结束且超过 15k 后，每 3k 次一次，30k 起停止。
    if (
        opt.vcp_enable
        and iteration % 3000 == 0
        and iteration > max(15000, opt.densify_until_iter)
        and iteration < 30000
    ):
        _, pruning = compute_vcd_vcp_scores(
            camlist=sample_vcd_cameras(cameras.copy(), opt.vcd_num_cams),
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            kernel_size=dataset.kernel_size,
            loss_thresh=opt.vcd_loss_thresh,
            need_vcd=False,
            need_vcp=True,
            frame_cache=frame_cache,
        )
        before = len(gaussians.get_xyz)
        gaussians.final_prune_fastgs(
            min_opacity=0.1, pruning_score=pruning, score_threshold=0.9
        )
        stats["late_prune_count"] += 1
        stats["late_prune_iterations"].append(iteration)
        stats["late_prune_gaussian_counts"].append(
            {"iteration": iteration, "before": before, "after": len(gaussians.get_xyz)}
        )
        refresh_filter(gaussians, dataset, cameras)

    if (
        iteration % 100 == 0
        and iteration > opt.densify_until_iter
        and not dataset.disable_filter3D
    ):
        if iteration < opt.iterations - 100:
            gaussians.compute_3D_filter(cameras=cameras)


def training(dataset, opt, pipe, args):
    """一次真实相机基线训练；每个 iteration 渲染一个相机。"""
    validate_options(opt, args)

    # 1. 初始化与可选续训。Scene 负责完整 K/pose、输入对应和 RGBD 初值。
    parameters, start_iteration, split = load_checkpoint(args.start_checkpoint)
    if start_iteration >= opt.iterations:
        raise ValueError(
            "Checkpoint iteration must be smaller than the requested iterations"
        )
    writer = prepare_output(dataset, opt, pipe, args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians)
    validate_training_priors(scene.getTrainCameras())
    gaussians.training_setup(opt)
    restore_checkpoint(
        gaussians, opt, args.start_checkpoint, parameters, split, scene.model_path
    )

    cameras = scene.getTrainCameras().copy()
    budget = dataset.frame_gpu_budget
    frame_cache = None if budget is None or budget == -1 else FramePayloadCache(budget)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    refresh_filter(gaussians, dataset, cameras)

    # 构造与调用 PatchMatch 使用同一份有效权重，关闭两项时不建立它。
    patchmatch = None
    if opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0:
        patchmatch = PatchMatch(
            opt.multi_view_patch_size,
            opt.multi_view_pixel_noise_th,
            kernel_size=dataset.kernel_size,
            pipe=pipe,
            debug=args.debug_images,
            model_path=dataset.model_path,
        )
    prior_counts = {
        "depth": sum(camera.has_depth_prior for camera in cameras),
        "normal": sum(camera.has_normal_prior for camera in cameras),
    }
    print(
        f"[Baseline] train={len(cameras)}, test={len(scene.getTestCameras())}, loaded priors={prior_counts}"
    )
    print(
        f"[Baseline] depth={opt.lambda_depth_prior}, normal_prior_base={opt.lambda_normal_prior} (dynamic schedule), depth_normal={opt.lambda_depth_normal}, ncc_base={opt.lambda_multi_view_ncc} (dynamic schedule), geo={opt.lambda_multi_view_geo}"
    )

    logger = TrainingLogger(scene, opt, args, start_iteration, writer)
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    progress = tqdm(total=opt.iterations - start_iteration, desc="Training")
    viewpoint_stack = []
    try:
        for iteration in range(start_iteration + 1, opt.iterations + 1):
            start_event.record()
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.unlockSGdegree(100)
                gaussians.oneupSHdegree()

            # 2. 无放回随机抽相机；一轮遍历完所有训练相机再重新建立列表。
            if not viewpoint_stack:
                viewpoint_stack = cameras.copy()
            camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            if frame_cache is not None:
                camera = frame_cache.ensure(camera)
            if iteration - 1 == args.debug_from:
                pipe.debug = True

            # 3. 渲染。需要 D/N 的阶段才请求 median depth，避免额外 CUDA 工作。
            schedule = loss_schedule(camera, opt, iteration)
            buffers = render(
                camera,
                gaussians,
                pipe,
                background,
                dataset.kernel_size,
                require_depth=schedule.needs_depth,
            )

            # 4. 同视角几何 / prior 和相邻视角 PatchMatch，统一计算与加权。
            nearest = None
            if (
                schedule.reg_enabled
                and (schedule.multi_view_ncc_weight > 0 or opt.lambda_multi_view_geo > 0)
                and patchmatch is not None
                and camera.nearest_id
            ):
                nearest = cameras[sample(camera.nearest_id, 1)[0]]
                if frame_cache is not None:
                    nearest = frame_cache.ensure(nearest, keep=(camera,))
            losses = compute_training_losses(
                buffers,
                camera,
                gaussians,
                opt,
                schedule,
                patchmatch=patchmatch,
                nearest_camera=nearest,
                iteration=iteration,
            )
            losses.total.backward()
            end_event.record()
            # 原循环通过 loss.item() 隐式等待 CUDA；这里显式等待后再读取事件。
            end_event.synchronize()

            # 5. 仅记录 / 评估 / 保存 PLY。保持旧版“optimizer 前 PLY”语义。
            with torch.no_grad():
                logger.record_loss(
                    iteration,
                    camera,
                    losses,
                    opt,
                    schedule,
                    start_event.elapsed_time(end_event),
                    gaussians,
                )
                progress.update(1)
                if iteration % args.log_interval == 0 or iteration == opt.iterations:
                    progress.set_postfix(
                        {
                            name: f"{logger.ema[name]:.4f}"
                            for name in (
                                "total",
                                "depth_prior",
                                "depth_normal",
                                "normal_prior",
                                "ncc",
                            )
                        }
                    )
                if iteration in args.test_iterations:
                    evaluate_views(
                        scene,
                        pipe,
                        background,
                        dataset.kernel_size,
                        iteration,
                        min(args.test_iterations),
                        writer,
                        frame_cache,
                    )
                if iteration in args.save_iterations:
                    print(f"[ITER {iteration}] Saving Gaussian PLY")
                    scene.save(iteration)

                # 6. 增密 / 剪枝 / opacity reset / filter。必须先于 optimizer。
                update_topology(
                    iteration,
                    gaussians,
                    scene,
                    dataset,
                    opt,
                    pipe,
                    background,
                    buffers,
                    logger.stats,
                    frame_cache,
                )

                # 7. 参数更新。不要在主循环额外 zero_grad：FastGS 依靠累积梯度。
                # 旧配方最后一个 iteration 仅计算/导出，不执行 optimizer.step。
                if iteration < opt.iterations:
                    gaussians.optimizer_step(iteration)

                # 8. checkpoint 在更新后保存，与只含模型的 PLY 区分开。
                if iteration in args.checkpoint_iterations:
                    save_checkpoint(gaussians, scene, iteration, args)
                if (
                    iteration % 100 == 0
                    or iteration in args.checkpoint_iterations
                    or iteration == opt.iterations
                ):
                    logger.flush_stats(gaussians)
        logger.stats["status"] = "completed"
    except Exception as error:
        logger.stats.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        logger.flush_stats(gaussians)
        logger.close()
        progress.close()


def main():
    parser = ArgumentParser(description="GGGS real-camera baseline training")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument(
        "--debug_from",
        type=int,
        default=-1,
        help="Enable CUDA rasterizer debug from this zero-based update",
    )
    parser.add_argument(
        "--debug_images",
        action="store_true",
        help="Save PatchMatch diagnostic images every 200 iterations",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable expensive Torch backward anomaly tracing",
    )
    parser.add_argument(
        "--test_iterations", nargs="*", type=int, default=[3000, 7000, 20000]
    )
    parser.add_argument(
        "--save_iterations", nargs="*", type=int, default=[3000, 7000, 20000]
    )
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="*",
        type=int,
        default=[],
        help="The final iteration is always checkpointed",
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--training_seed", type=int, default=0)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="JSONL / progress interval; each item includes component losses",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    opt = optimization.extract(args)
    try:
        validate_options(opt, args)
    except ValueError as error:
        parser.error(str(error))
    for name in ("test_iterations", "save_iterations", "checkpoint_iterations"):
        values = getattr(args, name)
        if any(value <= 0 for value in values):
            parser.error(f"{name} must contain positive iterations")
        setattr(
            args,
            name,
            sorted(set(value for value in values if value <= args.iterations)),
        )
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.checkpoint_iterations = sorted(
        set(args.checkpoint_iterations + [args.iterations])
    )

    safe_state(args.quiet, seed=args.training_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model.extract(args), opt, pipeline.extract(args), args)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
