#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr
#
"""GGGS 正式入口：真实基线，以及固定 GenRecon 虚拟几何联合流程。

阅读顺序：main → training → update_topology。
损失细节在 utils/training_losses.py；配置、日志、评估和 checkpoint 在
utils/training_io.py。这里保留训练阶段的先后关系，不隐藏在 Trainer 框架中。

默认初始化为真实相机的 RGBD→TSDF 点云；--init_mesh 自动采样并投影真实照片颜色，
--init_ply 复用已准备的点云。两者互斥，均不改变 filter/拓扑策略。指定 --virtual_geometry_config
后额外加入已审核机位的固定D/N；没有虚拟RGB、appearance ID或PatchMatch邻居。
真实监督保留，几何增密/补点在optimizer安全边界后统一提交。
未显式指定配置时，自动使用 dataset/virtual_geometry 中由离线 A 管道完成的产物；
没有产物或显式 --disable_virtual_geometry 时走原真实基线。训练不会生成虚拟机位。
详见 docs/dataset_pipeline.md 和 docs/virtual_geometry_training.md。
"""

import math
from pathlib import Path
from argparse import ArgumentParser
from random import randint, sample

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.cameras import FramePayloadCache
from utils.general_utils import safe_state
from utils.initialization import validate_initialization_options, resolve_checkpoint_initialization
from utils.dataset_virtual_geometry import resolve_virtual_geometry_config
from utils.loss_utils import PatchMatch
from utils.training_io import (
    TrainingLogger,
    evaluate_views,
    load_checkpoint,
    prepare_output,
    restore_checkpoint,
    record_initialization,
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


def training(dataset, opt, pipe, args, *, prepared_teacher=None):
    """每步必有真实相机，按日程附加虚拟相机，两路梯度共用一次参数更新。

    baseline保留backward→topology→step；joint使用step→安全拓扑事务。
    虚拟渲染不推进iteration，不增加独立optimizer step。
    """
    validate_options(opt, args)
    validate_initialization_options(dataset)
    if getattr(dataset, "init_mesh", "") and not Path(dataset.source_path, getattr(dataset, "colmap_dir", "sparse/0")).is_dir():
        raise ValueError("--init_mesh requires a COLMAP dataset")

    # A is an explicit offline step. This only selects its completed config;
    # no artifacts means the existing real-only pipeline, partial artifacts fail.
    explicit_virtual_config = getattr(args, "virtual_geometry_config", None)
    args.virtual_geometry_config = resolve_virtual_geometry_config(
        dataset.source_path, explicit_virtual_config,
        disabled=getattr(args, "disable_virtual_geometry", False),
    )
    if args.virtual_geometry_config:
        origin = "explicit" if explicit_virtual_config else "dataset"
        print(f"Virtual geometry ({origin}): {args.virtual_geometry_config}")
    else:
        print("Virtual geometry: disabled; using real-camera training")

    joint_config = teacher = joint = None
    if getattr(args, "virtual_geometry_config", None):
        if dataset.disable_filter3D:
            raise ValueError("Virtual geometry uses the joint 3D filter; --disable_filter3D is incompatible")
        from utils.virtual_geometry_config import load_joint_config, config_identity, virtual_schedule
        from utils.virtual_geometry_data import VirtualGeometryDataset
        from utils.virtual_geometry_losses import compute_virtual_geometry_losses
        from utils.geometry_topology import GeometryTopology
        from utils.joint_runtime import JointRuntime, JOINT_RUNTIME_KEY
        from utils.training_io import JOINT_RESUME_POLICY
        from utils.virtual_geometry_config import JOINT_RECIPE

        joint_config = load_joint_config(args.virtual_geometry_config)
        opt.dense_optimizer_updates = True
        args.dense_optimizer_updates = True
        if prepared_teacher is not None:
            if Path(prepared_teacher.path).resolve() != Path(joint_config["teacher_manifest"]).resolve():
                raise ValueError("Prepared teacher does not match the configured manifest")
            teacher = prepared_teacher
        else:
            teacher = VirtualGeometryDataset(
                joint_config["teacher_manifest"], device="cuda",
                gpu_cache_size=joint_config["gpu_cache_size"],
                cpu_cache_size=joint_config["cpu_cache_size"], seed=args.training_seed,
            )
    elif prepared_teacher is not None:
        raise ValueError("A prepared teacher requires virtual geometry training")

    # 1. 先识别续训，再决定初始化来源。Scene 在训练划分确定后处理 mesh；
    #    续训仅复用冻结点云，不再依赖原 mesh，也不重新运行 TSDF。
    if joint_config:
        parameters, start_iteration, split = load_checkpoint(args.start_checkpoint, joint=True)
        joint_identity = {"config_sha256": config_identity(joint_config), "teacher": teacher.identity}
        if parameters is not None and parameters[14][JOINT_RUNTIME_KEY]["identity"] != joint_identity:
            raise ValueError("Joint checkpoint teacher/config changed; use a fresh output/run")
    else:
        parameters, start_iteration, split = load_checkpoint(args.start_checkpoint)
    if start_iteration >= opt.iterations:
        raise ValueError(
            "Checkpoint iteration must be smaller than the requested iterations"
        )
    resolved_init_ply = None
    if split is not None:
        resolved_init_ply = resolve_checkpoint_initialization(dataset, args.start_checkpoint, split)
    if joint_config:
        writer = prepare_output(dataset, opt, pipe, args, joint_config=joint_config, joint_identity=joint_identity)
    else:
        writer = prepare_output(dataset, opt, pipe, args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, initialization_seed=args.training_seed,
                  initialization_record=split["initialization"] if split is not None else None,
                  resolved_init_ply=resolved_init_ply)
    # Keep requested CLI options distinct from the resolved cached PLY. This
    # prevents a saved --init_mesh config from also claiming --init_ply.
    record_initialization(scene.model_path)
    validate_training_priors(scene.getTrainCameras())
    if teacher is not None:
        teacher.validate_real_cameras(scene.getTrainCameras(), dataset.source_path)
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

    # 虚拟cache独立，不改变真实相机集合、scene extent和外观参数身份。
    viewpoint_stack = []
    if joint_config:
        topology = GeometryTopology(
            gaussians, scene, teacher, joint_config, pipe, background,
            dataset.kernel_size, frame_cache=frame_cache, opt=opt,
        )
        joint = JointRuntime(
            joint_config, teacher, topology, scene.model_path, start_iteration,
            detailed_virtual_log=getattr(args, "detailed_virtual_log", True),
            log_interval=args.log_interval, final_iteration=opt.iterations,
        )
        if parameters is not None:
            viewpoint_stack = joint.restore(parameters[14][JOINT_RUNTIME_KEY], gaussians, cameras)
            parameters = None  # 拓扑变动后不继续持有旧checkpoint的大张量。

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
    if joint:
        logger.stats.update(recipe=JOINT_RECIPE, rng_resume_policy=JOINT_RESUME_POLICY,
                            iteration_order="real/virtual backward -> optimizer -> safe topology -> save")
    step_timing = getattr(args, "step_timing", True)
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    ) if step_timing else (None, None)
    progress = tqdm(total=opt.iterations - start_iteration, desc="Training")
    try:
        for iteration in range(start_iteration + 1, opt.iterations + 1):
            if step_timing:
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
            render_options = {"depth_search_expansions": joint_config["depth_search_expansions"]} if joint else {}
            buffers = render(
                camera,
                gaussians,
                pipe,
                background,
                dataset.kernel_size,
                require_depth=schedule.needs_depth,
                **render_options,
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
                    nearest = frame_cache.ensure(
                        nearest, keep=(camera,),
                        fields=getattr(patchmatch, "neighbor_payload_fields", ("original_image",)),
                    )
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
            if joint and not bool(torch.isfinite(losses.total)):
                raise FloatingPointError(f"Nonfinite real loss before joint update at iteration {iteration}")
            losses.total.backward()
            if joint:
                joint.topology.collect_real(buffers)
                use_virtual, ramp, stage = virtual_schedule(iteration, joint_config)
                if use_virtual:
                    virtual_index, virtual_camera = teacher.next_view()
                    virtual_buffers = render(
                        virtual_camera, gaussians, pipe, background, dataset.kernel_size,
                        require_depth=True, **render_options,
                    )
                    virtual_losses = compute_virtual_geometry_losses(
                        virtual_buffers, virtual_camera, weights=joint_config["loss_weights"], ramp=ramp,
                        collect_metrics=joint.should_record(iteration),
                    )
                    if not bool(torch.isfinite(virtual_losses.total)):
                        raise FloatingPointError(f"Nonfinite virtual loss before update at iteration {iteration}")
                    virtual_losses.total.backward()
                    joint.topology.collect_virtual(virtual_buffers, virtual_camera)
                    joint.record(iteration, virtual_index, virtual_camera, virtual_losses, stage, ramp)
                    del virtual_losses, virtual_buffers
            if step_timing:
                end_event.record()
                # Only diagnostic timing requires this explicit event synchronization.
                end_event.synchronize()

            # 5. joint先更新，再到SH累积已消费的边界增删点，最后保存。
            # baseline保留原先optimizer前保存PLY的语义。
            with torch.no_grad():
                if joint:
                    gaussians.optimizer_step(iteration)
                    if iteration == joint_config["last_opacity_reset"]:
                        gaussians.reset_opacity()
                        logger.stats["opacity_reset_iterations"].append(iteration)
                    joint.topology.step(iteration)
                logger.record_loss(
                    iteration,
                    camera,
                    losses,
                    opt,
                    schedule,
                    start_event.elapsed_time(end_event) if step_timing else None,
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
                    scene.save(iteration, save_viewer=getattr(args, "save_viewer_ply", True))

                # 6. 原真实baseline专用：保持topology-before-step。joint已在上方提交。
                if not joint:
                    update_topology(
                        iteration, gaussians, scene, dataset, opt, pipe, background,
                        buffers, logger.stats, frame_cache,
                    )

                # 7. baseline末步仅导出；joint在上方每步（含末步）都已更新。
                # 两条路径都不额外zero_grad，以保留尚未消费的SH累计梯度。
                if not joint and iteration < opt.iterations:
                    gaussians.optimizer_step(iteration)

                # 8. checkpoint 在更新后保存，与只含模型的 PLY 区分开。
                if iteration in args.checkpoint_iterations:
                    if joint:
                        save_checkpoint(gaussians, scene, iteration, args,
                                        joint_runtime=joint.capture(gaussians, viewpoint_stack))
                    else:
                        save_checkpoint(gaussians, scene, iteration, args)
                if (
                    iteration % 100 == 0
                    or iteration in args.checkpoint_iterations
                    or iteration == opt.iterations
                ):
                    if joint:
                        joint.flush()
                        logger.stats["virtual_geometry"] = joint.summary()
                    logger.flush_stats(gaussians)
        logger.stats["status"] = "completed"
    except Exception as error:
        logger.stats.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if joint:
            joint.flush()
            logger.stats["virtual_geometry"] = joint.summary()
            joint.close()
        logger.flush_stats(gaussians)
        logger.close()
        progress.close()


def main():
    parser = ArgumentParser(description="GGGS real-camera and GenRecon virtual-geometry joint training")
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
        help="The final iteration is added unless --no_final_checkpoint is set",
    )
    parser.add_argument("--no_viewer_ply", action="store_false", dest="save_viewer_ply", default=True,
                        help="Skip the additional viewer PLY; keep the native Gaussian output")
    parser.add_argument("--no_final_checkpoint", action="store_false", dest="save_final_checkpoint", default=True,
                        help="Do not automatically add a final checkpoint; explicit checkpoints remain enabled")
    parser.add_argument("--keep_last_checkpoint", action="store_true", default=False,
                        help="After checkpoint and receipt succeed, remove older checkpoint pairs in this output directory")
    parser.add_argument("--no_step_timing", action="store_false", dest="step_timing", default=True,
                        help="Skip per-step CUDA timing events and their explicit synchronization")
    parser.add_argument("--compact_virtual_log", action="store_false", dest="detailed_virtual_log", default=True,
                        help="Write virtual diagnostics and TensorBoard scalars only at log intervals")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--virtual_geometry_config", type=str, default=None,
        help="显式虚拟D/N配置，优先于dataset/virtual_geometry自动发现；需先准备固定teacher缓存",
    )
    parser.add_argument(
        "--disable_virtual_geometry", action="store_true",
        help="即使dataset含虚拟视角产物也只训练真实相机；不能与显式虚拟配置同时使用",
    )
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
        validate_initialization_options(args)
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
    if getattr(args, "save_final_checkpoint", True):
        args.checkpoint_iterations = sorted(set(args.checkpoint_iterations + [args.iterations]))

    safe_state(args.quiet, seed=args.training_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model.extract(args), opt, pipeline.extract(args), args)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
