"""真实相机基线的配置、日志、评估与 checkpoint。

这里只负责观测和保存状态，不决定 loss、增密或 optimizer 的执行顺序。
"""

import hashlib
import json
import math
import os
import sys
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scene.train_split import (
    CHECKPOINT_STATE_KEY,
    checkpoint_training_state,
    read_checkpoint_split,
    validate_checkpoint_split,
)
from utils.photometric import get_photometric_backend


RECIPE = "gggs_real_baseline"
RESUME_POLICY = (
    "Same-dataset continuation; process RNG is reseeded. Camera sampling stack, "
    "RNG states and pending accumulated gradients are not restored."
)


def write_json(path, value):
    """同目录原子替换，避免中断后留下半个 JSON。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_baseline_checkpoint(state, iteration):
    """只恢复本基线；旧 R2C checkpoint 的更新语义不能静默混入。"""
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or state.get("recipe") != RECIPE
        or state.get("iteration") != iteration
    ):
        raise ValueError(
            "Resume requires a GGGS real-baseline checkpoint (recipe=gggs_real_baseline); legacy R2C checkpoints are not accepted"
        )
    if (
        state.get("camera_split", {}).get("initialization", {}).get("kind")
        != "train_only_rgbd"
    ):
        raise ValueError(
            "Baseline checkpoints require train-only RGBD initialization provenance"
        )
    return state


def load_checkpoint(path):
    """先验证配方与数据证明，再建立 Scene；不信任旁边的 sidecar 代替嵌入状态。"""
    if not path:
        return None, 0, None
    parameters, iteration = torch.load(path, weights_only=False)
    state = require_baseline_checkpoint(
        checkpoint_training_state(parameters, iteration), iteration
    )
    split = read_checkpoint_split(path, embedded_state=state)
    return parameters, iteration, split


def restore_checkpoint(gaussians, opt, path, parameters, split, output):
    if path:
        validate_checkpoint_split(path, output, previous_record=split)
        gaussians.restore(parameters, opt)


def prepare_output(dataset, opt, pipe, args):
    """保存实际生效配置；cfg_args 继续兼容现有 render / mesh_extract 入口。"""
    if not dataset.model_path:
        run_id = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())[:10]
        dataset.model_path = str(Path("output") / run_id)
    output = Path(dataset.model_path).resolve()
    if output == Path(dataset.source_path).resolve():
        raise ValueError("Training output must differ from the input dataset")
    if (output / "training_config.json").exists():
        raise ValueError(
            f"Output already contains a run: {output}. Use a new output directory, including for resume."
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset.model_path = str(output)
    (output / "cfg_args").write_text(str(Namespace(**vars(dataset))))
    groups = {
        "execution": vars(args),
        "model": vars(dataset),
        "optimization": vars(opt),
        "pipeline": vars(pipe),
    }
    write_json(
        output / "training_config.json",
        {
            "recipe": RECIPE,
            **groups,
            "resolved_args": {
                key: value for group in groups.values() for key, value in group.items()
            },
            "effective_loss_weights": {
                "lambda_dssim": opt.lambda_dssim,
                "lambda_mask": opt.lambda_mask,
                "lambda_depth_normal": opt.lambda_depth_normal,
                "lambda_normal_prior_base": opt.lambda_normal_prior,
                "lambda_depth_prior_max": opt.lambda_depth_prior,
                "lambda_multi_view_ncc_base": opt.lambda_multi_view_ncc,
                "lambda_multi_view_geo": opt.lambda_multi_view_geo,
            },
            "effective_real_depth_loss": {
                "mode": opt.real_depth_loss_mode,
                "relative_epsilon": opt.real_depth_relative_epsilon,
                "denominator_floor_scene_units": opt.real_depth_relative_floor,
                "loss_units": "scene"
                if opt.real_depth_loss_mode == "absolute"
                else "dimensionless_relative",
            },
            "normal_prior_schedule": {
                "source": "train_demo.py normal prior",
                "from_iter": opt.normal_prior_from_iter,
                "init_end_iter": min(7000, opt.iterations),
                "geometry_end_iter": min(15000, opt.iterations),
                "base_weight": opt.lambda_normal_prior,
                "after_geometry": "hold_base_weight",
                "loss": "signed cosine, prior_mask & alpha>0.5 & valid_points",
            },
            "multi_view_ncc_schedule": {
                "source": "train_demo.py with required normal/depth priors",
                "from_iter": max(7001, opt.regularization_from_iter),
                "geometry_end_iter": min(15000, opt.iterations),
                "base_weight": opt.lambda_multi_view_ncc,
                "after_geometry": "hold_quarter_base_weight",
            },
            "required_training_priors": ["normal", "depth"],
            "effective_optimizer_schedule": "dense_main_sh_every_16"
            if opt.dense_optimizer_updates
            else "legacy_fastgs",
            "photometric_backend": get_photometric_backend(),
            "iteration_order": "backward -> report/PLY -> topology/filter -> optimizer (except final iteration) -> checkpoint",
            "rng_resume_policy": RESUME_POLICY,
            "argv": sys.argv,
        },
    )
    print(f"Output folder: {output}")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Tensorboard unavailable; JSONL and JSON logs remain enabled.")
        return None
    return SummaryWriter(str(output))


def save_checkpoint(gaussians, scene, iteration, args):
    """checkpoint 位于 optimizer 之后；保留相机、标定、输入内容和初始化校验。"""
    output = Path(scene.model_path)
    state = {
        "schema_version": 1,
        "recipe": RECIPE,
        "iteration": iteration,
        "training_seed": args.training_seed,
        "rng_resume_policy": RESUME_POLICY,
        "camera_split": json.loads((output / "camera_split.json").read_text()),
    }
    require_baseline_checkpoint(state, iteration)
    captured = gaussians.capture()
    captured[14][CHECKPOINT_STATE_KEY] = state
    path = output / f"chkpnt{iteration}.pth"
    temporary = path.with_suffix(".pth.tmp")
    torch.save((captured, iteration), temporary)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    write_json(
        path.with_suffix(".state.json"),
        {**state, "checkpoint_sha256": digest.hexdigest()},
    )
    print(f"[ITER {iteration}] Saved checkpoint: {path}")


class TrainingLogger:
    """逐项记录 loss，并通过 optimizer hook 统计真正发生的参数更新。"""

    def __init__(self, scene, opt, args, start_iteration, tb_writer):
        self.output = Path(scene.model_path)
        self.writer = tb_writer
        self.log_interval = args.log_interval
        self.ema = {}
        self.loss_sums = {}
        self.stream = (self.output / "loss.jsonl").open("w", buffering=1)
        self.stats = {
            "recipe": RECIPE,
            "status": "running",
            "start_iteration": start_iteration,
            "training_seed": args.training_seed,
            "rng_resume_policy": RESUME_POLICY,
            "train_camera_count": len(scene.getTrainCameras()),
            "test_camera_count": len(scene.getTestCameras()),
            "real_samples": 0,
            "actual_main_optimizer_steps": 0,
            "actual_sh_optimizer_steps": 0,
            "densify_until_iter": opt.densify_until_iter,
            "late_prune_after_iteration": max(15000, opt.densify_until_iter),
            "opacity_reset_min_remaining": opt.opacity_reset_min_remaining,
            "opacity_reset_iterations": [],
            "opacity_reset_skipped_iterations": [],
            "densification_count": 0,
            "densification_iterations": [],
            "densification_gaussian_counts": [],
            "late_prune_count": 0,
            "late_prune_iterations": [],
            "late_prune_gaussian_counts": [],
        }
        self.hooks = []
        for optimizer, key in (
            (scene.gaussians.optimizer, "actual_main_optimizer_steps"),
            (scene.gaussians.shoptimizer, "actual_sh_optimizer_steps"),
        ):
            if optimizer is not None:

                def count_step(optimizer, step_args, step_kwargs, counter=key):
                    self.stats[counter] += 1

                self.hooks.append(optimizer.register_step_post_hook(count_step))

    def record_loss(
        self, iteration, camera, losses, opt, schedule, elapsed_ms, gaussians
    ):
        raw = {
            name: float(value.detach().item())
            for name, value in losses.raw_terms().items()
        }
        weighted = {
            name: float(value.detach().item())
            for name, value in losses.weighted_terms(opt, schedule).items()
        }
        total = float(losses.total.detach().item())
        if not all(
            math.isfinite(value) for value in [total, *raw.values(), *weighted.values()]
        ):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        self.stats["real_samples"] += 1
        self.stats["last_iteration"] = iteration
        for name, value in {**raw, "total": total}.items():
            self.ema[name] = 0.4 * value + 0.6 * self.ema.get(name, 0.0)
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + value
        if iteration % self.log_interval == 0 or iteration == opt.iterations:
            record = {
                "iteration": iteration,
                "camera": camera.image_name,
                "gaussians": len(gaussians.get_xyz),
                "raw": raw,
                "weighted": weighted,
                "normal_prior_weight": schedule.normal_prior_weight,
                "multi_view_ncc_weight": schedule.multi_view_ncc_weight,
                "total": total,
                "forward_backward_ms": elapsed_ms,
            }
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        if self.writer:
            for name, value in raw.items():
                self.writer.add_scalar("train_loss/" + name, value, iteration)
            self.writer.add_scalar("train_loss/total", total, iteration)
            self.writer.add_scalar("iter_time", elapsed_ms, iteration)

    def flush_stats(self, gaussians):
        samples = self.stats["real_samples"]
        self.stats["mean_losses"] = {
            name: value / max(samples, 1) for name, value in self.loss_sums.items()
        }
        self.stats["gaussians"] = len(gaussians.get_xyz)
        write_json(self.output / "training_stats.json", self.stats)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.stream.close()
        if self.writer:
            self.writer.close()


@torch.no_grad()
def evaluate_views(
    scene,
    pipe,
    background,
    kernel_size,
    iteration,
    first_test_iteration,
    writer=None,
    frame_cache=None,
):
    """仅评估，不更新几何。训练视角抽查的 PSNR 不是独立 mesh 质量证据。"""
    import cv2

    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss

    torch.cuda.empty_cache()
    train_cameras = scene.getTrainCameras()
    groups = {
        "test": scene.getTestCameras(),
        "train": [
            train_cameras[index % len(train_cameras)] for index in range(5, 30, 5)
        ],
    }
    results = {}
    for name, cameras in groups.items():
        if not cameras:
            continue
        l1_sum, psnr_sum = 0.0, 0.0
        for index, camera in enumerate(cameras):
            if frame_cache is not None:
                camera = frame_cache.ensure(camera)
            buffers = render(camera, scene.gaussians, pipe, background, kernel_size)
            image = buffers["render"].clamp(0, 1)
            target = camera.original_image_float()
            l1_sum += l1_loss(image, target).mean().double().item()
            psnr_sum += psnr(image, target).mean().double().item()
            if writer and index < 5:
                prefix = f"{name}_view_{camera.image_name}"
                writer.add_images(prefix + "/render", image[None], iteration)
                if iteration == first_test_iteration:
                    writer.add_images(prefix + "/ground_truth", target[None], iteration)
                depth = buffers["median_depth"].squeeze().cpu().numpy()
                depth_u8 = (
                    ((depth - depth.min()) / (depth.max() - depth.min() + 1e-20) * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET).transpose(
                    2, 0, 1
                )
                writer.add_images(prefix + "/depth", colored[None], iteration)
        results[name] = {
            "camera_count": len(cameras),
            "l1": l1_sum / len(cameras),
            "psnr": psnr_sum / len(cameras),
        }
        print(f"[ITER {iteration}] Evaluating {name}: {results[name]}")
        if writer:
            for metric in ("l1", "psnr"):
                writer.add_scalar(f"{name}/{metric}", results[name][metric], iteration)
    write_json(Path(scene.model_path) / f"evaluation_{iteration:06d}.json", results)
    torch.cuda.empty_cache()
