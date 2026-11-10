"""联合流程的可恢复状态和日志；训练顺序仍显式放在 train.py。"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from utils.training_io import write_json
from utils.virtual_geometry_config import config_identity


JOINT_RUNTIME_KEY = "gggs_virtual_geometry_runtime"


def camera_name(camera):
    return getattr(camera, "source_image_name", None) or camera.image_name


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Joint resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def pending_gradients(gaussians):
    """SH每16步才更新；非边界checkpoint必须保留尚未消费的梯度。"""
    result = {}
    for label, optimizer in (("main", gaussians.optimizer), ("sh", gaussians.shoptimizer)):
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            result[f"{label}/{group['name']}"] = [
                None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                for parameter in group["params"]
            ]
    return result


def restore_pending_gradients(gaussians, saved):
    found = set()
    for label, optimizer in (("main", gaussians.optimizer), ("sh", gaussians.shoptimizer)):
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            key = f"{label}/{group['name']}"
            if key not in saved or len(saved[key]) != len(group["params"]):
                raise ValueError(f"Joint checkpoint gradient layout changed: {key}")
            found.add(key)
            for parameter, gradient in zip(group["params"], saved[key]):
                if gradient is not None and (gradient.shape != parameter.shape or not torch.isfinite(gradient).all()):
                    raise ValueError(f"Invalid pending gradient in joint checkpoint: {key}")
                parameter.grad = None if gradient is None else gradient.to(parameter).clone()
    if found != set(saved):
        raise ValueError("Joint checkpoint has unknown optimizer gradient groups")


class JointRuntime:
    """只保存状态与记录结果，不隐藏 render/backward/optimizer/topology 顺序。"""

    def __init__(self, config, teacher, topology, output, start_iteration=0):
        self.config, self.teacher, self.topology = config, teacher, topology
        self.output = Path(output)
        self.identity = {"config_sha256": config_identity(config), "teacher": teacher.identity}
        self.start_iteration = start_iteration
        self.segment_virtual_samples = 0
        self.stats = {"virtual_samples": 0, "weighted_loss_sums": {}, "last_iteration": start_iteration}
        self.stream = (self.output / "virtual_loss.jsonl").open("w", buffering=1)

    def record(self, iteration, index, camera, losses, stage, ramp):
        raw = {key: float(value.detach().item()) for key, value in losses.raw.items()}
        weighted = {key: float(value.detach().item()) for key, value in losses.weighted.items()}
        total = float(losses.total.detach().item())
        if not all(math.isfinite(value) for value in [total, *raw.values(), *weighted.values()]):
            raise FloatingPointError(f"Nonfinite virtual geometry loss at iteration {iteration}")
        self.stats["virtual_samples"] += 1
        self.segment_virtual_samples += 1
        self.stats["last_iteration"] = iteration
        for key, value in weighted.items():
            self.stats["weighted_loss_sums"][key] = self.stats["weighted_loss_sums"].get(key, 0.) + value
        self.stream.write(json.dumps({
            "iteration": iteration, "view_index": int(index), "camera": camera_name(camera),
            "stage": stage, "ramp": ramp, "raw": raw, "weighted": weighted,
            "total": total, "fixed_domain_metrics": losses.metrics,
        }, allow_nan=False) + "\n")

    def summary(self):
        return {
            "identity": self.identity, "start_iteration": self.start_iteration,
            "virtual_camera_count": len(self.teacher),
            "segment_virtual_samples": self.segment_virtual_samples,
            **self.stats,
            "mean_weighted_losses": {key: value / max(self.stats["virtual_samples"], 1)
                                     for key, value in self.stats["weighted_loss_sums"].items()},
            "topology": self.topology.summary(),
        }

    def flush(self):
        write_json(self.output / "virtual_geometry_stats.json", self.summary())

    def capture(self, gaussians, real_stack):
        return {
            "schema_version": 1, "identity": self.identity,
            "teacher": self.teacher.state_dict(), "topology": self.topology.state_dict(),
            "stats": self.stats, "rng": capture_rng(),
            "real_stack": [camera_name(camera) for camera in real_stack],
            "pending_gradients": pending_gradients(gaussians),
            "filter_3D": gaussians.filter_3D.detach().cpu().clone(),
            "xyz_gradient_accum_abs": gaussians.xyz_gradient_accum_abs.detach().cpu().clone(),
        }

    def restore(self, saved, gaussians, real_cameras):
        if saved.get("schema_version") != 1 or saved.get("identity") != self.identity:
            raise ValueError("Joint teacher/config identity changed; use a new training run")
        by_name = {camera_name(camera): camera for camera in real_cameras}
        names = saved["real_stack"]
        if len(names) != len(set(names)) or set(names) - set(by_name):
            raise ValueError("Invalid real-camera sampling stack in joint checkpoint")
        self.teacher.load_state_dict(saved["teacher"])
        self.topology.load_state_dict(saved["topology"])
        self.stats = saved["stats"]
        for field in ("filter_3D", "xyz_gradient_accum_abs"):
            target = getattr(gaussians, field)
            value = saved[field]
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"Joint checkpoint {field} shape/value mismatch")
            setattr(gaussians, field, value.to(target).clone())
        restore_pending_gradients(gaussians, saved["pending_gradients"])
        # Scene/optimizers/cache construction may consume RNG; restore last.
        restore_rng(saved["rng"])
        return [by_name[name] for name in names]

    def close(self):
        self.stream.close()
