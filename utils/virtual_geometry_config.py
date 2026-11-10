"""正式联合流程的单份配置与真实 iteration 日程；不依赖 Torch/CUDA。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


JOINT_RECIPE = "gggs_virtual_geometry_joint_v1"
DEFAULTS = {
    "schema_version": 1,
    "teacher_manifest": "",
    "start_iteration": 3501,
    "ramp_end_iteration": 5000,
    "topology_until": 15000,
    "seed_from": 5001,
    "seed_until": 14000,
    "warm_interval": 4,
    "joint_interval": 2,
    "tail_interval": 4,
    "last_opacity_reset": 3000,
    "depth_search_expansions": 2,
    "gpu_cache_size": 2,
    "cpu_cache_size": 160,
    "audit_interval": 500,
    "audit_views": 16,
    "priority_views": 8,
    "topology_interval": 100,
    "max_gaussians": 2500000,
    "max_growth": 20000,
    "max_growth_fraction": 0.05,
    "geometry_reserve": 0.1,
    "geometry_reserve_release": 12000,
    "seed_batch_limit": 5000,
    "seed_fraction": 0.01,
    "seed_total_limit": 150000,
    "grace_updates": 500,
    "patch_prune_fraction": 0.1,
    "loss_weights": {
        "depth": 0.20,
        "normal_render": 0.05,
        "normal_depth": 0.05,
        "coverage": 0.05,
    },
}


def config_identity(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_joint_config(path):
    """未知键直接报错；短验证使用显式短日程，不隐式缩放正式阈值。"""
    path = Path(path).resolve()
    supplied = json.loads(path.read_text())
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown virtual geometry configuration keys: {sorted(unknown)}")
    config = deepcopy(DEFAULTS)
    weights = supplied.get("loss_weights", {})
    if not isinstance(weights, dict) or set(weights) - set(DEFAULTS["loss_weights"]):
        raise ValueError("Invalid virtual geometry loss_weights")
    config.update({key: value for key, value in supplied.items() if key != "loss_weights"})
    config["loss_weights"].update(weights)
    if config["schema_version"] != 1:
        raise ValueError("Unsupported virtual geometry config schema")
    manifest = config["teacher_manifest"]
    if not isinstance(manifest, str) or not manifest:
        raise ValueError("teacher_manifest must name a prepared geometry cache")
    manifest = Path(manifest)
    config["teacher_manifest"] = str((manifest if manifest.is_absolute() else path.parent / manifest).resolve())
    positive = (
        "start_iteration", "ramp_end_iteration", "topology_until", "seed_from", "seed_until",
        "warm_interval", "joint_interval", "tail_interval", "gpu_cache_size", "cpu_cache_size",
        "audit_interval", "audit_views", "priority_views", "topology_interval", "max_gaussians",
        "max_growth", "seed_batch_limit", "seed_total_limit", "grace_updates", "geometry_reserve_release",
    )
    for key in positive:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config["last_opacity_reset"] < config["start_iteration"]:
        raise ValueError("The last opacity reset must precede virtual supervision")
    if not config["start_iteration"] <= config["ramp_end_iteration"] < config["topology_until"]:
        raise ValueError("Virtual ramp must finish before fixed-topology convergence")
    if not config["start_iteration"] <= config["seed_from"] <= config["seed_until"] < config["topology_until"]:
        raise ValueError("Seed creation must be inside the joint topology stage")
    if config["depth_search_expansions"] not in (1, 2):
        raise ValueError("Joint depth_search_expansions must be 1 or 2")
    for key in ("max_growth_fraction", "geometry_reserve", "seed_fraction", "patch_prune_fraction"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{key} must lie strictly between zero and one")
    for key, value in config["loss_weights"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Virtual loss weight {key} must be finite and nonnegative")
    return config


def virtual_schedule(iteration, config):
    """频率按真实 step 计；新增虚拟图不推进真实学习率/增密日程。"""
    if iteration < config["start_iteration"]:
        return False, 0.0, "real_anchor"
    if iteration <= config["ramp_end_iteration"]:
        interval, stage = config["warm_interval"], "virtual_ramp"
    elif iteration <= config["topology_until"]:
        interval, stage = config["joint_interval"], "joint_geometry"
    else:
        interval, stage = config["tail_interval"], "fixed_topology"
    ramp = min(1.0, (iteration - config["start_iteration"] + 1) /
               (config["ramp_end_iteration"] - config["start_iteration"] + 1))
    return iteration % interval == 0, ramp, stage
