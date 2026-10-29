"""Optional, geometry-only supervision from immutable virtual-camera depth.

Targets/masks are prepared offline. In particular the training prediction's
alpha or depth validity never removes a target pixel from the objective. A local
sampler balances physical target groups before selecting their virtual views,
without consuming the real-camera sampler's random state.
"""

import copy
from collections import Counter
from functools import lru_cache
import importlib.util
import hashlib
import json
import math
from pathlib import Path
from random import Random

import numpy as np
import torch


@lru_cache(maxsize=1)
def _interval_operations():
    # CPU tests load this module by path without importing the CUDA Scene
    # package. The independent sibling also has no renderer/Scene dependency.
    path = Path(__file__).with_name("virtual_geometry_intervals.py")
    spec = importlib.util.spec_from_file_location("gggs_fixed_interval_operations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Virtual geometry paths must be nonempty strings")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _number(value, name, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid virtual geometry {name}")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"Invalid virtual geometry {name}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid virtual geometry {name}")
    return value


def load_virtual_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("virtual geometry config requires schema_version=1")
    _number(config.get("weight"), "weight", positive=True)
    for field, minimum in (("from_iter", 0), ("ramp_iters", 0), ("every_n_real_updates", 1)):
        _integer(config.get(field), field, minimum)
    config.setdefault("seed", 0)
    config.setdefault("gradient_diagnostics_interval", 100)
    _integer(config["seed"], "seed")
    _integer(config["gradient_diagnostics_interval"], "gradient_diagnostics_interval")
    config.setdefault("near", 0.01)
    config.setdefault("far", 100.0)
    _number(config["near"], "near", positive=True)
    _number(config["far"], "far", minimum=config["near"], positive=True)
    loss = config.setdefault("loss", {})
    loss.setdefault("mode", "relative_charbonnier")
    loss.setdefault("epsilon", 0.01)
    loss.setdefault("denominator_floor", 0.05)
    loss.setdefault("deadband_absolute", 0.0)
    loss.setdefault("deadband_relative", 0.0)
    if loss["mode"] not in ("relative_charbonnier", "absolute_charbonnier"):
        raise ValueError("Virtual loss mode must be relative_charbonnier or absolute_charbonnier")
    for field in ("epsilon", "denominator_floor", "deadband_absolute", "deadband_relative"):
        _number(loss[field], "loss." + field, positive=field in ("epsilon", "denominator_floor"))
    if loss.get("robust_saturation") is not None:
        _number(loss["robust_saturation"], "loss.robust_saturation", positive=True)
    interval = config.get("interval_repair")
    if interval is not None:
        if not isinstance(interval, dict) or not isinstance(interval.get("enabled", True), bool):
            raise ValueError("interval_repair must be an object with boolean enabled")
        interval.setdefault("repair_strength", 1.)
        interval.setdefault("weak_deadband_relative", .002)
        interval.setdefault("weak_deadband_absolute", 0.)
        for field in ("repair_strength", "weak_deadband_relative", "weak_deadband_absolute"):
            _number(interval[field], "interval_repair." + field)
        if interval["repair_strength"] > 1:
            raise ValueError("repair_strength must lie in [0,1]")
    scheduler = config.get("scheduler")
    if scheduler is not None:
        if not isinstance(scheduler, dict) or not isinstance(scheduler.get("enabled", True), bool):
            raise ValueError("Virtual scheduler must be an object with boolean enabled")
        if scheduler.get("enabled", True):
            history_explicit = "probe_outcome_history_limit" in scheduler
            for key in ("activation_iterations", "batch_sizes"):
                if not isinstance(scheduler.get(key), list) or not scheduler[key]:
                    raise ValueError("Virtual scheduler requires activation_iterations and batch_sizes")
                for value in scheduler[key]:
                    _integer(value, "scheduler." + key, 1)
            events = scheduler["activation_iterations"]
            if events != sorted(set(events)) or len(events) != len(scheduler["batch_sizes"]) or events[0] < config["from_iter"]:
                raise ValueError("Virtual activation iterations must be unique increasing, match batch_sizes and start no earlier than from_iter")
            scheduler.setdefault("local_ramp_iters", config["ramp_iters"])
            scheduler.setdefault("probe_every", 1000)
            scheduler.setdefault("probe_views", 24)
            scheduler.setdefault("probe_rank_weight", .25)
            scheduler.setdefault("probe_residual_scale", .05)
            scheduler.setdefault("probe_roughness_scale", .01)
            scheduler.setdefault("probe_outcome_history_limit", 4096)
            scheduler.setdefault("target_budget", "uniform")
            scheduler.setdefault("repair_pre_activation_probe", False)
            if not isinstance(scheduler["repair_pre_activation_probe"], bool):
                raise ValueError("repair_pre_activation_probe must be boolean")
            if scheduler["repair_pre_activation_probe"]:
                if not interval or not interval.get("enabled", True):
                    raise ValueError("Repair activation probes require interval_repair.enabled")
                scheduler.setdefault("repair_probe_limit", 2000)
                scheduler.setdefault("repair_min_excess_fraction", .01)
                scheduler.setdefault("repair_min_mean_relative_excess", 1e-5)
                _integer(scheduler["repair_probe_limit"], "repair_probe_limit", 1)
                for key in ("repair_min_excess_fraction", "repair_min_mean_relative_excess"):
                    _number(scheduler[key], key)
                if scheduler["repair_min_excess_fraction"] > 1:
                    raise ValueError("Repair excess fraction must lie in [0,1]")
                if not history_explicit:
                    scheduler["probe_outcome_history_limit"] = max(4096, scheduler["repair_probe_limit"] * len(events) + 1024)
            if scheduler["target_budget"] not in ("uniform", "bounded_sqrt_priority"):
                raise ValueError("Virtual scheduler target_budget must be uniform or bounded_sqrt_priority")
            for key, minimum in (("local_ramp_iters", 0), ("probe_every", 1), ("probe_views", 1), ("probe_outcome_history_limit", 1)):
                _integer(scheduler[key], "scheduler." + key, minimum)
            for key in ("real_camera_count", "max_active"):
                if key in scheduler:
                    _integer(scheduler[key], "scheduler." + key, 1)
            for key in ("probe_rank_weight", "probe_residual_scale", "probe_roughness_scale"):
                _number(scheduler[key], "scheduler." + key, positive=key != "probe_rank_weight")
            if scheduler["probe_rank_weight"] > 1:
                raise ValueError("Virtual feedback rank boost must be <=1; it never changes prior confidence")
    config["manifest"] = str(_path(path.parent, config.get("manifest")))
    return config, str(path), sha256_file(path)


def virtual_depth_loss(prediction, target, mask, config):
    """Weighted masked mean, with a fixed target denominator and explicit deadband.

    Zero/negative predicted depth stays penalized; nonfinite prediction is a hard
    failure rather than a way to escape supervision. Empty raster coverage may
    still lack a geometric gradient, which is separately accounted for.
    """
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Virtual depth prediction, target and mask grids differ")
    valid = mask > 0
    if not valid.any():
        raise ValueError("Virtual depth mask is empty")
    pred, truth, weights = prediction[valid], target[valid], mask[valid]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite predicted depth at a virtual target pixel")
    if not torch.isfinite(truth).all() or not (truth > 0).all():
        raise ValueError("Virtual target depth must be finite and positive")
    tolerance = config["deadband_absolute"] + config["deadband_relative"] * truth.detach()
    excess = (torch.abs(pred - truth) - tolerance).clamp_min(0)
    if config["mode"] == "relative_charbonnier":
        excess = excess / truth.detach().clamp_min(config["denominator_floor"])
    epsilon = config["epsilon"]
    saturation = config.get("robust_saturation")
    if saturation is None:
        # Keep the original unsaturated arithmetic when the option is absent.
        values = torch.sqrt(excess.square() + epsilon * epsilon) - epsilon
    else:
        values = torch.hypot(excess, excess.new_tensor(epsilon)) - epsilon
        # Bounded influence without mask/alpha gating. This stable form avoids
        # inf/inf for very large residuals and retains positive finite gradients.
        values = saturation - saturation * saturation / (saturation + values)
    return (values * weights).sum() / weights.sum()


class VirtualGeometryCamera:
    """Minimal camera using the same off-center projection/depth conversion as A."""

    def __init__(self, view, *, device="cuda", near=0.01, far=100.0):
        K = np.asarray(view["K"], dtype=np.float64)
        c2w = np.asarray(view["c2w"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        self.image_name = view["image_name"]
        self.uid = view["uid"]
        self.image_width, self.image_height = view["width"], view["height"]
        self.Fx, self.Fy = float(K[0, 0]), float(K[1, 1])
        self.Cx, self.Cy = float(K[0, 2] - .5), float(K[1, 2] - .5)
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.Fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.Fy))
        self.R = torch.as_tensor(w2c[:3, :3].T.copy(), dtype=torch.float32, device=device)
        self.T = torch.as_tensor(w2c[:3, 3].copy(), dtype=torch.float32, device=device)
        projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
        projection[0, 0], projection[1, 1] = 2 * self.Fx / self.image_width, 2 * self.Fy / self.image_height
        projection[0, 2], projection[1, 2] = 2 * K[0, 2] / self.image_width - 1, 2 * K[1, 2] / self.image_height - 1
        projection[2, 2], projection[2, 3], projection[3, 2] = far / (far - near), -far * near / (far - near), 1
        self.world_view_transform = torch.as_tensor(w2c.T.copy(), dtype=torch.float32, device=device)
        self.full_proj_transform = self.world_view_transform @ projection.T
        self.camera_center = torch.as_tensor(c2w[:3, 3].copy(), dtype=torch.float32, device=device)

    def camera_z_depth(self, depth):
        if tuple(depth.shape[-2:]) != (self.image_height, self.image_width):
            raise ValueError("Virtual raster grid differs from the camera")
        center_x, center_y = (self.image_width - 1) / 2, (self.image_height - 1) / 2
        if self.Cx == center_x and self.Cy == center_y:
            return depth
        y = torch.arange(self.image_height, device=depth.device, dtype=depth.dtype)[:, None]
        x = torch.arange(self.image_width, device=depth.device, dtype=depth.dtype)[None, :]
        raster_radius = torch.sqrt(1 + ((x - center_x) / self.Fx) ** 2 + ((y - center_y) / self.Fy) ** 2)
        camera_radius = torch.sqrt(1 + ((x - self.Cx) / self.Fx) ** 2 + ((y - self.Cy) / self.Fy) ** 2)
        return depth * (raster_radius / camera_radius)


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (tuple, list)) else value


class VirtualGeometryPriors:
    def __init__(self, config_path, *, recipe="real_only", checkpoint_state=None, real_camera_count=None):
        if recipe != "real_only":
            raise ValueError("Virtual geometry priors require the real_only recipe")
        self.config, self.config_path, self.config_sha256 = load_virtual_config(config_path)
        self.interval_enabled = bool(self.config.get("interval_repair") and self.config["interval_repair"].get("enabled", True))
        self.interval_config = {**self.config["loss"], **self.config.get("interval_repair", {})}
        self.manifest_path = Path(self.config["manifest"])
        manifest = json.loads(self.manifest_path.read_text())
        required = {"schema_version": 1, "depth_convention": "camera_z", "units": "scene", "pixel_center_convention": "colmap_half_integer"}
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError("Virtual manifest must declare camera-Z, scene units and COLMAP half-integer pixels")
        views = manifest.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError("Virtual geometry manifest requires nonempty views")
        self.views, self.groups, input_hashes = [], {}, {}
        input_hashes[str(self.manifest_path)] = sha256_file(self.manifest_path)
        for source in manifest.get("source_models", []):
            source_path = _path(self.manifest_path.parent, source.get("path"))
            digest = sha256_file(source_path)
            if source.get("sha256") != digest:
                raise ValueError(f"Virtual source model hash differs: {source_path}")
            input_hashes[str(source_path)] = digest
        names = set()
        for index, original in enumerate(views):
            view = copy.deepcopy(original)
            name, group = view.get("image_name"), view.get("target_id")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Virtual views require unique nonempty image_name")
            if not isinstance(group, (str, int)) or isinstance(group, bool) or str(group) == "":
                raise ValueError("Virtual views require a physical target_id for group-balanced budgets")
            view["target_id"] = str(group)
            mode = view.get("source_mode", "weak")
            if mode not in ("weak", "repair"):
                raise ValueError("Virtual source_mode must be weak or repair")
            if mode == "repair" and not self.interval_enabled:
                raise ValueError("Repair candidates require interval_repair.enabled")
            if self.interval_enabled:
                view["source_mode"] = mode
                physical = view.get("physical_patch_id", group)
                if not isinstance(physical, (str, int)) or isinstance(physical, bool):
                    raise ValueError("physical_patch_id must identify one shared physical budget")
                view["audit_target_id"] = str(group)
                view["physical_patch_id"] = str(physical)
                view["target_id"] = str(physical)
            for field in ("priority", "loss_weight", "target_priority"):
                view.setdefault(field, 1.)
                _number(view[field], "view." + field)
            if "offline_quality_priority" in view:
                _number(view["offline_quality_priority"], "view.offline_quality_priority")
            if view["loss_weight"] > 1:
                raise ValueError("Virtual offline loss_weight must lie in [0,1]")
            view.setdefault("spatial_group", view["target_id"])
            if not isinstance(view["spatial_group"], (str, int)) or isinstance(view["spatial_group"], bool):
                raise ValueError("Virtual spatial_group must be a string or integer")
            view["spatial_group"] = str(view["spatial_group"])
            names.add(name)
            for field in ("width", "height"):
                _integer(view.get(field), field, 1)
            K, c2w = np.asarray(view.get("K"), dtype=float), np.asarray(view.get("c2w"), dtype=float)
            if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError(f"Invalid virtual intrinsics: {name}")
            if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
                raise ValueError("Virtual intrinsics must be pinhole without skew")
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1]):
                raise ValueError(f"Invalid virtual c2w: {name}")
            rotation = c2w[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
                raise ValueError(f"Virtual c2w rotation is not rigid: {name}")
            arrays = {}
            fields = ["depth_path", "mask_path"]
            if mode == "repair":
                fields += ["interval_lower_path", "interval_upper_path"]
            for field in fields:
                source = _path(self.manifest_path.parent, view.get(field))
                digest = sha256_file(source)
                expected = view.get(field.replace("_path", "_sha256"))
                if expected is not None and expected != digest:
                    raise ValueError(f"Virtual {field} hash differs: {name}")
                input_hashes[str(source)] = digest
                view[field] = str(source)
                array = np.load(source, allow_pickle=False)
                if array.shape != (view["height"], view["width"]):
                    raise ValueError(f"Virtual {field} shape differs: {name}")
                arrays[field] = array
            depth, mask = arrays["depth_path"], arrays["mask_path"]
            if depth.dtype != np.float32 or mask.dtype.kind not in "bfiu":
                raise ValueError("Virtual depth must be float32 and mask numeric/binary")
            if not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any() or not (mask > 0).any():
                raise ValueError(f"Virtual mask must be nonempty finite [0,1]: {name}")
            selected = mask > 0
            if not np.isfinite(depth[selected]).all() or not (depth[selected] > self.config["near"]).all() or not (depth[selected] < self.config["far"]).all():
                raise ValueError(f"Invalid virtual target depth within fixed mask: {name}")
            view["uid"] = -(index + 1)
            view["depth_cpu"] = torch.from_numpy(np.where(selected, depth, 0).astype(np.float32, copy=True))[None]
            view["mask_cpu"] = torch.from_numpy(mask.astype(np.float32, copy=True))[None]
            if mode == "repair":
                for field in ("interval_lower", "interval_upper"):
                    if arrays[field + "_path"].dtype != np.float32:
                        raise ValueError("Repair camera-Z interval maps must be float32")
                    view[field + "_cpu"] = torch.from_numpy(np.where(selected, arrays[field + "_path"], 0).astype(np.float32, copy=True))[None]
                _interval_operations().validate_interval_targets(
                    view["depth_cpu"], view["mask_cpu"], view["mask_cpu"] > 0,
                    view["interval_lower_cpu"], view["interval_upper_cpu"],
                )
            view["valid_pixels"] = int(selected.sum())
            view["mask_weight_sum"] = float(mask.sum())
            self.groups.setdefault(view["target_id"], []).append(index)
            self.views.append(view)
        self.rng = Random(self.config["seed"])
        self.group_stack, self.view_stacks = [], {}
        self.scheduler = self.config.get("scheduler")
        self.scheduler_enabled = bool(self.scheduler and self.scheduler.get("enabled", True))
        self.pool_groups = copy.deepcopy(self.groups)
        self.target_priorities = {}
        for group, indices in self.pool_groups.items():
            values = [self.views[index]["target_priority"] for index in indices]
            if not np.allclose(values, values[0], rtol=1e-6, atol=1e-12):
                raise ValueError("All views of one physical target must share its fixed target_priority")
            self.target_priorities[group] = values[0]
        self.active_indices = set(range(len(self.views)))
        self.activation_iterations = {}
        self.probe_rng = Random(f"{self.config['seed']}:virtual_probe")
        self.probe_stack, self.probe_latest = [], {}
        self.next_activation_event, self.last_probe_iteration = 0, None
        self.current_iteration = 0
        if self.scheduler_enabled:
            count = real_camera_count if real_camera_count is not None else self.scheduler.get("real_camera_count")
            _integer(count, "actual real camera count", 1)
            if "real_camera_count" in self.scheduler and self.scheduler["real_camera_count"] != count:
                raise ValueError("Virtual scheduler real_camera_count differs from actual training cameras")
            self.active_cap = min(self.scheduler.get("max_active", count * 2 // 3), count * 2 // 3)
            if self.active_cap < 1:
                raise ValueError("Virtual scheduler camera cap leaves no eligible virtual cameras")
            if not any(self._eligible(index) for index in range(len(self.views))):
                raise ValueError("Virtual scheduler has no positive-priority, positive-confidence eligible candidates")
            self.active_indices, self.groups = set(), {}
        identity = {"config_sha256": self.config_sha256, "input_sha256": input_hashes}
        self.record = {
            "enabled": True, "config": self.config, "config_path": self.config_path, **identity,
            "view_count": len(self.views), "target_count": len(self.groups),
            "budget_normalization": "Uniform shuffled target-group cycles; uniform shuffled view cycles within target; per-view fixed-mask weighted mean",
            "virtual_rgb": False, "virtual_normal": False, "virtual_densification": False,
            "prediction_mask_policy": "Fixed offline target mask, never gated by predicted alpha or predicted depth",
            "render_samples": 0, "loss_samples": 0, "actual_optimizer_updates": 0,
            "loss_sum": 0., "weighted_loss_sum": 0., "valid_pixels_sum": 0,
            "zero_or_negative_prediction_pixels_sum": 0,
            "by_target": {}, "gradient_diagnostics": [],
        }
        self.pending_samples = 0
        if self.interval_enabled:
            self.record["interval_repair"] = {"enabled": True, "by_mode": {},
                "potential_policy": "Offline fixed masks and positive source intervals; prediction/alpha never change potential or its denominator",
                "inside_policy": "Exactly zero auxiliary loss and gradient inside fixed intervals", "budget": "One shared physical_patch_id budget across weak/repair mode views"}
        if self.scheduler_enabled:
            self.record["by_view"] = {}
            self.record["scheduler"] = {
                "enabled": True, "real_camera_count": count, "cap_floor_two_thirds_real_cameras": self.active_cap,
                "pool_view_count": len(self.views), "pool_target_count": len(self.pool_groups),
                "activation_events": [], "probe_outcomes": [], "probe_render_samples": 0,
                "repair_pre_activation_events": [],
                "ranking_policy": ("Minimum active spatial load, then minimum physical target load; shared offline area-quality units: repair multiplied by clipped [.05,1] interval-excess need after stability gate, weak boosted by at most probe_rank_weight"
                                   if self.interval_enabled else "Minimum active spatial load, then minimum target load, then offline priority with bounded GS-need rank boost"),
                "confidence_policy": "GS probe does not alter offline loss_weight or fixed masks; per-view confidence multiplies outside masked mean",
            }
        if checkpoint_state is not None:
            previous = checkpoint_state.get("record", {})
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError("Virtual geometry checkpoint input/config identity differs")
            self.record = copy.deepcopy(previous)
            self.rng.setstate(_tuples(checkpoint_state["rng_state"]))
            self.group_stack = checkpoint_state["group_stack"].copy()
            self.view_stacks = copy.deepcopy(checkpoint_state["view_stacks"])
            self.pending_samples = checkpoint_state.get("pending_samples", 0)
            if self.scheduler_enabled:
                state = checkpoint_state.get("scheduler_state")
                if not isinstance(state, dict):
                    raise ValueError("Progressive virtual checkpoint lacks scheduler state")
                self.active_indices = set(state["active_indices"])
                if len(self.active_indices) > self.active_cap or any(index < 0 or index >= len(self.views) for index in self.active_indices):
                    raise ValueError("Invalid progressive virtual active camera set")
                self.activation_iterations = {int(key): value for key, value in state["activation_iterations"].items()}
                if set(self.activation_iterations) != self.active_indices:
                    raise ValueError("Virtual activation timestamps do not match active cameras")
                self.next_activation_event = state["next_activation_event"]
                self.last_probe_iteration = state["last_probe_iteration"]
                self.probe_stack = state["probe_stack"].copy()
                self.probe_latest = copy.deepcopy(state["probe_latest"])
                self.probe_rng.setstate(_tuples(state["probe_rng_state"]))
                self.current_iteration = state["current_iteration"]
                self._rebuild_active_groups(clear_stacks=False)
        if self.scheduler_enabled:
            self._refresh_scheduler_record()

    def _eligible(self, index):
        view = self.views[index]
        return view["priority"] > 0 and view["loss_weight"] > 0

    def _repair_needs_action(self, index, *, require_current=False):
        latest = self.probe_latest.get(self.views[index]["image_name"])
        if latest is None or (require_current and latest["iteration"] != self.current_iteration):
            return False
        return (latest.get("outside_interval_fraction", 0.) >= self.scheduler.get("repair_min_excess_fraction", .01)
                and latest.get("relative_interval_excess", 0.) > self.scheduler.get("repair_min_mean_relative_excess", 1e-5))

    def _activation_eligible(self, index):
        if not self._eligible(index):
            return False
        if self.scheduler_enabled and self.scheduler.get("repair_pre_activation_probe") and self.views[index].get("source_mode") == "repair":
            return self._repair_needs_action(index, require_current=True)
        return True

    def _rebuild_active_groups(self, *, clear_stacks):
        self.groups = {}
        for index in sorted(self.active_indices):
            self.groups.setdefault(self.views[index]["target_id"], []).append(index)
        if clear_stacks:
            self.group_stack, self.view_stacks = [], {}
        else:
            tickets = self._target_tickets()
            if any(group not in self.groups or count > tickets[group] for group, count in Counter(self.group_stack).items()):
                raise ValueError("Virtual checkpoint group sampler differs from active groups")
            for group, indices in self.view_stacks.items():
                if group not in self.groups or len(set(indices)) != len(indices) or any(index not in self.groups[group] for index in indices):
                    raise ValueError("Virtual checkpoint view sampler differs from active groups")

    def _target_tickets(self):
        if not self.scheduler_enabled or self.scheduler["target_budget"] == "uniform" or not self.groups:
            return {group: 1 for group in self.groups}
        positive = [self.target_priorities[group] for group in self.groups if self.target_priorities[group] > 0]
        median = float(np.median(positive)) if positive else 1.
        return {group: max(1, min(4, int(round(2 * min(2., max(.5, math.sqrt(self.target_priorities[group] / median)))))))
                for group in self.groups}

    def _rank_score(self, index):
        view = self.views[index]
        latest = self.probe_latest.get(view["image_name"], {})
        # Opt-in interval pools mix weak and repair candidates.  Their planner
        # greedy rank has different units from observed area times quality, so
        # both modes must use the same offline quantity before GS need adjusts
        # it.  Legacy pools retain their original priority and exact ordering.
        offline_priority = view.get("offline_quality_priority", view["priority"]) if self.interval_enabled else view["priority"]
        if self.interval_enabled and self.scheduler.get("repair_pre_activation_probe") and view.get("source_mode") == "repair":
            # The planner's greedy ordering is a coverage reference.  A repair
            # camera must instead earn quota through its fixed offline quality
            # and the current GS's excess beyond the trusted source interval.
            # The separate activation gate still excludes stable/unprobed views.
            return offline_priority * min(1., max(.05, latest.get("need_rank", 0.)))
        return offline_priority * (1 + self.scheduler["probe_rank_weight"] * latest.get("need_rank", 0.))

    def _select_new_indices(self, count, candidates=None):
        remaining = ({index for index in range(len(self.views)) if index not in self.active_indices and self._activation_eligible(index)}
                     if candidates is None else set(candidates))
        spatial_counts, target_counts = {}, {}
        for index in self.active_indices:
            view = self.views[index]
            spatial_counts[view["spatial_group"]] = spatial_counts.get(view["spatial_group"], 0) + 1
            target_counts[view["target_id"]] = target_counts.get(view["target_id"], 0) + 1
        chosen = []
        for _ in range(min(count, len(remaining))):
            # Spatial/physical-patch fairness precedes the bounded need rank.
            index = min(remaining, key=lambda i: (
                spatial_counts.get(self.views[i]["spatial_group"], 0),
                target_counts.get(self.views[i]["target_id"], 0),
                -self._rank_score(i), self.views[i]["image_name"],
            ))
            chosen.append(index)
            remaining.remove(index)
            view = self.views[index]
            spatial_counts[view["spatial_group"]] = spatial_counts.get(view["spatial_group"], 0) + 1
            target_counts[view["target_id"]] = target_counts.get(view["target_id"], 0) + 1
        return chosen

    def _probe_indices(self):
        eligible = {index for index in range(len(self.views)) if index not in self.active_indices and self._eligible(index)}
        self.probe_stack = [index for index in self.probe_stack if index in eligible]
        if not self.probe_stack:
            buckets = {}
            for index in eligible:
                buckets.setdefault(self.views[index]["spatial_group"], []).append(index)
            for values in buckets.values():
                values.sort(key=lambda i: (-self.views[i]["priority"], self.views[i]["image_name"]), reverse=True)
            ordered = []
            while buckets:
                groups = sorted(buckets)
                self.probe_rng.shuffle(groups)
                for group in groups:
                    ordered.append(buckets[group].pop())
                    if not buckets[group]:
                        del buckets[group]
            self.probe_stack = list(reversed(ordered))
        selected = []
        for _ in range(min(self.scheduler["probe_views"], len(self.probe_stack))):
            selected.append(self.probe_stack.pop())
        return selected

    @staticmethod
    def probe_statistics(prediction, target, mask):
        """Fixed-mask need proxies; never create confidence or change masks."""
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Virtual probe camera grids differ")
        valid = mask > 0
        if not valid.any() or not torch.isfinite(prediction[valid]).all():
            raise FloatingPointError("Virtual probe has empty targets or nonfinite predicted target depth")
        weights = mask[valid]
        relative = (prediction[valid] - target[valid]).abs() / target[valid].clamp_min(.05)
        depth, truth, fixed_mask = prediction.squeeze(0), target.squeeze(0), valid.squeeze(0)
        if min(depth.shape) >= 3:
            stencil = fixed_mask[1:-1, 1:-1] & fixed_mask[:-2, 1:-1] & fixed_mask[2:, 1:-1] & fixed_mask[1:-1, :-2] & fixed_mask[1:-1, 2:]
            lap = (depth[:-2, 1:-1] + depth[2:, 1:-1] + depth[1:-1, :-2] + depth[1:-1, 2:] - 4 * depth[1:-1, 1:-1]).abs()
            roughness = float((lap[stencil] / truth[1:-1, 1:-1][stencil].clamp_min(.05)).mean().item()) if stencil.any() else 0.
        else:
            roughness = 0.
        return {"relative_abs_error": float((relative * weights).sum().item() / weights.sum().item()),
                "relative_camera_z_laplacian": roughness,
                "zero_prediction_fraction": float((prediction[valid] <= 0).float().mean().item()),
                "fixed_valid_pixels": int(valid.sum().item())}

    def _repair_bounds(self, view, device):
        return view["interval_lower_cpu"].to(device), view["interval_upper_cpu"].to(device)

    def depth_loss_for(self, view, prediction, target, mask):
        if self.interval_enabled and view.get("source_mode") == "repair":
            lower, upper = self._repair_bounds(view, prediction.device)
            return _interval_operations().virtual_interval_depth_loss(
                prediction, target, mask, self.interval_config, repair_mask=mask > 0, lower=lower, upper=upper,
            )
        return virtual_depth_loss(prediction, target, mask, self.config["loss"])

    def probe_feedback(self, view, prediction, target, mask):
        result = self.probe_statistics(prediction, target, mask)
        if self.interval_enabled and view.get("source_mode") == "repair":
            lower, upper = self._repair_bounds(view, prediction.device)
            interval = _interval_operations().interval_excess_statistics(
                prediction, target, mask, self.interval_config, repair_mask=mask > 0, lower=lower, upper=upper,
            )
            result["relative_center_abs_error"] = result["relative_abs_error"]
            result.update(interval)
            result["relative_abs_error"] = interval["relative_interval_excess"]
            result["outside_interval_fraction"] = interval["outside_interval_pixels"] / interval["potential_pixels"]
            result["source_mode"] = "repair"
        return result

    def _record_probe(self, index, iteration, outcome):
        for key in ("relative_abs_error", "relative_camera_z_laplacian", "zero_prediction_fraction"):
            _number(outcome.get(key), "probe." + key)
        if self.interval_enabled and self.views[index].get("source_mode") == "repair":
            _number(outcome.get("relative_interval_excess"), "probe.relative_interval_excess")
            _number(outcome.get("outside_interval_fraction"), "probe.outside_interval_fraction")
            need = min(outcome["relative_interval_excess"] / self.scheduler["probe_residual_scale"], 1.)
        else:
            need = (.75 * min(outcome["relative_abs_error"] / self.scheduler["probe_residual_scale"], 1.)
                    + .25 * min(outcome["relative_camera_z_laplacian"] / self.scheduler["probe_roughness_scale"], 1.))
        result = {"iteration": iteration, "image_name": self.views[index]["image_name"],
                  "target_id": self.views[index]["target_id"], "need_rank": need, **outcome}
        self.probe_latest[self.views[index]["image_name"]] = result
        record = self.record["scheduler"]
        record["probe_outcomes"].append(result)
        record["probe_render_samples"] += 1
        record["probe_outcomes"] = record["probe_outcomes"][-self.scheduler["probe_outcome_history_limit"]:]

    def _probe_repairs_before_activation(self, iteration, callback):
        candidates = {index for index in range(len(self.views)) if index not in self.active_indices and self._eligible(index)
                      and self.views[index].get("source_mode") == "repair"}
        selected = self._select_new_indices(min(self.scheduler["repair_probe_limit"], len(candidates)), candidates=candidates)
        if selected and callback is None:
            raise ValueError("Repair activation requires actual in-process GS feedback")
        for index in selected:
            self._record_probe(index, iteration, callback(self.views[index]))
        eligible = sum(self._repair_needs_action(index, require_current=True) for index in candidates)
        summary = {"iteration": iteration, "inactive_repair_candidates": len(candidates), "probe_views": len(selected),
                   "repair_above_interval_gate": eligible, "repair_dormant_after_probe": len(selected) - eligible,
                   "repair_not_probed_this_stage": len(candidates) - len(selected),
                   "minimum_outside_fraction": self.scheduler["repair_min_excess_fraction"],
                   "minimum_mean_relative_interval_excess": self.scheduler["repair_min_mean_relative_excess"]}
        self.record["scheduler"]["repair_pre_activation_events"].append(summary)
        return summary

    def maybe_update_schedule(self, iteration, probe_callback=None):
        """Activate additions only at explicit stages, using no-grad feedback for rank."""
        if not self.scheduler_enabled:
            return
        self.current_iteration = iteration
        events = self.scheduler["activation_iterations"]
        if iteration < events[0]:
            return
        repair_event_due = (self.scheduler.get("repair_pre_activation_probe") and self.next_activation_event < len(events)
                            and iteration >= events[self.next_activation_event])
        if (not repair_event_due and 0 < self.next_activation_event < len(events) and len(self.active_indices) < self.active_cap
                and (self.last_probe_iteration is None or iteration - self.last_probe_iteration >= self.scheduler["probe_every"])):
            indices = self._probe_indices()
            if indices and probe_callback is None:
                raise ValueError("Progressive virtual additions require the no-grad GS probe callback")
            for index in indices:
                self._record_probe(index, iteration, probe_callback(self.views[index]))
            self.last_probe_iteration = iteration
        while self.next_activation_event < len(events) and iteration >= events[self.next_activation_event]:
            repair_summary = self._probe_repairs_before_activation(iteration, probe_callback) if self.scheduler.get("repair_pre_activation_probe") else None
            requested = self.scheduler["batch_sizes"][self.next_activation_event]
            selected = self._select_new_indices(min(requested, self.active_cap - len(self.active_indices)))
            self.active_indices.update(selected)
            self.activation_iterations.update({index: iteration for index in selected})
            self._rebuild_active_groups(clear_stacks=True)
            self.record["scheduler"]["activation_events"].append({
                "scheduled_iteration": events[self.next_activation_event], "actual_iteration": iteration,
                "requested_views": requested, "added_views": len(selected),
                "cumulative_activated_views": len(self.active_indices),
                "views": [{"image_name": self.views[index]["image_name"], "target_id": self.views[index]["target_id"],
                           "spatial_group": self.views[index]["spatial_group"], "priority": self.views[index]["priority"],
                           "offline_loss_weight": self.views[index]["loss_weight"], "rank_score": self._rank_score(index),
                           **({"source_mode": self.views[index].get("source_mode", "weak")} if self.interval_enabled else {})} for index in selected],
            })
            if repair_summary is not None:
                repair_summary["activated_repair_views"] = sum(self.views[index].get("source_mode") == "repair" for index in selected)
            if self.next_activation_event == 0 and self.last_probe_iteration is None:
                self.last_probe_iteration = iteration
            self.next_activation_event += 1
        if iteration % 100 == 0 or iteration in events:
            self._refresh_scheduler_record()

    def _refresh_scheduler_record(self):
        if not self.scheduler_enabled:
            return
        self.record["scheduler"].update({
            "current_iteration": self.current_iteration,
            "cumulative_activated_views": len(self.active_indices), "active_target_count": len(self.groups),
            "completed_activation_events": self.next_activation_event,
            "active_weights": {self.views[index]["image_name"]: {
                "activated_iteration": self.activation_iterations[index], "offline_loss_weight": self.views[index]["loss_weight"],
                "effective_loss_weight": self.weight_for(self.views[index], self.current_iteration),
            } for index in sorted(self.active_indices)},
            "target_budget_tickets": self._target_tickets(),
            "target_budget_policy": "1 ticket each" if self.scheduler["target_budget"] == "uniform" else "round(2*clip(sqrt(fixed_target_priority/active_target_positive_median),.5,2)); 1..4 tickets per physical target, independent of view count",
        })
        if self.interval_enabled:
            inactive_repairs = [index for index in range(len(self.views)) if index not in self.active_indices and self.views[index].get("source_mode") == "repair"]
            self.record["scheduler"].update({
                "active_views_by_mode": dict(Counter(self.views[index].get("source_mode", "weak") for index in self.active_indices)),
                "dormant_repair_views_latest_probe": sum(not self._repair_needs_action(index) for index in inactive_repairs if self.views[index]["image_name"] in self.probe_latest),
                "unprobed_repair_views": sum(self.views[index]["image_name"] not in self.probe_latest for index in inactive_repairs),
                "activated_never_used_views": [self.views[index]["image_name"] for index in sorted(self.active_indices) if self.record["by_view"].get(self.views[index]["image_name"], {}).get("loss_samples", 0) == 0],
                "sampled_but_always_zero_aux_loss_views": [self.views[index]["image_name"] for index in sorted(self.active_indices)
                    if self.record["by_view"].get(self.views[index]["image_name"], {}).get("loss_samples", 0) > 0
                    and self.record["by_view"].get(self.views[index]["image_name"], {}).get("nonzero_loss_samples", 0) == 0],
            })

    def weight_at(self, iteration):
        start, ramp = self.config["from_iter"], self.config["ramp_iters"]
        if iteration < start:
            return 0.
        return self.config["weight"] * (1. if ramp == 0 else min((iteration - start + 1) / ramp, 1.))

    def due(self, iteration, real_samples):
        return (not self.scheduler_enabled or bool(self.groups)) and self.weight_at(iteration) > 0 and real_samples % self.config["every_n_real_updates"] == 0

    def weight_for(self, view, iteration):
        value = self.weight_at(iteration) * view["loss_weight"]
        if self.scheduler_enabled:
            index = -view["uid"] - 1
            if index not in self.active_indices:
                return 0.
            ramp = self.scheduler["local_ramp_iters"]
            value *= 1. if ramp == 0 else min(max(iteration - self.activation_iterations[index] + 1, 0) / ramp, 1.)
        return value

    def payload(self, view, device="cuda"):
        return (VirtualGeometryCamera(view, device=device, near=self.config["near"], far=self.config["far"]),
                view["depth_cpu"].to(device), view["mask_cpu"].to(device))

    def draw(self, device="cuda"):
        if not self.group_stack:
            if self.scheduler_enabled:
                tickets = self._target_tickets()
                self.group_stack = [group for group in sorted(self.groups) for _ in range(tickets[group])]
            else:
                self.group_stack = sorted(self.groups)
            self.rng.shuffle(self.group_stack)
        group = self.group_stack.pop()
        if not self.view_stacks.get(group):
            self.view_stacks[group] = self.groups[group].copy()
            self.rng.shuffle(self.view_stacks[group])
        view = self.views[self.view_stacks[group].pop()]
        self.record["render_samples"] += 1
        camera, depth, mask = self.payload(view, device)
        return view, camera, depth, mask

    def diagnose_gradients(self, weighted_loss, gaussians, iteration):
        interval = self.config["gradient_diagnostics_interval"]
        if self.record["gradient_diagnostics"] and (not interval or iteration % interval):
            return
        parameters = [(name, getattr(gaussians, attribute)) for name, attribute in
                      (("xyz", "_xyz"), ("scaling", "_scaling"), ("rotation", "_rotation"), ("opacity", "_opacity"))]
        gradients = torch.autograd.grad(weighted_loss, [item[1] for item in parameters], retain_graph=True, allow_unused=True)
        norms = {}
        for (name, _), gradient in zip(parameters, gradients):
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite virtual {name} gradient")
            norms[name] = None if gradient is None else float(gradient.detach().norm().item())
        self.record["gradient_diagnostics"].append({"iteration": iteration, "weighted_virtual_only_gradient_norms": norms})

    def record_loss(self, view, prediction, mask, loss, weight, iteration, will_update=True):
        self.record["loss_samples"] += 1
        zero_count = int(((prediction.detach() <= 0) & (mask > 0)).sum().item())
        values = {"loss_sum": float(loss.detach().item()), "weighted_loss_sum": float(loss.detach().item()) * weight,
                  "valid_pixels_sum": view["valid_pixels"], "zero_or_negative_prediction_pixels_sum": zero_count}
        target = self.record["by_target"].setdefault(view["target_id"], {"loss_samples": 0, **{key: 0 for key in values}})
        target["loss_samples"] += 1
        for current in (self.record, target):
            for key, value in values.items():
                current[key] += value
            current["loss_mean"] = current["loss_sum"] / current["loss_samples"]
            current["weighted_loss_mean"] = current["weighted_loss_sum"] / current["loss_samples"]
        self.record["last_iteration"] = iteration
        if self.interval_enabled:
            mode = view.get("source_mode", "weak")
            entry = self.record["interval_repair"]["by_mode"].setdefault(mode, {"loss_samples": 0, "nonzero_loss_samples": 0, "weighted_loss_sum": 0., "potential_pixels_sum": 0, "inside_interval_pixels_sum": 0, "outside_interval_pixels_sum": 0})
            entry["loss_samples"] += 1
            entry["nonzero_loss_samples"] += int(values["loss_sum"] > 0)
            entry["weighted_loss_sum"] += values["weighted_loss_sum"]
            entry["potential_pixels_sum"] += view["valid_pixels"]
            if mode == "repair":
                lower, upper = self._repair_bounds(view, prediction.device)
                options = {"repair_mask": mask > 0, "lower": lower, "upper": upper}
            else:
                options = {}
            summary = _interval_operations().interval_excess_statistics(prediction, view["depth_cpu"].to(prediction.device), mask, self.interval_config, **options)
            entry["outside_interval_pixels_sum"] += summary["outside_interval_pixels"]
            entry["inside_interval_pixels_sum"] += summary["potential_pixels"] - summary["outside_interval_pixels"]
        if self.scheduler_enabled:
            current = self.record["by_view"].setdefault(view["image_name"], {"loss_samples": 0, "weighted_loss_sum": 0., "effective_weight_sum": 0.})
            current["loss_samples"] += 1
            current["weighted_loss_sum"] += values["weighted_loss_sum"]
            current["effective_weight_sum"] += weight
            if self.interval_enabled:
                current["nonzero_loss_samples"] = current.get("nonzero_loss_samples", 0) + int(values["loss_sum"] > 0)
        if will_update:
            self.pending_samples += 1

    def record_optimizer_step(self):
        if self.pending_samples:
            self.record["actual_optimizer_updates"] += 1
            self.record["loss_samples_reaching_optimizer"] = self.record.get("loss_samples_reaching_optimizer", 0) + self.pending_samples
            self.pending_samples = 0

    def checkpoint_state(self):
        result = {"record": self.record, "rng_state": self.rng.getstate(), "group_stack": self.group_stack,
                  "view_stacks": self.view_stacks, "pending_samples": self.pending_samples}
        if self.scheduler_enabled:
            self._refresh_scheduler_record()
            result["scheduler_state"] = {
                "active_indices": sorted(self.active_indices), "activation_iterations": self.activation_iterations,
                "next_activation_event": self.next_activation_event, "last_probe_iteration": self.last_probe_iteration,
                "probe_stack": self.probe_stack, "probe_latest": self.probe_latest,
                "probe_rng_state": self.probe_rng.getstate(), "current_iteration": self.current_iteration,
            }
        return result

    def write_snapshot(self, output):
        output = Path(output)
        snapshot = {"schema_version": 1, "config_path": self.config_path, "config_sha256": self.config_sha256,
                    "config": self.config, "input_sha256": self.record["input_sha256"],
                    "views": [{key: value for key, value in view.items() if not key.endswith("_cpu")} for view in self.views]}
        (output / "virtual_geometry_inputs.json").write_text(json.dumps(snapshot, indent=2))
