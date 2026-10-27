"""One-shot, frozen real-camera depth completion from an in-process snapshot.

The optional hook is isolated from camera sampling and restores all process RNGs.
Original priors are never replaced; completion is a separate supervision source.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def preserve_rng_states():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def load_completion_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("real depth completion requires schema_version=1")
    for field in ("trigger_iteration", "ramp_iters", "edge_dilation"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if field == "trigger_iteration" else 0):
            raise ValueError(f"Invalid completion {field}")
    for field in ("weight", "alpha_threshold", "edge_relative_threshold", "min_depth"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid completion {field}")
        if value < 0 or (field in ("weight", "edge_relative_threshold", "min_depth") and value == 0):
            raise ValueError(f"Invalid completion {field}")
    if config["alpha_threshold"] > 1:
        raise ValueError("completion alpha_threshold must lie in [0,1]")
    if not isinstance(config.get("use_fp16"), bool):
        raise ValueError("completion use_fp16 must be boolean")
    for field in ("output", "python", "script"):
        value = config.get(field)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"completion {field} must be an absolute path")
        if field != "output" and not Path(value).is_file():
            raise ValueError(f"completion {field} does not exist: {value}")
    if not isinstance(config.get("model_name"), str) or not config["model_name"]:
        raise ValueError("completion model_name is required")
    return config, str(path), sha256_file(path)


def confident_depth_mask(depth, alpha, config):
    """Mask unreliable coverage and relative 3x3 discontinuities, on CPU or GPU."""
    depth, alpha = depth.squeeze(), alpha.squeeze()
    if depth.ndim != 2 or alpha.shape != depth.shape:
        raise ValueError("completion depth/alpha must share a 2D camera grid")
    finite_positive = torch.isfinite(depth) & (depth > config["min_depth"])
    safe = torch.where(finite_positive, depth, torch.zeros_like(depth))
    upper = F.max_pool2d(safe[None, None], 3, stride=1, padding=1)[0, 0]
    # Invalid pixels should not turn the whole neighbourhood into an artificial edge.
    lower_input = torch.where(finite_positive, depth, torch.full_like(depth, float("inf")))
    lower = -F.max_pool2d(-lower_input[None, None], 3, stride=1, padding=1)[0, 0]
    relative_range = (upper - lower) / safe.clamp_min(config["min_depth"])
    edges = finite_positive & (relative_range > config["edge_relative_threshold"])
    dilation = config["edge_dilation"]
    if dilation:
        edges = F.max_pool2d(edges.float()[None, None], 2 * dilation + 1, stride=1, padding=dilation)[0, 0].bool()
    return finite_positive & torch.isfinite(alpha) & (alpha >= config["alpha_threshold"]) & ~edges


def attach_completed_priors(cameras, manifest, root, minimum_depth):
    """Validate the entire result before changing any camera payload."""
    root = Path(root)
    expected = {camera.image_name: camera for camera in cameras}
    views = manifest.get("views")
    if manifest.get("schema_version") != 1 or not isinstance(views, list):
        raise ValueError("completion output requires schema_version=1 and views")
    names = [view.get("image_name") for view in views]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("completion output camera set differs from the original missing-depth set")
    prepared, records = [], []
    for view in views:
        camera = expected[view["image_name"]]
        if camera.has_depth_prior:
            raise ValueError(f"completion cannot replace an existing prior: {camera.image_name}")
        paths = {name: Path(view[name]) if Path(view[name]).is_absolute() else root / view[name] for name in ("depth", "mask")}
        depth = np.load(paths["depth"], allow_pickle=False)
        mask = np.load(paths["mask"], allow_pickle=False)
        shape = (camera.image_height, camera.image_width)
        if depth.dtype != np.float32 or depth.shape != shape or mask.shape != shape:
            raise ValueError(f"completion output must be float32 depth and matching mask: {camera.image_name}")
        if mask.dtype != np.bool_ and (mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1]).all()):
            raise ValueError(f"completion output mask must be binary: {camera.image_name}")
        mask = mask.astype(bool, copy=False)
        if not mask.any() or not np.isfinite(depth[mask]).all() or not (depth[mask] > minimum_depth).all():
            raise ValueError(f"completion output has invalid or empty valid depth: {camera.image_name}")
        depth = np.where(mask, depth, 0).astype(np.float32)
        prepared.append((camera, torch.from_numpy(np.ascontiguousarray(depth))[None]))
        records.append({"image_name": camera.image_name, "depth": str(paths["depth"].resolve()),
                        "depth_sha256": sha256_file(paths["depth"]), "mask": str(paths["mask"].resolve()),
                        "mask_sha256": sha256_file(paths["mask"]), "valid_pixels": int(mask.sum()),
                        "pixel_count": int(mask.size)})
    for camera, depth in prepared:
        camera._store_payload("depth_prior", depth)
        camera.has_depth_prior = True
        camera.real_depth_completion = True
    return records


class RealDepthCompletion:
    def __init__(self, config_path, cameras, *, recipe, checkpoint=None):
        if recipe != "real_only":
            raise ValueError("real depth completion requires the real_only recipe")
        if checkpoint:
            raise ValueError("real depth completion currently requires continuous training; checkpoint resume is unsupported")
        self.config, self.config_path, self.config_sha256 = load_completion_config(config_path)
        self.cameras = [camera for camera in cameras if not camera.has_depth_prior]
        if not self.cameras:
            raise ValueError("real depth completion requires at least one missing-depth real camera")
        names = [camera.image_name for camera in cameras]
        if len(names) != len(set(names)):
            raise ValueError("real depth completion requires unique real camera image names")
        self.output = Path(self.config["output"])
        if (self.output / "input_manifest.json").exists() or (self.output / "manifest.json").exists():
            raise ValueError("completion output already contains a run; use a fresh output directory")
        self.done = False
        self.record = {"enabled": True, "status": "pending", "config": self.config,
                       "config_path": self.config_path, "config_sha256": self.config_sha256,
                       "missing_image_names": [camera.image_name for camera in self.cameras],
                       "original_prior_count": len(cameras) - len(self.cameras),
                       "samples": 0, "loss_sum": 0.0, "weighted_loss_sum": 0.0,
                       "by_image": {camera.image_name: {"samples": 0, "loss_sum": 0.0, "weighted_loss_sum": 0.0}
                                    for camera in self.cameras}}

    def weight_for(self, camera, iteration, original_weight):
        if not getattr(camera, "real_depth_completion", False):
            return original_weight
        start = self.config["trigger_iteration"] + 1
        if iteration < start:
            return 0.0
        ramp = self.config["ramp_iters"]
        return self.config["weight"] * (1.0 if ramp == 0 else min((iteration - start + 1) / ramp, 1.0))

    def record_loss(self, camera, iteration, loss, weight):
        if not getattr(camera, "real_depth_completion", False):
            return
        value = float(loss.detach().item())
        for record in (self.record, self.record["by_image"][camera.image_name]):
            record["samples"] += 1
            record["loss_sum"] += value
            record["weighted_loss_sum"] += value * weight
            record["loss_mean"] = record["loss_sum"] / record["samples"]
            record["weighted_loss_mean"] = record["weighted_loss_sum"] / record["samples"]
        frame = self.record["by_image"][camera.image_name]
        frame.setdefault("first_sample_iteration", iteration)
        frame["last_sample_iteration"] = iteration

    def maybe_run(self, iteration, optimizer_steps, gaussians, render_fn, pipe, background, kernel_size, frame_cache):
        if self.done or iteration != self.config["trigger_iteration"]:
            return
        if optimizer_steps != iteration:
            raise ValueError("completion trigger must follow the matching number of full optimizer updates")
        if sha256_file(self.config_path) != self.config_sha256:
            raise ValueError("completion config changed during training")
        started = time.monotonic()
        with preserve_rng_states(), torch.no_grad():
            self.output.mkdir(parents=True, exist_ok=True)
            input_dir = self.output / "inputs"
            input_dir.mkdir(exist_ok=True)
            snapshot = self.output / f"gaussians_after_{iteration}_updates.ply"
            gaussians.save_ply(str(snapshot))
            snapshot_record = {"path": str(snapshot.resolve()), "sha256": sha256_file(snapshot)}
            views = []
            for index, camera in enumerate(self.cameras):
                key = f"{index:04d}-{hashlib.sha256(camera.image_name.encode()).hexdigest()[:12]}"
                paths = {field: input_dir / f"{key}_{field}.npy" for field in ("depth", "raw_depth", "alpha", "input_mask")}
                package = render_fn(camera, gaussians, pipe, background, kernel_size, require_depth=True)
                raw_depth = package["median_depth"].detach().squeeze()
                alpha = package["mask"].detach().squeeze()
                mask = confident_depth_mask(raw_depth, alpha, self.config)
                if not mask.any().item():
                    raise ValueError(f"completion guide has no reliable depth anchors: {camera.image_name}")
                depth = torch.where(mask, raw_depth, torch.zeros_like(raw_depth))
                for field, value in (("depth", depth), ("raw_depth", raw_depth), ("alpha", alpha), ("input_mask", mask)):
                    np.save(paths[field], value.cpu().numpy().astype(np.bool_ if field == "input_mask" else np.float32), allow_pickle=False)
                rgb = getattr(camera, "original_image_cpu", None)
                if rgb is None:
                    rgb = camera.original_image
                if rgb is None:
                    raise ValueError(f"completion camera has no real RGB payload: {camera.image_name}")
                rgb = rgb.detach().cpu()
                if rgb.dtype != torch.uint8:
                    rgb = (rgb.clamp(0, 1) * 255).round().to(torch.uint8)
                rgb_path = input_dir / f"{key}_rgb.png"
                Image.fromarray(rgb.permute(1, 2, 0).numpy()).save(rgb_path)
                views.append({"image_name": camera.image_name, "key": key, "rgb": str(rgb_path.resolve()),
                              **{field: str(path.resolve()) for field, path in paths.items()},
                              "intrinsics_px": camera.source_K_colmap.tolist(),
                              "width": camera.image_width, "height": camera.image_height,
                              "input_valid_pixels": int(mask.sum().item()), "pixel_count": int(mask.numel())})
                del package, raw_depth, alpha, mask, depth
            input_manifest = {"schema_version": 1, "trigger_iteration": iteration,
                              "snapshot": snapshot_record, "config_sha256": self.config_sha256,
                              "depth_convention": "camera_z", "depth_units": "scene",
                              "pixel_center_convention": "colmap_half_integer", "views": views}
            input_path = self.output / "input_manifest.json"
            input_path.write_text(json.dumps(input_manifest, indent=2))
            if frame_cache is not None:
                frame_cache.clear()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            render_seconds = time.monotonic() - started
            subprocess.run([self.config["python"], self.config["script"], "--config", self.config_path,
                            "--input-manifest", str(input_path.resolve())], check=True)
            output_path = self.output / "manifest.json"
            output_manifest = json.loads(output_path.read_text())
            if sha256_file(snapshot) != snapshot_record["sha256"] or sha256_file(self.config_path) != self.config_sha256:
                raise ValueError("completion source snapshot or configuration changed during inference")
            if output_manifest.get("status") != "complete":
                raise ValueError("completion output must have status=complete")
            if output_manifest.get("config_sha256") != self.config_sha256:
                raise ValueError("completion output config identity differs from the active configuration")
            if output_manifest.get("input_manifest_sha256") != sha256_file(input_path):
                raise ValueError("completion output input identity differs from the rendered snapshot")
            if output_manifest.get("snapshot") != snapshot_record:
                raise ValueError("completion output snapshot identity differs from the rendered snapshot")
            frame_records = attach_completed_priors(self.cameras, output_manifest, self.output, self.config["min_depth"])
            self.record.update({"status": "complete", "trigger_iteration": iteration,
                                "actual_optimizer_updates_at_trigger": optimizer_steps,
                                "snapshot": snapshot_record, "input_manifest": str(input_path.resolve()),
                                "input_manifest_sha256": sha256_file(input_path),
                                "output_manifest": str(output_path.resolve()), "output_manifest_sha256": sha256_file(output_path),
                                "render_seconds": render_seconds, "elapsed_seconds": time.monotonic() - started,
                                "completed_prior_count": len(frame_records), "views": frame_records,
                                "rng_policy": "Python, NumPy, Torch CPU and all initialized CUDA RNG states preserved"})
            self.done = True
        print(f"[RealDepthCompletion] Added {len(self.cameras)} frozen priors after update {iteration} in {self.record['elapsed_seconds']:.1f}s")
