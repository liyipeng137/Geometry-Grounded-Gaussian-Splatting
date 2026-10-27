"""Load fixed external camera-Z priors without inference or geometry rendering.

Only real cameras that originally lack depth are eligible. The external producer
manifest stays unchanged; a separate, content-bound camera identity proof binds
its outputs to the actual training RGB, calibration and world poses.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_fixed_prior_options(config_path, completion_path, recipe, checkpoint):
    """Reject unsupported combinations before constructing a GPU scene."""
    if not config_path:
        return
    if completion_path:
        raise ValueError("real_depth_prior_config and real_depth_completion_config are mutually exclusive")
    if recipe != "real_only":
        raise ValueError("fixed real depth priors require the real_only recipe")
    if checkpoint:
        raise ValueError("fixed real depth priors currently require continuous training; checkpoint resume is unsupported")


def _absolute_file(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
        raise ValueError(f"fixed depth {label} must name an existing absolute file")
    return Path(value).resolve()


def _checked_file(path, checksum, label):
    path = _absolute_file(path, label)
    if (not isinstance(checksum, str) or len(checksum) != 64
            or sha256_file(path) != checksum):
        raise ValueError(f"fixed depth {label} SHA256 mismatch")
    return path


def load_fixed_prior_config(path):
    path = _absolute_file(str(path), "configuration")
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("fixed real depth config requires schema_version=1")
    for field in ("from_iter", "ramp_iters"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if field == "from_iter" else 0):
            raise ValueError(f"Invalid fixed depth {field}")
    for field in ("weight", "min_depth"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid fixed depth {field}")
    if config["min_depth"] < 0.01:
        raise ValueError("fixed depth min_depth must be at least the training threshold 0.01")
    manifest = _absolute_file(config.get("manifest"), "manifest")
    identity = _absolute_file(config.get("identity_manifest", str(manifest.parent / "fixed_real_depth_identity.json")), "camera identity")
    return config, path, manifest, identity


def _views_by_name(document, label):
    views = document.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"fixed depth {label} requires nonempty views")
    result = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"Invalid fixed depth {label} view")
        name = view.get("image_name")
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or ".." in Path(name).parts or name in result):
            raise ValueError(f"fixed depth {label} has invalid or duplicate camera names")
        result[name] = view
    return result


def _cpu_array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _camera_rgb(camera):
    payload = getattr(camera, "original_image_cpu", None)
    if payload is None:
        payload = getattr(camera, "original_image", None)
    if payload is None:
        raise ValueError(f"fixed depth camera has no RGB payload: {camera.image_name}")
    array = _cpu_array(payload)
    if array.shape != (3, camera.image_height, camera.image_width):
        raise ValueError(f"fixed depth camera RGB dimensions differ: {camera.image_name}")
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.moveaxis(array, 0, -1)


class FixedRealDepthPriors:
    def __init__(self, config_path, cameras, *, recipe, checkpoint=None, completion_config=""):
        validate_fixed_prior_options(config_path, completion_config, recipe, checkpoint)
        self.config, config_path, manifest_path, identity_path = load_fixed_prior_config(config_path)
        config_sha, manifest_sha, identity_sha = [sha256_file(path) for path in (config_path, manifest_path, identity_path)]
        manifest = json.loads(manifest_path.read_text())
        identity = json.loads(identity_path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("fixed depth producer manifest must be complete schema1")
        if identity.get("schema_version") != 1 or identity.get("kind") != "fixed_real_depth_camera_identity":
            raise ValueError("fixed depth camera identity requires its own schema1 kind")
        preserved_manifest = identity.get("manifest", {})
        if _checked_file(preserved_manifest.get("path"), preserved_manifest.get("sha256"), "producer manifest") != manifest_path:
            raise ValueError("fixed depth camera identity belongs to another manifest")
        for document, label in ((manifest, "producer"), (identity, "camera identity")):
            if document.get("depth_convention") != "camera_z" or document.get("depth_units") != "scene":
                raise ValueError(f"fixed depth {label} must declare camera_z in scene units")
        if identity.get("pixel_center_convention") != "colmap_half_integer":
            raise ValueError("fixed depth camera identity must declare COLMAP half-integer pixels")
        sources = identity.get("source_sha256")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("fixed depth camera identity requires source SHA256 assets")
        for path, checksum in sources.items():
            _checked_file(path, checksum, "source")
        if any(getattr(camera, "is_pseudo", False) for camera in cameras):
            raise ValueError("fixed depth inputs must contain real cameras only")
        all_cameras = {camera.image_name: camera for camera in cameras}
        if len(all_cameras) != len(cameras):
            raise ValueError("fixed depth cameras must have unique complete image names")
        missing = {name: camera for name, camera in all_cameras.items() if not camera.has_depth_prior}
        output_views = _views_by_name(manifest, "producer")
        camera_views = _views_by_name(identity, "camera identity")
        if not missing or set(output_views) != set(missing) or set(camera_views) != set(missing):
            raise ValueError("fixed depth target camera set differs from the entire original missing-depth set")
        prepared, records = [], []
        for name, camera in missing.items():
            view, proof = output_views[name], camera_views[name]
            shape = (camera.image_height, camera.image_width)
            for description in (view, proof):
                if (description.get("height"), description.get("width")) != shape:
                    raise ValueError(f"fixed depth resolution mismatch: {name}")
            intrinsic = np.asarray(proof.get("intrinsics_px"), dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or not np.allclose(intrinsic, camera.source_K_colmap, atol=1e-9, rtol=0):
                raise ValueError(f"fixed depth intrinsic calibration mismatch: {name}")
            world_to_camera = np.asarray(proof.get("world_to_camera"), dtype=np.float64)
            actual_pose = np.eye(4)
            actual_pose[:3, :3] = _cpu_array(camera.R).T
            actual_pose[:3, 3] = _cpu_array(camera.T)
            if (world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all()
                    or not np.allclose(world_to_camera, actual_pose, atol=2e-6, rtol=1e-6)):
                raise ValueError(f"fixed depth world pose mismatch: {name}")
            rgb_proof = proof.get("rgb", {})
            rgb_path = _checked_file(rgb_proof.get("path"), rgb_proof.get("sha256"), "RGB")
            with Image.open(rgb_path) as image:
                if image.size != shape[::-1] or not np.array_equal(np.asarray(image.convert("RGB")), _camera_rgb(camera)):
                    raise ValueError(f"fixed depth real training RGB mismatch: {name}")
            paths = {field: _checked_file(view.get(field), view.get(field + "_sha256"), field) for field in ("depth", "mask")}
            depth = np.load(paths["depth"], allow_pickle=False)
            mask = np.load(paths["mask"], allow_pickle=False)
            if depth.dtype != np.float32 or depth.shape != shape or mask.shape != shape or mask.dtype != np.bool_:
                raise ValueError(f"fixed depth requires float32 depth and bool mask on the exact camera grid: {name}")
            if not np.isfinite(depth).all() or not mask.any() or not (depth[mask] > self.config["min_depth"]).all():
                raise ValueError(f"fixed depth has nonfinite, empty or invalid valid depth: {name}")
            # Mask is the sole pixel eligibility rule; no new alpha/edge test.
            masked_depth = np.where(mask, depth, 0).astype(np.float32)
            prepared.append((camera, torch.from_numpy(np.ascontiguousarray(masked_depth))[None]))
            records.append({"image_name": name, "depth": str(paths["depth"]), "depth_sha256": view["depth_sha256"],
                            "mask": str(paths["mask"]), "mask_sha256": view["mask_sha256"],
                            "rgb": str(rgb_path), "rgb_sha256": rgb_proof["sha256"],
                            "valid_pixels": int(mask.sum()), "pixel_count": int(mask.size)})
        if [sha256_file(path) for path in (config_path, manifest_path, identity_path)] != [config_sha, manifest_sha, identity_sha]:
            raise ValueError("fixed depth configuration or manifest changed while loading")
        for record in records:
            for field in ("depth", "mask", "rgb"):
                _checked_file(record[field], record[field + "_sha256"], field)
        # Mutate payloads only after every result/identity has passed validation.
        for camera, depth in prepared:
            camera._store_payload("depth_prior", depth)
            camera.has_depth_prior = True
            camera.real_depth_fixed_prior = True
        self.names = set(missing)
        self.record = {"enabled": True, "status": "complete", "source_kind": manifest.get("kind", "external_fixed"),
                       "config": self.config, "config_path": str(config_path), "config_sha256": config_sha,
                       "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
                       "identity_manifest": str(identity_path), "identity_sha256": identity_sha,
                       "source_sha256": sources, "total_real_camera_count": len(cameras),
                       "original_prior_count": len(cameras) - len(missing), "completed_prior_count": len(missing),
                       "missing_image_names": sorted(missing), "views": records,
                       "samples": 0, "inactive_samples": 0, "loss_sum": 0.0, "weighted_loss_sum": 0.0,
                       "sampled_camera_count": 0, "all_targets_sampled": False,
                       "by_image": {name: {"samples": 0, "inactive_samples": 0, "loss_sum": 0.0, "weighted_loss_sum": 0.0}
                                    for name in sorted(missing)},
                       "policy": "Fixed outputs loaded after original initialization; original priors, real camera schedule and model parameters unchanged. No inference/rendering."}

    def weight_for(self, camera, iteration, original_weight):
        if camera.image_name not in self.names:
            return original_weight
        start = self.config["from_iter"]
        if iteration < start:
            return 0.0
        ramp = self.config["ramp_iters"]
        return self.config["weight"] * (1 if ramp == 0 else min((iteration - start + 1) / ramp, 1.0))

    def record_loss(self, camera, iteration, loss, weight):
        if camera.image_name not in self.names:
            return
        frame = self.record["by_image"][camera.image_name]
        if weight <= 0:
            self.record["inactive_samples"] += 1
            frame["inactive_samples"] += 1
            return
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise ValueError("fixed real depth loss must be finite")
        first_sample = frame["samples"] == 0
        for record in (self.record, frame):
            record["samples"] += 1
            record["loss_sum"] += value
            record["weighted_loss_sum"] += value * weight
            record["loss_mean"] = record["loss_sum"] / record["samples"]
            record["weighted_loss_mean"] = record["weighted_loss_sum"] / record["samples"]
        if first_sample:
            self.record["sampled_camera_count"] += 1
        self.record["all_targets_sampled"] = self.record["sampled_camera_count"] == len(self.names)
        frame.setdefault("first_sample_iteration", iteration)
        frame["last_sample_iteration"] = iteration
