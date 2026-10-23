import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.sh_utils import RGB2SH


def load_seed_manifest(manifest_path, train_image_names):
    path = Path(manifest_path).resolve()
    manifest_bytes = path.read_bytes()
    if path.suffix.lower() == ".npz":
        arrays_path = path
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive:
                raise ValueError("Direct seed NPZ requires a Unicode scalar metadata_json")
            metadata = json.loads(str(archive["metadata_json"].item()))
    else:
        metadata = json.loads(manifest_bytes)
        arrays_path = (path.parent / metadata["arrays_path"]).resolve()
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Seed metadata must have schema_version=1")
    source = metadata.get("source")
    if source not in ("real_supported", "generated"):
        raise ValueError("Seed source must distinguish real_supported from generated")
    if metadata.get("coordinate_system") != "world" or metadata.get("units") != "scene":
        raise ValueError("Seeds must use world coordinates and unchanged scene units")
    if not isinstance(metadata.get("provenance"), str) or not metadata["provenance"].strip():
        raise ValueError("Seed provenance must identify the point/depth producer")
    support_names = metadata.get("support_image_names")
    if not isinstance(support_names, list) or len(set(support_names)) != len(support_names):
        raise ValueError("Seed support_image_names must be a unique list of full training image names")
    if any(not isinstance(name, str) or name not in train_image_names for name in support_names):
        raise ValueError("Seed support contains unknown or heldout images")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in (
            "positions", "colors", "scales", "quaternions_wxyz", "confidence", "support_indices"
        )}
    positions = arrays["positions"]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Seed positions must have shape Nx3")
    count = positions.shape[0]
    for name, columns in (("positions", 3), ("colors", 3), ("scales", 3), ("quaternions_wxyz", 4)):
        values = arrays[name]
        if values.shape != (count, columns) or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"Seed {name} must be a finite Nx{columns} array")
        arrays[name] = values.astype(np.float32)
    confidence = arrays["confidence"].reshape(-1)
    if confidence.shape != (count,) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Seed confidence must contain one finite value in [0,1] per point")
    arrays["confidence"] = confidence.astype(np.float32)
    if np.any((arrays["colors"] < 0) | (arrays["colors"] > 1)) or np.any(arrays["scales"] <= 0):
        raise ValueError("Seed RGB must lie in [0,1] and scales must be positive scene-unit lengths")
    quaternion_norm = np.linalg.norm(arrays["quaternions_wxyz"], axis=1)
    if np.any(quaternion_norm < 1e-6):
        raise ValueError("Seed quaternions must have nonzero norm")
    arrays["quaternions_wxyz"] /= quaternion_norm[:, None]
    support_indices = arrays["support_indices"]
    if support_indices.ndim != 2 or support_indices.shape[0] != count or not np.issubdtype(support_indices.dtype, np.integer):
        raise ValueError("Seed support_indices must have shape NxM and integer dtype")
    if np.any((support_indices < -1) | (support_indices >= len(support_names))):
        raise ValueError("Seed support index is outside support_image_names")
    if source == "real_supported":
        for indices in support_indices:
            if len(set(indices[indices >= 0].tolist())) < 2:
                raise ValueError("Every real_supported seed needs at least two distinct real support images")
    return metadata, arrays, hashlib.sha256(manifest_bytes + arrays_path.read_bytes()).hexdigest()


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _camera_depth(camera):
    depth = getattr(camera, "depth_prior_cpu", None)
    if depth is None:
        depth = getattr(camera, "depth_prior", None)
    return None if depth is None else np.asarray(_numpy(depth), dtype=np.float32).squeeze()


def validate_seed_depth_evidence(arrays, metadata, train_cameras, tolerance):
    positions = arrays["positions"]
    supports = arrays["support_indices"]
    support_name_to_index = {name: index for index, name in enumerate(metadata["support_image_names"])}
    confirmed = np.zeros(len(positions), dtype=np.int32)
    free_space_violation = np.zeros(len(positions), dtype=bool)
    for camera in train_cameras:
        depth = _camera_depth(camera)
        if depth is None:
            continue
        if depth.shape != (camera.image_height, camera.image_width):
            raise ValueError(f"Seed evidence camera depth shape mismatch: {camera.image_name}")
        rotation = _numpy(camera.R).T
        translation = _numpy(camera.T)
        camera_points = positions @ rotation.T + translation
        camera_z = camera_points[:, 2]
        positive = camera_z > 1e-5
        denominator = np.maximum(camera_z, 1e-5)
        intrinsic = getattr(camera, "source_K_colmap", None)
        if intrinsic is None:
            intrinsic = np.array([[camera.Fx, 0, camera.Cx + 0.5], [0, camera.Fy, camera.Cy + 0.5], [0, 0, 1]])
        colmap_x = intrinsic[0, 0] * camera_points[:, 0] / denominator + intrinsic[0, 2]
        colmap_y = intrinsic[1, 1] * camera_points[:, 1] / denominator + intrinsic[1, 2]
        pixel_x = np.rint(colmap_x - 0.5).astype(np.int64)
        pixel_y = np.rint(colmap_y - 0.5).astype(np.int64)
        inside = positive & (pixel_x >= 0) & (pixel_x < camera.image_width) & (pixel_y >= 0) & (pixel_y < camera.image_height)
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        observed = depth[pixel_y[indices], pixel_x[indices]]
        known = np.isfinite(observed) & (observed > 0.01)
        valid_indices = indices[known]
        observed = observed[known]
        relative_error = np.abs(camera_z[valid_indices] - observed) / observed
        free_space_violation[valid_indices] |= camera_z[valid_indices] < observed * (1 - tolerance)
        support_index = support_name_to_index.get(camera.source_image_name)
        if support_index is not None:
            claimed = (supports[valid_indices] == support_index).any(axis=1)
            confirmed[valid_indices] += claimed & (relative_error <= tolerance)
    return (confirmed >= 2) & ~free_space_violation, {
        "verified_two_real_depth_views": int((confirmed >= 2).sum()),
        "free_space_violations": int(free_space_violation.sum()),
    }


def select_seed_indices(arrays, existing_positions, options):
    count = len(existing_positions)
    fraction = options.pseudo_seed_max_fraction
    maximum = options.pseudo_seed_max_points
    if not math.isfinite(fraction) or not 0 <= fraction <= 0.02 or not 0 <= maximum <= 20000:
        raise ValueError("Seed cap cannot exceed 2% of existing Gaussians or 20000 points")
    for name in ("pseudo_seed_voxel_size", "pseudo_seed_min_distance", "pseudo_seed_opacity", "pseudo_seed_depth_tolerance"):
        value = getattr(options, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= options.pseudo_seed_min_confidence <= 1 or options.pseudo_seed_opacity > 0.05:
        raise ValueError("Seed confidence threshold must lie in [0,1] and opacity cannot exceed 0.05")
    if options.pseudo_seed_depth_tolerance > 0.05:
        raise ValueError("Seed real-depth tolerance cannot exceed 5%")
    cap = min(int(count * fraction), int(maximum))
    candidates = np.flatnonzero(
        (arrays["confidence"] > 0)
        & (arrays["confidence"] >= options.pseudo_seed_min_confidence)
        & (arrays["scales"].max(axis=1) <= 2 * options.pseudo_seed_voxel_size)
    )
    if not cap or not len(candidates):
        return np.zeros(0, dtype=np.int64), {"cap": cap, "confidence_and_scale_candidates": int(len(candidates))}
    distances = cKDTree(existing_positions).query(arrays["positions"][candidates], k=1, workers=1)[0]
    minimum_distance = np.maximum(options.pseudo_seed_min_distance, 2 * arrays["scales"][candidates].max(axis=1))
    candidates = candidates[distances >= minimum_distance]
    order = candidates[np.argsort(-arrays["confidence"][candidates], kind="stable")]
    selected = []
    occupied_voxels = set()
    for index in order:
        voxel = tuple(np.floor(arrays["positions"][index] / options.pseudo_seed_voxel_size).astype(np.int64).tolist())
        if voxel in occupied_voxels:
            continue
        occupied_voxels.add(voxel)
        selected.append(int(index))
        if len(selected) == cap:
            break
    return np.asarray(selected, dtype=np.int64), {
        "cap": cap,
        "confidence_and_scale_candidates": int(len(candidates)),
        "unique_new_seed_voxels": len(selected),
    }


@torch.no_grad()
def append_seed_gaussians(gaussians, arrays, selected_indices, opacity):
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if not len(selected_indices):
        return 0
    existing = gaussians.get_xyz
    device, dtype = existing.device, existing.dtype
    count = len(selected_indices)
    def tensor(name):
        return torch.as_tensor(arrays[name][selected_indices], dtype=dtype, device=device)
    colors = RGB2SH(tensor("colors"))[:, None, :]
    confidence = tensor("confidence")[:, None]
    initial_opacity = (opacity * confidence).clamp(min=1e-5, max=0.05)
    extensions = {
        "xyz": tensor("positions"),
        "f_dc": colors,
        "f_rest": torch.zeros((count,) + gaussians._features_rest.shape[1:], dtype=dtype, device=device),
        "opacity": torch.logit(initial_opacity),
        "scaling": tensor("scales").log(),
        "rotation": tensor("quaternions_wxyz"),
        "sg_axis": torch.zeros((count,) + gaussians._sg_axis.shape[1:], dtype=dtype, device=device),
        "sg_sharpness": torch.zeros((count,) + gaussians._sg_sharpness.shape[1:], dtype=dtype, device=device),
        "sg_color": torch.zeros((count,) + gaussians._sg_color.shape[1:], dtype=dtype, device=device),
    }
    optimized = gaussians.cat_tensors_to_optimizer(extensions)
    for parameter_name, attribute in (
        ("xyz", "_xyz"), ("f_dc", "_features_dc"), ("f_rest", "_features_rest"),
        ("opacity", "_opacity"), ("scaling", "_scaling"), ("rotation", "_rotation"),
        ("sg_axis", "_sg_axis"), ("sg_sharpness", "_sg_sharpness"), ("sg_color", "_sg_color"),
    ):
        setattr(gaussians, attribute, optimized[parameter_name])
    for attribute in ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom", "max_radii2D"):
        previous = getattr(gaussians, attribute)
        padding = torch.zeros((count,) + previous.shape[1:], dtype=previous.dtype, device=previous.device)
        setattr(gaussians, attribute, torch.cat((previous, padding), dim=0))
    if hasattr(gaussians, "filter_3D"):
        previous = gaussians.filter_3D
        padding = torch.zeros((count,) + previous.shape[1:], dtype=previous.dtype, device=previous.device)
        gaussians.filter_3D = torch.cat((previous, padding), dim=0)
    return count


def admit_pseudo_seeds(manifest_path, gaussians, train_cameras, options, model_path, iteration, previous_admissions=()):
    if not manifest_path:
        return None
    train_names = {camera.source_image_name for camera in train_cameras}
    metadata, arrays, source_hash = load_seed_manifest(manifest_path, train_names)
    if source_hash in previous_admissions:
        return {"source_sha256": source_hash, "appended_gaussians": 0, "skipped_already_applied": True}
    if metadata["source"] != "real_supported":
        raise ValueError("Generated seed hypotheses are reserved but disabled until a separate multi-center geometry gate is implemented")
    verified, evidence_report = validate_seed_depth_evidence(arrays, metadata, train_cameras, options.pseudo_seed_depth_tolerance)
    arrays["confidence"] = arrays["confidence"] * verified
    selected, selection_report = select_seed_indices(arrays, _numpy(gaussians.get_xyz), options)
    start_count = len(gaussians.get_xyz)
    appended = append_seed_gaussians(gaussians, arrays, selected, options.pseudo_seed_opacity)
    report = {
        "schema_version": 1,
        "manifest_path": str(Path(manifest_path).resolve()),
        "source_sha256": source_hash,
        "metadata": metadata,
        "applied_after_iteration": iteration,
        "input_candidates": len(arrays["positions"]),
        "existing_gaussians": start_count,
        "appended_gaussians": appended,
        "initial_index_start": start_count,
        "index_note": "Indices describe admission only; later native densification/pruning changes indices. The saved world points retain admission provenance.",
        "evidence_is_independent_ground_truth": False,
        **evidence_report,
        **selection_report,
    }
    output = Path(model_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "seed_admission.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "admitted_seeds.npz", **{name: values[selected] for name, values in arrays.items()})
    return report
