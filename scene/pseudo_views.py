import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_unit_map(path, width, height, label, binary=False):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if values.shape == (1, height, width):
            values = values[0]
    else:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if values.shape != (height, width) or not np.isfinite(values).all():
        raise ValueError(f"{label}: expected a finite {height}x{width} map")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{label}: map values must lie in [0,1]")
    if binary and not np.isin(values, [0, 1]).all():
        raise ValueError(f"{label}: valid mask must be binary")
    return torch.from_numpy(values.copy()).unsqueeze(0)


def load_pseudo_manifest(manifest_path, real_cameras):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Pseudo manifest must be an object with schema_version=1")
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Pseudo manifest must contain a nonempty views list")
    references = {camera.image_name: camera for camera in real_cameras}
    if len(references) != len(real_cameras):
        raise ValueError("Real camera image names must be unique for pseudo references")
    loaded_views = []
    image_names = set(references)
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Each pseudo view must be an object")
        image_name = view["image_name"]
        if not isinstance(image_name, str) or not image_name or image_name in image_names:
            raise ValueError(f"Pseudo image_name must be nonempty and unique: {image_name!r}")
        image_names.add(image_name)
        reference_name = view["reference_image_name"]
        if reference_name not in references:
            raise ValueError(f"Pseudo view {image_name}: unknown real reference {reference_name!r}")
        if not isinstance(view.get("source"), str) or not view["source"].strip():
            raise ValueError(f"Pseudo view {image_name}: source must be a nonempty string")
        if view.get("pixel_center_convention", "array_integer") not in ("array_integer", "colmap_half_integer"):
            raise ValueError(f"Pseudo view {image_name}: unsupported pixel center convention")
        width, height = view["width"], view["height"]
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError(f"Pseudo view {image_name}: width/height must be positive integers")
        intrinsic = np.asarray(view["K"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"Pseudo view {image_name}: K must be a finite 3x3 matrix")
        if not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-7) or not np.allclose(
            [intrinsic[0, 1], intrinsic[1, 0]], 0, atol=1e-7
        ) or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
            raise ValueError(f"Pseudo view {image_name}: only positive-focal pinhole K without skew is supported")
        for principal, size in ((intrinsic[0, 2], width), (intrinsic[1, 2], height)):
            if not any(math.isclose(principal, center, abs_tol=1e-5, rel_tol=0) for center in (size / 2, (size - 1) / 2)):
                raise ValueError(f"Pseudo view {image_name}: off-center K is unsupported; reproject to a centered image")
        world_to_camera = np.asarray(view["world_to_camera"], dtype=np.float64)
        if world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all():
            raise ValueError(f"Pseudo view {image_name}: world_to_camera must be a finite 4x4 OpenCV matrix")
        rotation = world_to_camera[:3, :3]
        if not np.allclose(world_to_camera[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=1e-4
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"Pseudo view {image_name}: world_to_camera must be a rigid OpenCV transform")
        image_path = (manifest_path.parent / view["image_path"]).resolve()
        mask_path = (manifest_path.parent / view["valid_mask_path"]).resolve()
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(f"Pseudo view {image_name}: RGB dimensions disagree with K metadata")
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        valid_mask = load_unit_map(mask_path, width, height, f"Pseudo view {image_name} valid mask", binary=True)
        if not valid_mask.any():
            raise ValueError(f"Pseudo view {image_name}: valid mask must contain valid pixels")
        confidence_key = "confidence_mask_path" if view.get("confidence_mask_path") else "confidence_path"
        confidence_path = view.get(confidence_key)
        confidence = None
        if confidence_path:
            confidence_path = (manifest_path.parent / confidence_path).resolve()
            confidence = load_unit_map(confidence_path, width, height, f"Pseudo view {image_name} confidence")
            if not (confidence * valid_mask).any():
                raise ValueError(f"Pseudo view {image_name}: RGB confidence has no valid support")
        depth_path = view.get("depth_path")
        depth, depth_mask, depth_confidence = None, None, None
        resolved_depth_paths = {}
        if depth_path:
            if view.get("depth_convention") != "camera_z" or view.get("depth_units") != "scene":
                raise ValueError(f"Pseudo view {image_name}: depth requires camera_z convention and scene units")
            if not isinstance(view.get("depth_source"), str) or not view["depth_source"].strip():
                raise ValueError(f"Pseudo view {image_name}: depth_source must identify the geometry evidence")
            depth_path = (manifest_path.parent / depth_path).resolve()
            if depth_path.suffix.lower() != ".npy":
                raise ValueError(f"Pseudo view {image_name}: camera-Z depth must be a floating-point NPY")
            depth_values = np.load(depth_path, allow_pickle=False)
            if depth_values.shape == (1, height, width):
                depth_values = depth_values[0]
            if depth_values.shape != (height, width) or not np.issubdtype(depth_values.dtype, np.floating):
                raise ValueError(f"Pseudo view {image_name}: depth must be floating point with the RGB dimensions")
            depth_mask_key = "depth_valid_mask_path" if view.get("depth_valid_mask_path") else "depth_mask_path"
            if not view.get(depth_mask_key):
                raise ValueError(f"Pseudo view {image_name}: depth requires an explicit depth_valid_mask_path")
            depth_mask_path = (manifest_path.parent / view[depth_mask_key]).resolve()
            depth_mask = load_unit_map(depth_mask_path, width, height, f"Pseudo view {image_name} depth mask", binary=True)
            valid_depth = depth_values[depth_mask[0].numpy().astype(bool)]
            if not valid_depth.size or not np.isfinite(valid_depth).all() or np.any(valid_depth <= 0):
                raise ValueError(f"Pseudo view {image_name}: supervised depth pixels must be finite and positive")
            depth_values = np.where(depth_mask[0].numpy() > 0, depth_values, 0).astype(np.float32)
            depth = torch.from_numpy(depth_values.copy()).unsqueeze(0)
            resolved_depth_paths = {"depth_path": str(depth_path), depth_mask_key: str(depth_mask_path)}
            if view.get("depth_confidence_path"):
                depth_confidence_path = (manifest_path.parent / view["depth_confidence_path"]).resolve()
                depth_confidence = load_unit_map(depth_confidence_path, width, height, f"Pseudo view {image_name} depth confidence")
                if not (depth_confidence * depth_mask).any():
                    raise ValueError(f"Pseudo view {image_name}: depth confidence has no valid support")
                resolved_depth_paths["depth_confidence_path"] = str(depth_confidence_path)
        elif any(view.get(key) for key in ("depth_valid_mask_path", "depth_mask_path", "depth_confidence_path")):
            raise ValueError(f"Pseudo view {image_name}: depth masks/confidence require depth_path")
        densification_mask = None
        geometry_source = view.get("geometry_support_source", view.get("depth_source"))
        if depth is not None and geometry_source == "real_aligned_depth_reprojection" and view.get("depth_source") == "real_aligned_depth_reprojection":
            densification_mask = depth_mask.clone()
            if view.get("geometry_support_mask_path"):
                support_path = (manifest_path.parent / view["geometry_support_mask_path"]).resolve()
                densification_mask *= load_unit_map(support_path, width, height, f"Pseudo view {image_name} geometry support", binary=True)
                resolved_depth_paths["geometry_support_mask_path"] = str(support_path)
            densification_mask *= valid_mask
        loaded_views.append({
            **view,
            "image_path": str(image_path),
            "valid_mask_path": str(mask_path),
            **({confidence_key: str(confidence_path)} if confidence_path else {}),
            **resolved_depth_paths,
            "K": intrinsic,
            "world_to_camera": world_to_camera,
            "reference_uid": references[reference_name].uid,
            "image": torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
            "valid_mask": valid_mask,
            "confidence": confidence,
            "depth": depth,
            "depth_mask": depth_mask,
            "depth_confidence": depth_confidence,
            "densification_mask": densification_mask,
        })
    return loaded_views, hashlib.sha256(manifest_bytes).hexdigest()


def make_pseudo_cameras(views, data_device, lazy_payload):
    from scene.cameras import Camera

    cameras = []
    for view in views:
        intrinsic = view["K"]
        world_to_camera = view["world_to_camera"]
        camera = Camera(
            colmap_id=-1,
            R=world_to_camera[:3, :3].T,
            T=world_to_camera[:3, 3],
            FoVx=2 * math.atan(view["width"] / (2 * intrinsic[0, 0])),
            FoVy=2 * math.atan(view["height"] / (2 * intrinsic[1, 1])),
            image=view["image"],
            gt_alpha_mask=None,
            image_name=view["image_name"],
            uid=view["reference_uid"],
            data_device=data_device,
            lazy_payload=lazy_payload,
        )
        camera.Fx, camera.Fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        pixel_offset = 0.5 if view.get("pixel_center_convention") == "colmap_half_integer" else 0.0
        camera.Cx, camera.Cy = float(intrinsic[0, 2]) - pixel_offset, float(intrinsic[1, 2]) - pixel_offset
        camera.pixel_center_convention = view.get("pixel_center_convention", "array_integer")
        camera.projection_matrix[2, 0] = 2 * (camera.Cx - (camera.image_width - 1) / 2) / camera.image_width
        camera.projection_matrix[2, 1] = 2 * (camera.Cy - (camera.image_height - 1) / 2) / camera.image_height
        camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
        camera.pseudo_valid_mask_cpu = view["valid_mask"]
        camera.pseudo_confidence_cpu = view["confidence"]
        camera.pseudo_depth_cpu = view["depth"]
        camera.pseudo_depth_mask_cpu = view["depth_mask"]
        camera.pseudo_depth_confidence_cpu = view["depth_confidence"]
        camera.pseudo_has_depth = view["depth"] is not None
        camera.pseudo_source = view["source"]
        camera.pseudo_densification_mask_cpu = view["densification_mask"]
        cameras.append(camera)
    return cameras


def pseudo_rgb_l1(image, target, valid_mask, gaussians, reference_uid, confidence=None):
    target = target.float().div(255) if target.dtype == torch.uint8 else target
    prediction = image
    model_name = gaussians.app_model.name
    if model_name != "NO":
        embedding = gaussians.get_apperance_embedding(reference_uid)
        if model_name == "GS":
            prediction = torch.addmm(embedding[:3, 3, None], embedding[:3, :3], image.reshape(3, -1)).reshape_as(image)
        elif model_name == "PGSR":
            prediction = torch.addcmul(embedding[1], torch.exp(embedding[0]), image)
        elif model_name == "GOF":
            original_height, original_width = image.shape[1:]
            height, width = original_height // 32 * 32, original_width // 32 * 32
            if min(height, width) == 0:
                raise ValueError("GOF pseudo images must be at least 32 pixels in each dimension")
            top, left = (original_height - height) // 2, (original_width - width) // 2
            prediction = image[:, top:top + height, left:left + width]
            target = target[:, top:top + height, left:left + width]
            valid_mask = valid_mask[:, top:top + height, left:left + width]
            if confidence is not None:
                confidence = confidence[:, top:top + height, left:left + width]
            downsampled = torch.nn.functional.interpolate(
                prediction[None], size=(height // 32, width // 32), mode="bilinear", align_corners=True
            )[0]
            embedding_map = embedding[None].repeat(height // 32, width // 32, 1).permute(2, 0, 1)
            network_input = torch.cat([downsampled, embedding_map], dim=0)[None]
            prediction = gaussians.appearance_network(network_input) * prediction
        else:
            raise ValueError(f"Unsupported pseudo appearance model: {model_name}")
    valid_mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
    weights = valid_mask if confidence is None else valid_mask * confidence.to(device=prediction.device, dtype=prediction.dtype)
    return ((prediction - target).abs() * weights).sum() / (3 * valid_mask.sum()).clamp_min(1)


def pseudo_depth_l1(rendered_depth, camera):
    if not camera.pseudo_has_depth:
        return rendered_depth.new_tensor(0.0)
    target = camera.pseudo_depth_cpu.to(device=rendered_depth.device, dtype=rendered_depth.dtype)
    valid_mask = camera.pseudo_depth_mask_cpu.to(device=rendered_depth.device, dtype=rendered_depth.dtype)
    confidence = camera.pseudo_depth_confidence_cpu
    weights = valid_mask if confidence is None else valid_mask * confidence.to(device=rendered_depth.device, dtype=rendered_depth.dtype)
    height, width = rendered_depth.shape[-2:]
    pixel_x = torch.arange(width, device=rendered_depth.device, dtype=rendered_depth.dtype)[None, :]
    pixel_y = torch.arange(height, device=rendered_depth.device, dtype=rendered_depth.dtype)[:, None]
    raster_radius = (1 + ((pixel_x - (width - 1) / 2) / camera.Fx).square() + ((pixel_y - (height - 1) / 2) / camera.Fy).square()).sqrt()
    camera_radius = (1 + ((pixel_x - camera.Cx) / camera.Fx).square() + ((pixel_y - camera.Cy) / camera.Fy).square()).sqrt()
    camera_depth = rendered_depth * raster_radius / camera_radius
    return ((camera_depth - target).abs() * weights).sum() / valid_mask.sum().clamp_min(1)


def ramp_weight(iteration, start_iteration, ramp_iterations):
    if iteration < start_iteration:
        return 0.0
    if ramp_iterations <= 0:
        return 1.0
    return min((iteration - start_iteration + 1) / ramp_iterations, 1.0)


def accumulate_pseudo_densification(gaussians, render_package, camera, weight, depth_tolerance, minimum_confidence):
    support_mask = camera.pseudo_densification_mask_cpu
    points = gaussians.get_xyz.detach()
    screen_points = render_package["viewspace_points"]
    gradient = screen_points.grad
    if support_mask is None or gradient is None or not len(points):
        return {"eligible_gaussians": 0, "screen_gradient_norm_sum": 0.0}
    device, dtype = points.device, points.dtype
    transform = camera.world_view_transform.to(device=device, dtype=dtype)
    camera_points = points @ transform[:3, :3] + transform[3, :3]
    camera_depth = camera_points[:, 2]
    positive = torch.isfinite(camera_points).all(dim=1) & (camera_depth > 0)
    safe_depth = torch.where(positive, camera_depth, torch.ones_like(camera_depth))
    pixel_x = camera_points[:, 0] / safe_depth * camera.Fx + camera.Cx
    pixel_y = camera_points[:, 1] / safe_depth * camera.Fy + camera.Cy
    height, width = support_mask.shape[-2:]
    in_frame = positive & (pixel_x >= 0) & (pixel_x <= width - 1) & (pixel_y >= 0) & (pixel_y <= height - 1)
    columns = torch.nan_to_num(pixel_x).round().clamp(0, width - 1).long()
    rows = torch.nan_to_num(pixel_y).round().clamp(0, height - 1).long()
    support = support_mask.to(device=device)[0, rows, columns] > 0
    targets = camera.pseudo_depth_cpu.to(device=device, dtype=dtype)[0, rows, columns]
    depth_agrees = torch.isfinite(targets) & (targets > 0) & ((camera_depth - targets).abs() <= depth_tolerance * targets)
    eligible = render_package["visibility_filter"] & in_frame & support & depth_agrees & torch.isfinite(gradient).all(dim=1)
    for confidence_map in (camera.pseudo_confidence_cpu, camera.pseudo_depth_confidence_cpu):
        if confidence_map is not None:
            confidence = confidence_map.to(device=device)[0, rows, columns]
            eligible &= confidence >= minimum_confidence
    count = int(eligible.sum().item())
    if count:
        gaussians.max_radii2D[eligible] = torch.maximum(gaussians.max_radii2D[eligible], render_package["radii"][eligible])
        gaussians.xyz_gradient_accum[eligible] += weight * torch.linalg.vector_norm(gradient[eligible, :2], dim=-1, keepdim=True)
        gaussians.xyz_gradient_accum_abs[eligible] += weight * torch.linalg.vector_norm(gradient[eligible, 2:], dim=-1, keepdim=True)
        gaussians.denom[eligible] += weight
    return {
        "eligible_gaussians": count,
        "screen_gradient_norm_sum": float(torch.linalg.vector_norm(gradient[eligible, :2], dim=-1).sum().item()) if count else 0.0,
    }


def screen_gradient_statistics(render_package):
    gradient = render_package["viewspace_points"].grad
    if gradient is None:
        return {"visible_gaussians": 0, "screen_gradient_norm_sum": 0.0}
    valid = render_package["visibility_filter"] & torch.isfinite(gradient).all(dim=1)
    return {
        "visible_gaussians": int(valid.sum().item()),
        "screen_gradient_norm_sum": float(torch.linalg.vector_norm(gradient[valid, :2], dim=-1).sum().item()),
    }
