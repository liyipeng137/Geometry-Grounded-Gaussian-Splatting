# Bounded mesh extraction using Marching Tetrahedra
# Combines mesh_extract_tetrahedra.py with boundary estimation from export_mesh.py

import os
import math
import gc
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, integrate
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from tetranerf.utils.extension import cpp
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from utils.system_utils import searchForMaxIteration
from utils.tetmesh import marching_tetrahedra


def post_process_mesh(mesh, cluster_to_keep=1, min_triangles=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    
    Args:
        mesh: input mesh
        cluster_to_keep: number of largest clusters to keep
        min_triangles: minimum number of triangles for a cluster to be kept
    """
    import copy

    print("post processing the mesh to have {} clusters, min_triangles={}".format(cluster_to_keep, min_triangles))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    cluster_to_keep = min(cluster_to_keep, len(cluster_n_triangles))
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, min_triangles)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def smooth_mesh(mesh, iterations=0, lambda_filter=0.5, mu=-0.53):
    if iterations <= 0:
        return mesh

    print(
        "smoothing mesh with Taubin filter: "
        f"iterations={iterations}, lambda={lambda_filter}, mu={mu}"
    )
    mesh_smooth = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=lambda_filter,
        mu=mu,
    )
    mesh_smooth.remove_degenerate_triangles()
    mesh_smooth.remove_duplicated_triangles()
    mesh_smooth.remove_duplicated_vertices()
    mesh_smooth.remove_unreferenced_vertices()
    return mesh_smooth


class MinimalMeshCamera:
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_name, uid, image_width, image_height, gt_mask=None, K=None, device="cuda"):
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        if K is None:
            K = np.array([
                [self.image_width / (2 * math.tan(self.FoVx / 2.0)), 0, self.image_width / 2.0],
                [0, self.image_height / (2 * math.tan(self.FoVy / 2.0)), self.image_height / 2.0],
                [0, 0, 1],
            ])
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Mesh camera K must be finite with positive focal lengths")
        if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
            raise ValueError("Mesh camera K must be an undistorted pinhole camera without skew")
        self.source_K_colmap = K.copy()
        self.pixel_center_convention = "colmap_half_integer"
        self.Fx, self.Fy = float(K[0, 0]), float(K[1, 1])
        self.Cx, self.Cy = float(K[0, 2]) - 0.5, float(K[1, 2]) - 0.5
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.Fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.Fy))

        self.gt_mask = gt_mask
        self._gt_mask_cuda = None

        self.zfar = 100.0
        self.znear = 0.01
        w2c = getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
        projection = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        )
        # COLMAP principal points refer to half-integer pixel centers, as in training.
        projection[0, 2] = 2 * K[0, 2] / self.image_width - 1
        projection[1, 2] = 2 * K[1, 2] / self.image_height - 1
        self.projection_matrix = projection.transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def _resolve_image_resolution(orig_w, orig_h, dataset, resolution_scale=1.0):
    if dataset.resolution in [1, 2, 4, 8]:
        return (
            round(orig_w / (resolution_scale * dataset.resolution)),
            round(orig_h / (resolution_scale * dataset.resolution)),
        )

    if dataset.resolution == -1:
        global_down = (orig_w / 1600) if orig_w > 1600 else 1
    else:
        global_down = orig_w / dataset.resolution
    scale = float(global_down) * float(resolution_scale)
    return int(orig_w / scale), int(orig_h / scale)


def _load_gt_mask_for_camera(dataset, image_name, resolution):
    mask_dir = dataset.mask_dir.strip()
    if not mask_dir:
        return None

    if os.path.isabs(mask_dir):
        mask_root = mask_dir
    else:
        mask_root = os.path.join(dataset.source_path, mask_dir)

    mask_format = dataset.mask_format.lower().lstrip(".")
    mask_path = os.path.join(mask_root, f"{image_name}.{mask_format}")
    if not os.path.exists(mask_path):
        return None

    mask_img = Image.open(mask_path)
    loaded_mask = PILtoTorch(mask_img, resolution)[:1]
    return (loaded_mask > 0.5).float()


def load_mesh_cameras_from_colmap(dataset: ModelParams, load_gt_mask=True, llffhold=8):
    sparse_path = os.path.join(dataset.source_path, "sparse", "0")
    images_subdir = "images" if dataset.images is None else dataset.images
    images_folder = os.path.join(dataset.source_path, images_subdir)

    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))

    cam_infos = []
    for key in extrinsics:
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            fov_y = focal2fov(focal_length_y, height)
            fov_x = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fov_y = focal2fov(focal_length_y, height)
            fov_x = focal2fov(focal_length_x, width)
        else:
            raise ValueError(
                f"Unsupported COLMAP camera model: {intr.model}. Only PINHOLE/SIMPLE_PINHOLE are supported."
            )

        if intr.model == "SIMPLE_PINHOLE":
            cx, cy = intr.params[1:3]
        else:
            cx, cy = intr.params[2:4]
        K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]])

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_infos.append(
            {
                "uid": intr.id,
                "R": np.transpose(qvec2rotmat(extr.qvec)),
                "T": np.array(extr.tvec),
                "FoVx": fov_x,
                "FoVy": fov_y,
                "image_name": image_name,
                "width": width,
                "height": height,
                "K": K,
            }
        )

    cam_infos = sorted(cam_infos, key=lambda x: x["image_name"])
    if dataset.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos

    cameras = []
    for idx, cam in enumerate(train_cam_infos):
        resolution = _resolve_image_resolution(cam["width"], cam["height"], dataset, resolution_scale=1.0)
        gt_mask = _load_gt_mask_for_camera(dataset, cam["image_name"], resolution) if load_gt_mask else None
        cameras.append(
            MinimalMeshCamera(
                colmap_id=cam["uid"],
                R=cam["R"],
                T=cam["T"],
                FoVx=cam["FoVx"],
                FoVy=cam["FoVy"],
                image_name=cam["image_name"],
                uid=idx,
                image_width=resolution[0],
                image_height=resolution[1],
                gt_mask=gt_mask,
                K=np.diag([resolution[0] / cam["width"], resolution[1] / cam["height"], 1]) @ cam["K"],
            )
        )
    print(f"Loaded {len(cameras)} mesh cameras from COLMAP (minimal mode)")
    return cameras


def load_gaussians_from_iteration(dataset: ModelParams, gaussians: GaussianModel, iteration: int):
    if iteration == -1:
        point_cloud_root = os.path.join(dataset.model_path, "point_cloud")
        loaded_iter = searchForMaxIteration(point_cloud_root)
    else:
        loaded_iter = iteration
    ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing gaussian checkpoint: {ply_path}")
    print(f"Loading trained model at iteration {loaded_iter}")
    gaussians.load_ply(ply_path)
    return loaded_iter


def _camera_centers_and_forwards(cameras):
    c2ws = np.array([
        np.linalg.inv(np.asarray((cam.world_view_transform.T).detach().cpu().numpy()))
        for cam in cameras
    ])
    camera_centers = c2ws[:, :3, 3]
    # COLMAP/OpenCV camera space looks along +Z; this is the +Z axis in world space.
    camera_forwards = c2ws[:, :3, 2]
    camera_forwards = camera_forwards / np.clip(
        np.linalg.norm(camera_forwards, axis=-1, keepdims=True),
        1e-8,
        None,
    )
    return camera_centers, camera_forwards


def _estimate_ray_focus(camera_centers, camera_forwards):
    identity = np.eye(3, dtype=np.float64)
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for center, forward in zip(camera_centers, camera_forwards):
        forward = forward.astype(np.float64)
        projector = identity - np.outer(forward, forward)
        lhs += projector
        rhs += projector @ center.astype(np.float64)
    focus, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return focus


def detect_object_centric_camera_layout(
    cameras,
    center=None,
    radius=None,
    min_center_looking_ratio=0.65,
    center_max_angle_deg=45.0,
    min_mean_alignment=0.45,
    max_focus_residual_ratio=0.35,
):
    """
    Detect whether most cameras look inward at a shared center/object.

    This intentionally targets object-centric captures. Indoor captures with
    cameras looking outward from a room should produce low or negative
    center-alignment scores and keep the wider default boundary.
    """
    camera_centers, camera_forwards = _camera_centers_and_forwards(cameras)
    metrics = {"num_cameras": len(camera_centers)}
    if len(camera_centers) < 3:
        metrics["reason"] = "need at least 3 cameras"
        return False, metrics

    if center is None:
        center = np.mean(camera_centers, axis=0)
    if radius is None:
        radius = np.max(np.linalg.norm(camera_centers - center[None, :], axis=-1))
    radius = max(float(radius), 1e-6)

    to_center = center[None, :] - camera_centers
    to_center_norm = np.linalg.norm(to_center, axis=-1, keepdims=True)
    valid = to_center_norm[:, 0] > 1e-8
    to_center_dir = np.zeros_like(to_center)
    to_center_dir[valid] = to_center[valid] / to_center_norm[valid]

    alignment = np.sum(camera_forwards * to_center_dir, axis=-1)
    alignment = alignment[valid]
    cos_threshold = math.cos(math.radians(center_max_angle_deg))
    center_looking_ratio = float(np.mean(alignment >= cos_threshold)) if len(alignment) else 0.0
    mean_alignment = float(np.mean(alignment)) if len(alignment) else -1.0
    median_alignment = float(np.median(alignment)) if len(alignment) else -1.0

    focus = _estimate_ray_focus(camera_centers, camera_forwards)
    focus_delta = focus[None, :] - camera_centers
    focus_depth = np.sum(focus_delta * camera_forwards, axis=-1)
    focus_perp = np.linalg.norm(focus_delta - focus_depth[:, None] * camera_forwards, axis=-1)
    focus_residual_ratio = float(np.median(focus_perp) / radius)
    focus_front_ratio = float(np.mean(focus_depth > 0))

    metrics.update(
        {
            "center_looking_ratio": center_looking_ratio,
            "center_max_angle_deg": center_max_angle_deg,
            "mean_alignment": mean_alignment,
            "median_alignment": median_alignment,
            "focus": focus,
            "focus_residual_ratio": focus_residual_ratio,
            "focus_front_ratio": focus_front_ratio,
        }
    )

    is_object_centric = (
        center_looking_ratio >= min_center_looking_ratio
        and mean_alignment >= min_mean_alignment
        and focus_front_ratio >= min_center_looking_ratio
        and focus_residual_ratio <= max_focus_residual_ratio
    )
    return is_object_centric, metrics


def print_object_centric_detection(is_object_centric, metrics):
    print("Object-centric camera layout detection:")
    print(f"  is_object_centric: {is_object_centric}")
    if "reason" in metrics:
        print(f"  reason: {metrics['reason']}")
        return
    focus = metrics["focus"]
    print(f"  cameras: {metrics['num_cameras']}")
    print(
        "  center-looking cameras: "
        f"{metrics['center_looking_ratio']:.3f} "
        f"(angle <= {metrics['center_max_angle_deg']:.1f} deg)"
    )
    print(f"  mean center alignment: {metrics['mean_alignment']:.3f}")
    print(f"  median center alignment: {metrics['median_alignment']:.3f}")
    print(f"  focus point: [{focus[0]:.4f}, {focus[1]:.4f}, {focus[2]:.4f}]")
    print(f"  focus in-front ratio: {metrics['focus_front_ratio']:.3f}")
    print(f"  focus residual / radius: {metrics['focus_residual_ratio']:.3f}")


def estimate_bounding_sphere(cameras):
    """
    Estimate the bounding sphere given camera poses
    Returns center and radius that encompasses the scene
    """
    camera_centers, _ = _camera_centers_and_forwards(cameras)
    
    # Estimate scene center as mean of camera positions
    center = np.mean(camera_centers, axis=0)
    
    # Estimate radius as distance to furthest camera
    distances = np.linalg.norm(camera_centers - center, axis=-1)
    radius = np.max(distances) * 1.2  # Add 20% margin
    
    print(f"Estimated bounding sphere:")
    print(f"  Center: {center}")
    print(f"  Radius: {radius:.4f}")
    
    return center, radius


@torch.no_grad()
def evaluation_validation(view, points, inside):
    if view.gt_mask is None:
        return inside

    points_cam = points @ view.R + view.T
    pts2d = points_cam[:, :2] / points_cam[:, 2:]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
        pts2d,
    )
    if view._gt_mask_cuda is None:
        view._gt_mask_cuda = view.gt_mask.to(device="cuda")
    sampled_mask = torch.nn.functional.grid_sample(view._gt_mask_cuda[None], pts2d[None, None], align_corners=True)
    return (sampled_mask.squeeze() > 0.5) & inside


@torch.no_grad()
def evaluage_alpha_cull(
    points, views, gaussians, pipeline, kernel_size,
    alpha_threshold=0.5, depth_trunc=-1.0, depth_fade_ratio=0.15,
):
    """
    Evaluate alpha-derived SDF with optional per-view depth truncation.

    depth_trunc:
        <= 0 disables depth truncation.
    depth_fade_ratio:
        Fraction of depth_trunc used as a soft fade zone near far depth.
        0.0 means hard cutoff; 0.15 means the last 15% is softly penalized.
    """
    final_sdf = []
    any_valid = []
    chunk_size = 10000000
    use_depth_trunc = depth_trunc > 0
    fade_ratio = max(0.0, min(0.99, float(depth_fade_ratio)))
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        final_weight_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device="cuda")
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")
        for view in tqdm(views, desc="Rendering progress"):
            ret = integrate(point_chunk, view, gaussians, pipeline, kernel_size)
            valid_points = evaluation_validation(view, point_chunk, ret["inside"])

            # Optional per-view depth truncation in camera space.
            if use_depth_trunc:
                points_cam = point_chunk @ view.R + view.T
                z = points_cam[:, 2]
                depth_mask = (z > 0) & (z < depth_trunc)
                valid_points = valid_points & depth_mask

                if fade_ratio > 0:
                    fade_start = depth_trunc * (1.0 - fade_ratio)
                    fade_denom = max(depth_trunc - fade_start, 1e-6)
                    depth_weight = torch.clamp((depth_trunc - z) / fade_denom, min=0.0, max=1.0)
                    # Push far-end alpha towards threshold instead of hard dropping to zero.
                    alpha_effective = depth_weight * ret["alpha_integrated"] + (1.0 - depth_weight) * alpha_threshold
                else:
                    alpha_effective = ret["alpha_integrated"]
            else:
                alpha_effective = ret["alpha_integrated"]

            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
            final_weight_chunk = torch.where(
                valid_points,
                torch.min(alpha_effective, final_weight_chunk),
                final_weight_chunk,
            )
        final_weight_chunk[torch.logical_not(any_valid_chunk)] = 0
        final_sdf_chunk = alpha_threshold - final_weight_chunk
        final_sdf.append(final_sdf_chunk)
        any_valid.append(any_valid_chunk)
    return torch.cat(final_sdf), torch.cat(any_valid)


@torch.no_grad()
def marching_tetrahedra_with_binary_search_bounded(
    model_path, views, gaussians, pipeline, kernel_size, move_cpu, num_cluster,
    alpha_threshold=0.5, scale_factor=1.0, min_triangles=50,
    center=None, radius=None, boundary_margin=1.2,
    depth_trunc=-1.0, depth_fade_ratio=0.15,
    target_faces=-1,
    smooth_iterations=0, smooth_lambda=0.5, smooth_mu=-0.53,
):
    """
    Marching tetrahedra with boundary constraints
    
    Args:
        center: bounding sphere center (numpy array)
        radius: bounding sphere radius (float)
        boundary_margin: margin factor for boundary (default 1.2 = 20% margin)
    """
    # Generate tetra points
    points, points_scale = gaussians.get_tetra_points()
    
    # Apply boundary filtering if provided
    if center is not None and radius is not None:
        center_torch = torch.from_numpy(center).float().cuda()
        boundary_radius = radius * boundary_margin
        
        # Filter points outside boundary
        distances = torch.norm(points - center_torch[None, :], dim=1)
        boundary_mask = distances < boundary_radius
        
        print(f"Boundary filtering:")
        print(f"  Total points: {points.shape[0]}")
        print(f"  Points within boundary: {boundary_mask.sum().item()}")
        print(f"  Filtered out: {(~boundary_mask).sum().item()}")
        
        # Triangulation uses the filtered vertex array directly. No original-index
        # mapping is needed; retaining it and the unfiltered seeds wastes GPU RAM.
        points_filtered = points[boundary_mask]
        points_scale_filtered = points_scale[boundary_mask]
        del boundary_mask, distances, center_torch
    else:
        points_filtered = points
        points_scale_filtered = points_scale
        print("No boundary constraint applied")
    del points, points_scale

    print("construct cell")
    # The extension triangulates on CPU and returns connectivity on its input
    # device. Keep connectivity on CPU when marching there to avoid a large,
    # unnecessary GPU allocation.
    triangulation_points = points_filtered.cpu().contiguous() if move_cpu else points_filtered
    cells = cpp.triangulate(triangulation_points)
    del triangulation_points

    torch.save(cells, os.path.join(model_path, "cells_bounded.pt"))

    sdf, valid = evaluage_alpha_cull(
        points_filtered, views, gaussians, pipeline, kernel_size,
        alpha_threshold=alpha_threshold,
        depth_trunc=depth_trunc,
        depth_fade_ratio=depth_fade_ratio,
    )

    # Move the retained tensors before marching, then release their GPU storage.
    # empty_cache alone cannot release tensors still referenced in this frame.
    if move_cpu:
        points_filtered = points_filtered.cpu()
        points_scale_filtered = points_scale_filtered.cpu()
        cells = cells.cpu()
        sdf, valid = sdf.cpu(), valid.cpu()
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, edge_ids = marching_tetrahedra(
        points_filtered[None], cells.long(), sdf[None],
        points_scale_filtered[None], valid[None]
    )
    del points_filtered, points_scale_filtered, cells, sdf, valid, edge_ids

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()

    faces = faces_list[0].cpu().numpy()
    del verts_list, scale_list, faces_list
    torch.cuda.empty_cache()
    points_mid = (end_points[:, 0, :] + end_points[:, 1, :]) * 0.5

    mesh = trimesh.Trimesh(vertices=points_mid.cpu().numpy(), faces=faces, process=False)
    mesh.export(os.path.join(model_path, "recon_init_bounded.ply"))
    del mesh, points_mid

    # Binary search refinement
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    # n_binary_steps = 10
    n_binary_steps = 5
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) * 0.5
        mid_sdf, _ = evaluage_alpha_cull(
            mid_points, views, gaussians, pipeline, kernel_size,
            alpha_threshold=alpha_threshold,
            depth_trunc=depth_trunc,
            depth_fade_ratio=depth_fade_ratio,
        )
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points_final = (left_points + right_points) * 0.5

    # Scale-based filtering
    mesh = trimesh.Trimesh(vertices=points_final.cpu().numpy(), faces=faces, process=False)
    vertice_mask = distance <= (scale * scale_factor)
    face_mask = vertice_mask.cpu().numpy()[faces].all(axis=1) 
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    # Additional boundary filtering on final vertices
    if center is not None and radius is not None:
        vertices = mesh.vertices
        vertex_distances = np.linalg.norm(vertices - center, axis=1)
        vertex_boundary_mask = vertex_distances < (radius * boundary_margin)
        
        # Create face mask based on vertices within boundary
        face_boundary_mask = vertex_boundary_mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_boundary_mask)
        mesh.remove_unreferenced_vertices()
        
        print(f"Final boundary filtering:")
        print(f"  Vertices within boundary: {vertex_boundary_mask.sum()}")
        print(f"  Final mesh vertices: {len(mesh.vertices)}")

    mesh.export(os.path.join(model_path, "recon_bounded.ply"))

    # Post-processing
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

    print("remove flyers")
    mesh_post = post_process_mesh(o3d_mesh, cluster_to_keep=num_cluster, min_triangles=min_triangles)
    # o3d.io.write_triangle_mesh(os.path.join(model_path, "recon_bounded_post_full.ply"), mesh_post)

    # if target_faces > 0 and len(mesh_post.triangles) > target_faces:
        # print(f"simplifying mesh: {len(mesh_post.triangles)} -> {target_faces} faces")
        # mesh_post = mesh_post.simplify_quadric_decimation(target_faces)
        # mesh_post.remove_degenerate_triangles()
        # mesh_post.remove_duplicated_triangles()
        # mesh_post.remove_duplicated_vertices()
        # mesh_post.remove_unreferenced_vertices()
        # print(f"simplified: {len(mesh_post.vertices)} verts, {len(mesh_post.triangles)} faces")
        # # o3d.io.write_triangle_mesh(os.path.join(model_path, "recon_bounded_post_simplified.ply"), mesh_post)
    if target_faces > 0 and len(mesh_post.triangles) > target_faces:
        import pyvista
        import fast_simplification
        print(f"simplifying mesh: {len(mesh_post.triangles)} -> {target_faces} faces")

        try:
            # mesh post to pyvista mesh
            verts = np.asarray(mesh_post.vertices)
            faces = np.asarray(mesh_post.triangles)
            pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
            pv_mesh = pyvista.PolyData(verts, pv_faces)
            pv_simple = fast_simplification.simplify_mesh(pv_mesh, target_count=target_faces, verbose=True)

            # pyvista mesh to o3d mesh
            sv = pv_simple.points
            sf = pv_simple.faces.reshape(-1, 4)[:, 1:]
            mesh_post = o3d.geometry.TriangleMesh()
            mesh_post.vertices = o3d.utility.Vector3dVector(sv)
            mesh_post.triangles = o3d.utility.Vector3iVector(sf)
            mesh_post.remove_degenerate_triangles()
            mesh_post.remove_duplicated_triangles()
            mesh_post.remove_duplicated_vertices()
            mesh_post.remove_unreferenced_vertices()
            print(f"simplified: {len(mesh_post.vertices)} verts, {len(mesh_post.triangles)} faces")
        except Exception as e:
            print(f"Error simplifying mesh: {e}, fallback to o3d simplify")
            mesh_post = mesh_post.simplify_quadric_decimation(target_faces)
            mesh_post.remove_degenerate_triangles()
            mesh_post.remove_duplicated_triangles()
            mesh_post.remove_duplicated_vertices()
            mesh_post.remove_unreferenced_vertices()
            print(f"open3d simplified: {len(mesh_post.vertices)} verts, {len(mesh_post.triangles)} faces")
        # except Exception as e:
        #     print(f"Error simplifying mesh: {e}, fallback to open3d simplify")

    mesh_smooth = smooth_mesh(
        mesh_post,
        iterations=smooth_iterations,
        lambda_filter=smooth_lambda,
        mu=smooth_mu,
    )
    if smooth_iterations > 0:
        o3d.io.write_triangle_mesh(os.path.join(model_path, "recon_bounded_post_smooth.ply"), mesh_smooth)
    # clean cache
    torch.cuda.empty_cache()
    del mesh_post, mesh_smooth
    gc.collect()
    print("done!")


def extract_mesh(
    dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
    move_cpu: bool, num_cluster: int,
    alpha_threshold: float = 0.5, scale_factor: float = 1.0, min_triangles: int = 50,
    use_boundary: bool = True, boundary_margin: float = 1.2,
    depth_trunc: float = -1.0, depth_trunc_factor: float = 2.0, depth_fade_ratio: float = 0.15,
    load_gt_mask: bool = True,
    target_faces: int = -1,
    smooth_iterations: int = 0, smooth_lambda: float = 0.5, smooth_mu: float = -0.53,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        load_gaussians_from_iteration(dataset, gaussians, iteration)
        kernel_size = dataset.kernel_size
        cams = load_mesh_cameras_from_colmap(dataset, load_gt_mask=load_gt_mask)
        
        # Estimate bounding sphere when needed by boundary and/or automatic depth truncation.
        center, radius = None, None
        need_scene_scale = use_boundary or (depth_trunc <= 0 and depth_trunc_factor > 0)
        if need_scene_scale:
            center, radius = estimate_bounding_sphere(cams)

        is_object_centric, object_centric_metrics = detect_object_centric_camera_layout(
            cams,
            center=center,
            radius=radius,
        )
        print_object_centric_detection(is_object_centric, object_centric_metrics)
        if is_object_centric:
            boundary_margin = 1.1
            depth_trunc_factor = 1.07
            num_cluster = 3
            # 创建一个临时文件到模型路径下
            temp_file = os.path.join(dataset.model_path, "object_centric.txt")
            with open(temp_file, "w") as f:
                f.write(f"boundary_margin: {boundary_margin}\n")
                f.write(f"depth_trunc_factor: {depth_trunc_factor}\n")
                f.write(f"num_cluster: {num_cluster}\n")
        # Resolve depth truncation.
        resolved_depth_trunc = depth_trunc
        if resolved_depth_trunc <= 0 and radius is not None and depth_trunc_factor > 0:
            resolved_depth_trunc = radius * depth_trunc_factor
            print(f"Auto depth truncation enabled: {resolved_depth_trunc:.4f} (radius * {depth_trunc_factor})")
        elif resolved_depth_trunc > 0:
            print(f"Manual depth truncation enabled: {resolved_depth_trunc:.4f}")
        else:
            print("Depth truncation disabled")

        if not use_boundary:
            center, radius = None, None
        
        marching_tetrahedra_with_binary_search_bounded(
            dataset.model_path, cams, gaussians, pipeline, kernel_size, 
            move_cpu, num_cluster, alpha_threshold, scale_factor, min_triangles,
            center, radius, boundary_margin,
            resolved_depth_trunc, depth_fade_ratio,
            target_faces,
            smooth_iterations, smooth_lambda, smooth_mu,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Bounded mesh extraction using Marching Tetrahedra")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=5, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--move_cpu", action="store_true")
    
    # Mesh reduction parameters
    parser.add_argument("--alpha_threshold", default=0.5, type=float, 
                       help="Alpha threshold for surface extraction (0.5-0.7, higher = less volume)")
    parser.add_argument("--scale_factor", default=0.7, type=float,
                       help="Scale factor for edge filtering (0.7-1.0, lower = fewer faces)")
    parser.add_argument("--min_triangles", default=50, type=int,
                       help="Minimum triangles per cluster (50-500, higher = remove more floaters)")
    
    # Boundary parameters
    parser.add_argument("--no_boundary", action="store_true",
                       help="Disable boundary constraint (extract unbounded mesh)")
    parser.add_argument("--boundary_margin", default=2.0, type=float,
                       help="Boundary margin factor (1.0-2.0, larger = more margin)")
    parser.add_argument("--depth_trunc", default=-1.0, type=float,
                       help="Per-view depth truncation distance in camera space. <=0 uses auto or disables")
    parser.add_argument("--depth_trunc_factor", default=1.8, type=float,
                       help="Auto depth truncation factor when --depth_trunc<=0 (depth_trunc = radius * factor)")
    parser.add_argument("--depth_fade_ratio", default=0.15, type=float,
                       help="Soft fade zone ratio near depth_trunc. 0 = hard cutoff, 0.1~0.25 recommended")
    parser.add_argument("--no_load_gt_mask", action="store_true",
                       help="Disable loading GT masks in minimal camera mode")
    parser.add_argument("--target_faces", default=-1, type=int,
                       help="Target face count after decimation (-1 disables, e.g. 200000)")
    parser.add_argument("--smooth_iterations", default=8, type=int,
                       help="Taubin smoothing iterations after mesh extraction (0 disables smoothing)")
    parser.add_argument("--smooth_lambda", default=0.6, type=float,
                       help="Taubin smoothing lambda parameter")
    parser.add_argument("--smooth_mu", default=-0.53, type=float,
                       help="Taubin smoothing mu parameter")
    
    args = get_combined_args(parser)

    extract_mesh(
        model.extract(args), args.iteration, pipeline.extract(args), 
        args.move_cpu, args.num_cluster,
        args.alpha_threshold, args.scale_factor, args.min_triangles,
        use_boundary=not args.no_boundary, boundary_margin=args.boundary_margin,
        depth_trunc=args.depth_trunc, depth_trunc_factor=args.depth_trunc_factor, depth_fade_ratio=args.depth_fade_ratio,
        load_gt_mask=not args.no_load_gt_mask,
        target_faces=args.target_faces,
        smooth_iterations=args.smooth_iterations,
        smooth_lambda=args.smooth_lambda,
        smooth_mu=args.smooth_mu,
    )
