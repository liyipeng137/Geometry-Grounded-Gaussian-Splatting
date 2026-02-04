# Bounded mesh extraction using Marching Tetrahedra
# Combines mesh_extract_tetrahedra.py with boundary estimation from export_mesh.py

import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, evaluate_sdf, integrate
from scene import Scene
from tetranerf.utils.extension import cpp
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
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, min_triangles)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def estimate_bounding_sphere(cameras):
    """
    Estimate the bounding sphere given camera poses
    Returns center and radius that encompasses the scene
    """
    # Extract camera centers (C2W positions)
    c2ws = np.array([
        np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) 
        for cam in cameras
    ])
    
    # Camera centers in world space
    camera_centers = c2ws[:, :3, 3]
    
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
    sampled_mask = torch.nn.functional.grid_sample(view.gt_mask[None].cuda(), pts2d[None, None], align_corners=True)
    return (sampled_mask.squeeze() > 0.5) & inside


@torch.no_grad()
def evaluage_alpha_cull(points, views, gaussians, pipeline, kernel_size):
    final_sdf = []
    any_valid = []
    chunk_size = 10000000
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        final_weight_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device="cuda")
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")
        for view in tqdm(views, desc="Rendering progress"):
            ret = integrate(point_chunk, view, gaussians, pipeline, kernel_size)
            valid_points = evaluation_validation(view, point_chunk, ret["inside"])
            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
            final_weight_chunk = torch.where(
                valid_points,
                torch.min(ret["alpha_integrated"], final_weight_chunk),
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
    center=None, radius=None, boundary_margin=1.2
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
        
        # Keep original indices for later cell filtering
        original_indices = torch.arange(points.shape[0], device=points.device)
        valid_indices = original_indices[boundary_mask]
        
        # Filter points and scales
        points_filtered = points[boundary_mask]
        points_scale_filtered = points_scale[boundary_mask]
        
        # Create index mapping for cell filtering
        index_map = torch.full((points.shape[0],), -1, dtype=torch.long, device=points.device)
        index_map[valid_indices] = torch.arange(valid_indices.shape[0], device=points.device)
    else:
        points_filtered = points
        points_scale_filtered = points_scale
        index_map = None
        print("No boundary constraint applied")

    print("construct cell")
    cells = cpp.triangulate(points_filtered)
    
    # # Filter cells if boundary was applied
    # if index_map is not None:
    #     # Remove cells that reference filtered-out vertices
    #     valid_cells_mask = torch.all(cells >= 0, dim=1)
    #     cells = cells[valid_cells_mask]
    #     print(f"Cells after boundary filtering: {cells.shape[0]}")
    
    torch.save(cells, os.path.join(model_path, "cells_bounded.pt"))

    sdf, valid = evaluage_alpha_cull(points_filtered, views, gaussians, pipeline, kernel_size)

    torch.cuda.empty_cache()
    # Marching tetrahedra
    if move_cpu:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            points_filtered.cpu()[None], cells.cpu().long(), sdf[None].cpu(), 
            points_scale_filtered[None].cpu(), valid[None].cpu()
        )
    else:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            points_filtered[None], cells.long(), sdf[None], 
            points_scale_filtered[None], valid[None]
        )
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()

    faces = faces_list[0].cpu().numpy()
    points_mid = (end_points[:, 0, :] + end_points[:, 1, :]) * 0.5

    mesh = trimesh.Trimesh(vertices=points_mid.cpu().numpy(), faces=faces, process=False)
    mesh.export(os.path.join(model_path, "recon_init_bounded.ply"))

    # Binary search refinement
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 10
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) * 0.5
        mid_sdf, _ = evaluage_alpha_cull(mid_points, views, gaussians, pipeline, kernel_size)
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
    mesh_post = post_process_mesh(o3d_mesh, cluster_to_keep=1, min_triangles=min_triangles)
    o3d.io.write_triangle_mesh(os.path.join(model_path, "recon_bounded_post.ply"), mesh_post)
    print("done!")


def extract_mesh(
    dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
    move_cpu: bool, num_cluster: int,
    alpha_threshold: float = 0.5, scale_factor: float = 1.0, min_triangles: int = 50,
    use_boundary: bool = True, boundary_margin: float = 1.2
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        kernel_size = dataset.kernel_size

        cams = scene.getTrainCameras()
        
        # Estimate bounding sphere
        center, radius = None, None
        if use_boundary:
            center, radius = estimate_bounding_sphere(cams)
        
        marching_tetrahedra_with_binary_search_bounded(
            dataset.model_path, cams, gaussians, pipeline, kernel_size, 
            move_cpu, num_cluster, alpha_threshold, scale_factor, min_triangles,
            center, radius, boundary_margin
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Bounded mesh extraction using Marching Tetrahedra")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--move_cpu", action="store_true")
    
    # Mesh reduction parameters
    parser.add_argument("--alpha_threshold", default=0.5, type=float, 
                       help="Alpha threshold for surface extraction (0.5-0.7, higher = less volume)")
    parser.add_argument("--scale_factor", default=0.8, type=float,
                       help="Scale factor for edge filtering (0.7-1.0, lower = fewer faces)")
    parser.add_argument("--min_triangles", default=50, type=int,
                       help="Minimum triangles per cluster (50-500, higher = remove more floaters)")
    
    # Boundary parameters
    parser.add_argument("--no_boundary", action="store_true",
                       help="Disable boundary constraint (extract unbounded mesh)")
    parser.add_argument("--boundary_margin", default=1.2, type=float,
                       help="Boundary margin factor (1.0-2.0, larger = more margin)")
    
    args = get_combined_args(parser)

    # Fix global alpha_threshold for evaluage_alpha_cull
    global alpha_threshold
    alpha_threshold = args.alpha_threshold

    extract_mesh(
        model.extract(args), args.iteration, pipeline.extract(args), 
        args.move_cpu, args.num_cluster,
        args.alpha_threshold, args.scale_factor, args.min_triangles,
        use_boundary=not args.no_boundary, boundary_margin=args.boundary_margin
    )
