"""Offline, CPU-only virtual camera selection against one immutable mesh.

This produces review assets, not training supervision. Surface regions are a
spatial/normal proxy over area samples, not connected components of the mesh.
"""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


SCHEMA_VERSION = 2
EVIDENCE_NAMES = {0: "unknown", 1: "support", 2: "mixed", 3: "conflict"}


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def camera_rays(camera):
    """Unnormalized rays: intersection t is camera-Z, not ray length."""
    x, y = np.meshgrid(np.arange(camera["width"]) + .5, np.arange(camera["height"]) + .5)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    pose = np.asarray(camera["c2w"])
    directions = pixels @ np.linalg.inv(camera["K"]).T @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], directions.shape)
    return np.ascontiguousarray(np.concatenate((origins, directions), axis=-1), dtype=np.float32)


def look_at_pose(center, target, reference_c2w):
    """Keep reference camera roll; never assume a world up axis."""
    center, target = np.asarray(center), np.asarray(target)
    forward = unit(target - center)
    if np.linalg.norm(target - center) < 1e-8:
        raise ValueError("Look-at target equals camera center")
    reference = np.asarray(reference_c2w)
    right = np.cross(reference[:3, 1], forward)
    if np.linalg.norm(right) < .1:
        right = reference[:3, 0] - forward * np.dot(reference[:3, 0], forward)
    right = unit(right)
    down = unit(np.cross(forward, right))
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = center
    return pose


def camera_position_groups(cameras, distance):
    representatives, labels = [], []
    for camera in cameras:
        center = np.asarray(camera["c2w"])[:3, 3]
        distances = np.linalg.norm(np.asarray(representatives) - center, axis=1) if representatives else np.empty(0)
        if len(distances) and distances.min() <= distance:
            labels.append(int(distances.argmin()))
        else:
            labels.append(len(representatives))
            representatives.append(center)
    return np.asarray(labels, np.int32)


def depth_evidence(points, camera, depth, scene_scale):
    """Classify relative to a reliable first hit; behind it is not conflict."""
    points = np.asarray(points)
    pose, intrinsic = np.asarray(camera["c2w"]), np.asarray(camera["K"])
    q = (points - pose[:3, 3]) @ pose[:3, :3]
    projected = q @ intrinsic.T
    uv = projected[:, :2] / np.maximum(q[:, 2:3], 1e-10) - .5
    x, y = np.rint(uv).astype(np.int64).T
    height, width = depth.shape
    inside = (q[:, 2] > 0) & (x >= 1) & (y >= 1) & (x < width - 1) & (y < height - 1)
    output = {key: np.zeros(len(points), bool) for key in ("measured", "support", "free", "behind")}
    selected = np.flatnonzero(inside)
    if not len(selected):
        return output
    xx, yy = x[selected], y[selected]
    values = np.stack([depth[yy, xx], depth[yy, xx - 1], depth[yy, xx + 1], depth[yy - 1, xx], depth[yy + 1, xx]], axis=1)
    valid = np.isfinite(values).all(1) & (values > 0).all(1)
    values = np.where(np.isfinite(values), values, 0.)
    tolerance = np.maximum(scene_scale / 64, .02 * values[:, 0])
    valid &= np.ptp(values, axis=1) <= np.maximum(scene_scale / 64, .04 * values[:, 0])
    # Reject grazing depth stencils as strong geometric evidence.
    inverse = np.linalg.inv(intrinsic)
    neighbors = []
    for dx, dy, column in ((-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)):
        pixels = np.column_stack((xx + .5 + dx, yy + .5 + dy, np.ones(len(xx))))
        neighbors.append((pixels @ inverse.T) * values[:, column, None])
    normals = unit(np.cross(neighbors[3] - neighbors[2], neighbors[1] - neighbors[0]))
    valid &= np.abs(np.sum(normals * unit(q[selected]), axis=1)) >= .25
    residual = q[selected, 2] - values[:, 0]
    output["measured"][selected] = valid
    output["support"][selected] = valid & (np.abs(residual) <= tolerance)
    output["free"][selected] = valid & (residual < -tolerance)
    output["behind"][selected] = valid & (residual > tolerance)
    return output


def classify_evidence(support_groups, conflict_groups):
    support, conflict = np.asarray(support_groups), np.asarray(conflict_groups)
    result = np.zeros(support.shape, np.uint8)
    result[support > 0] = 1
    result[conflict > 0] = 2
    result[(conflict >= 2) & (support == 0)] = 3
    return result


def aggregate_evidence(points, cameras, depths, scene_scale):
    groups = camera_position_groups(cameras, .02 * scene_scale)
    support, conflict = np.zeros(len(points), np.uint16), np.zeros(len(points), np.uint16)
    for group in np.unique(groups):
        group_support, group_free = np.zeros(len(points), bool), np.zeros(len(points), bool)
        for index in np.flatnonzero(groups == group):
            evidence = depth_evidence(points, cameras[index], depths[index], scene_scale)
            group_support |= evidence["support"]
            group_free |= evidence["free"]
        support += group_support
        conflict += group_free & ~group_support
        if int(group) % 20 == 0:
            log("surface_evidence", position_group=int(group), total_groups=int(groups.max()) + 1)
    return {"support_groups": support, "conflict_groups": conflict,
            "evidence_class": classify_evidence(support, conflict)}


def known_free_space(points, cameras, depths, scene_scale):
    points = np.asarray(points)
    safe = np.zeros(len(points), bool)
    for camera, depth in zip(cameras, depths):
        pending = np.flatnonzero(~safe)
        if not len(pending):
            break
        # Exact measured capture centers are known reachable. This tiny tolerance
        # does not turn a neighborhood of an unobserved center into free space.
        center = np.asarray(camera["c2w"])[:3, 3]
        safe[pending] |= np.linalg.norm(points[pending] - center, axis=1) < scene_scale * 1e-5
        safe[pending] |= depth_evidence(points[pending], camera, depth, scene_scale)["free"]
    return safe


def path_safety(start, end, scene, cameras, depths, scene_scale, threads=8):
    """Direct depth tests along a path, plus exact-mesh distance/intersection."""
    del threads  # backend owns its thread count
    start, end = np.asarray(start), np.asarray(end)
    length = float(np.linalg.norm(end - start))
    step = scene_scale / 256
    points = np.linspace(start, end, max(2, int(np.ceil(length / step)) + 1))
    free = known_free_space(points, cameras, depths, scene_scale)
    distances = scene.distance(points)
    clearance = scene_scale / 64
    first = np.inf
    if length > 1e-10:
        first = float(scene.first_hit(start[None], ((end - start) / length)[None])[0])
    crossed = np.isfinite(first) and first < length - 1e-6
    reason = None
    if not free.all():
        reason = "path_not_in_observed_free_space"
    elif not np.isfinite(distances).all() or (distances < clearance).any():
        reason = "mesh_clearance"
    elif crossed:
        reason = "path_crosses_mesh"
    return {"passed": reason is None, "reason": reason, "connection_checked": True,
            "path_length": length, "path_samples": len(points),
            "known_free_fraction": float(free.mean()), "minimum_mesh_distance": float(distances.min()),
            "required_clearance": clearance, "segment_crosses_mesh": bool(crossed)}


def load_cameras(dataset, long_edge):
    path = Path(__file__).resolve().parents[1] / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("virtual_selection_colmap", path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    sparse = dataset / "sparse/0"
    binary = (sparse / "cameras.bin").is_file()
    intrinsics = loader.read_intrinsics_binary(sparse / "cameras.bin") if binary else loader.read_intrinsics_text(sparse / "cameras.txt")
    images = loader.read_extrinsics_binary(sparse / "images.bin") if binary else loader.read_extrinsics_text(sparse / "images.txt")
    cameras, depths, samples, depth_paths = [], [], [], []
    for index, image in enumerate(sorted(images.values(), key=lambda row: row.name)):
        camera = intrinsics[image.camera_id]
        if camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        else:
            raise ValueError("Undistorted PINHOLE/SIMPLE_PINHOLE COLMAP input required")
        scale = long_edge / max(camera.width, camera.height)
        width, height = max(2, round(camera.width * scale)), max(2, round(camera.height * scale))
        intrinsic = np.diag([width / camera.width, height / camera.height, 1]) @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        pose = np.eye(4)
        pose[:3, :3] = image.qvec2rotmat().T
        pose[:3, 3] = -pose[:3, :3] @ image.tvec
        depth_path = dataset / "depth" / Path(image.name).with_suffix(".npy")
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        if depth.ndim != 2:
            raise ValueError(f"Expected HxW camera-Z depth: {depth_path}")
        yy = np.minimum(((np.arange(height) + .5) * depth.shape[0] / height).astype(int), depth.shape[0] - 1)
        xx = np.minimum(((np.arange(width) + .5) * depth.shape[1] / width).astype(int), depth.shape[1] - 1)
        resized = np.asarray(depth[np.ix_(yy, xx)], np.float32)
        valid = resized[np.isfinite(resized) & (resized > 0)]
        if not len(valid):
            raise ValueError(f"No valid real depth: {depth_path}")
        samples.append(valid[::max(1, len(valid) // 2048)])
        depths.append(resized)
        depth_paths.append(depth_path)
        cameras.append({"id": f"real_{index:04d}", "name": image.name, "kind": "real",
                        "selection_source": "real", "width": width, "height": height,
                        "K": intrinsic.tolist(), "c2w": pose.tolist(), "anchor_id": None,
                        "target": (pose[:3, 3] + pose[:3, 2] * float(np.median(valid))).tolist(),
                        "image_path": str(dataset / "images" / image.name), "depth_path": str(depth_path),
                        "stats": {"median_real_depth": float(np.median(valid))}})
    return cameras, depths, float(np.median(np.concatenate(samples))), depth_paths, binary


class MeshScene:
    def __init__(self, vertices, faces, threads):
        import open3d as o3d
        self.o3d, self.threads = o3d, threads
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)),
                                 o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))

    def cast(self, rays):
        return {key: value.numpy() for key, value in self.scene.cast_rays(
            self.o3d.core.Tensor(np.ascontiguousarray(rays, np.float32)), nthreads=self.threads).items()}

    def distance(self, points):
        return self.scene.compute_distance(self.o3d.core.Tensor(np.ascontiguousarray(points, np.float32)),
                                           nthreads=self.threads).numpy()

    def first_hit(self, origins, directions):
        return self.cast(np.concatenate((origins, directions), axis=-1))["t_hit"]


def load_mesh_and_samples(path, count, seed, scene_scale):
    """Exact full-face area sampling in bounded blocks, not face-index sampling."""
    from plyfile import PlyData
    mesh = PlyData.read(path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    records = mesh["vertex"].data
    vertices = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(np.float32)
    faces = mesh["face"].data["vertex_indices"]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Triangle mesh required")
    areas = np.empty(len(faces), np.float64)
    block_size = 250000
    for start in range(0, len(faces), block_size):
        triangles = vertices[faces[start:start + block_size]]
        areas[start:start + len(triangles)] = .5 * np.linalg.norm(np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        if start % (block_size * 20) == 0:
            log("mesh_area_scan", done=start, total=len(faces))
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Mesh has invalid area")
    total_area = float(areas.sum())
    np.cumsum(areas, out=areas)
    rng = np.random.default_rng(seed)
    ids = np.searchsorted(areas, rng.random(count) * total_area).astype(np.int64)
    del areas
    triangles = vertices[faces[ids]]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    xyz = (1 - u[:, None]) * a + (u * (1 - v))[:, None] * b + (u * v)[:, None] * c
    normals = unit(np.cross(b - a, c - a))
    # Unsigned normal bin + spatial cell is a surface-region proxy only.
    dominant = np.argmax(np.abs(normals), axis=1)
    canonical = normals * np.where(normals[np.arange(count), dominant] < 0, -1., 1.)[:, None]
    keys = np.column_stack((np.floor(xyz / (.15 * scene_scale)).astype(np.int32),
                            np.rint(canonical * 4).astype(np.int32)))
    _, patch = np.unique(keys, axis=0, return_inverse=True)
    atlas = {"xyz": xyz.astype(np.float32), "normal_world": normals.astype(np.float32),
             "face_id": ids, "patch_id": patch.astype(np.int32),
             "area_weight": np.full(count, total_area / count, np.float32)}
    return vertices, faces, atlas, {"vertices": len(vertices), "triangles": len(faces),
                                   "surface_area": total_area, "surface_samples": count,
                                   "patch_method": "spatial_cells_plus_unsigned_normal_bins; not mesh connectivity"}


def _camera_record(source, pose, target, source_kind, index):
    return {"id": f"virtual_{index:05d}", "name": f"virtual_{index:05d}", "kind": "virtual",
            "selection_source": source_kind, "width": source["width"], "height": source["height"],
            "K": source["K"], "c2w": np.asarray(pose).tolist(), "target": np.asarray(target).tolist(),
            "anchor_id": source["id"], "stats": {}}


def propose_candidates(cameras, atlas, scene_scale, candidate_limit, seed):
    rng = np.random.default_rng(seed)
    poses = np.asarray([c["c2w"] for c in cameras])
    centers = poses[:, :3, 3]
    # Farthest-point sampling in position plus orientation features.
    features = np.column_stack((centers / scene_scale, poses[:, :3, 2] * .25))
    anchors, distances = [0], np.full(len(cameras), np.inf)
    while len(anchors) < min(48, len(cameras)):
        distances = np.minimum(distances, np.sum((features - features[anchors[-1]]) ** 2, axis=1))
        distances[anchors] = -1
        anchors.append(int(distances.argmax()))
    result = []
    for anchor in anchors:
        source = cameras[anchor]
        pose = poses[anchor]
        anchor_depth = source["stats"].get("median_teacher_depth") or source["stats"]["median_real_depth"]
        radius = min(.08 * anchor_depth, .08 * scene_scale)
        for offset in ((-radius, 0, 0), (radius, 0, 0), (0, -radius, 0), (0, radius, 0)):
            candidate_pose = pose.copy()
            candidate_pose[:3, 3] += pose[:3, :3] @ offset
            result.append(_camera_record(source, candidate_pose, candidate_pose[:3, 3] + pose[:3, 2] * scene_scale,
                                         "trajectory_offset", len(result)))
    for index in np.linspace(0, max(0, len(cameras) - 2), min(96, len(cameras) - 1), dtype=int):
        if np.linalg.norm(centers[index + 1] - centers[index]) > .3 * scene_scale:
            continue
        pose = np.eye(4)
        pose[:3, 3] = (centers[index] + centers[index + 1]) / 2
        pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(poses[index:index + 2, :3, :3]))([.5]).as_matrix()[0]
        result.append(_camera_record(cameras[index], pose, pose[:3, 3] + pose[:3, 2] * scene_scale,
                                     "trajectory_interpolation", len(result)))
    patch_ids, counts = np.unique(atlas["patch_id"], return_counts=True)
    patch_order = patch_ids[np.argsort(-counts, kind="stable")]
    # Stratify over region sizes. Tiny regions still enter the candidate pool.
    if len(patch_order) > 400:
        patch_order = patch_order[np.linspace(0, len(patch_order) - 1, 400, dtype=int)]
    for patch_id in patch_order:
        indices = np.flatnonzero(atlas["patch_id"] == patch_id)
        if np.all(atlas["evidence_class"][indices] == 3):
            continue
        target = np.mean(atlas["xyz"][indices], axis=0)
        to_target = target - centers
        distance = np.linalg.norm(to_target, axis=1)
        n = atlas["normal_world"][indices[0]]
        score = np.abs(unit(to_target) @ n) / np.maximum(distance / scene_scale, .2)
        ranked = np.argsort(-score)
        used = []
        for anchor in ranked:
            if distance[anchor] < .15 * scene_scale:
                continue
            if any(np.linalg.norm(centers[anchor] - centers[old]) < .03 * scene_scale for old in used):
                continue
            used.append(int(anchor))
            source = cameras[anchor]
            pose = look_at_pose(centers[anchor], target, poses[anchor])
            result.append(_camera_record(source, pose, target, "surface_lookat", len(result)))
            result[-1]["target_patch_id"] = int(patch_id)
            # Short move toward the target, validated later against real free space.
            shifted = centers[anchor] + unit(to_target[anchor]) * min(.06 * scene_scale, .08 * distance[anchor])
            shifted_pose = look_at_pose(shifted, target, poses[anchor])
            result.append(_camera_record(source, shifted_pose, target, "surface_lookat", len(result)))
            result[-1]["target_patch_id"] = int(patch_id)
            if len(used) == 2:
                break
    # Deduplicate exact pose/K grids before any rendering; no patch-specific teacher.
    unique, seen = [], set()
    for candidate in result:
        signature = np.round(np.r_[np.asarray(candidate["c2w"]).ravel(), np.asarray(candidate["K"]).ravel()], 7).tobytes()
        if signature not in seen:
            unique.append(candidate)
            seen.add(signature)
    # Stable sampling prevents a late surface region from disappearing merely by ID.
    if len(unique) > candidate_limit:
        trajectory = [c for c in unique if c["selection_source"] != "surface_lookat"]
        surface = [c for c in unique if c["selection_source"] == "surface_lookat"]
        trajectory_count = min(len(trajectory), max(1, candidate_limit // 2))
        trajectory_ids = np.linspace(0, len(trajectory) - 1, trajectory_count, dtype=int)
        surface_count = min(len(surface), candidate_limit - trajectory_count)
        keep = sorted(rng.choice(len(surface), surface_count, replace=False))
        unique = [trajectory[i] for i in trajectory_ids] + [surface[i] for i in keep]
    return unique


def direction_bins(directions):
    # Twelve approximately uniform bins; opposite viewing directions are distinct.
    phi = (1 + np.sqrt(5)) / 2
    axes = np.array([(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
                    [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
                    [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)])
    return np.argmax(unit(directions) @ unit(axes).T, axis=1).astype(np.uint8)


def render_coverage(scene, camera, atlas, tree, scene_scale, near=.2):
    rays = camera_rays(camera)
    rendered = scene.cast(rays)
    depth = rendered["t_hit"]
    normals = rendered["primitive_normals"]
    valid = np.isfinite(depth) & (depth > near)
    incidence = np.abs(np.sum(normals * unit(rays[..., 3:]), axis=-1))
    valid &= incidence >= .25
    hit = rays[valid, :3] + rays[valid, 3:] * depth[valid, None]
    coverage = {"indices": np.empty(0, np.int32), "directions": np.empty(0, np.uint8)}
    stats = {"visible_fraction": float(valid.mean()), "teacher_hit_fraction": float(np.isfinite(depth).mean()),
             "near_clip_fraction": float((np.isfinite(depth) & (depth <= near)).mean()),
             "unknown_fraction": 0., "covered_surface_samples": 0}
    if len(hit):
        distance, indices = tree.query(hit, workers=scene.threads)
        compatible = (distance <= scene_scale / 32) & (np.abs(np.sum(atlas["normal_world"][indices] * normals[valid], axis=1)) >= .8)
        indices = np.unique(indices[compatible]).astype(np.int32)
        if not len(indices):
            return coverage, stats, rendered
        # A first-hit pixel near a surfel does not prove that surfel is visible:
        # it could lie on the far side of a thin wall. Recast the exact sampled
        # point and require numerical-distance agreement (and face identity when
        # the backend supplies it), rather than the broad NN association radius.
        center = np.asarray(camera["c2w"])[:3, 3]
        offsets = atlas["xyz"][indices] - center
        sample_distances = np.linalg.norm(offsets, axis=1)
        sample_rays = np.column_stack((np.broadcast_to(center, offsets.shape), unit(offsets)))
        sample_hits = scene.cast(sample_rays)
        same_hit = np.isfinite(sample_hits["t_hit"]) & (np.abs(sample_hits["t_hit"] - sample_distances) <= max(1e-6, scene_scale * 2e-5))
        if "primitive_ids" in sample_hits and "face_id" in atlas:
            same_hit &= sample_hits["primitive_ids"] == atlas["face_id"][indices]
        indices = indices[same_hit]
        directions = np.asarray(camera["c2w"])[:3, 3] - atlas["xyz"][indices]
        coverage = {"indices": indices, "directions": direction_bins(directions)}
        evidence = atlas["evidence_class"][indices]
        stats.update(unknown_fraction=float((evidence == 0).mean()) if len(indices) else 0.,
                     conflict_fraction=float((evidence == 3).mean()) if len(indices) else 0.,
                     covered_surface_samples=len(indices),
                     nonconflict_surface_area=float(atlas["area_weight"][indices[evidence != 3]].sum()))
    return coverage, stats, rendered


def greedy_select(candidates, coverage, atlas, target_count=160):
    count = len(atlas["xyz"])
    exposure, bits = np.zeros(count, np.uint16), np.zeros(count, np.uint16)
    patches = atlas["patch_id"]
    patch_area = np.bincount(patches, weights=atlas["area_weight"])
    # Approximate sqrt-area patch allocation while all actual coverage is raycast.
    weights = atlas["area_weight"] / np.sqrt(np.maximum(patch_area[patches], 1e-12))
    weights *= np.choose(atlas["evidence_class"], [.5, 1., .25, 0.])
    selected, remaining = [], set(range(len(candidates)))
    family_counts = {"trajectory": 0, "surface": 0}
    family = ["surface" if c["selection_source"] == "surface_lookat" else "trajectory" for c in candidates]
    while remaining and len(selected) < target_count:
        scores = []
        for index in sorted(remaining):
            ids, directions = coverage[index]["indices"], coverage[index]["directions"]
            novel = (bits[ids] & (np.uint16(1) << directions)) == 0
            gain = np.where(exposure[ids] == 0, 1., np.where(novel, .35, .02))
            gain *= np.clip((4 - exposure[ids].astype(np.float64)) / 4, 0, 1)
            score = float(np.sum(weights[ids] * gain))
            # Prefer balanced source families but never fill with zero-gain views.
            if family_counts[family[index]] > len(selected) / 2 + 2:
                score *= .6
            scores.append((score, index))
        score, index = max(scores, key=lambda value: (value[0], -value[1]))
        if score <= 1e-12:
            break
        selected.append(index)
        remaining.remove(index)
        ids, directions = coverage[index]["indices"], coverage[index]["directions"]
        exposure[ids] += 1
        bits[ids] |= np.uint16(1) << directions
        family_counts[family[index]] += 1
        if len(selected) % 20 == 0:
            log("greedy_selection", selected=len(selected), requested=target_count)
    eligible = atlas["evidence_class"] != 3
    area = atlas["area_weight"]
    direction_count = np.array([int(value).bit_count() for value in bits], np.uint8)
    denominator = float(area[eligible].sum())
    return selected, {"selected_count": len(selected), "requested_count": target_count,
                      "source_counts": family_counts,
                      "nonconflict_area": denominator,
                      "covered_nonconflict_area_fraction": float(area[eligible & (exposure > 0)].sum() / max(denominator, 1e-12)),
                      "two_direction_nonconflict_area_fraction": float(area[eligible & (direction_count >= 2)].sum() / max(denominator, 1e-12)),
                      "four_view_nonconflict_area_fraction": float(area[eligible & (exposure >= 4)].sum() / max(denominator, 1e-12)),
                      "exposure": exposure, "direction_count": direction_count}


def resized_camera(camera, long_edge):
    scale = long_edge / max(camera["width"], camera["height"])
    width, height = max(2, round(camera["width"] * scale)), max(2, round(camera["height"] * scale))
    intrinsic = np.diag([width / camera["width"], height / camera["height"], 1]) @ np.asarray(camera["K"])
    return {**camera, "width": width, "height": height, "K": intrinsic.tolist()}


def save_preview(output, camera, rendered, scene_scale, render_camera=None):
    from PIL import Image
    render_camera = render_camera or camera
    pose = np.asarray(camera["c2w"])
    normals = rendered["primitive_normals"] @ pose[:3, :3]
    rays_camera = camera_rays(render_camera)[..., 3:] @ pose[:3, :3]
    normals *= np.where(np.sum(normals * rays_camera, axis=-1) > 0, -1., 1.)[..., None]
    depth = rendered["t_hit"]
    valid = np.isfinite(depth) & (depth > .2)
    light = unit(np.array([-.4, -.6, -1.]))
    shade = .24 + .65 * np.maximum(0, np.sum(normals * light, axis=-1))
    shading = np.repeat(np.rint(np.clip(shade, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    normal_image = np.rint(np.clip((normals + 1) / 2, 0, 1) * 255).astype(np.uint8)
    depth_gray = np.rint(np.clip(np.where(valid, depth, 0) / (3 * scene_scale), 0, 1) * 255).astype(np.uint8)
    shading[~valid], normal_image[~valid] = 20, 20
    images = {"shading": shading, "normal": normal_image, "depth": depth_gray}
    camera["preview"] = {}
    for kind, image in images.items():
        relative = f"previews/{camera['id']}_{kind}.png"
        Image.fromarray(image).save(output / relative)
        camera["preview"][kind] = relative
    camera["preview"].update(width=render_camera["width"], height=render_camera["height"], purpose="review_only")


def run_selection(dataset, mesh_path, output, target_count=160, candidate_limit=1200,
                  long_edge=256, surface_samples=120000, threads=8, seed=0, preview_long_edge=512):
    dataset, mesh_path, output = Path(dataset).resolve(), Path(mesh_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    if not 1 <= target_count <= candidate_limit or min(long_edge, preview_long_edge) < 32 or surface_samples < 100:
        raise ValueError("Invalid selection dimensions/counts")
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir()
    started = time.monotonic()
    config = dict(target_count=target_count, candidate_limit=candidate_limit, long_edge=long_edge,
                  surface_samples=surface_samples, threads=threads, seed=seed, preview_long_edge=preview_long_edge)
    write_json(output / "state.json", {"status": "running", "config": config})
    log("loading_real_cameras")
    cameras, depths, scale, depth_paths, binary = load_cameras(dataset, long_edge)
    log("real_inputs_loaded", cameras=len(cameras), scene_scale=scale)
    suffix = "bin" if binary else "txt"
    sources = [mesh_path, dataset / f"sparse/0/cameras.{suffix}", dataset / f"sparse/0/images.{suffix}",
               Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "scripts/select_virtual_cameras.py", *depth_paths]
    fingerprints = {str(path): sha256(path) for path in sources}
    write_json(output / "source_fingerprints.json", fingerprints)
    vertices, faces, atlas, mesh_info = load_mesh_and_samples(mesh_path, surface_samples, seed, scale)
    atlas.update(aggregate_evidence(atlas["xyz"], cameras, depths, scale))
    np.savez_compressed(output / "sample_atlas.npz", **atlas)
    colors = np.array([[145, 154, 168], [105, 190, 145], [230, 174, 82], [225, 88, 91]], np.uint8)
    np.savez_compressed(output / "scene_preview.npz", xyz=atlas["xyz"], rgb=colors[atlas["evidence_class"]],
                        normal_world=atlas["normal_world"], evidence_class=atlas["evidence_class"], patch_id=atlas["patch_id"])
    log("building_exact_mesh_acceleration", **mesh_info)
    scene = MeshScene(vertices, faces, threads)
    del vertices, faces
    tree = cKDTree(atlas["xyz"])
    # Anchor displacement uses teacher depth, as defined by the design.
    for index, camera in enumerate(cameras):
        rendered = scene.cast(camera_rays(camera))
        valid = rendered["t_hit"][np.isfinite(rendered["t_hit"]) & (rendered["t_hit"] > .2)]
        camera["stats"]["median_teacher_depth"] = float(np.median(valid)) if len(valid) else None
        if (index + 1) % 20 == 0:
            log("real_teacher_depths", done=index + 1, total=len(cameras))
    candidates = propose_candidates(cameras, atlas, scale, candidate_limit, seed)
    log("candidates_proposed", candidates=len(candidates))
    by_id = {camera["id"]: camera for camera in cameras}
    accepted, coverage, audits = [], [], []
    for index, candidate in enumerate(candidates):
        anchor = by_id[candidate["anchor_id"]]
        audit = path_safety(np.asarray(anchor["c2w"])[:3, 3], np.asarray(candidate["c2w"])[:3, 3],
                            scene, cameras, depths, scale, threads)
        candidate["stats"].update(audit)
        candidate["stats"]["path_passed"] = audit["passed"]
        if audit["passed"]:
            visible, statistics, _ = render_coverage(scene, candidate, atlas, tree, scale)
            candidate["stats"].update(statistics)
            if statistics["visible_fraction"] < .05 or not len(visible["indices"]):
                candidate["rejection_reason"] = "insufficient_renderable_teacher_coverage"
            elif not np.any(atlas["evidence_class"][visible["indices"]] != 3):
                candidate["rejection_reason"] = "only_conflicting_surface"
            else:
                candidate["accepted"] = True
                accepted.append(candidate)
                coverage.append(visible)
        else:
            candidate["rejection_reason"] = audit["reason"]
        candidate.setdefault("accepted", False)
        audits.append(candidate)
        # Always leave reviewable progress even if a long mesh job is interrupted.
        with (output / "candidate_audit.jsonl").open("a") as stream:
            stream.write(json.dumps(candidate, allow_nan=False) + "\n")
        if (index + 1) % 20 == 0 or index + 1 == len(candidates):
            log("candidate_audit", done=index + 1, total=len(candidates), accepted=len(accepted))
    if not accepted:
        write_json(output / "candidates.json", {"schema_version": 2, "candidates": audits})
        write_json(output / "state.json", {"status": "failed_no_safe_candidates", "config": config})
        raise RuntimeError("No safe candidates; inspect candidate_audit.jsonl")
    chosen, report = greedy_select(accepted, coverage, atlas, target_count)
    selected = [accepted[index] for index in chosen]
    np.savez_compressed(output / "coverage.npz", exposure=report.pop("exposure"), direction_count=report.pop("direction_count"))
    chosen_ids = {camera["id"] for camera in selected}
    for candidate in audits:
        candidate["selected"] = candidate["id"] in chosen_ids
        if candidate["accepted"] and not candidate["selected"]:
            candidate["selection_rejection_reason"] = "redundant_or_budget_limited"
    all_preview_cameras = cameras + selected
    for index, camera in enumerate(all_preview_cameras):
        preview_camera = resized_camera(camera, preview_long_edge)
        save_preview(output, camera, scene.cast(camera_rays(preview_camera)), scale, preview_camera)
        if (index + 1) % 20 == 0:
            log("camera_previews", done=index + 1, total=len(all_preview_cameras))
    write_json(output / "real_cameras.json", {"schema_version": 2, "cameras": cameras})
    write_json(output / "virtual_cameras.json", {"schema_version": 2, "cameras": selected})
    write_json(output / "candidates.json", {"schema_version": 2, "candidates": audits})
    rejections = {}
    for camera in audits:
        if not camera["accepted"]:
            reason = camera["rejection_reason"]
            rejections[reason] = rejections.get(reason, 0) + 1
    report.update(schema_version=2, status="complete", config=config, scene_scale=scale, mesh=mesh_info,
                  real_camera_count=len(cameras), proposed_count=len(candidates), safe_useful_count=len(accepted),
                  rejection_counts=rejections,
                  evidence_counts={name: int((atlas["evidence_class"] == value).sum()) for value, name in EVIDENCE_NAMES.items()},
                  elapsed_seconds=time.monotonic() - started,
                  limitations=["Selection/preview only; no training or 512-pixel D/N cache.",
                               "Surface patches are spatial/normal sample proxies, not full-mesh connectivity.",
                               "Coverage is an area-sample estimate associated with exact first-hit raycasts.",
                               "Free-space paths use direct real-depth tests at discrete samples plus exact mesh intersection; estimated priors are not ground truth.",
                               "Mesh geometry is fixed globally; no per-patch or per-view alignment.",
                               "Unknown surfaces remain eligible and are not independent ground truth."])
    write_json(output / "report.json", report)
    manifest = {"schema_version": 2, "status": "complete", "dataset": str(dataset), "genrecon_mesh": str(mesh_path),
                "scene_scale": scale, "pixel_center_convention": "colmap_half_integer", "depth_convention": "camera_z",
                "normal_convention": "camera_space_unit_face_normal_toward_camera", "units": "scene",
                "real_cameras_path": "real_cameras.json", "virtual_cameras_path": "virtual_cameras.json",
                "scene_preview_path": "scene_preview.npz", "atlas_path": "sample_atlas.npz",
                "report_path": "report.json", "candidates_path": "candidates.json", "coverage_path": "coverage.npz",
                "source_fingerprints_path": "source_fingerprints.json", "preview_colors": "evidence categories; not texture"}
    write_json(output / "manifest.json", manifest)
    receipt_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in ("state.json", "receipt.json"))
    write_json(output / "receipt.json", {"schema_version": 2, "files": {str(path.relative_to(output)): sha256(path) for path in receipt_paths}})
    write_json(output / "state.json", {"status": "complete", "config": config, "manifest_sha256": sha256(output / "manifest.json")})
    log("complete", selected=len(selected), output=str(output), elapsed_seconds=report["elapsed_seconds"])
    return manifest
