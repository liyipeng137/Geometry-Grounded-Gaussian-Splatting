"""全局 GenRecon 面积采样 20 万 XYZ/RGB 点；不改变训练/filter/teacher 配方。"""

from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.virtual_camera_selection import MeshScene, log, sha256, unit, write_json


POINT_COUNT = 200000


def load_real_views(dataset, images="images", eval_mode=False, train_exclude_images=""):
    """直接读取 COLMAP，避免 import scene 触发 CUDA 训练依赖。"""
    dataset = Path(dataset).resolve()
    module_path = Path(__file__).resolve().parents[1] / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("genrecon_init_colmap", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sparse = Path(dataset) / "sparse/0"
    if (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file():
        cameras = module.read_intrinsics_binary(sparse / "cameras.bin")
        extrinsics = module.read_extrinsics_binary(sparse / "images.bin")
        calibration_files = [sparse / "cameras.bin", sparse / "images.bin"]
    else:
        cameras = module.read_intrinsics_text(sparse / "cameras.txt")
        extrinsics = module.read_extrinsics_text(sparse / "images.txt")
        calibration_files = [sparse / "cameras.txt", sparse / "images.txt"]
    views = []
    for image in sorted(extrinsics.values(), key=lambda value: value.name):
        if Path(image.name).is_absolute() or any(part in ("", ".", "..") for part in image.name.split("/")) or "\\" in image.name:
            raise ValueError(f"Unsafe full COLMAP image.name: {image.name}")
        camera = cameras[image.camera_id]
        if camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        else:
            raise ValueError("Undistorted SIMPLE_PINHOLE/PINHOLE cameras required")
        pose = np.eye(4)
        quaternion = np.asarray(image.qvec, np.float64)
        length = np.linalg.norm(quaternion)
        if not np.isfinite(quaternion).all() or length <= 0:
            raise ValueError(f"Invalid camera rotation: {image.name}")
        pose[:3, :3] = module.qvec2rotmat(quaternion / length).T
        pose[:3, 3] = -pose[:3, :3] @ image.tvec
        image_path = Path(dataset) / (images or "images") / image.name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        views.append({"name": image.name, "image_path": str(image_path),
                      "width": int(camera.width), "height": int(camera.height),
                      "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), "c2w": pose})
    split_path = Path(__file__).resolve().parents[1] / "scene/train_split.py"
    spec = importlib.util.spec_from_file_location("mesh_init_train_split", split_path)
    split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split)
    excluded, _ = split.load_excluded_image_names(train_exclude_images)
    if train_exclude_images and eval_mode:
        raise ValueError("Explicit train_exclude_images cannot be combined with implicit --eval")
    wrappers = [SimpleNamespace(colmap_image_name=view["name"], image_name=view["name"], view=view) for view in views]
    train, _ = split.partition_cameras(wrappers, excluded, eval_mode, llffhold=8)
    views = [camera.view for camera in train]
    if not views or len(views) >= np.iinfo(np.uint16).max:
        raise ValueError("Require 1..65534 real cameras")
    return views, calibration_files


def views_from_cameras(train_cameras):
    """Scene CameraInfo.R 已是 c2w 旋转；不做 inverse 或降精度转换。"""
    views = []
    for camera in train_cameras:
        name = camera.colmap_image_name or camera.image_name
        intrinsic = np.asarray(camera.K, np.float64)
        rotation, translation = np.asarray(camera.R, np.float64), np.asarray(camera.T, np.float64)
        if (intrinsic.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,)
                or not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation))
                or min(camera.width, camera.height, intrinsic[0, 0], intrinsic[1, 1]) <= 0):
            raise ValueError(f"Invalid training camera geometry: {name}")
        if not np.allclose(intrinsic[2], [0, 0, 1]) or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
            raise ValueError("Mesh initialization requires undistorted pinhole K")
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = -rotation @ translation
        path = Path(camera.image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        views.append({"name": name, "image_path": str(path), "width": int(camera.width), "height": int(camera.height),
                      "K": intrinsic.copy(), "c2w": pose})
    views.sort(key=lambda view: view["name"])
    if not views or len(views) >= np.iinfo(np.uint16).max or len({view["name"] for view in views}) != len(views):
        raise ValueError("Require 1..65534 uniquely named actual training cameras")
    return views


def sample_surface(vertices, faces, count, seed, block_size=250000):
    """精确全部三角面积 CDF；只在固定大小块中展开三角坐标。"""
    if count < 1 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Positive count and triangle faces required")
    cdf = np.empty(len(faces), np.float64)
    for start in range(0, len(faces), block_size):
        ids = faces[start:start + block_size]
        if (ids < 0).any() or (ids >= len(vertices)).any():
            raise ValueError("Invalid mesh face index")
        triangle = vertices[ids]
        a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
        cdf[start:start + len(ids)] = .5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        if start % (20 * block_size) == 0:
            log("gen_init_area_scan", done=start, total=len(faces))
    if not np.isfinite(cdf).all() or cdf.sum() <= 0:
        raise ValueError("Mesh has nonfinite or zero total surface area")
    degenerate_faces = int((cdf == 0).sum())
    np.cumsum(cdf, out=cdf)
    total_area = float(cdf[-1])
    rng = np.random.default_rng(seed)
    face_ids = np.searchsorted(cdf, rng.random(count) * total_area, side="right")
    del cdf
    triangle = vertices[faces[face_ids]]
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    points = (1 - u[:, None]) * a + (u * (1 - v))[:, None] * b + (u * v)[:, None] * c
    normals = unit(np.cross(b - a, c - a))
    return points.astype(np.float32), normals.astype(np.float32), face_ids.astype(np.int64), {
        "total_area": total_area, "degenerate_faces": degenerate_faces,
        "sampled_distinct_faces": int(len(np.unique(face_ids))),
        "method": "all-triangle exact area CDF and uniform barycentric sampling; no visibility selection",
    }


def project_points(points, view):
    pose = np.asarray(view["c2w"])
    camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
    projected = camera_points @ np.asarray(view["K"]).T
    # COLMAP 坐标以半整数为像素中心；数组采样坐标必须减 .5。
    pixels = projected[:, :2] / np.maximum(camera_points[:, 2:3], 1e-12) - .5
    valid = ((camera_points[:, 2] > 1e-6) & np.isfinite(pixels).all(axis=1)
             & (pixels[:, 0] >= 0) & (pixels[:, 0] <= view["width"] - 1)
             & (pixels[:, 1] >= 0) & (pixels[:, 1] <= view["height"] - 1))
    return pixels, valid


def bilinear_rgb(image, pixels):
    x, y = pixels[:, 0], pixels[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    result = ((1 - wx) * (1 - wy) * image[y0, x0] + wx * (1 - wy) * image[y0, x1]
              + (1 - wx) * wy * image[y1, x0] + wx * wy * image[y1, x1])
    return np.rint(result).clip(0, 255).astype(np.uint8)


def visible_samples(scene, points, face_ids, center):
    """使用单位方向，t_hit 是欧氏距离；不能把投影在图内当无遮挡。"""
    delta = points - center
    distance = np.linalg.norm(delta, axis=1)
    rays = np.column_stack((np.broadcast_to(center, points.shape), unit(delta))).astype(np.float32)
    hit = scene.cast(rays)
    tolerance = np.maximum(1e-5, distance * 2e-5)
    visible = (np.isfinite(hit["t_hit"]) & (np.abs(hit["t_hit"] - distance) <= tolerance))
    # 同面身份进一步防止邻近薄层拿到前景颜色；样本位于三角形内部。
    visible &= hit["primitive_ids"].astype(np.int64) == face_ids
    return visible


def project_colors(points, normals, face_ids, views, scene, image_reader=None):
    """每点选一个可见真实机位；失败继续所有候选，未见点保留中性灰。"""
    count = len(points)
    scores = np.full((len(views), count), -np.inf, np.float32)
    for index, view in enumerate(views):
        _, valid = project_points(points, view)
        toward_camera = np.asarray(view["c2w"])[:3, 3] - points
        squared_distance = np.sum(toward_camera ** 2, axis=1)
        incidence = np.abs(np.sum(normals * unit(toward_camera), axis=1))
        scores[index, valid] = (incidence[valid] / np.maximum(squared_distance[valid], 1e-12)).astype(np.float32)
    candidate_counts = np.isfinite(scores).sum(axis=0)
    order = np.argsort(-scores, axis=0, kind="stable").astype(np.uint16)
    del scores
    colors = np.full((count, 3), 128, np.uint8)
    source_view = np.full(count, -1, np.int32)
    attempts = np.zeros(count, np.uint16)
    image_cache = OrderedDict()

    def read_image(view_index):
        if view_index not in image_cache:
            view = views[view_index]
            image = image_reader(view) if image_reader else np.asarray(Image.open(view["image_path"]).convert("RGB"))
            if image.shape != (view["height"], view["width"], 3):
                raise ValueError(f"Real image dimensions differ from COLMAP: {view['name']}")
            image_cache[view_index] = image
            if len(image_cache) > 16:
                image_cache.popitem(last=False)
        image_cache.move_to_end(view_index)
        return image_cache[view_index]

    for rank in range(int(candidate_counts.max(initial=0))):
        pending = np.flatnonzero((source_view < 0) & (candidate_counts > rank))
        if not len(pending):
            break
        chosen_views = order[rank, pending]
        for view_index in np.unique(chosen_views):
            indices = pending[chosen_views == view_index]
            view = views[int(view_index)]
            visible = visible_samples(scene, points[indices], face_ids[indices], np.asarray(view["c2w"])[:3, 3])
            attempts[indices] += 1
            indices = indices[visible]
            if not len(indices):
                continue
            pixels, valid = project_points(points[indices], view)
            if not valid.all():
                raise RuntimeError("Camera candidate changed during RGB projection")
            colors[indices] = bilinear_rgb(read_image(int(view_index)), pixels)
            source_view[indices] = int(view_index)
        if rank < 3 or (rank + 1) % 10 == 0:
            log("gen_init_rgb_projection", candidate_rank=rank + 1, colored=int((source_view >= 0).sum()),
                total=count, checked_rays=int(attempts.sum(dtype=np.int64)))
    colored = source_view >= 0
    stats = {"colored_points": int(colored.sum()), "neutral_points": int((~colored).sum()),
             "colored_fraction": float(colored.mean()), "no_image_candidate_points": int((candidate_counts == 0).sum()),
             "all_image_candidates_occluded_points": int(((~colored) & (candidate_counts > 0)).sum()),
             "checked_rays": int(attempts.sum(dtype=np.int64)),
             "attempt_quantiles_0_50_90_99_100": np.quantile(attempts, [0, .5, .9, .99, 1]).tolist(),
             "source_camera_counts": {views[i]["name"]: int((source_view == i).sum()) for i in np.unique(source_view[colored])}}
    return colors, source_view, stats


def write_point_cloud(path, points, normals, colors):
    records = np.empty(len(points), dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
                       + [(name, "u1") for name in ("red", "green", "blue")])
    for index, name in enumerate(("x", "y", "z")):
        records[name] = points[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        records[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, index]
    PlyData([PlyElement.describe(records, "vertex")], text=False).write(path)


def _prepare_views(dataset, mesh_path, output, views, calibration_files, seed=0, threads=8):
    dataset, mesh_path, output = Path(dataset).resolve(), Path(mesh_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    input_files = [mesh_path, *calibration_files, *(Path(view["image_path"]) for view in views)]
    fingerprints = {str(path): sha256(path) for path in input_files}
    producers = {str(Path(__file__).resolve()): sha256(__file__),
                 str(Path(__file__).resolve().parents[1] / "utils/virtual_camera_selection.py"):
                     sha256(Path(__file__).resolve().parents[1] / "utils/virtual_camera_selection.py")}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "state.json", {"status": "running", "seed": seed, "point_count": POINT_COUNT})
    try:
        mesh = PlyData.read(mesh_path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
        records = mesh["vertex"].data
        vertices = np.column_stack([records[key] for key in ("x", "y", "z")]).astype(np.float32)
        faces = mesh["face"].data["vertex_indices"]
        points, normals, face_ids, sampling = sample_surface(vertices, faces, POINT_COUNT, seed)
        log("gen_init_exact_mesh_acceleration", vertices=len(vertices), triangles=len(faces))
        scene = MeshScene(vertices, faces, threads)
        mesh_counts = {"vertices": len(vertices), "triangles": len(faces)}
        del vertices, faces, mesh
        colors, source_views, coloring = project_colors(points, normals, face_ids, views, scene)
        if len(points) != POINT_COUNT or not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise RuntimeError("Invalid initialization sample count/coordinates")
        ply_path = output / "point_cloud.ply"
        write_point_cloud(ply_path, points, normals, colors)
        np.savez_compressed(output / "sampling_provenance.npz", face_id=face_ids, color_view_index=source_views)
        for path, expected in {**fingerprints, **producers}.items():
            if sha256(path) != expected:
                raise RuntimeError(f"Input changed during Gen initialization preparation: {path}")
        camera_records = [{**view, "K": np.asarray(view["K"]).tolist(), "c2w": np.asarray(view["c2w"]).tolist()} for view in views]
        camera_hash = hashlib.sha256(json.dumps(camera_records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        metadata = {"schema_version": 1, "status": "complete", "initialization_source": "genrecon_global_surface",
                    "dataset": str(dataset), "genrecon_mesh": str(mesh_path), "coordinate_system": "unchanged COLMAP world",
                    "point_count": POINT_COUNT, "seed": seed, "real_camera_count": len(views), "mesh": mesh_counts,
                    "sampling": sampling, "coloring": coloring, "source_files_sha256": fingerprints,
                    "producer_files_sha256": producers, "point_cloud_path": str(ply_path), "point_cloud_sha256": sha256(ply_path),
                    "provenance_path": "sampling_provenance.npz", "provenance_sha256": sha256(output / "sampling_provenance.npz"),
                    "color_camera_names": [view["name"] for view in views],
                    "color_cameras": camera_records, "color_camera_fingerprint": camera_hash,
                    "camera_scope": "actual training camera subset and its supplied image paths only",
                    "color_policy": "first exact-mesh-visible real camera by absolute-incidence/inverse-distance-squared rank; exhaust all in-image candidates; bilinear RGB; unseen=128",
                    "pixel_center_convention": "COLMAP half-integer projected coordinates minus 0.5 for image indexing",
                    "normal_policy": "source mesh face normals saved as metadata; training decides whether to consume them",
                    "limitations": ["Global area sampling keeps invisible shells/backs; visibility only chooses color, never filters XYZ.",
                                    "Visibility is first-hit consistency with the Gen mesh, not independently measured real-scene occlusion.",
                                    "No Gen texture or learned exposure correction is used.",
                                    "No filter, protection, Gaussian scale, rotation, or opacity recipe is changed here."]}
        write_json(output / "metadata.json", metadata)
        write_json(output / "state.json", {"status": "complete", "metadata_sha256": sha256(output / "metadata.json")})
        log("gen_init_complete", output=str(output), point_count=POINT_COUNT,
            colored_fraction=coloring["colored_fraction"], neutral_points=coloring["neutral_points"],
            checked_rays=coloring["checked_rays"])
        return ply_path
    except Exception as error:
        write_json(output / "state.json", {"status": "failed", "error": str(error)})
        raise


def prepare_from_cameras(dataset, mesh_path, output, train_cameras, seed=0, threads=8):
    """训练入口：只用已划分的 CameraInfo 着色，返回该次独占输出的 PLY。"""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    views = views_from_cameras(train_cameras)
    sparse = Path(dataset).resolve() / "sparse/0"
    extension = "bin" if (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file() else "txt"
    calibration_files = [sparse / f"cameras.{extension}", sparse / f"images.{extension}"]
    return _prepare_views(dataset, mesh_path, output, views, calibration_files, seed, threads)


def prepare(dataset, mesh_path, output, seed=0, threads=8, images="images", eval_mode=False, train_exclude_images=""):
    """手动 CLI 的兼容入口，与正式 reader 使用相同的训练划分规则。"""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    views, calibration_files = load_real_views(dataset, images, eval_mode, train_exclude_images)
    return _prepare_views(dataset, mesh_path, output, views, calibration_files, seed, threads)
