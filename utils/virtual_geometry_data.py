"""冻结虚拟几何 teacher 的离线准备、校验和有限设备缓存。"""

import copy
import hashlib
import importlib.util
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree

from utils.virtual_camera_selection import (
    MeshScene, camera_position_groups, camera_rays, classify_evidence, depth_evidence,
    log, resized_camera, sha256, unit, write_json,
)


ARRAY_KEYS = {"depth", "normal", "valid_depth", "valid_normal", "q", "weight_depth",
              "weight_normal", "weight_coverage", "face_id", "patch_id", "evidence_class"}
SCALAR_IMAGES = {"depth", "valid_depth", "valid_normal", "q", "weight_depth", "weight_normal", "weight_coverage"}


def _arrays_fingerprint(arrays):
    """同进程预载交接的内容证明；不重新压缩，也不改变磁盘 schema。"""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _source_fingerprint(path, preverified_sources):
    """仅复用本进程调用者已验证的前置 SHA；阶段末仍完整重验来源。"""
    key = str(Path(path).resolve())
    if key not in preverified_sources:
        return sha256(path)
    expected = preverified_sources[key]
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid preverified source SHA: {path}")
    if not Path(path).is_file():
        raise ValueError(f"Preverified source is missing: {path}")
    return expected


def validate_camera(record):
    width, height = record["width"], record["height"]
    intrinsic, pose = np.asarray(record["K"], float), np.asarray(record["c2w"], float)
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 3:
        raise ValueError("Teacher camera needs integer dimensions >= 3")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise ValueError("Invalid teacher intrinsic")
    if not np.allclose(intrinsic[2], [0, 0, 1]) or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
        raise ValueError("Teacher camera must be an undistorted pinhole without skew")
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid teacher pose")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6):
        raise ValueError("Teacher pose is not a proper rotation")


def validate_arrays(arrays, record):
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Teacher arrays do not match schema 3")
    shape = (record["height"], record["width"])
    for key, value in arrays.items():
        expected_shape = shape + (3,) if key == "normal" else shape
        if value.shape != expected_shape:
            raise ValueError(f"Teacher {key} has wrong dimensions")
        if key in ("valid_depth", "valid_normal"):
            if value.dtype != np.bool_:
                raise ValueError(f"Teacher {key} must be bool")
        elif key in ("face_id", "patch_id", "evidence_class"):
            if value.dtype.kind not in "iu":
                raise ValueError(f"Teacher {key} must be integer")
        elif value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Teacher {key} must be finite float32")
    valid, normal_valid = arrays["valid_depth"], arrays["valid_normal"]
    if np.any(normal_valid & ~valid) or np.any(valid & (arrays["depth"] <= .2)):
        raise ValueError("Invalid teacher depth/normal domain")
    if np.any(~valid & (arrays["depth"] != 0)) or np.any(valid & ((arrays["face_id"] < 0) | (arrays["patch_id"] < 0))):
        raise ValueError("Teacher first-hit identity/depth differs from validity")
    if np.any(valid & ((arrays["evidence_class"] < 0) | (arrays["evidence_class"] > 3))):
        raise ValueError("Unknown teacher evidence label")
    lengths = np.linalg.norm(arrays["normal"], axis=-1)
    if np.any(normal_valid & (np.abs(lengths - 1) > 2e-4)):
        raise ValueError("Teacher normals must be unit face normals")
    for key in ("q", "weight_depth", "weight_normal", "weight_coverage"):
        if (arrays[key] < 0).any() or np.any(~valid & (arrays[key] != 0)):
            raise ValueError(f"Invalid {key} outside teacher domain")


def region_keys(points, normals, scene_scale):
    normals = unit(normals)
    dominant = np.argmax(np.abs(normals), axis=1)
    normals *= np.where(normals[np.arange(len(normals)), dominant] < 0, -1., 1.)[:, None]
    return np.column_stack((np.floor(points / (.15 * scene_scale)).astype(np.int32),
                            np.rint(normals * 4).astype(np.int32)))


def teacher_arrays(scene, camera, atlas, tree, region_ids, scene_scale):
    """渲染原 mesh；近邻 atlas 只传证据，不能把薄层另一侧误当对应面。"""
    rays = camera_rays(camera)
    rendered = scene.cast(rays)
    depth = rendered["t_hit"].astype(np.float32)
    world_normal = rendered["primitive_normals"].astype(np.float32)
    valid = np.isfinite(depth) & (depth > .2)
    depth[~valid] = 0
    face_id = rendered["primitive_ids"].astype(np.int64)
    face_id[~valid] = -1
    camera_normal = world_normal @ np.asarray(camera["c2w"])[:3, :3]
    camera_rays_direction = rays[..., 3:] @ np.asarray(camera["c2w"])[:3, :3]
    camera_normal *= np.where(np.sum(camera_normal * camera_rays_direction, axis=-1) > 0, -1., 1.)[..., None]
    camera_normal = unit(camera_normal).astype(np.float32)
    camera_normal[~valid] = 0
    low = minimum_filter(np.where(valid, depth, np.inf), size=3)
    high = maximum_filter(np.where(valid, depth, 0), size=3)
    incidence = np.abs(np.sum(world_normal * unit(rays[..., 3:]), axis=-1))
    full_stencil = minimum_filter(valid.astype(np.uint8), size=3, mode="constant", cval=0) == 1
    normal_valid = full_stencil & (low > .2) & np.isfinite(low) & ((high - low) <= np.maximum(scene_scale / 64, .02 * depth)) & (incidence >= .25)
    # 一像素 stencil 边界无 N，不删除对应的有效 D。
    normal_valid[[0, -1], :] = False
    normal_valid[:, [0, -1]] = False
    evidence = np.zeros(depth.shape, np.uint8)
    patch_id = np.full(depth.shape, -1, np.int32)
    points = rays[valid, :3] + depth[valid, None] * rays[valid, 3:]
    normals = world_normal[valid]
    associated_fraction = 0.
    if len(points):
        distance, nearest = tree.query(points, workers=scene.threads)
        delta = points - atlas["xyz"][nearest]
        normal_distance = np.abs(np.sum(delta * atlas["normal_world"][nearest], axis=1))
        compatible = ((distance <= scene_scale / 32)
                      & (np.abs(np.sum(normals * atlas["normal_world"][nearest], axis=1)) >= np.cos(np.radians(30)))
                      & (normal_distance <= np.maximum(scene_scale * .002, .002 * depth[valid])))
        ids = np.unique(nearest[compatible])
        visible_atlas = np.zeros(len(atlas["xyz"]), bool)
        if len(ids):
            center = np.asarray(camera["c2w"])[:3, 3]
            offsets = atlas["xyz"][ids] - center
            sample_rays = np.column_stack((np.broadcast_to(center, offsets.shape), unit(offsets)))
            first = scene.cast(sample_rays)
            visible = (np.isfinite(first["t_hit"])
                       & (np.abs(first["t_hit"] - np.linalg.norm(offsets, axis=1)) <= max(1e-6, scene_scale * 2e-5)))
            visible &= first["primitive_ids"] == atlas["face_id"][ids]
            visible_atlas[ids] = visible
        compatible &= visible_atlas[nearest]
        labels = np.zeros(len(points), np.uint8)
        labels[compatible] = atlas["evidence_class"][nearest[compatible]]
        evidence[valid] = labels
        associated_fraction = float(compatible.mean())
        keys, inverse = np.unique(region_keys(points, normals, scene_scale), axis=0, return_inverse=True)
        region_values = []
        for key in keys:
            identity = tuple(int(value) for value in key)
            if identity not in region_ids:
                region_ids[identity] = len(region_ids)
            region_values.append(region_ids[identity])
        patch_id[valid] = np.asarray(region_values, np.int32)[inverse]
    weight_depth = np.choose(evidence, [.25, 1., .1, 0.]).astype(np.float32) * valid
    weight_normal = np.choose(evidence, [.5, 1., .25, 0.]).astype(np.float32) * valid
    arrays = {"depth": depth, "normal": camera_normal, "valid_depth": valid, "valid_normal": normal_valid,
              "q": np.zeros(depth.shape, np.float32), "weight_depth": weight_depth,
              "weight_normal": weight_normal, "weight_coverage": weight_depth.copy(),
              "face_id": face_id, "patch_id": patch_id, "evidence_class": evidence}
    stats = {"depth_pixels": int(valid.sum()), "normal_pixels": int(normal_valid.sum()),
             "atlas_associated_fraction": associated_fraction,
             "evidence_pixels": {str(value): int((valid & (evidence == value)).sum()) for value in range(4)}}
    return arrays, stats


def exposure_budget(pixel_counts, patch_areas):
    """sqrt(area) 给区域预算，再分到该区域在整个固定池的所有像素。"""
    pixel_counts, patch_areas = np.asarray(pixel_counts), np.asarray(patch_areas)
    active = pixel_counts > 0
    result = np.zeros(len(pixel_counts), np.float64)
    if active.any():
        budgets = np.sqrt(patch_areas[active])
        median = np.median(budgets)
        budgets = np.clip(budgets, median * .25, median * 4)
        # 平均 q 约为 1，避免很小的数值尺度；归一化仍由全池固定 Z 决定。
        budgets *= pixel_counts.sum() / budgets.sum()
        result[active] = budgets / pixel_counts[active]
    return result.astype(np.float32)


def prepare_teacher(approved_manifest, output, dataset=None, long_edge=512, threads=8,
                    *, mesh_resources=None, preverified_sources=None, input_paths=None, prepared_arrays=None):
    """冻结 D/N 缓存；可复用同进程 mesh/来源证明，最终数组只压缩写盘一次。

    prepared_arrays 是可选空 dict，成功后填入 index -> 内部预载 bundle，供
    VirtualGeometryDataset(preloaded_arrays=...) 接管。公开默认路径仍完整校验。
    """
    approved_manifest, output = Path(approved_manifest).resolve(), Path(output).resolve()
    if prepared_arrays is not None and (not isinstance(prepared_arrays, dict) or prepared_arrays):
        raise ValueError("prepared_arrays must be an empty dictionary")
    preverified_sources = {str(Path(path).resolve()): value for path, value in (preverified_sources or {}).items()}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty teacher directory: {output}")
    approved = json.loads(approved_manifest.read_text())
    if approved.get("schema_version") != 2 or approved.get("status") != "complete":
        raise ValueError("A completed approved camera schema-2 manifest is required")
    root = approved_manifest.parent
    dataset = Path(dataset or approved["dataset"]).resolve()
    if str(dataset) != str(Path(approved["dataset"]).resolve()):
        raise ValueError("Teacher dataset must equal the approved pose dataset")
    source_pose_path = root / approved["virtual_cameras_path"]
    source_real_path = root / approved["real_cameras_path"]
    source_atlas_path = root / approved["atlas_path"]
    approved_receipt = json.loads((root / "receipt.json").read_text())["files"]
    for path in (approved_manifest, source_pose_path, source_real_path, source_atlas_path,
                 root / approved["source_fingerprints_path"]):
        relative = str(path.relative_to(root))
        if approved_receipt.get(relative) != sha256(path):
            raise ValueError(f"Approved camera receipt does not match {relative}")
    views = json.loads(source_pose_path.read_text())["cameras"]
    real = json.loads(source_real_path.read_text())["cameras"]
    if not views or len({row["id"] for row in views}) != len(views):
        raise ValueError("Approved virtual camera IDs must be nonempty and unique")
    mesh_path = Path(approved["genrecon_mesh"])
    paths = input_paths if input_paths is not None else approved.get("input_paths", {})
    image_root = Path(paths.get("images", dataset / "images"))
    normal_root = Path(paths.get("normals", dataset / "normals"))
    if not image_root.is_absolute() or not normal_root.is_absolute():
        raise ValueError("Resolved teacher image/normal input paths must be absolute")
    if mesh_resources is not None and Path(mesh_resources.mesh_path).resolve() != mesh_path.resolve():
        raise ValueError("Shared mesh resources differ from the approved teacher mesh")
    previous = json.loads((root / approved["source_fingerprints_path"]).read_text())
    approved_producers = {path: digest for path, digest in previous.items() if Path(path).suffix in (".py", ".sh", ".cu", ".cpp", ".h", ".so")}
    sources = {path: digest for path, digest in previous.items() if path not in approved_producers}
    for path, expected in sources.items():
        if _source_fingerprint(path, preverified_sources) != expected:
            raise ValueError(f"Approved source changed: {path}")
    for path in (approved_manifest, source_pose_path, source_real_path, source_atlas_path,
                 root / approved["source_fingerprints_path"]):
        sources[str(path)] = sha256(path)
    producer_paths = (Path(__file__).resolve(), Path(__file__).resolve().with_name("virtual_camera_selection.py"),
                      Path(__file__).resolve().parents[1] / "scripts/prepare_virtual_geometry_teacher.py")
    producers = {str(path): sha256(path) for path in producer_paths}
    for camera in real:
        name = Path(camera["name"])
        for path in (image_root / name, normal_root / name.with_suffix(".png")):
            sources[str(path)] = _source_fingerprint(path, preverified_sources)
    output.mkdir(parents=True, exist_ok=True)
    (output / "views").mkdir()
    write_json(output / "state.json", {"status": "running", "approved_manifest_sha256": sources[str(approved_manifest)]})
    write_json(output / "real_cameras.json", {"schema_version": 3, "cameras": real})
    shutil.copyfile(source_atlas_path, output / "source_atlas.npz")
    atlas = dict(np.load(source_atlas_path, allow_pickle=False))
    scale = float(approved["scene_scale"])
    keys = region_keys(atlas["xyz"], atlas["normal_world"], scale)
    unique_keys, atlas_regions = np.unique(keys, axis=0, return_inverse=True)
    region_ids = {tuple(int(value) for value in key): index for index, key in enumerate(unique_keys)}
    patch_areas = np.bincount(atlas_regions, weights=atlas["area_weight"])
    fallback_area = float(np.median(atlas["area_weight"]))
    tree = cKDTree(atlas["xyz"])
    if mesh_resources is None:
        from plyfile import PlyData
        mesh = PlyData.read(mesh_path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
        vertex = mesh["vertex"].data
        vertices = np.column_stack([vertex[key] for key in ("x", "y", "z")]).astype(np.float32)
        log("teacher_exact_mesh_load", vertices=len(vertices), faces=len(mesh["face"]))
        scene = MeshScene(vertices, mesh["face"].data["vertex_indices"], threads)
        del vertices, mesh
    else:
        log("teacher_exact_mesh_reused", vertices=len(mesh_resources.vertices), faces=len(mesh_resources.faces))
        scene = mesh_resources.scene
    pixel_counts = np.zeros(len(region_ids), np.int64)
    records, rendered_arrays = [], []
    for index, approved_view in enumerate(views):
        camera = resized_camera(approved_view, long_edge)
        validate_camera(camera)
        arrays, statistics = teacher_arrays(scene, camera, atlas, tree, region_ids, scale)
        if not arrays["valid_depth"].any():
            raise ValueError(f"Approved camera has no usable teacher depth: {camera['id']}")
        pixel_counts = np.pad(pixel_counts, (0, len(region_ids) - len(pixel_counts)))
        pixel_counts += np.bincount(arrays["patch_id"][arrays["valid_depth"]], minlength=len(pixel_counts))
        relative = f"views/{index:04d}.npz"
        rendered_arrays.append(arrays)
        records.append({"id": camera["id"], "virtual_index": index, "width": camera["width"], "height": camera["height"],
                        "K": camera["K"], "c2w": camera["c2w"], "arrays_path": relative,
                        "selection_source": camera["selection_source"], "anchor_id": camera["anchor_id"], "stats": statistics})
        if (index + 1) % 10 == 0 or index + 1 == len(views):
            log("teacher_render", done=index + 1, total=len(views))
    patch_areas = np.pad(patch_areas, (0, len(region_ids) - len(patch_areas)), constant_values=fallback_area)
    per_pixel = exposure_budget(pixel_counts, patch_areas)
    normalization = {"depth": 0., "normal": 0., "coverage": 0.}
    for index, record in enumerate(records):
        path = output / record["arrays_path"]
        arrays = rendered_arrays[index]
        valid = arrays["valid_depth"]
        arrays["q"][valid] = per_pixel[arrays["patch_id"][valid]]
        validate_arrays(arrays, record)
        normalization["depth"] += float(arrays["q"][valid].sum(dtype=np.float64)) / len(records)
        normalization["coverage"] += float(arrays["q"][valid].sum(dtype=np.float64)) / len(records)
        normalization["normal"] += float(arrays["q"][arrays["valid_normal"]].sum(dtype=np.float64)) / len(records)
        np.savez_compressed(path, **arrays)
        record["arrays_sha256"] = sha256(path)
        if (index + 1) % 20 == 0:
            log("teacher_fixed_pool_budget", done=index + 1, total=len(records))
    # seed 仍采用原 mesh 的精确表面样本，但 patch 身份与训练缓存保持一致。
    atlas["patch_id"] = atlas_regions.astype(np.int32)
    np.savez_compressed(output / "atlas.npz", **atlas)
    np.savez_compressed(output / "patch_budget.npz", pixel_counts=pixel_counts, area_estimate=patch_areas, q_per_pixel=per_pixel)
    for path, expected in {**sources, **producers}.items():
        if sha256(path) != expected:
            raise ValueError(f"Source changed while preparing teacher: {path}")
    manifest = {"schema_version": 3, "status": "complete", "dataset": str(dataset), "genrecon_mesh": str(mesh_path),
                "approved_manifest_path": str(approved_manifest), "approved_manifest_sha256": sources[str(approved_manifest)],
                "approved_camera_sha256": sources[str(source_pose_path)], "scene_scale": scale, "view_count": len(records),
                "depth_convention": "camera_z", "pixel_center_convention": "colmap_half_integer",
                "normal_convention": "camera_space_unit_face_normal_toward_camera", "units": "scene", "renderer_near": .2,
                "normalizers": normalization, "views": records, "real_cameras_path": "real_cameras.json",
                "real_cameras_sha256": sha256(output / "real_cameras.json"), "atlas_path": "atlas.npz",
                "atlas_sha256": sha256(output / "atlas.npz"), "source_files_sha256": sources,
                "producer_files_sha256": producers, "approved_producer_files_sha256": approved_producers,
                "budget_policy": "bounded sqrt sampled-region area divided by all pool pixels; fixed per-loss mean-pool Z; reliability only multiplies numerator",
                "region_method": "spatial/unsigned-normal bins; not full mesh connectivity; newly visible unsampled regions use one area-sample fallback",
                "evidence_transfer": "nearby compatible normal/normal-offset plus exact first-hit atlas witness; unmatched remains unknown",
                "virtual_rgb": False, "teacher_geometry_transform": "identity"}
    write_json(output / "manifest.json", manifest)
    write_json(output / "receipt.json", {"manifest_sha256": sha256(output / "manifest.json"),
                                         "files": {str(path.relative_to(output)): sha256(path) for path in output.rglob("*.npz")}})
    write_json(output / "state.json", {"status": "complete", "manifest_sha256": sha256(output / "manifest.json")})
    if prepared_arrays is not None:
        prepared_arrays.update({index: {"arrays": arrays, "arrays_sha256": records[index]["arrays_sha256"],
                                        "content_sha256": _arrays_fingerprint(arrays)}
                               for index, arrays in enumerate(rendered_arrays)})
    log("teacher_complete", output=str(output), views=len(records), normalizers=normalization)
    return output / "manifest.json"


class ShuffledViewSampler:
    def __init__(self, count, seed=0):
        if count < 1:
            raise ValueError("Sampler requires at least one view")
        self.count, self.rng = int(count), np.random.default_rng(seed)
        self.remaining, self.draws = [], 0

    def next(self):
        if not self.remaining:
            self.remaining = self.rng.permutation(self.count).tolist()
        self.draws += 1
        return self.remaining.pop()

    def state_dict(self):
        return {"count": self.count, "remaining": list(self.remaining), "draws": self.draws,
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        remaining = state["remaining"]
        if state["count"] != self.count or len(set(remaining)) != len(remaining) or any(not 0 <= value < self.count for value in remaining):
            raise ValueError("Sampler state does not belong to this view pool")
        self.remaining, self.draws = list(remaining), int(state["draws"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class VirtualGeometryDataset:
    def __init__(self, manifest_path, device="cuda", gpu_cache_size=2, cpu_cache_size=16, verify=True, seed=0,
                 *, preloaded_arrays=None):
        if preloaded_arrays is not None and (not verify or not isinstance(preloaded_arrays, dict)):
            raise ValueError("Preloaded teacher arrays require normal on-disk verification and a dictionary")
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        document = self.manifest
        if document.get("schema_version") != 3 or document.get("status") != "complete" or document.get("virtual_rgb") is not False:
            raise ValueError("Completed schema-3 geometry-only teacher required")
        self.cameras = document["views"]
        if not self.cameras or len(self.cameras) != document["view_count"] or len({row["id"] for row in self.cameras}) != len(self.cameras):
            raise ValueError("Teacher view identity mismatch")
        if (document["depth_convention"] != "camera_z" or document["pixel_center_convention"] != "colmap_half_integer"
                or document["normal_convention"] != "camera_space_unit_face_normal_toward_camera"
                or document.get("renderer_near") != .2 or document.get("teacher_geometry_transform") != "identity"):
            raise ValueError("Unsupported teacher coordinate convention")
        self.scene_scale = float(document["scene_scale"])
        self.normalizers = dict(document["normalizers"])
        if (set(self.normalizers) != {"depth", "normal", "coverage"}
                or not all(np.isfinite(value) and value >= 0 for value in self.normalizers.values())
                or self.normalizers["depth"] <= 0 or self.normalizers["coverage"] <= 0):
            raise ValueError("Invalid fixed-pool normalization")
        self.view_ids = [row["id"] for row in self.cameras]
        self.identity = {"manifest_sha256": sha256(self.path), "approved_camera_sha256": document["approved_camera_sha256"],
                         "approved_manifest_sha256": document["approved_manifest_sha256"], "view_ids": self.view_ids}
        self.source_identity = {"dataset": document["dataset"],
                                "source_files_sha256": copy.deepcopy(document["source_files_sha256"]),
                                "approved_camera_sha256": document["approved_camera_sha256"],
                                "approved_manifest_sha256": document["approved_manifest_sha256"]}
        self.device, self.gpu_cache_size, self.cpu_cache_size = str(device), int(gpu_cache_size), int(cpu_cache_size)
        if min(self.gpu_cache_size, self.cpu_cache_size) < 1:
            raise ValueError("Virtual device and CPU caches must be bounded positive counts")
        self._cpu, self._device = OrderedDict(), OrderedDict()
        self.sampler = ShuffledViewSampler(len(self.cameras), seed)
        if verify:
            for path, expected in document["source_files_sha256"].items():
                if sha256(path) != expected:
                    raise ValueError(f"Teacher source changed: {path}")
            for key in ("real_cameras", "atlas"):
                if sha256(self.root / document[key + "_path"]) != document[key + "_sha256"]:
                    raise ValueError(f"Teacher {key} cache changed")
        self.real_cameras = json.loads((self.root / document["real_cameras_path"]).read_text())["cameras"]
        self.atlas = dict(np.load(self.root / document["atlas_path"], allow_pickle=False))
        self._real_depths = None
        for record in self.cameras:
            validate_camera(record)
            if verify and sha256(self.root / record["arrays_path"]) != record["arrays_sha256"]:
                raise ValueError(f"Teacher view cache changed: {record['id']}")
        if preloaded_arrays is not None:
            for index, bundle in preloaded_arrays.items():
                if not isinstance(index, int) or not 0 <= index < len(self.cameras):
                    raise ValueError("Preloaded teacher view index is outside this pool")
                if not isinstance(bundle, dict) or set(bundle) != {"arrays", "arrays_sha256", "content_sha256"}:
                    raise ValueError("Invalid preloaded teacher array bundle")
                record, arrays = self.cameras[index], bundle["arrays"]
                if bundle["arrays_sha256"] != record["arrays_sha256"]:
                    raise ValueError("Preloaded teacher arrays belong to another cache")
                if not isinstance(arrays, dict):
                    raise ValueError("Preloaded teacher arrays must be a dictionary")
                validate_arrays(arrays, record)
                if _arrays_fingerprint(arrays) != bundle["content_sha256"]:
                    raise ValueError("Preloaded teacher arrays changed after preparation")
                self._cpu[index] = arrays
                while len(self._cpu) > self.cpu_cache_size:
                    self._cpu.popitem(last=False)
        self._virtual_camera_class = None

    def __len__(self):
        return len(self.cameras)

    def validate_real_cameras(self, cameras, dataset_path=None):
        """拒绝 eval/排除帧后仍使用全真实集合生成的 teacher。"""
        if dataset_path is not None and Path(dataset_path).resolve() != Path(self.manifest["dataset"]).resolve():
            raise ValueError("Real dataset path differs from frozen teacher")
        names = []
        for camera in cameras:
            if isinstance(camera, dict):
                name = camera.get("source_image_name", camera.get("name"))
            else:
                name = getattr(camera, "source_image_name", None)
            if not name:
                raise ValueError("Real camera must expose full source_image_name")
            names.append(str(name))
        expected = [camera["name"] for camera in self.real_cameras]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("Real camera split differs from frozen teacher; rebuild teacher for this split")
        return True

    def _arrays(self, index):
        if index in self._cpu:
            self._cpu.move_to_end(index)
            return self._cpu[index]
        record = self.cameras[index]
        path = self.root / record["arrays_path"]
        if sha256(path) != record["arrays_sha256"]:
            raise ValueError(f"Teacher cache changed after validation: {record['id']}")
        arrays = dict(np.load(path, allow_pickle=False))
        validate_arrays(arrays, record)
        self._cpu[index] = arrays
        while len(self._cpu) > self.cpu_cache_size:
            self._cpu.popitem(last=False)
        return arrays

    def view(self, index, device=None):
        import torch
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        device = str(device or self.device)
        key = (index, device)
        if key in self._device:
            self._device.move_to_end(key)
            return self._device[key]
        if self._virtual_camera_class is None:
            path = Path(__file__).resolve().parents[1] / "scene/virtual_cameras.py"
            spec = importlib.util.spec_from_file_location("virtual_geometry_camera", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._virtual_camera_class = module.VirtualCamera
        teacher = {}
        for name, array in self._arrays(index).items():
            if name == "normal":
                array = np.moveaxis(array, -1, 0)
            elif name in SCALAR_IMAGES:
                array = array[None]
            teacher[name] = torch.from_numpy(np.ascontiguousarray(array))
        camera = self._virtual_camera_class(self.cameras[index], teacher, self.normalizers, device, index)
        self._device[key] = camera
        while len(self._device) > self.gpu_cache_size:
            self._device.popitem(last=False)
        return camera

    get = view
    get_view = view

    def next_view(self):
        index = self.sampler.next()
        return index, self.view(index)

    def state_dict(self):
        return {"identity": copy.deepcopy(self.identity), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Virtual checkpoint belongs to another immutable teacher")
        self.sampler.load_state_dict(state["sampler"])

    @property
    def real_depths(self):
        if self._real_depths is None:
            self._real_depths = []
            for camera in self.real_cameras:
                raw = np.load(camera["depth_path"], mmap_mode="r", allow_pickle=False)
                y = np.minimum(((np.arange(camera["height"]) + .5) * raw.shape[0] / camera["height"]).astype(int), raw.shape[0] - 1)
                x = np.minimum(((np.arange(camera["width"]) + .5) * raw.shape[1] / camera["width"]).astype(int), raw.shape[1] - 1)
                self._real_depths.append(np.asarray(raw[np.ix_(y, x)], np.float32))
        return self._real_depths

    def real_evidence(self, points):
        groups = camera_position_groups(self.real_cameras, .02 * self.scene_scale)
        support, conflict = np.zeros(len(points), np.uint16), np.zeros(len(points), np.uint16)
        for group in np.unique(groups):
            surface, free = np.zeros(len(points), bool), np.zeros(len(points), bool)
            for index in np.flatnonzero(groups == group):
                ev = depth_evidence(points, self.real_cameras[index], self.real_depths[index], self.scene_scale)
                surface |= ev["support"]
                free |= ev["free"]
            support += surface
            conflict += free & ~surface
        return {"support_groups": support, "conflict_groups": conflict, "evidence_class": classify_evidence(support, conflict)}
