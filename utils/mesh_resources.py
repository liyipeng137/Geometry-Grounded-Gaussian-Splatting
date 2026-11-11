"""共享完整 CPU mesh、面积 CDF 和 raycast 场；每次采样使用独立 RNG。"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _log(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def _unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _area_distribution(vertices, faces, block_size=250000):
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Nonempty triangle mesh required")
    cdf = np.empty(len(faces), np.float64)
    for start in range(0, len(faces), block_size):
        ids = faces[start:start + block_size]
        if (ids < 0).any() or (ids >= len(vertices)).any():
            raise ValueError("Invalid mesh face index")
        triangle = vertices[ids]
        cdf[start:start + len(ids)] = .5 * np.linalg.norm(np.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
        if start % (20 * block_size) == 0:
            _log("shared_mesh_area_scan", done=start, total=len(faces))
    if not np.isfinite(cdf).all() or cdf.sum() <= 0:
        raise ValueError("Mesh has nonfinite or zero total surface area")
    # 旧 atlas 用 pairwise sum，旧 init 用顺序 cumsum 末值；保留各自舍入习惯。
    totals = {"atlas": float(cdf.sum()), "degenerate_faces": int((cdf == 0).sum())}
    np.cumsum(cdf, out=cdf)
    totals["initialization"] = float(cdf[-1])
    return cdf, totals


def _sample(vertices, faces, cdf, totals, count, seed, kind):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Surface sample count must be a positive integer")
    if kind not in ("atlas", "initialization"):
        raise ValueError("Sample kind must be atlas or initialization")
    total = totals[kind]
    rng = np.random.default_rng(seed)
    ids = np.searchsorted(cdf, rng.random(count) * total, side="left" if kind == "atlas" else "right")
    triangle = vertices[faces[ids]]
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    xyz = (1 - u[:, None]) * a + (u * (1 - v))[:, None] * b + (u * v)[:, None] * c
    normals = _unit(np.cross(b - a, c - a))
    # atlas 的旧分区在 float64 中间坐标上计算，不能在 floor/round 前提前截断。
    if kind == "initialization":
        xyz, normals = xyz.astype(np.float32), normals.astype(np.float32)
    return xyz, normals, ids.astype(np.int64), {
        "total_area": total, "degenerate_faces": totals["degenerate_faces"],
        "sampled_distinct_faces": int(len(np.unique(ids))),
        "method": "all-triangle exact area CDF and uniform barycentric sampling; no visibility selection",
    }


def sample_surface_from_arrays(vertices, faces, count, seed, kind="initialization", block_size=250000):
    cdf, totals = _area_distribution(vertices, faces, block_size)
    return _sample(vertices, faces, cdf, totals, count, seed, kind)


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


class MeshResources:
    def __init__(self, mesh_path, threads=8):
        self.mesh_path = Path(mesh_path).expanduser().resolve()
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("Mesh threads must be positive")
        self._ply = self._vertices = self._faces = self._scene = self._cdf = self._totals = None
        self._mesh_sha256 = None

    @property
    def mesh_sha256(self):
        if self._mesh_sha256 is None:
            digest = hashlib.sha256()
            with self.mesh_path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            self._mesh_sha256 = digest.hexdigest()
        return self._mesh_sha256

    def _load(self):
        if self._vertices is not None:
            return
        from plyfile import PlyData
        # 在第一次加载前冻结身份，后续调用者可与各自输入 SHA 对照。
        self.mesh_sha256
        self._ply = PlyData.read(self.mesh_path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
        records = self._ply["vertex"].data
        self._vertices = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(np.float32)
        self._faces = self._ply["face"].data["vertex_indices"]
        if (self._faces.ndim != 2 or self._faces.shape[1] != 3 or not len(self._faces)
                or not np.isfinite(self._vertices).all() or self._faces.min() < 0 or self._faces.max() >= len(self._vertices)):
            raise ValueError("Invalid complete triangle mesh")
        _log("shared_mesh_loaded", vertices=len(self._vertices), triangles=len(self._faces))

    @property
    def vertices(self):
        self._load()
        return self._vertices

    @property
    def faces(self):
        self._load()
        return self._faces

    @property
    def area_cdf(self):
        if self._cdf is None:
            self._cdf, self._totals = _area_distribution(self.vertices, self.faces)
        return self._cdf

    @property
    def scene(self):
        if self._scene is None:
            self._scene = MeshScene(self.vertices, self.faces, self.threads)
        return self._scene

    def sample_surface(self, count, seed, kind="initialization"):
        cdf = self.area_cdf
        return _sample(self.vertices, self.faces, cdf, self._totals, count, seed, kind)
