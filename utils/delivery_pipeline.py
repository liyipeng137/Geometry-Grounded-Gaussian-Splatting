"""Delivery orchestration: versioned caches and one resolved production recipe."""

import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import sys
import subprocess
import traceback
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np
import yaml

from utils.virtual_camera_selection import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _merge(base, update, prefix=""):
    if not isinstance(update, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        default = base[key]
        if isinstance(default, dict):
            _merge(default, value, prefix + key + ".")
        else:
            valid = (
                isinstance(value, bool)
                if isinstance(default, bool)
                else isinstance(value, int) and not isinstance(value, bool)
                if isinstance(default, int)
                else isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                if isinstance(default, float)
                else isinstance(value, type(default))
            )
            if not valid:
                raise ValueError(f"Invalid configuration type: {prefix}{key}")
            base[key] = value
    return base


def load_config(path=None):
    default = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg = (
        _merge(deepcopy(default), yaml.safe_load(Path(path).read_text()))
        if path
        else default
    )
    if cfg["version"] != 1 or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("Unsupported config version or seed")
    init, views, train, geo, out = (
        cfg[k]
        for k in ("initialization", "virtual_views", "training", "geometry", "outputs")
    )
    if (
        init["method"] not in ("gen_mesh", "tsdf")
        or not 0 < init["num_points"] <= geo["max_gaussians"]
    ):
        raise ValueError(
            "Initialization method/count is invalid or exceeds max_gaussians"
        )
    if (
        not 0 < views["trajectory_offset_fraction"] <= 1
        or not 0 < views["max_count"] <= views["candidate_limit"]
    ):
        raise ValueError("Invalid virtual view count or offset")
    if (
        min(views["selection_long_edge"], views["teacher_long_edge"]) < 32
        or views["atlas_points"] < 100
    ):
        raise ValueError("Invalid view resolution/atlas size")
    if train["appearance_mode"] not in (0, 1, 2, 3, 4) or train["iterations"] < 1:
        raise ValueError("Invalid training budget or appearance mode")
    if not train["dense_optimizer_updates"]:
        raise ValueError(
            "Delivery joint training requires dense_optimizer_updates=true"
        )
    if (
        not 0
        <= geo["last_opacity_reset"]
        < geo["start_iteration"]
        <= geo["ramp_end_iteration"]
        < geo["topology_until"]
        <= train["iterations"]
    ):
        raise ValueError("Virtual geometry schedule must fit the training budget")
    if (
        not geo["start_iteration"]
        <= geo["seed_from"]
        <= geo["seed_until"]
        < geo["topology_until"]
    ):
        raise ValueError("Invalid seed schedule")
    for name, value in geo.items():
        if isinstance(value, int) and name != "last_opacity_reset" and value < 1:
            raise ValueError(f"geometry.{name} must be positive")
        if isinstance(value, float) and not 0 < value < 1:
            raise ValueError(f"geometry.{name} must lie in (0,1)")
    if geo["depth_search_expansions"] not in (1, 2):
        raise ValueError("depth_search_expansions must be 1 or 2")
    for name, value in train.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value < 0
        ):
            raise ValueError(f"training.{name} must be nonnegative")
    if not 0 <= train["lambda_dssim"] <= 1:
        raise ValueError("lambda_dssim must lie in [0,1]")
    if any(v < 0 for v in geo["loss_weights"].values()):
        raise ValueError("Loss weights must be nonnegative")
    if (
        cfg["runtime"]["cpu_threads"] < 1
        or cfg["runtime"]["real_gpu_cache"] < 1
        or cfg["runtime"]["rgb_cache_gb"] <= 0
    ):
        raise ValueError("Runtime cache/thread budgets must be positive")
    if out["checkpoint_interval"] < 0 or out["log_interval"] < 1:
        raise ValueError("Invalid checkpoint/log intervals")
    for key in ("evaluation_iterations", "save_iterations"):
        if any(
            isinstance(v, bool)
            or not isinstance(v, int)
            or not 0 < v <= train["iterations"]
            for v in out[key]
        ):
            raise ValueError(f"Invalid {key}")
    return cfg


def read_inputs(dataset, cfg):
    dataset = Path(dataset).expanduser().resolve()
    paths = {k: (dataset / Path(v)).resolve() for k, v in cfg["inputs"].items()}
    if not paths["mesh"].is_file() or any(
        not paths[k].is_dir() for k in ("images", "depth", "normals", "colmap")
    ):
        raise ValueError(
            "Dataset requires images, depth, normals, COLMAP model and genrecon mesh"
        )
    spec = importlib.util.spec_from_file_location(
        "delivery_colmap", ROOT / "scene/colmap_loader.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    ext = (
        "bin"
        if (paths["colmap"] / "cameras.bin").is_file()
        and (paths["colmap"] / "images.bin").is_file()
        else "txt"
    )
    camera_file, image_file = (
        paths["colmap"] / f"{name}.{ext}" for name in ("cameras", "images")
    )
    intrinsics = (
        loader.read_intrinsics_binary if ext == "bin" else loader.read_intrinsics_text
    )(camera_file)
    extrinsics = (
        loader.read_extrinsics_binary if ext == "bin" else loader.read_extrinsics_text
    )(image_file)
    groups = {k: {} for k in ("mesh", "calibration", "images", "depth", "normals")}
    groups["mesh"][str(paths["mesh"])] = sha256(paths["mesh"])
    for path in (camera_file, image_file):
        groups["calibration"][str(path)] = sha256(path)
    from PIL import Image

    cameras = []
    for image in sorted(extrinsics.values(), key=lambda x: x.name):
        name = image.name
        if (
            Path(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or "\\" in name
        ):
            raise ValueError(f"Unsafe COLMAP name: {name}")
        intr = intrinsics[image.camera_id]
        if intr.model == "SIMPLE_PINHOLE":
            fx, cx, cy = intr.params
            fy = fx
        elif intr.model == "PINHOLE":
            fx, fy, cx, cy = intr.params
        else:
            raise ValueError(
                "Delivery requires undistorted PINHOLE/SIMPLE_PINHOLE cameras"
            )
        rgb = paths["images"] / name
        depth = paths["depth"] / Path(name).with_suffix(".npy")
        normal = paths["normals"] / Path(name).with_suffix(".png")
        for group, path in (("images", rgb), ("depth", depth), ("normals", normal)):
            groups[group][str(path)] = sha256(path)
        with Image.open(rgb) as im:
            if im.size != (intr.width, intr.height):
                raise ValueError(f"Image/COLMAP dimensions differ: {name}")
        d = np.load(depth, mmap_mode="r", allow_pickle=False)
        if d.shape != (intr.height, intr.width):
            raise ValueError(f"Depth/COLMAP dimensions differ: {name}")
        with Image.open(normal) as im:
            if im.size != (intr.width, intr.height):
                raise ValueError(f"Normal/COLMAP dimensions differ: {name}")
        q = np.asarray(image.qvec)
        qnorm = np.linalg.norm(q)
        if not np.isfinite(q).all() or not np.isfinite(qnorm) or qnorm <= 0:
            raise ValueError(f"Invalid COLMAP rotation: {name}")
        rot = loader.qvec2rotmat(q / qnorm).T
        cameras.append(
            SimpleNamespace(
                uid=intr.id,
                R=rot,
                T=np.asarray(image.tvec),
                K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                width=intr.width,
                height=intr.height,
                colmap_image_name=name,
                image_name=name,
                image_path=str(rgb),
                image=None,
                mask=None,
                FovX=2 * np.arctan(intr.width / (2 * fx)),
                FovY=2 * np.arctan(intr.height / (2 * fy)),
            )
        )
    if not cameras:
        raise ValueError("No registered cameras")
    return paths, cameras, intrinsics, [camera_file, image_file], groups


def cache_keys(cfg, paths, groups):
    # Separate identities: changing initialization N never invalidates cameras.
    views = {k: v for k, v in cfg["virtual_views"].items() if k != "teacher_long_edge"}
    camera = fingerprint(
        {
            "recipe": "delivery_camera_v1",
            "sources": {k: groups[k] for k in ("mesh", "calibration", "depth")},
            "paths": {k: str(v) for k, v in paths.items()},
            "views": views,
            "previews": cfg["outputs"]["camera_preview"],
            "seed": cfg["seed"],
        }
    )
    # Legacy schema3 source proof also binds photos/normals. Keep that strong contract.
    teacher = fingerprint(
        {
            "recipe": "teacher_schema3_v1",
            "cameras": camera,
            "rgb": groups["images"],
            "normals": groups["normals"],
            "long_edge": cfg["virtual_views"]["teacher_long_edge"],
        }
    )
    needed = (
        ("mesh", "calibration", "images")
        if cfg["initialization"]["method"] == "gen_mesh"
        else ("calibration", "images", "depth")
    )
    initial = fingerprint(
        {
            "recipe": "delivery_init_v1",
            "source": {k: groups[k] for k in needed},
            "initialization": cfg["initialization"],
            "seed": cfg["seed"],
        }
    )
    return {"cameras": camera, "teacher": teacher, "initialization": initial}


def _ready(path, key, only_files=None):
    if not path.exists():
        return False
    receipt = path / "delivery_cache.json"
    if not receipt.is_file():
        raise ValueError(
            f"Incomplete cache preserved: {path}; move it aside before retrying"
        )
    record = json.loads(receipt.read_text())
    if record.get("status") != "complete" or record.get("key") != key:
        raise ValueError(f"Cache identity mismatch: {path}")
    selected = (
        record["files"]
        if only_files is None
        else {name: record["files"][name] for name in only_files}
    )
    for name, digest in selected.items():
        if sha256(path / name) != digest:
            raise ValueError(f"Cache file changed: {path / name}")
    return True


def _publish(path, key):
    files = {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "delivery_cache.json"
    }
    write_json(
        path / "delivery_cache.json", {"status": "complete", "key": key, "files": files}
    )


def prepare(dataset, cfg, workspace, summary):
    from utils.mesh_resources import MeshResources
    from utils.virtual_camera_selection import run_selection
    from utils.virtual_geometry_data import prepare_teacher, VirtualGeometryDataset
    from utils.mesh_initialization import prepare_from_cameras

    input_started = time.monotonic()
    paths, cameras, intrinsics, calib, groups = read_inputs(dataset, cfg)
    summary.setdefault("stage_seconds", {})["inputs"] = time.monotonic() - input_started
    keys = cache_keys(cfg, paths, groups)
    sources = {p: h for values in groups.values() for p, h in values.items()}
    cache = workspace / "cache"
    dirs = {name: cache / name / key[:20] for name, key in keys.items()}
    if not cfg["cache"]["reuse"]:
        dirs = {
            name: cache / name / (key[:20] + "_" + uuid.uuid4().hex[:8])
            for name, key in keys.items()
        }
    hits = {
        name: _ready(
            dirs[name], keys[name], ["manifest.json"] if name == "teacher" else None
        )
        for name in dirs
    }
    summary.update(
        input_fingerprint=fingerprint(groups),
        cache_keys=keys,
        cache_hits=hits,
        cache_paths={k: str(v) for k, v in dirs.items()},
    )
    resources = None

    def mesh():
        nonlocal resources
        if resources is None:
            resources = MeshResources(
                paths["mesh"], threads=cfg["runtime"]["cpu_threads"]
            )
        return resources

    def timed(name, function):
        start = time.monotonic()
        result = function()
        summary.setdefault("stage_seconds", {})[name] = time.monotonic() - start
        print(f"[Pipeline] {name}: {summary['stage_seconds'][name]:.2f}s", flush=True)
        return result

    v = cfg["virtual_views"]
    if not hits["cameras"]:
        timed(
            "camera_selection",
            lambda: run_selection(
                dataset,
                paths["mesh"],
                dirs["cameras"],
                target_count=v["max_count"],
                candidate_limit=v["candidate_limit"],
                long_edge=v["selection_long_edge"],
                surface_samples=v["atlas_points"],
                threads=cfg["runtime"]["cpu_threads"],
                seed=cfg["seed"],
                trajectory_offset_fraction=v["trajectory_offset_fraction"],
                mesh_resources=mesh(),
                export_previews=cfg["outputs"]["camera_preview"],
                detailed_audit=False,
                input_paths=paths,
            ),
        )
        if cfg["outputs"]["camera_preview"]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/export_virtual_camera_preview.py"),
                    "--manifest",
                    str(dirs["cameras"] / "manifest.json"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            result.check_returncode()
        _publish(dirs["cameras"], keys["cameras"])
    prepared_arrays = {}
    if not hits["teacher"]:
        timed(
            "teacher",
            lambda: prepare_teacher(
                dirs["cameras"] / "manifest.json",
                dirs["teacher"],
                dataset=dataset,
                long_edge=v["teacher_long_edge"],
                threads=cfg["runtime"]["cpu_threads"],
                mesh_resources=mesh(),
                preverified_sources=sources,
                prepared_arrays=prepared_arrays,
                input_paths=paths,
            ),
        )
        _publish(dirs["teacher"], keys["teacher"])
    if not hits["initialization"]:
        d = dirs["initialization"]
        if cfg["initialization"]["method"] == "gen_mesh":
            timed(
                "initialization",
                lambda: prepare_from_cameras(
                    dataset,
                    paths["mesh"],
                    d,
                    cameras,
                    seed=cfg["seed"],
                    threads=cfg["runtime"]["cpu_threads"],
                    num_points=cfg["initialization"]["num_points"],
                    mesh_resources=mesh(),
                    rgb_cache_bytes=int(cfg["runtime"]["rgb_cache_gb"] * 1024**3),
                    calibration_files=calib,
                ),
            )
        else:
            from utils.pd_utils import generate_ply_from_rgbd

            d.mkdir(parents=True)
            previous = np.random.get_state()
            np.random.seed(cfg["seed"])
            try:
                timed(
                    "initialization",
                    lambda: generate_ply_from_rgbd(
                        cameras,
                        cfg["initialization"]["num_points"],
                        str(d / "point_cloud.ply"),
                        cam_intrinsics=intrinsics,
                        source_path=str(dataset),
                        depth_prior_dir=str(paths["depth"]),
                        pixel_center_convention="colmap_half_integer",
                    ),
                )
            finally:
                np.random.set_state(previous)
            write_json(
                d / "metadata.json",
                {
                    "method": "tsdf",
                    "requested_points": cfg["initialization"]["num_points"],
                    "seed": cfg["seed"],
                },
            )
        _publish(d, keys["initialization"])
    resources = None
    gc.collect()
    teacher = timed(
        "teacher_load",
        lambda: VirtualGeometryDataset(
            dirs["teacher"] / "manifest.json",
            device="cuda",
            gpu_cache_size=cfg["geometry"]["gpu_cache_size"],
            cpu_cache_size=cfg["geometry"]["cpu_cache_size"],
            seed=cfg["seed"],
            preloaded_arrays=prepared_arrays or None,
        ),
    )
    prepared_arrays.clear()
    from plyfile import PlyData

    count = len(PlyData.read(dirs["initialization"] / "point_cloud.ply")["vertex"].data)
    summary["initialization"] = {
        "method": cfg["initialization"]["method"],
        "requested_points": cfg["initialization"]["num_points"],
        "actual_points": count,
    }
    return paths, dirs, teacher


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, value):
        self.logfile.write(value)
        return self.console.write(value)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def run(dataset, config=None, output=None, resume=None):
    cfg = load_config(config)
    os.environ["OMP_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
    if resume and not cfg["cache"]["reuse"]:
        raise ValueError(
            "Resume requires cache.reuse=true to preserve the fixed teacher identity"
        )
    dataset = Path(dataset).expanduser().resolve()
    workspace = dataset / "gggs"
    output = (
        Path(output).expanduser().resolve()
        if output
        else workspace
        / "runs"
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"New output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    summary = {
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "output": str(output),
    }
    started = time.monotonic()
    streams = ExitStack()
    logfile = streams.enter_context((output / "run.log").open("w"))
    streams.enter_context(redirect_stdout(_Tee(sys.stdout, logfile)))
    streams.enter_context(redirect_stderr(_Tee(sys.stderr, logfile)))
    write_json(output / "summary.json", summary)
    try:
        paths, dirs, teacher = prepare(dataset, cfg, workspace, summary)
        write_json(output / "summary.json", summary)
        geometry = {
            "schema_version": 1,
            "teacher_manifest": str(dirs["teacher"] / "manifest.json"),
            **cfg["geometry"],
        }
        write_json(output / "joint_config.json", geometry)
        from arguments import ModelParams, OptimizationParams, PipelineParams

        parser = argparse.ArgumentParser()
        model = ModelParams(parser)
        optimization = OptimizationParams(parser)
        pipeline = PipelineParams(parser)
        args = parser.parse_args([])
        for k, v in cfg["training"].items():
            if k != "appearance_mode":
                setattr(args, k, v)
        for k, v in {
            "source_path": str(dataset),
            "model_path": str(output),
            "images": str(paths["images"]),
            "depth_prior_dir": str(paths["depth"]),
            "normal_prior_dir": str(paths["normals"]),
            "colmap_dir": str(paths["colmap"]),
            "init_ply": str(dirs["initialization"] / "point_cloud.ply"),
            "init_num_points": cfg["initialization"]["num_points"],
            "use_decoupled_appearance": cfg["training"]["appearance_mode"],
            "frame_gpu_budget": cfg["runtime"]["real_gpu_cache"],
            "virtual_geometry_config": str(output / "joint_config.json"),
            "disable_virtual_geometry": False,
            "training_seed": cfg["seed"],
            "start_checkpoint": str(Path(resume).resolve()) if resume else None,
            "debug_from": -1,
            "debug_images": False,
            "quiet": False,
            "log_interval": cfg["outputs"]["log_interval"],
            "test_iterations": cfg["outputs"]["evaluation_iterations"],
            "save_iterations": sorted(
                set(cfg["outputs"]["save_iterations"] + [cfg["training"]["iterations"]])
            ),
            "keep_last_checkpoint": True,
            "save_viewer_ply": cfg["outputs"]["viewer_ply"],
            "save_final_checkpoint": cfg["outputs"]["save_final_checkpoint"],
            "step_timing": False,
            "detailed_virtual_log": False,
        }.items():
            setattr(args, k, v)
        interval = cfg["outputs"]["checkpoint_interval"]
        end = cfg["training"]["iterations"]
        args.checkpoint_iterations = (
            list(range(interval, end + 1, interval)) if interval else []
        )
        if cfg["outputs"]["save_final_checkpoint"]:
            args.checkpoint_iterations = sorted(set(args.checkpoint_iterations + [end]))
        import torch

        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        from utils.general_utils import safe_state

        safe_state(False, seed=cfg["seed"])
        from train import training

        summary["status"] = "training"
        write_json(output / "summary.json", summary)
        begin = time.monotonic()
        training(
            model.extract(args),
            optimization.extract(args),
            pipeline.extract(args),
            args,
            prepared_teacher=teacher,
        )
        summary.setdefault("stage_seconds", {})["training"] = time.monotonic() - begin
        summary.update(
            status="completed",
            final_model=str(output / f"point_cloud/iteration_{end}/point_cloud.ply"),
        )
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        summary.update(
            elapsed_seconds=time.monotonic() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(output / "summary.json", summary)
        streams.close()
    return output
