#!/usr/bin/env python3
"""手动准备 dataset/virtual_geometry；训练只消费完整产物，不自动运行本步骤。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_camera_selection import log, run_selection, sha256, write_json  # noqa: E402
from utils.virtual_geometry_data import VirtualGeometryDataset, prepare_teacher  # noqa: E402


PREVIEW_FILES = ("preview.html", "contact_sheet.jpg", "contact_sheet_trajectory.jpg",
                 "contact_sheet_surface.jpg", "contact_sheet_real.jpg")


def validate_dataset_pipeline(dataset, root):
    # 统一验证器由训练入口共用；延迟导入不要求先构造 Scene/Torch。
    from utils.dataset_virtual_geometry import validate_dataset_pipeline as validate
    return validate(dataset, root)


def _require_previews(root):
    cameras = json.loads((root / "cameras/virtual_cameras.json").read_text())["cameras"]
    real_cameras = json.loads((root / "cameras/real_cameras.json").read_text())["cameras"]
    for camera in real_cameras + cameras:
        for kind in ("shading", "normal", "depth"):
            reference = camera.get("preview", {}).get(kind)
            if not isinstance(reference, str) or not reference:
                raise ValueError(f"Incomplete camera preview reference: {camera.get('id')} {kind}")
            path = Path(reference).expanduser()
            path = path if path.is_absolute() else root / "cameras" / path
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Incomplete camera preview: {path}")
    families = {"surface" if camera["selection_source"] == "surface_lookat" else "trajectory" for camera in cameras}
    required = {"preview.html", "contact_sheet.jpg", "contact_sheet_real.jpg"}
    required.update(f"contact_sheet_{family}.jpg" for family in families)
    for name in sorted(required):
        path = root / "cameras" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete virtual-camera preview: {path}")


def _verify_teacher(dataset, mesh, manifest_path):
    teacher = VirtualGeometryDataset(manifest_path, device="cpu")
    if Path(teacher.manifest["dataset"]).expanduser().resolve() != dataset:
        raise ValueError("Teacher is bound to another dataset")
    if Path(teacher.manifest["genrecon_mesh"]).expanduser().resolve() != mesh:
        raise ValueError("Teacher is bound to another mesh")


def prepare_pipeline(dataset, mesh=None, output=None, threads=8, seed=0):
    dataset = Path(dataset).expanduser().resolve()
    mesh = Path(mesh).expanduser().resolve() if mesh else dataset / "genrecon_mesh.ply"
    root = Path(output).expanduser().resolve() if output else dataset / "virtual_geometry"
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    if threads < 1 or seed < 0:
        raise ValueError("threads must be positive and seed must be nonnegative")
    marker_path = root / "pipeline.json"
    if root.exists() and any(root.iterdir()):
        if not marker_path.is_file():
            raise FileExistsError(f"Nonempty incomplete output is preserved: {root}")
        existing = json.loads(marker_path.read_text())
        if existing.get("schema_version") != 1 or existing.get("status") != "complete":
            raise FileExistsError(f"Nonempty incomplete output is preserved: {root}")
        if Path(existing.get("dataset", "")).expanduser().resolve() != dataset or Path(existing.get("mesh", "")).expanduser().resolve() != mesh:
            raise ValueError("Existing virtual geometry belongs to another dataset or mesh")
        if existing.get("trajectory_offset_fraction", .08) != .16:
            raise ValueError("Existing cache has a different trajectory offset; choose a new output directory")
        if existing.get("seed") != seed:
            raise ValueError("Existing virtual geometry uses a different seed; choose a new output directory")
        config_path = validate_dataset_pipeline(dataset, root)
        _require_previews(root)
        _verify_teacher(dataset, mesh, root / "teacher/manifest.json")
        log("virtual_geometry_reused", root=str(root), config=str(config_path))
        return Path(config_path)

    root.mkdir(parents=True, exist_ok=True)
    marker = {"schema_version": 1, "status": "running", "dataset": str(dataset), "mesh": str(mesh),
              "stage": "camera_selection", "seed": seed, "threads": threads, "trajectory_offset_fraction": .16}
    write_json(marker_path, marker)
    try:
        log("virtual_geometry_stage", stage=marker["stage"], root=str(root))
        run_selection(dataset, mesh, root / "cameras", target_count=160, candidate_limit=1200,
                      long_edge=256, surface_samples=120000, threads=threads, seed=seed, preview_long_edge=512)
        marker["stage"] = "camera_preview"
        write_json(marker_path, marker)
        log("virtual_geometry_stage", stage=marker["stage"], root=str(root))
        script = Path(__file__).resolve().with_name("export_virtual_camera_preview.py")
        subprocess.run([sys.executable, str(script), "--manifest", str(root / "cameras/manifest.json")], check=True)
        _require_previews(root)

        marker["stage"] = "teacher_preparation"
        write_json(marker_path, marker)
        log("virtual_geometry_stage", stage=marker["stage"], root=str(root))
        prepare_teacher(root / "cameras/manifest.json", root / "teacher", dataset=dataset, long_edge=512, threads=threads)
        marker["stage"] = "teacher_validation"
        write_json(marker_path, marker)
        log("virtual_geometry_stage", stage=marker["stage"], root=str(root))
        teacher_path = root / "teacher/manifest.json"
        _verify_teacher(dataset, mesh, teacher_path)

        config_path = root / "train_config.json"
        write_json(config_path, {"schema_version": 1, "teacher_manifest": "teacher/manifest.json"})
        # complete 最后原子发布；中断/失败的中间产物不能被训练当作完成。
        marker.update(status="complete", stage="complete", config_sha256=sha256(config_path),
                      teacher_manifest_sha256=sha256(teacher_path), config_path="train_config.json",
                      teacher_manifest_path="teacher/manifest.json", camera_manifest_path="cameras/manifest.json",
                      preview_path="cameras/preview.html")
        write_json(marker_path, marker)
        log("virtual_geometry_complete", root=str(root), config=str(config_path), preview=str(root / "cameras/preview.html"))
        return config_path
    except Exception as error:
        marker.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(marker_path, marker)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare_pipeline(args.dataset, args.mesh, args.output, args.threads, args.seed)


if __name__ == "__main__":
    main()
