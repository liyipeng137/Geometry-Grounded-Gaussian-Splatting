import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

import pipeline
from arguments import ModelParams
from utils.depth_normal_prior import (
    NORMAL_DIR,
    depth_normal_preparation,
    normals_from_depth,
)


def test_calibrated_normal_orientation_scale_and_discontinuity():
    K = np.array([[82.0, 0.0, 23.3], [0.0, 65.0, 17.7], [0.0, 0.0, 1.0]])
    y, x = torch.meshgrid(torch.arange(41.0), torch.arange(53.0), indexing="ij")
    ray_x, ray_y = (x + 0.5 - K[0, 2]) / K[0, 0], (y + 0.5 - K[1, 2]) / K[1, 1]
    z = 2 / (1 + 0.15 * ray_x - 0.25 * ray_y)
    z[:, 27:] *= 2
    z[10, 10] = float("nan")
    normal, count = normals_from_depth(z, K)
    expected = torch.tensor([-0.15, 0.25, -1.0])
    expected /= expected.norm()
    valid = normal.norm(dim=-1) > 0.9
    torch.testing.assert_close(
        normal[valid], expected.expand_as(normal[valid]), atol=2e-5, rtol=0
    )
    assert count == int(valid.sum()) and valid[1:-1, 26:28].all()
    assert not valid[10, 10] and not valid[0].any()
    torch.testing.assert_close(
        normals_from_depth(2 * z, K)[0], normal, atol=2e-6, rtol=0
    )


def test_prior_refresh_repairs_one_frame_and_failed_batch_preserves_previous_files(
    tmp_path,
):
    args = SimpleNamespace(
        source_path=str(tmp_path),
        depth_prior_dir="depth",
        depth_prior_format="png",
        depth_prior_scale=1000.0,
    )
    K = np.array([[70.0, 0.0, 6.2], [0.0, 65.0, 5.3], [0.0, 0.0, 1.0]])
    cams = [
        SimpleNamespace(
            colmap_image_name=f"{i}/view.part.jpg", K=K, height=10, width=14
        )
        for i in ("a", "b")
    ]
    native = np.full((10, 14), 2.0, np.float32)
    for name in ("a", "b"):
        (tmp_path / "depth" / name).mkdir(parents=True)
    np.save(tmp_path / "depth/a/view.part.npy", native)
    Image.fromarray((native[::2, ::2] * 1000).astype(np.uint16)).save(
        tmp_path / "depth/b/view.part.png"
    )

    def prepare():
        with depth_normal_preparation(args, device="cpu") as generate:
            return generate(cams)

    output = tmp_path / NORMAL_DIR
    assert prepare()["generated"] == 2
    manifest = output / "manifest.json"
    a, b = (output / key / "view.part.png" for key in ("a", "b"))
    assert np.array_equal(np.array(Image.open(a)), np.array(Image.open(b)))
    assert tuple(np.array(Image.open(a))[3, 4]) == (128, 128, 255)
    assert tuple(np.array(Image.open(a))[0, 0]) == (128, 128, 128)
    before = (manifest.stat().st_mtime_ns, a.stat().st_mtime_ns, b.stat().st_mtime_ns)
    assert prepare()["reused"] == 2
    assert before == (
        manifest.stat().st_mtime_ns,
        a.stat().st_mtime_ns,
        b.stat().st_mtime_ns,
    )
    a.write_bytes(b"partial")
    assert prepare()["generated"] == 1 and b.stat().st_mtime_ns == before[2]
    old = (manifest.read_bytes(), a.read_bytes(), b.read_bytes())
    np.save(
        tmp_path / "depth/a/view.part.npy",
        native + np.arange(14, dtype=np.float32) * 0.01,
    )
    np.save(tmp_path / "depth/b/view.part.npy", np.zeros_like(native))
    with pytest.raises(ValueError, match="No usable"):
        prepare()
    assert (manifest.read_bytes(), a.read_bytes(), b.read_bytes()) == old
    np.save(tmp_path / "depth/b/view.part.npy", native)
    assert prepare()["generated"] == 2
    cams[0].K = K.copy()
    cams[0].K[0, 2] += 1.0
    assert prepare()["generated"] == 1


def test_trainer_fixed_normals_do_not_change_other_model_parsers():
    parser = argparse.ArgumentParser()
    group = ModelParams(parser, fixed_normal_dir=NORMAL_DIR)
    args = group.extract(parser.parse_args([]))
    assert (args.normal_prior_dir, args.normal_prior_format) == (NORMAL_DIR, "png")
    with pytest.raises(SystemExit):
        parser.parse_args(["--normal_prior_dir", "other"])
    parser = argparse.ArgumentParser()
    original = ModelParams(parser)
    assert (
        original.extract(
            parser.parse_args(["--normal_prior_dir", "other"])
        ).normal_prior_dir
        == "other"
    )


def test_real_pipeline_stage_order_and_reserved_virtual_input(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    dataset.mkdir()
    output = tmp_path / "out"
    initial = tmp_path / "init.ply"
    initial.touch()
    calls = []

    def stage(command, *, cwd, env, check):
        assert check and cwd == pipeline.ROOT
        calls.append(command)
        if len(calls) == 1:
            assert command[command.index("--init_ply") + 1] == str(initial)
            assert command[command.index("-s") + 1] == str(dataset)
            model = output / "point_cloud/iteration_20000/point_cloud.ply"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            (output / "depth_normal_prior.json").write_text('{"frames": 3}')
        else:
            assert command[command.index("-m") + 1] == str(output)
            (output / "recon_bounded_post_smooth.ply").write_bytes(b"mesh")

    monkeypatch.setattr(pipeline.subprocess, "run", stage)
    with pytest.raises(NotImplementedError, match="Virtual supervision"):
        pipeline.run(dataset, output, virtual_geometry_config=tmp_path / "teacher.json")
    assert not calls and not output.exists()
    mesh = pipeline.run(dataset, output, init_ply=initial)
    assert [Path(c[2]).name for c in calls] == [
        "train_real_geometry.py",
        "mesh_extract_mini.py",
    ]
    assert json.loads((output / "pipeline_summary.json").read_text())["mesh"] == str(
        mesh
    )
    with pytest.raises(FileExistsError):
        pipeline.run(dataset, output)
