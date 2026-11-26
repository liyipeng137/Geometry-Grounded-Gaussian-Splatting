import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

import pipeline
from arguments import ModelParams, OptimizationParams
from utils.affinegs_depth import affinegs_depth_loss
from utils.depth_normal_prior import (
    NORMAL_DIR,
    normals_from_depth,
    prepare_depth_normal_priors,
)
from utils.training_losses import compute_training_losses, loss_schedule


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


def test_missing_normals_are_generated_and_existing_ones_reused(tmp_path):
    K = np.array([[70.0, 0.0, 6.2], [0.0, 65.0, 5.3], [0.0, 0.0, 1.0]])
    cams = [
        SimpleNamespace(colmap_image_name=f"{i}/view.part.jpg", K=K, height=10, width=14)
        for i in ("a", "b")
    ]
    depth = tmp_path / "depth"
    native = np.full((10, 14), 2.0, np.float32)
    for name in ("a", "b"):
        (depth / name).mkdir(parents=True)
    np.save(depth / "a/view.part.npy", native)
    # A half-resolution PNG depth is resampled onto the camera grid.
    Image.fromarray((native[::2, ::2] * 1000).astype(np.uint16)).save(depth / "b/view.part.png")

    def prepare():
        return prepare_depth_normal_priors(tmp_path, depth, cams, "png", 1000.0)

    root = tmp_path / NORMAL_DIR
    assert prepare()["generated"] == 2
    a, b = (root / key / "view.part.png" for key in ("a", "b"))
    assert np.array_equal(np.array(Image.open(a)), np.array(Image.open(b)))
    assert tuple(np.array(Image.open(a))[3, 4]) == (128, 128, 255)
    assert tuple(np.array(Image.open(a))[0, 0]) == (128, 128, 128)
    before = (a.stat().st_mtime_ns, b.stat().st_mtime_ns)
    assert prepare() == {"directory": str(root), "frames": 2, "generated": 0, "reused": 2}
    assert before == (a.stat().st_mtime_ns, b.stat().st_mtime_ns)
    a.unlink()
    assert prepare()["generated"] == 1 and b.stat().st_mtime_ns == before[1]
    a.unlink()
    (depth / "a/view.part.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing aligned depth"):
        prepare()
    np.save(depth / "a/view.part.npy", np.zeros_like(native))
    with pytest.raises(ValueError, match="No usable"):
        prepare()


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


@pytest.mark.parametrize(
    "options,mode,merge_sparse",
    [
        ({}, "absolute", True),
        ({"real_depth_loss_mode": "affinegs", "merge_sparse_init": False}, "affinegs", False),
    ],
)
def test_real_pipeline_stage_order_and_reserved_virtual_input(
    tmp_path, monkeypatch, options, mode, merge_sparse
):
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
            flag = "--merge-sparse-init" if merge_sparse else "--no-merge-sparse-init"
            assert flag in command
            assert command[command.index("--init_ply") + 1] == str(initial)
            assert command[command.index("-s") + 1] == str(dataset)
            assert command[command.index("--real_depth_loss_mode") + 1] == mode
            model = output / "point_cloud/iteration_20000/point_cloud.ply"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            (output / "camera_split.json").write_text(json.dumps({
                "initialization": {"kind": "external_ply", "point_count": 2},
                "normal_prior": {"frames": 3},
            }))
        else:
            assert command[command.index("-m") + 1] == str(output)
            (output / "recon_bounded_post_smooth.ply").write_bytes(b"mesh")

    monkeypatch.setattr(pipeline.subprocess, "run", stage)
    with pytest.raises(NotImplementedError, match="Virtual supervision"):
        pipeline.run(dataset, output, virtual_geometry_config=tmp_path / "teacher.json")
    assert not calls and not output.exists()
    mesh = pipeline.run(dataset, output, init_ply=initial, **options)
    assert [Path(c[2]).name for c in calls] == [
        "train_real_geometry.py",
        "mesh_extract_mini.py",
    ]
    summary = json.loads((output / "pipeline_summary.json").read_text())
    assert summary["mesh"] == str(mesh) and summary["real_depth_loss_mode"] == mode
    assert summary["merge_sparse_init"] is merge_sparse
    assert summary["initialization"]["kind"] == "external_ply"
    assert summary["normal_prior"] == {"frames": 3}
    with pytest.raises(FileExistsError):
        pipeline.run(dataset, output)


def test_color_free_training_uses_the_selected_weighted_depth_loss():
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 17), torch.linspace(0, 1, 23), indexing="ij"
    )
    prior = (1.5 + x + 0.2 * y)[None]
    z = (prior * 1.7 + 0.15 * torch.sin(x * 9)[None]).requires_grad_()
    camera = SimpleNamespace(normal_prior=None, depth_prior=prior, gt_mask=None)
    for mode in ("absolute", "affinegs"):
        opt = group.extract(
            parser.parse_args(
                ["--real_depth_loss_mode", mode, "--depth_prior_from_iter", "1"]
            )
        )
        schedule = loss_schedule(camera, opt, 1)
        terms = compute_training_losses(
            {"median_depth": z},
            camera,
            SimpleNamespace(),
            opt,
            schedule,
            supervise_rgb=False,
        )
        expected = (
            affinegs_depth_loss(z, prior)
            if mode == "affinegs"
            else (z - prior).abs().mean()
        )
        expected = expected * schedule.depth_prior_weight
        torch.testing.assert_close(terms.total, expected)
        (grad,) = torch.autograd.grad(terms.total, z)
        (wanted,) = torch.autograd.grad(expected, z)
        torch.testing.assert_close(grad, wanted)
        assert grad.abs().sum() > 0
