"""Exercise PPISP integration boundaries without importing Gaussian CUDA code."""

import ast
from enum import Enum
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils import training_losses


ROOT = Path(__file__).resolve().parents[1]


class AppModel(Enum):
    NO = 0
    GS = 1
    GOF = 2
    PGSR = 3
    PPISP = 4


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def test_ppisp_rgb_and_vcd_use_same_corrected_image_without_legacy_embedding():
    image = torch.full((3, 2, 2), 0.25, requires_grad=True)
    target = torch.zeros_like(image)
    calls = []

    def apply(raw, index):
        calls.append((raw, index))
        return raw * 2 + 0.1

    def forbidden_embedding(*args):
        raise AssertionError("PPISP must not access a PGSR/GS embedding")

    gaussians = SimpleNamespace(
        App_model=AppModel, app_model=AppModel.PPISP,
        ppisp=SimpleNamespace(apply=apply), get_apperance_embedding=forbidden_embedding,
    )
    photo_calls = []

    def photo(prediction, gt, weight):
        photo_calls.append((prediction, gt, weight))
        l1 = (prediction - gt).abs().mean()
        # Different terms expose whether L1 and SSIM are sent different images.
        structural = (prediction - gt).square().mean()
        return (1 - weight) * l1 + weight * structural, l1, structural

    rgb_loss = load_function("utils/loss_utils.py", "photometric_rgb_loss", {
        "GaussianModel": SimpleNamespace(App_model=AppModel), "photometric_loss": photo,
    })
    vcd = load_function("utils/vcd_utils.py", "apply_appearance_correction", {})
    value, l1 = rgb_loss(image, target, gaussians, 7, 0.25)
    corrected = vcd(image, gaussians, 7)
    torch.testing.assert_close(corrected, torch.full_like(image, 0.6))
    assert len(photo_calls) == 1 and photo_calls[0][2] == 0.25
    torch.testing.assert_close(photo_calls[0][0], corrected)
    assert [index for _, index in calls] == [7, 7]
    assert l1.item() == pytest.approx(0.6)
    assert value.item() == pytest.approx(0.54)
    value.backward()
    torch.testing.assert_close(image.grad, torch.full_like(image, 2 * (0.75 + 0.5 * 0.6) / image.numel()))


def test_ppisp_regularizer_is_added_once_and_keeps_its_gradient(monkeypatch):
    image = torch.full((3, 2, 2), 0.2, requires_grad=True)
    parameter = torch.tensor(4.0, requires_grad=True)
    calls = []

    def regularizer():
        calls.append(True)
        return 0.25 * parameter.square()

    gaussians = SimpleNamespace(ppisp=SimpleNamespace(regularization_loss=regularizer))
    camera = SimpleNamespace(uid=0, original_image=torch.zeros_like(image), normal_prior=None, depth_prior=None, gt_mask=None)
    opt = SimpleNamespace(
        iterations=20000, regularization_from_iter=7000, normal_prior_from_iter=1,
        depth_prior_from_iter=3000, depth_prior_ramp_iters=0, lambda_dssim=0.2,
        lambda_normal_prior=0, lambda_depth_prior=0, lambda_depth_normal=0,
        lambda_multi_view_ncc=0, lambda_multi_view_geo=0, lambda_mask=0,
    )
    monkeypatch.setattr(training_losses, "_photometric_rgb_loss", lambda raw, *args: (raw.mean(), raw.mean()))
    schedule = training_losses.loss_schedule(camera, opt, 1)
    result = training_losses.compute_training_losses({"render": image}, camera, gaussians, opt, schedule)
    assert len(calls) == 1
    assert result.total.item() == pytest.approx(4.2)
    assert result.weighted_terms(opt, schedule)["appearance_regularization"].item() == pytest.approx(4.0)
    result.total.backward()
    assert parameter.grad.item() == pytest.approx(2.0)
    torch.testing.assert_close(image.grad, torch.full_like(image, 1 / image.numel()))


def test_ppisp_steps_during_fastgs_accumulation_but_is_not_a_gaussian_optimizer():
    calls = []
    main = SimpleNamespace(step=lambda: calls.append("main"), zero_grad=lambda **kwargs: calls.append("main.clear"))
    sh = SimpleNamespace(step=lambda: calls.append("sh"), zero_grad=lambda **kwargs: calls.append("sh.clear"))
    isp_optimizer = object()
    model = SimpleNamespace(
        optimizer=main, shoptimizer=sh, dense_optimizer_updates=False,
        ppisp=SimpleNamespace(step=lambda: calls.append("ppisp"), optimizers=[isp_optimizer]),
    )
    step = load_function("scene/gaussian_model.py", "optimizer_step", {}, "GaussianModel")
    iter_optimizers = load_function("scene/gaussian_model.py", "_iter_optimizers", {}, "GaussianModel")
    step(model, 15001)
    assert calls == ["ppisp"]
    step(model, 15008)
    assert calls == ["ppisp", "ppisp", "main", "main.clear", "sh", "sh.clear"]
    assert iter_optimizers(model) == [main, sh]
    assert isp_optimizer not in iter_optimizers(model)


@pytest.mark.parametrize("mode", [AppModel.PGSR, AppModel.PPISP])
def test_scene_save_pairs_ppisp_with_ply_and_preserves_mode3_export(tmp_path, mode):
    calls = []

    def save_file(kind, path):
        calls.append((kind, Path(path).name))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"ply fixture")

    def torch_save(state, path):
        calls.append(("isp", Path(path).name))
        Path(path).write_text(json.dumps(state))

    gaussians = SimpleNamespace(
        app_model=mode,
        save_ply=lambda path: save_file("ply", path),
        save_3dgsviewer_ply=lambda path: save_file("viewer", path),
        ppisp=SimpleNamespace(export_state=lambda: {"completed_updates": 4}),
    )
    scene = SimpleNamespace(model_path=str(tmp_path), gaussians=gaussians)
    save = load_function("scene/__init__.py", "save", {
        "os": os, "torch": SimpleNamespace(save=torch_save),
        "GaussianModel": SimpleNamespace(App_model=AppModel),
    }, "Scene")
    save(scene, 5)
    output = tmp_path / "point_cloud/iteration_5"
    assert calls[0] == ("ply", "point_cloud.ply")
    if mode is AppModel.PPISP:
        assert calls == [("ply", "point_cloud.ply"), ("viewer", "point_cloud_canonical_3dgsviewer.ply"), ("isp", "ppisp.pth.tmp")]
        assert json.loads((output / "ppisp.pth").read_text()) == {"completed_updates": 4}
        assert not (output / "ppisp.pth.tmp").exists()
    else:
        assert calls == [("ply", "point_cloud.ply"), ("viewer", "point_cloud_3dgsviewer.ply")]
        assert not (output / "ppisp.pth").exists()


def test_render_uses_full_names_unclamped_rgb_and_never_heldout_uid(tmp_path):
    raw = torch.tensor([[[-0.2, 1.5]], [[0.2, 0.3]], [[0.4, 0.5]]])
    saved, applied = [], []
    views = [
        SimpleNamespace(source_image_name="train/frame.png", image_name="frame", uid=0, colmap_id=5),
        SimpleNamespace(source_image_name="heldout/frame.png", image_name="frame", uid=3, colmap_id=5),
    ]
    for view in views:
        view.original_image_float = lambda: torch.zeros_like(raw)

    def apply(image, frame_idx, camera_id=None):
        applied.append((image.clone(), frame_idx, camera_id))
        return image * 0.5

    gaussians = SimpleNamespace(
        app_model=AppModel.PPISP,
        ppisp=SimpleNamespace(frame_index_by_name={"train/frame.png": 3}, apply=apply),
    )
    render_set = load_function("render.py", "render_set", {
        "os": os, "json": json,
        "GaussianModel": SimpleNamespace(App_model=AppModel),
        "tqdm": lambda items, **kwargs: items,
        "render": lambda *args, **kwargs: {"render": raw.clone()},
        "torchvision": SimpleNamespace(utils=SimpleNamespace(save_image=lambda image, path: saved.append((image.clone(), path)))),
    })
    render_set(str(tmp_path), "test", 5, views, gaussians, None, None, 0.0, None)
    assert [(frame, cam) for _, frame, cam in applied] == [(3, 5), (-1, 5)]
    torch.testing.assert_close(applied[0][0], raw)
    assert len(saved) == 6
    torch.testing.assert_close(saved[0][0], raw)
    report = json.loads((tmp_path / "test/ours_5/appearance_rendering.json").read_text())
    assert [view["appearance"] for view in report["views"]] == ["learned-frame", "camera-only"]
    assert report["controller"] is False
