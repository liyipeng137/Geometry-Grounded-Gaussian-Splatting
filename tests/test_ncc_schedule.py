"""NCC 日程、实际加权及必需 prior 的 CPU 回归。"""

import ast
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from arguments import OptimizationParams
from utils import training_losses as losses


def options(**overrides):
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    opt = group.extract(parser.parse_args([]))
    for name, value in overrides.items():
        setattr(opt, name, value)
    return opt


def test_ncc_default_and_demo_boundaries():
    opt = options()
    assert opt.lambda_multi_view_ncc == 0.2
    for iteration, expected in (
        (1, 0), (6999, 0), (7000, 0), (7001, 0.2),
        (11000, 0.10001250156269534), (11001, 0.09999374921865234),
        (15000, 0.05), (15001, 0.05), (20000, 0.05),
    ):
        assert losses.multi_view_ncc_weight(iteration, opt) == pytest.approx(expected)


def test_short_budgets_and_zero_weight():
    for budget, iteration, expected in (
        (200, 200, 0), (7000, 7000, 0), (7001, 7001, 0.2),
        (10000, 7001, 0.2), (10000, 8500, 0.1000333444481494),
        (10000, 8501, 0.09998332777592531), (10000, 10000, 0.05),
    ):
        assert losses.multi_view_ncc_weight(iteration, options(iterations=budget)) == pytest.approx(expected)
    for iteration in (1, 7001, 11001, 15000, 20000):
        assert losses.multi_view_ncc_weight(iteration, options(lambda_multi_view_ncc=0)) == 0


def test_effective_weight_controls_patchmatch_and_loss_gradient():
    cam = SimpleNamespace(
        uid=0, original_image=torch.zeros(3, 2, 2),
        normal_prior=None, depth_prior=None, gt_mask=None,
    )
    image = torch.zeros(3, 2, 2, requires_grad=True)
    normal = torch.zeros_like(image)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    pkg = {"render": image, "median_depth": torch.ones(1, 2, 2), "normal": normal}
    opt = options(lambda_normal_prior=0, lambda_depth_prior=0, lambda_depth_normal=0,
                  lambda_multi_view_geo=0)
    ncc_value = torch.tensor(2.0, requires_grad=True)
    patchmatch = Mock(return_value=(ncc_value, image.new_tensor(3.0)))

    with (
        patch.object(losses, "_photometric_rgb_loss", side_effect=lambda pred, *_: (pred.sum() * 0, pred.sum() * 0)),
        patch.object(losses, "_depth_to_normal", return_value=(normal, valid)),
    ):
        before = losses.loss_schedule(cam, opt, 7000)
        result = losses.compute_training_losses(pkg, cam, None, opt, before, patchmatch=patchmatch)
        assert before.multi_view_ncc_weight == 0
        assert result.total.item() == 0
        patchmatch.assert_not_called()

        tail = losses.loss_schedule(cam, opt, 20000)
        result = losses.compute_training_losses(pkg, cam, None, opt, tail, patchmatch=patchmatch)
        assert tail.multi_view_ncc_weight == pytest.approx(0.05)
        assert result.total.item() == pytest.approx(0.1)
        result.total.backward()
        assert ncc_value.grad.item() == pytest.approx(0.05)
        assert patchmatch.call_count == 1

    delayed = losses.loss_schedule(cam, options(regularization_from_iter=8000), 7001)
    assert delayed.multi_view_ncc_weight == 0


def test_required_priors_use_loaded_flags_not_gpu_residency():
    path = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_training_priors")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    validate = namespace["validate_training_priors"]
    cameras = [SimpleNamespace(
        image_name=f"frame_{index}.png", has_normal_prior=True, has_depth_prior=True,
        normal_prior=None, depth_prior=None,
    ) for index in range(2)]
    # 合法 lazy-payload 相机尚未搬到 GPU，不能因此误报缺失。
    assert validate(cameras) is None
    for prior in ("normal", "depth"):
        setattr(cameras[1], f"has_{prior}_prior", False)
        with pytest.raises(ValueError) as error:
            validate(cameras)
        assert "frame_1.png" in str(error.value) and prior in str(error.value)
        setattr(cameras[1], f"has_{prior}_prior", True)
