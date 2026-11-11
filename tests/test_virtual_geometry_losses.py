import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_geometry_losses import compute_virtual_geometry_losses  # noqa: E402


def make_camera(reliability=1.):
    path = Path(__file__).resolve().parents[1] / "scene/virtual_cameras.py"
    spec = importlib.util.spec_from_file_location("test_virtual_camera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {"id": "virtual_0", "width": 7, "height": 7, "K": [[10., 0., 3.5], [0., 10., 3.5], [0., 0., 1.]],
              "c2w": np.eye(4).tolist()}
    depth = torch.full((1, 7, 7), 2.)
    normal = torch.zeros((3, 7, 7))
    normal[2] = -1
    valid_normal = torch.zeros_like(depth, dtype=torch.bool)
    valid_normal[:, 1:-1, 1:-1] = True
    teacher = {"depth": depth, "normal": normal, "valid_depth": torch.ones_like(depth, dtype=torch.bool),
               "valid_normal": valid_normal, "q": torch.ones_like(depth),
               "weight_depth": torch.full_like(depth, reliability), "weight_normal": torch.full_like(depth, reliability),
               "weight_coverage": torch.full_like(depth, reliability), "face_id": torch.zeros((7, 7), dtype=torch.int64),
               "patch_id": torch.zeros((7, 7), dtype=torch.int64), "evidence_class": torch.zeros((7, 7), dtype=torch.uint8)}
    return module.VirtualCamera(record, teacher, {"depth": 49., "normal": 25., "coverage": 49.}, device="cpu")


def package(depth=2.1, alpha=.9):
    normal = torch.zeros((3, 7, 7))
    normal[2] = -1
    return {"median_depth": torch.full((1, 7, 7), depth, requires_grad=True),
            "mask": torch.full((1, 7, 7), alpha, requires_grad=True), "normal": normal.requires_grad_()}


def test_unknown_reliability_is_not_cancelled_by_normalization():
    full = compute_virtual_geometry_losses(package(), make_camera())
    unknown = compute_virtual_geometry_losses(package(), make_camera(.25))
    assert torch.allclose(unknown.total, full.total * .25)


def test_matching_plane_has_zero_geometry_loss_and_no_rgb_attribute():
    cam = make_camera()
    result = compute_virtual_geometry_losses(package(2., .9), cam)
    assert not hasattr(cam, "original_image")
    assert result.total.item() == pytest.approx(0., abs=1e-7)
    assert result.metrics["missing_fraction"] == 0


def test_all_invalid_students_have_finite_loss_gradient_and_fixed_domain_misses():
    render = package(float("nan"), .2)
    render["normal"] = torch.full((3, 7, 7), float("nan"), requires_grad=True)
    result = compute_virtual_geometry_losses(render, make_camera())
    result.total.backward()
    assert torch.isfinite(result.total)
    assert result.metrics["missing_fraction"] == 1
    assert result.metrics["fixed_depth_relative_error"] == 1
    assert result.raw["depth"] == 0
    assert result.raw["coverage"] > 0
    for value in render.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_root_failure_does_not_supply_depth_or_normal_gradients():
    render = package()
    render["depth_status"] = torch.full((1, 7, 7), 2, dtype=torch.int32)
    result = compute_virtual_geometry_losses(render, make_camera())
    result.total.backward()
    assert result.metrics["root_search_failed_fraction"] == 1
    assert render["median_depth"].grad.abs().max() == 0
    assert render["normal"].grad.abs().max() == 0


def test_valid_near_student_surface_still_receives_depth_correction_gradient():
    render = package(.1, .9)
    render["depth_status"] = torch.full((1, 7, 7), 3, dtype=torch.int32)
    result = compute_virtual_geometry_losses(render, make_camera())
    result.total.backward()
    assert result.metrics["missing_fraction"] == 0
    assert result.raw["depth"] > 0
    assert (render["median_depth"].grad < 0).all()


def test_noncentral_K_camera_z_conversion_uses_same_physical_ray():
    cam = make_camera()
    cam.Cx += .3
    depth = torch.ones((1, 7, 7))
    converted = cam.camera_z_depth(depth)
    assert not torch.allclose(converted, depth)
    assert torch.isfinite(converted).all()


def test_skipping_diagnostics_preserves_all_losses_maps_and_gradients():
    full_render, compact_render = package(2.12, .7), package(2.12, .7)
    full = compute_virtual_geometry_losses(full_render, make_camera(), collect_metrics=True)
    compact = compute_virtual_geometry_losses(compact_render, make_camera(), collect_metrics=False)
    assert full.metrics and compact.metrics == {}
    for name in full.raw:
        torch.testing.assert_close(compact.raw[name], full.raw[name], rtol=0, atol=0)
        torch.testing.assert_close(compact.weighted[name], full.weighted[name], rtol=0, atol=0)
    for name in full.maps:
        torch.testing.assert_close(compact.maps[name], full.maps[name], rtol=0, atol=0)
    full.total.backward()
    compact.total.backward()
    for name in full_render:
        torch.testing.assert_close(compact_render[name].grad, full_render[name].grad, rtol=0, atol=0)
