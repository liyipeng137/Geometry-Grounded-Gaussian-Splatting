from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene.cameras import Camera
from scene.geometry_gaussians import GeometryGaussians
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity test")
def test_native_zero_color_matches_rgb_geometry_and_gradients(tmp_path):
    torch.manual_seed(202)
    h, w = 64, 80
    k = np.array([[72.0, 0.0, 35.7], [0.0, 75.0, 29.3], [0.0, 0.0, 1.0]])
    rgb = torch.randint(256, (3, h, w), dtype=torch.uint8)
    mask = torch.rand(1, h, w)

    def camera(gray):
        c = Camera(
            0,
            np.eye(3),
            np.zeros(3),
            1.0,
            1.0,
            rgb,
            mask,
            "parity",
            0,
            K=k,
            lazy_payload=True,
            cache_gray=gray,
        )
        c.load_payload_to_device()
        return c

    color_camera, gray_camera = camera(False), camera(True)
    assert gray_camera.original_image_cpu is None and gray_camera.original_image is None
    torch.testing.assert_close(
        gray_camera.gray_image, color_camera.gray_image, rtol=0, atol=1e-7
    )
    n = 256
    xyz = torch.rand(n, 3, device="cuda") - 0.5
    xyz[:, 2] = 1.8 + xyz[:, 0] * 0.1
    q = torch.zeros(n, 4, device="cuda")
    q[:, 0] = 1.0
    values = {
        "_xyz": xyz,
        "_rotation": q,
        "_scaling": xyz.new_tensor([0.12, 0.1, 0.024]).log().expand(n, 3),
        "_opacity": xyz.new_full((n, 1), 0.9),
    }
    old, new = GaussianModel(3, 0), GeometryGaussians()
    for m in (old, new):
        for key, val in values.items():
            setattr(m, key, torch.nn.Parameter(val.clone().contiguous()))
        m.filter_3D = xyz.new_full((n, 1), 0.003)
    old._features_dc = torch.randn(n, 1, 3, device="cuda")
    old._features_rest = torch.randn(n, 15, 3, device="cuda")
    old._sg_axis = old._sg_color = xyz.new_empty((n, 0, 3))
    old._sg_sharpness = xyz.new_empty((n, 0))
    pipe = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    bg = xyz.new_zeros(3)
    first = render(color_camera, old, pipe, bg, 0.0, depth_search_expansions=2)
    second = render(gray_camera, new, pipe, bg, 0.0, depth_search_expansions=2)
    assert (
        "render" not in second
        and "render" in first
        and (second["depth_status"] == 3).any()
    )
    for k in ("median_depth", "mask", "normal", "radii", "depth_status"):
        torch.testing.assert_close(first[k], second[k], rtol=2e-5, atol=2e-6)
    weights = [torch.randn_like(first[k]) for k in ("median_depth", "normal", "mask")]

    def gradients(pkg, m):
        loss = sum(
            (pkg[k] * v).mean()
            for k, v in zip(("median_depth", "normal", "mask"), weights)
        )
        return torch.autograd.grad(loss, tuple(getattr(m, k) for k in values))

    for a, b in zip(gradients(first, old), gradients(second, new)):
        assert a.isfinite().all() and b.isfinite().all() and a.abs().sum() > 0
        torch.testing.assert_close(a, b, rtol=3e-4, atol=5e-6)
    from plyfile import PlyData

    path = tmp_path / "geometry.ply"
    new.save_ply(path)
    assert len(PlyData.read(str(path))["vertex"].data.dtype.names) == 12
    loaded = GeometryGaussians()
    loaded.load_ply(path)
    for key in (*values, "filter_3D"):
        torch.testing.assert_close(
            getattr(loaded, key), getattr(new, key), rtol=0, atol=0
        )
    assert not any(
        hasattr(new, name)
        for name in (
            "_features_dc",
            "_features_rest",
            "_sg_color",
            "_appearance_embeddings",
        )
    )
    assert set(new.point_parameters) == {"xyz", "scaling", "rotation", "opacity"}
    new.export_neutral_viewer(tmp_path / "viewer.ply")
    visible = PlyData.read(str(tmp_path / "viewer.ply"))["vertex"].data
    assert (
        "filter_3D" not in visible.dtype.names and "f_rest_0" not in visible.dtype.names
    )
    assert all(np.all(visible[f"f_dc_{i}"] == 0) for i in range(3))
    np.testing.assert_allclose(
        np.exp(visible["scale_0"]),
        new.get_scaling_with_3D_filter[:, 0].detach().cpu().numpy(),
        rtol=2e-6,
    )


def test_20k_refinement_boundaries():
    from train_real_geometry import refinement_policy

    for step, weight, lr, active in (
        (6600, 0.1, 1, True),
        (6800, 0.1, 1, False),
        (7000, 0.1, 1, False),
        (9000, 0.15, 1, True),
        (15000, 0.15, 1, True),
        (20000, 0.06, 0.25, False),
    ):
        ncc, actual_lr, topology = refinement_policy(step, 20000)
        assert (
            ncc == pytest.approx(weight)
            and actual_lr == pytest.approx(lr)
            and topology is active
        )
