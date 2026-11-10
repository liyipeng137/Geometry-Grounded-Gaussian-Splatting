"""Renderer contracts: CPU camera conversion and optional real CUDA fixtures.

CUDA tests exercise native code, including a root outside the original bracket;
they are skipped explicitly when Torch/CUDA is unavailable.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_geometry_score(torch_module, settings, rasterizer):
    path = ROOT / "gaussian_renderer/__init__.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "geometry_score")
    function.decorator_list = []
    namespace = {"torch": torch_module, "math": math,
                 "GaussianRasterizationSettings": settings, "GaussianRasterizer": rasterizer}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["geometry_score"]


def contribution_reference(alpha, peak, target, tolerance, weight=1, error=1, support=1):
    """Analytical single-ray compositing oracle, independent of CUDA tensors."""
    output = np.zeros((5, len(alpha)), dtype=np.float64)
    transmittance = 1.0
    for index, (opacity, depth) in enumerate(zip(alpha, peak)):
        opacity = min(opacity, .99)
        if opacity < 1 / 255:
            continue
        next_transmittance = transmittance * (1 - opacity)
        if next_transmittance < .0001:
            break
        contribution = weight * opacity * transmittance
        output[3, index] = contribution
        if abs(depth - target) <= tolerance:
            output[:3, index] = contribution * np.array([error, support, 1])
        else:
            output[4, index] = contribution
        transmittance = next_transmittance
    return output


class Array(np.ndarray):
    device = "cpu"

    def numel(self):
        return self.size

    def contiguous(self):
        return self

    def square(self):
        return self ** 2


def array(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype).view(Array)


class NumpyTorch:
    float32, int32 = np.float32, np.int32

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        return array(value, dtype)

    @staticmethod
    def arange(count, device=None, dtype=None):
        return np.arange(count, dtype=dtype).view(Array)

    @staticmethod
    def empty(count, dtype=None, device=None):
        return np.empty(count, dtype=dtype).view(Array)

    isfinite = staticmethod(np.isfinite)
    sqrt = staticmethod(np.sqrt)

    @staticmethod
    def where(condition, a, b):
        return np.where(condition, a, b).view(Array)


def test_camera_z_band_uses_actual_principal_point_and_masks_invalid_teacher():
    captured = {}

    class Rasterizer:
        def __init__(self, settings):
            captured["settings"] = settings

        def geometry_score(self, xyz, opacity, scales, rotation, depth, tolerance, error, support, weight):
            captured.update(depth=depth, tolerance=tolerance, weight=weight)
            return [array([i]) for i in range(5)]

    score = load_geometry_score(NumpyTorch, SimpleNamespace, Rasterizer)
    camera = SimpleNamespace(image_height=2, image_width=3, Cx=.2, Cy=.8, Fx=2., Fy=3.,
                             FoVx=1., FoVy=1., world_view_transform=None,
                             full_proj_transform=None, camera_center=None)
    pc = SimpleNamespace(get_xyz=array([[0, 0, 2]]), get_scaling_n_opacity_with_3D_filter=(None, None), get_rotation=None)
    depth = array([[2, 2, np.nan], [2, 2, 2]])
    tolerance = np.full((2, 3), .1)
    result = score(camera, pc, SimpleNamespace(debug=False), None, 0, depth, tolerance,
                   np.ones((2, 3)), np.ones((2, 3)), np.full((2, 3), .25))
    expected_norm = np.sqrt(1 + ((np.arange(3)[None] - .2) / 2.) ** 2 + ((np.arange(2)[:, None] - .8) / 3.) ** 2)
    np.testing.assert_allclose(captured["depth"][1], 2 * expected_norm[1])
    np.testing.assert_allclose(captured["tolerance"][1], .1 * expected_norm[1])
    assert captured["depth"][0, 2] == captured["weight"][0, 2] == 0
    assert captured["weight"][1, 1] == .25  # Reliability is not normalized away.
    assert set(result) == {"error_sum", "support_sum", "near_weight_sum", "weight_sum", "off_surface_sum"}


def test_two_layer_reference_keeps_visibility_weight_and_depth_ownership():
    value = contribution_reference([.6, .7], [2, 3], 3, .05, weight=2, error=3, support=.8)
    np.testing.assert_allclose(value[3], [1.2, .56])
    np.testing.assert_allclose(value[2], [0, .56])
    np.testing.assert_allclose(value[0], [0, 1.68])
    np.testing.assert_allclose(value[1], [0, .448])
    np.testing.assert_allclose(value[4], [1.2, 0])
    np.testing.assert_allclose(value[2] + value[4], value[3])


@pytest.fixture
def cuda_backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    import diff_gaussian_rasterization as backend
    assert hasattr(backend._C, "geometry_score_gaussians"), "rebuild the modified CUDA extension"
    return torch, backend


def cuda_scene(cuda_backend, means=((0., 0., 2.),), opacity=(.9,), scales=None, expansions=0, cx=16., cy=16.):
    torch, backend = cuda_backend
    width = height = 33
    fx, fy = 30., 32.
    projection = torch.zeros((4, 4), device="cuda")
    projection[0, 0], projection[1, 1] = 2 * fx / width, 2 * fy / height
    projection[0, 2] = 2 * (cx + .5) / width - 1
    projection[1, 2] = 2 * (cy + .5) / height - 1
    projection[2, 2], projection[2, 3], projection[3, 2] = 100 / 99.8, -20 / 99.8, 1
    settings = backend.GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=width / (2 * fx), tanfovy=height / (2 * fy),
        kernel_size=0., bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=projection.T.contiguous(),
        sh_degree=0, sg_degree=0, campos=torch.zeros(3, device="cuda"),
        prefiltered=False, require_depth=True, debug=True, get_flag=False,
        metric_map=torch.empty(0, dtype=torch.int32, device="cuda"), depth_search_expansions=expansions,
    )
    count = len(means)
    tensors = {
        "means3D": torch.tensor(means, dtype=torch.float32, device="cuda").reshape(count, 3).requires_grad_(),
        "means2D": torch.zeros((count, 3), device="cuda", requires_grad=True),
        "opacities": torch.tensor(opacity, device="cuda").reshape(count, 1).requires_grad_(),
        # Existing compact-box binning rejects exactly diagonal 2D conics.
        # A rotated anisotropic XY footprint avoids that pre-existing corner
        # case without changing the center-ray alpha or Z-depth oracle.
        "scales": torch.tensor(scales if scales is not None else [[.12, .08, .12]] * count, device="cuda").reshape(count, 3).requires_grad_(),
        "rotations": torch.tensor([[math.cos(.3), 0., 0., math.sin(.3)]] * count, device="cuda").reshape(count, 4).requires_grad_(),
        "colors_precomp": torch.full((count, 3), .5, device="cuda", requires_grad=True),
        "sg_axis": torch.empty((count, 0, 3), device="cuda"),
        "sg_sharpness": torch.empty((count, 0), device="cuda"),
        "sg_color": torch.empty((count, 0, 3), device="cuda"),
    }
    return backend.GaussianRasterizer(settings), tensors


def test_cuda_geometry_score_two_layers_matches_analytic_alpha_T(cuda_backend):
    torch, _ = cuda_backend
    rasterizer, tensors = cuda_scene(cuda_backend, means=((0., 0., 2.), (0., 0., 3.)), opacity=(.6, .7))
    rendered = rasterizer(**tensors)
    assert (rendered[1] > 0).all()
    torch.testing.assert_close(rendered[3][0, 16, 16], torch.tensor(.88, device="cuda"))
    maps = [torch.zeros((33, 33), device="cuda") for _ in range(5)]
    for image, value in zip(maps, [3., .05, 3., .8, 2.]):
        image[16, 16] = value
    output = rasterizer.geometry_score(tensors["means3D"], tensors["opacities"], tensors["scales"], tensors["rotations"], *maps)
    expected = contribution_reference([.6, .7], [2, 3], 3, .05, weight=2, error=3, support=.8)
    np.testing.assert_allclose(torch.stack(output).cpu().numpy(), expected, rtol=2e-5, atol=1e-6)
    assert not any(value.requires_grad for value in output)
    assert all(tensor.grad is None for tensor in tensors.values())


def test_cuda_geometry_score_empty_scene_and_empty_rays(cuda_backend):
    torch, _ = cuda_backend
    for means, opacity in (((), ()), (((100., 0., 2.),), (.9,))):
        rasterizer, tensors = cuda_scene(cuda_backend, means=means, opacity=opacity)
        maps = [torch.full((33, 33), value, device="cuda") for value in [2., .05, 1., 1., 1.]]
        output = rasterizer.geometry_score(tensors["means3D"], tensors["opacities"], tensors["scales"], tensors["rotations"], *maps)
        assert all(torch.isfinite(value).all() and (value == 0).all() for value in output)


def test_cuda_geometry_score_noncentered_K_preserves_camera_z_band(cuda_backend):
    torch, backend = cuda_backend
    rasterizer, tensors = cuda_scene(cuda_backend, means=((.4, 0., 2.),), cx=10.)
    settings = rasterizer.raster_settings
    camera = SimpleNamespace(
        image_height=33, image_width=33, Cx=10., Cy=16., Fx=30., Fy=32.,
        FoVx=2 * math.atan(settings.tanfovx), FoVy=2 * math.atan(settings.tanfovy),
        world_view_transform=settings.viewmatrix, full_proj_transform=settings.projmatrix,
        camera_center=settings.campos,
    )
    pc = SimpleNamespace(get_xyz=tensors["means3D"],
                         get_scaling_n_opacity_with_3D_filter=(tensors["scales"], tensors["opacities"]),
                         get_rotation=tensors["rotations"])
    score = load_geometry_score(torch, backend.GaussianRasterizationSettings, backend.GaussianRasterizer)
    weight = torch.zeros((33, 33), device="cuda")
    weight[16, 16] = 1
    output = score(camera, pc, SimpleNamespace(debug=True), settings.bg, 0, 2., .003, 1., 1., weight)
    assert output["weight_sum"].item() > .85
    torch.testing.assert_close(output["near_weight_sum"], output["weight_sum"])
    assert output["off_surface_sum"].item() == 0


def test_cuda_retry_recovers_outside_bracket_without_changing_valid_pixels(cuda_backend):
    torch, _ = cuda_backend
    default, inputs = cuda_scene(cuda_backend, scales=[[.15, .10, 1.2]])
    joint, joint_inputs = cuda_scene(cuda_backend, scales=[[.15, .10, 1.2]], expansions=2)
    base_output, retry_output = default(**inputs), joint(**joint_inputs)
    assert base_output[3][0, 16, 16] > .55
    assert base_output[2][0, 16, 16] == 0
    assert 1.1 < retry_output[2][0, 16, 16] < 1.5
    for index in (0, 3, 4):  # RGB, alpha and normal are not recomputed.
        assert torch.equal(base_output[index], retry_output[index])
    hit = base_output[2] > 0
    assert torch.equal(base_output[2][hit], retry_output[2][hit])


def test_cuda_valid_depth_gradient_and_joint_invalid_depth_guard(cuda_backend):
    torch, _ = cuda_backend
    rasterizer, tensors = cuda_scene(cuda_backend, expansions=2)
    output = rasterizer(**tensors)
    assert (output[1] > 0).all()
    assert output[2][0, 16, 16] > 0
    assert output[3][0, 16, 16] > .85
    output[2][0, 16, 16].backward()
    analytic = tensors["means3D"].grad[0, 2].item()
    assert analytic > .5
    baseline, baseline_inputs = cuda_scene(cuda_backend, expansions=0)
    baseline_output = baseline(**baseline_inputs)
    baseline_output[2][0, 16, 16].backward()
    assert torch.equal(baseline_output[2], output[2])
    for name in ("means3D", "opacities", "scales", "rotations"):
        torch.testing.assert_close(baseline_inputs[name].grad, tensors[name].grad, atol=1e-7, rtol=1e-6)
    values = []
    for offset in (-.001, .001):
        shifted, shifted_inputs = cuda_scene(cuda_backend, means=((0., 0., 2 + offset),), expansions=2)
        values.append(shifted(**shifted_inputs)[2][0, 16, 16].item())
    assert abs(analytic - (values[1] - values[0]) / .002) < .05
    assert all(tensor.grad is None or torch.isfinite(tensor.grad).all() for tensor in tensors.values())

    low, low_inputs = cuda_scene(cuda_backend, opacity=(.2,), expansions=2)
    low_output = low(**low_inputs)
    assert (low_output[1] > 0).all()
    assert .15 < low_output[3][0, 16, 16] < .25
    assert low_output[2][0, 16, 16] == 0
    low_output[2][0, 16, 16].backward()
    for name in ("means3D", "opacities", "scales", "rotations"):
        assert torch.isfinite(low_inputs[name].grad).all()
        assert torch.count_nonzero(low_inputs[name].grad) == 0

    coverage, coverage_inputs = cuda_scene(cuda_backend, opacity=(.2,), expansions=2)
    coverage_output = coverage(**coverage_inputs)
    (torch.relu(.8 - coverage_output[3][0, 16, 16]) ** 2).backward()
    assert coverage_inputs["opacities"].grad.item() < 0
    assert all(tensor.grad is None or torch.isfinite(tensor.grad).all() for tensor in coverage_inputs.values())
