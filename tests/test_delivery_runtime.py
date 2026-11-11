"""Production I/O switches, binary PLY parity, partial payloads and scalar logs."""

import ast
from collections import OrderedDict
import importlib.util
import io
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace, owner=None):
    source = ROOT/path
    body = ast.parse(source.read_text()).body
    if owner:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    nodes = [node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("rows", [0, 1, 37])
@pytest.mark.parametrize("widths", [(3, 3, 3, 45, 1, 3, 4, 0, 0, 0, 1), (3, 3, 3, 45, 1, 3, 4)])
def test_vectorized_ply_records_match_original_tuple_conversion_bytes(rows, widths):
    helper = definitions("scene/gaussian_model.py", {"_ply_elements"}, {"np": np})["_ply_elements"]
    rng = np.random.default_rng(91)
    columns = [rng.standard_normal((rows, width)).astype(np.float32) for width in widths]
    if rows:
        columns[0][0, 0] = -0.0
    dtype = [(f"property_{i}", "f4") for i in range(sum(widths))]
    packed = np.concatenate(columns, axis=1)
    expected = np.empty(rows, dtype=dtype)
    expected[:] = list(map(tuple, packed))
    actual = helper(columns, dtype)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_vectorized_ply_keeps_header_and_file_bytes():
    plyfile = pytest.importorskip("plyfile")
    helper = definitions("scene/gaussian_model.py", {"_ply_elements"}, {"np": np})["_ply_elements"]
    columns = [np.array([[1.234567890123, -0., .5], [3., -4., 8.]], np.float64), np.ones((2, 1), np.float32)]
    dtype = [(name, "f4") for name in ("x", "y", "z", "opacity")]
    old = np.empty(2, dtype=dtype)
    old[:] = list(map(tuple, np.concatenate(columns, axis=1)))
    result = []
    for data in (old, helper(columns, dtype)):
        buffer = io.BytesIO()
        plyfile.PlyData([plyfile.PlyElement.describe(data, "vertex")]).write(buffer)
        result.append(buffer.getvalue())
    assert result[0] == result[1]


@pytest.fixture
def payload_api():
    fields = ("original_image", "gt_mask", "normal_prior", "normal_prior_mask", "depth_prior", "depth_confidence")
    ns = {"OrderedDict": OrderedDict, "_PAYLOAD_FIELDS": fields}
    definitions("scene/cameras.py", {"FramePayloadCache"}, ns)
    definitions("scene/cameras.py", {"load_payload_to_device", "release_payload_from_device"}, ns, owner="Camera")

    def make_camera():
        calls = []

        class Payload:
            def __init__(self, name):
                self.name = name

            def to(self, device, non_blocking):
                assert device == "cuda" and non_blocking
                calls.append(self.name)
                return object()

        camera = SimpleNamespace(lazy_payload=True, data_device="cuda", calls=calls)
        for field in fields:
            setattr(camera, field, None)
            setattr(camera, field+"_cpu", Payload(field))
        for method in ("load_payload_to_device", "release_payload_from_device"):
            setattr(camera, method, MethodType(ns[method], camera))
        return camera

    return SimpleNamespace(Cache=ns["FramePayloadCache"], make_camera=make_camera, fields=fields)


def test_rgb_neighbor_cache_hit_is_upgraded_when_it_becomes_real_frame(payload_api):
    cache, camera = payload_api.Cache(2), payload_api.make_camera()
    cache.ensure(camera, fields=("original_image",))
    rgb = camera.original_image
    assert camera.calls == ["original_image"]
    assert camera.normal_prior is None and camera.depth_prior is None
    cache.ensure(camera)
    assert camera.original_image is rgb
    assert camera.calls == list(payload_api.fields)
    assert all(getattr(camera, name) is not None for name in payload_api.fields)
    cache.ensure(camera)
    assert camera.calls == list(payload_api.fields)


def test_neighbor_eviction_preserves_keep_and_cpu_payload(payload_api):
    cache = payload_api.Cache(1)
    current, neighbor, other = [payload_api.make_camera() for _ in range(3)]
    cpu = neighbor.depth_prior_cpu
    cache.ensure(current)
    cache.ensure(neighbor, keep=(current,), fields=("original_image",))
    assert current.depth_prior is not None and neighbor.depth_prior is None
    cache.ensure(current)
    cache.ensure(other, keep=(current,), fields=("original_image",))
    assert current.normal_prior is not None and other.original_image is not None
    assert neighbor.original_image is None and neighbor.depth_prior_cpu is cpu
    cache.clear()
    assert current.original_image is None and other.original_image is None


def test_unknown_payload_field_is_rejected(payload_api):
    with pytest.raises(ValueError, match="Unknown camera payload fields"):
        payload_api.make_camera().load_payload_to_device(fields=("depth_typo",))


@pytest.mark.parametrize("compact", [False, True])
def test_cli_delivery_flags_keep_legacy_defaults_without_running_training(monkeypatch, compact):
    spec = importlib.util.spec_from_file_location("delivery_arguments", ROOT/"arguments/__init__.py")
    args_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args_module)
    called = []
    ns = {name: getattr(args_module, name) for name in ("ArgumentParser", "ModelParams", "OptimizationParams", "PipelineParams")}
    ns.update(validate_options=lambda *args: None, validate_initialization_options=lambda *args: None,
              safe_state=lambda *args, **kwargs: None,
              torch=SimpleNamespace(autograd=SimpleNamespace(set_detect_anomaly=lambda value: None)),
              training=lambda dataset, opt, pipe, args: called.append(args))
    definitions("train.py", {"main"}, ns)
    command = ["train.py", "--iterations", "32", "--checkpoint_iterations", "16"]
    if compact:
        command += ["--no_viewer_ply", "--no_final_checkpoint", "--no_step_timing", "--compact_virtual_log", "--keep_last_checkpoint"]
    monkeypatch.setattr("sys.argv", command)
    ns["main"]()
    args = called[0]
    assert args.save_viewer_ply is (not compact)
    assert args.step_timing is (not compact)
    assert args.detailed_virtual_log is (not compact)
    assert args.save_final_checkpoint is (not compact)
    assert args.keep_last_checkpoint is compact
    assert args.checkpoint_iterations == ([16] if compact else [16, 32])
    assert args.save_iterations == [32]


@pytest.fixture
def logger_api():
    torch = pytest.importorskip("torch")
    ns = {"torch": torch, "json": json, "Path": Path, "RECIPE": "test", "RESUME_POLICY": "test"}
    definitions("utils/training_io.py", {"write_json", "TrainingLogger"}, ns)
    return SimpleNamespace(**ns)


def test_device_accumulated_logger_matches_reference_sums_and_ema(logger_api, tmp_path, monkeypatch):
    torch = logger_api.torch
    model = SimpleNamespace(optimizer=None, shoptimizer=None, ppisp=None, get_xyz=[0, 1])
    scene = SimpleNamespace(model_path=str(tmp_path), gaussians=model, getTrainCameras=lambda: [0], getTestCameras=lambda: [])
    opt = SimpleNamespace(iterations=5, densify_until_iter=3, opacity_reset_min_remaining=0)
    args = SimpleNamespace(log_interval=3, training_seed=0, detailed_virtual_log=False)
    logger = logger_api.TrainingLogger(scene, opt, args, 0, None)
    camera, schedule = SimpleNamespace(image_name="real"), SimpleNamespace(normal_prior_weight=.1, multi_view_ncc_weight=.2)
    sums, ema, copies = {}, {}, []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(value, *args, **kwargs):
        copies.append(value.numel())
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    parameter = torch.tensor(.13, requires_grad=True)
    rng = torch.get_rng_state().clone()
    for iteration in range(1, 6):
        raw = {"rgb": parameter*iteration, "depth_prior": torch.tensor(.03*iteration)}
        weighted = {"rgb": raw["rgb"], "depth_prior": raw["depth_prior"]*.4}
        total = sum(weighted.values())
        losses = SimpleNamespace(total=total, raw_terms=lambda: raw, weighted_terms=lambda *unused: weighted)
        for name, value in {**raw, "total": total}.items():
            scalar = float(value.detach())
            sums[name] = sums.get(name, 0.)+scalar
            ema[name] = .4*scalar+.6*ema.get(name, 0.)
        logger.record_loss(iteration, camera, losses, opt, schedule, None, model)
        if iteration == 2:
            assert copies == []
    logger.flush_stats(model)
    assert len(copies) == 2
    assert logger.loss_sums == pytest.approx(sums, abs=1e-14)
    assert logger.ema == pytest.approx(ema, abs=1e-14)
    assert parameter.grad is None and not logger._sum_values.requires_grad
    assert torch.equal(torch.get_rng_state(), rng)
    rows = [json.loads(row) for row in (tmp_path/"loss.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [3, 5]
    assert all(row["forward_backward_ms"] is None for row in rows)
    logger.close()


def test_compact_real_logger_still_rejects_nonfinite_unused_raw_term(logger_api, tmp_path):
    torch = logger_api.torch
    model = SimpleNamespace(optimizer=None, shoptimizer=None, ppisp=None, get_xyz=[0])
    scene = SimpleNamespace(model_path=str(tmp_path), gaussians=model, getTrainCameras=lambda: [0], getTestCameras=lambda: [])
    opt = SimpleNamespace(iterations=20, densify_until_iter=10, opacity_reset_min_remaining=0)
    logger = logger_api.TrainingLogger(scene, opt, SimpleNamespace(log_interval=100, training_seed=0, detailed_virtual_log=False), 0, None)
    losses = SimpleNamespace(total=torch.tensor(1.), raw_terms=lambda: {"unused": torch.tensor(float("nan"))},
                             weighted_terms=lambda *args: {"rgb": torch.tensor(1.)})
    with pytest.raises(FloatingPointError):
        logger.record_loss(1, SimpleNamespace(image_name="real"), losses, opt, None, None, model)
    assert logger.stats["real_samples"] == 0
    logger.close()
