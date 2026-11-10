"""Exercise joint continuation with actual CPU Torch tensors and Adam state.

Production definitions are loaded without importing the CUDA Scene stack.
The view sampler is the production ShuffledViewSampler; teacher I/O and costly
geometry audits are replaced by small stateful collaborators.
"""

import ast
import copy
import io
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def production_definitions(relative_path, namespace, names=None):
    path = ROOT / relative_path
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def runtime_api(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    namespace = {"json": json, "math": math, "random": random, "Path": Path,
                 "np": np, "torch": torch, "copy": copy}
    production_definitions("utils/virtual_geometry_config.py", namespace, {"config_identity"})
    import hashlib
    namespace["hashlib"] = hashlib
    production_definitions("utils/training_io.py", namespace, {"write_json"})
    production_definitions("utils/joint_runtime.py", namespace)
    production_definitions("utils/virtual_geometry_data.py", namespace, {"ShuffledViewSampler"})
    saved = namespace["capture_rng"]()
    yield SimpleNamespace(**namespace)
    namespace["restore_rng"](saved)


class FakeTeacher:
    def __init__(self, sampler_class, seed=19):
        self.identity = {"manifest_sha256": "frozen-teacher", "cache_identity": "fixed-inputs"}
        self.sampler = sampler_class(5, seed)
        self.load_calls = 0

    def __len__(self):
        return 5

    def state_dict(self):
        return {"identity": copy.deepcopy(self.identity), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state):
        self.load_calls += 1
        if state["identity"] != self.identity:
            raise ValueError("FakeTeacher identity changed")
        self.sampler.load_state_dict(state["sampler"])


class FakeTopology:
    def __init__(self):
        self.rng = np.random.default_rng(23)
        self.pending = True
        self.queue = {"surface-cell": {"views": [1, 4], "checks": 2}}
        self.audit_cursor = 3
        self.load_calls = 0

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "pending": self.pending,
                "queue": copy.deepcopy(self.queue), "audit_cursor": self.audit_cursor}

    def load_state_dict(self, state):
        self.load_calls += 1
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.pending = state["pending"]
        self.queue = copy.deepcopy(state["queue"])
        self.audit_cursor = state["audit_cursor"]

    def summary(self):
        return {"pending": self.pending, "queued_cells": len(self.queue)}


def make_model():
    xyz = torch.nn.Parameter(torch.tensor([[.1, .2, .3], [.4, .5, .6]]))
    opacity = torch.nn.Parameter(torch.tensor([[-1.], [-.5]]))
    sh = torch.nn.Parameter(torch.full((2, 3), .1))
    main = torch.optim.Adam([{"params": [xyz], "name": "xyz"},
                             {"params": [opacity], "name": "opacity"}], lr=.01)
    shoptimizer = torch.optim.Adam([{"params": [sh], "name": "f_rest"}], lr=.003)
    # Initialize genuine Adam moments before the checkpoint boundary.
    (xyz.square().sum() + opacity.square().sum() + sh.square().sum()).backward()
    main.step()
    shoptimizer.step()
    main.zero_grad(set_to_none=True)
    shoptimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(
        optimizer=main, shoptimizer=shoptimizer, xyz=xyz, opacity=opacity, sh=sh,
        filter_3D=torch.tensor([[.03], [.07]]),
        xyz_gradient_accum_abs=torch.tensor([[.2], [.4]]),
    )


@pytest.fixture
def runtime_factory(tmp_path, runtime_api):
    opened = []

    def create(config=None):
        output = tmp_path / f"segment-{len(opened)}"
        output.mkdir()
        config = config or {"teacher_manifest": "/teacher/manifest.json", "warm_interval": 4}
        teacher, topology = FakeTeacher(runtime_api.ShuffledViewSampler), FakeTopology()
        runtime = runtime_api.JointRuntime(config, teacher, topology, output, start_iteration=7)
        opened.append(runtime)
        return runtime

    yield create
    for runtime in opened:
        runtime.close()


def cameras():
    return [SimpleNamespace(source_image_name=f"folder/frame_{i}.png", image_name=f"frame_{i}")
            for i in range(4)]


def serialized(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False, map_location="cpu")


def draw_sequence(runtime, real_cameras, real_stack, count=17):
    sequence = []
    for _ in range(count):
        if not real_stack:
            real_stack = real_cameras.copy()
        camera = real_stack.pop(random.randrange(len(real_stack)))
        sequence.append((camera.source_image_name, runtime.teacher.sampler.next(),
                         random.random(), np.random.random(3).tolist(), torch.rand(3).tolist(),
                         int(runtime.topology.rng.integers(100000))))
    return sequence


def test_roundtrip_restores_real_virtual_rng_and_topology(runtime_api, runtime_factory):
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    runtime, model = runtime_factory(), make_model()
    real = cameras()
    for _ in range(3):
        runtime.teacher.sampler.next()
    runtime.stats = {"virtual_samples": 3, "weighted_loss_sums": {"depth": .6}, "last_iteration": 7}
    saved = serialized(runtime.capture(model, [real[3], real[1]]))
    assert saved["rng"]["cuda"] == []
    expected = draw_sequence(runtime, real, [real[3], real[1]])

    resumed, restored_model = runtime_factory(), make_model()
    random.random()
    np.random.rand(100)
    torch.rand(100)
    resumed.topology.pending = False
    resumed.topology.queue.clear()
    resumed.topology.audit_cursor = 0
    stack = resumed.restore(saved, restored_model, real)
    assert [camera.source_image_name for camera in stack] == saved["real_stack"]
    assert draw_sequence(resumed, real, stack) == expected
    assert resumed.topology.pending is True
    assert resumed.topology.queue == saved["topology"]["queue"]
    assert resumed.topology.audit_cursor == 3
    assert resumed.summary()["virtual_samples"] == 3
    assert resumed.summary()["segment_virtual_samples"] == 0


def test_pending_sh_survives_checkpoint_and_next_adam_update(runtime_factory):
    runtime, model = runtime_factory(), make_model()
    model.sh.grad = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
    saved = serialized(runtime.capture(model, cameras()[1:]))
    parameter_state = {name: getattr(model, name).detach().clone() for name in ("xyz", "opacity", "sh")}
    optimizer_state = copy.deepcopy(model.shoptimizer.state_dict())

    resumed, restored_model = runtime_factory(), make_model()
    for name, value in parameter_state.items():
        getattr(restored_model, name).data.copy_(value)
    restored_model.shoptimizer.load_state_dict(optimizer_state)
    restored_model.filter_3D.zero_()
    restored_model.xyz_gradient_accum_abs.zero_()
    resumed.restore(saved, restored_model, cameras())
    assert restored_model.xyz.grad is None and restored_model.opacity.grad is None
    torch.testing.assert_close(restored_model.sh.grad, model.sh.grad, rtol=0, atol=0)
    torch.testing.assert_close(restored_model.filter_3D, model.filter_3D, rtol=0, atol=0)
    torch.testing.assert_close(restored_model.xyz_gradient_accum_abs, model.xyz_gradient_accum_abs, rtol=0, atol=0)
    for candidate in (model, restored_model):
        (candidate.sh.square().sum() * .7).backward()
        candidate.shoptimizer.step()
        candidate.shoptimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(restored_model.sh, model.sh, rtol=0, atol=0)
    left, right = model.shoptimizer.state[model.sh], restored_model.shoptimizer.state[restored_model.sh]
    for name in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


def test_capture_and_restore_tensors_do_not_alias(runtime_factory):
    runtime, model = runtime_factory(), make_model()
    model.sh.grad = torch.ones_like(model.sh)
    saved = runtime.capture(model, cameras())
    model.sh.grad.fill_(8)
    model.filter_3D.fill_(9)
    model.xyz_gradient_accum_abs.fill_(10)
    assert (saved["pending_gradients"]["sh/f_rest"][0] == 1).all()
    assert (saved["filter_3D"] < 1).all()
    assert (saved["xyz_gradient_accum_abs"] < 1).all()
    resumed, restored = runtime_factory(), make_model()
    resumed.restore(saved, restored, cameras())
    saved["pending_gradients"]["sh/f_rest"][0].fill_(19)
    saved["filter_3D"].fill_(19)
    saved["xyz_gradient_accum_abs"].fill_(19)
    assert (restored.sh.grad == 1).all()
    assert (restored.filter_3D < 1).all()
    assert (restored.xyz_gradient_accum_abs < 1).all()


@pytest.mark.parametrize("field", ["schema_version", "config_sha256", "teacher"])
def test_identity_mismatch_rejected_before_collaborator_restore(runtime_factory, field):
    runtime, model = runtime_factory(), make_model()
    saved = serialized(runtime.capture(model, cameras()))
    if field == "schema_version":
        saved[field] = 99
    else:
        saved["identity"][field] = "different"
    resumed = runtime_factory()
    with pytest.raises(ValueError, match="identity changed"):
        resumed.restore(saved, make_model(), cameras())
    assert resumed.teacher.load_calls == resumed.topology.load_calls == 0


@pytest.mark.parametrize("names", [["missing.png"], ["folder/frame_1.png", "folder/frame_1.png"]])
def test_real_sampling_stack_rejects_unknown_or_duplicate_frames(runtime_factory, names):
    runtime, model = runtime_factory(), make_model()
    saved = serialized(runtime.capture(model, cameras()))
    saved["real_stack"] = names
    with pytest.raises(ValueError, match="sampling stack"):
        runtime_factory().restore(saved, make_model(), cameras())


@pytest.mark.parametrize("field,bad", [("filter_3D", "shape"), ("filter_3D", "nan"),
                                       ("xyz_gradient_accum_abs", "shape"), ("xyz_gradient_accum_abs", "nan")])
def test_filter_and_absolute_gradient_restore_validate_values(runtime_factory, field, bad):
    runtime, model = runtime_factory(), make_model()
    saved = serialized(runtime.capture(model, cameras()))
    saved[field] = torch.zeros((3, 1)) if bad == "shape" else torch.full_like(saved[field], float("nan"))
    with pytest.raises(ValueError, match="shape/value mismatch"):
        runtime_factory().restore(saved, make_model(), cameras())


@pytest.mark.parametrize("bad", ["missing", "extra", "arity", "shape", "nonfinite"])
def test_pending_gradient_restore_rejects_incompatible_payload(runtime_api, bad):
    model = make_model()
    model.sh.grad = torch.ones_like(model.sh)
    saved = runtime_api.pending_gradients(model)
    if bad == "missing":
        saved.pop("sh/f_rest")
    elif bad == "extra":
        saved["sh/unrecognized"] = [None]
    elif bad == "arity":
        saved["sh/f_rest"] = []
    elif bad == "shape":
        saved["sh/f_rest"] = [torch.zeros((9, 9))]
    else:
        saved["sh/f_rest"] = [torch.full_like(model.sh, float("inf"))]
    with pytest.raises(ValueError, match="gradient"):
        runtime_api.restore_pending_gradients(make_model(), saved)


def test_cuda_rng_device_count_mismatch_is_explicit(runtime_api, monkeypatch):
    saved = runtime_api.capture_rng()
    saved["cuda"] = [torch.zeros(8, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="same CUDA device count"):
        runtime_api.restore_rng(saved)


def test_production_mode3_restore_binds_optimizer_to_restored_appearance():
    """The restored model and its optimizer must share the same Parameter."""
    namespace = {"torch": torch, "nn": torch.nn}
    production_definitions("scene/gaussian_model.py", namespace,
                           {"_restore_saturated_opacity_logits", "_require_finite_restore_values"})
    path = ROOT / "scene/gaussian_model.py"
    model_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
    method = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "restore")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class Harness:
        App_model = SimpleNamespace(PGSR=3, GOF=2, PPISP=4)
        restore = namespace["restore"]

        def __init__(self, appearance):
            self.app_model = self.App_model.PGSR
            self._appearance_embeddings = appearance
            self._xyz = torch.nn.Parameter(torch.zeros((2, 3)))
            self._features_rest = torch.nn.Parameter(torch.zeros((2, 3)))

        def training_setup(self, options):
            self.optimizer = torch.optim.Adam([
                {"params": [self._xyz], "name": "xyz"},
                {"params": [self._appearance_embeddings], "name": "appearance_embeddings"},
            ], lr=.01)
            self.shoptimizer = torch.optim.Adam([
                {"params": [self._features_rest], "name": "f_rest"},
            ], lr=.01)

    appearance = torch.nn.Parameter(torch.tensor([[.2, .1], [.4, -.2]]))
    source = Harness(appearance)
    source.training_setup(None)
    source._appearance_embeddings.square().sum().backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    def parameter(shape):
        return torch.nn.Parameter(torch.zeros(shape))

    checkpoint = (
        0, 0, source._xyz, parameter((2, 1, 3)), source._features_rest,
        parameter((2, 3)), parameter((2, 4)), parameter((2, 1)),
        parameter((2, 0, 3)), parameter((2, 0)), parameter((2, 0, 3)),
        torch.zeros(2), torch.zeros((2, 1)), torch.ones((2, 1)),
        source.optimizer.state_dict(), source.shoptimizer.state_dict(), 1., 3, None, appearance,
    )
    resumed = Harness(torch.nn.Parameter(torch.zeros_like(appearance)))
    resumed.restore(serialized(checkpoint), None)
    group = next(group for group in resumed.optimizer.param_groups if group["name"] == "appearance_embeddings")
    assert group["params"][0] is resumed._appearance_embeddings
    before = resumed._appearance_embeddings.detach().clone()
    resumed._appearance_embeddings.sum().backward()
    resumed.optimizer.step()
    resumed.optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, resumed._appearance_embeddings)
    assert resumed._appearance_embeddings.grad is None
