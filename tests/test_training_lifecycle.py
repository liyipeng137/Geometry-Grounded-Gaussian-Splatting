"""Run the real orchestration on CPU fakes, without importing Torch/CUDA.

The optimizer schedule is extracted and executed from GaussianModel itself.
These checks catch accidental reordering, extra gradient clearing, unprotected
neighbor payloads, and querying CUDA event timing before completion.
"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def execute_definitions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


def make_harness(start=0, stop=3, multiview=False, failure_at=None):
    trace = []
    state = SimpleNamespace(iteration=start, models=[], scenes=[], loggers=[], caches=[], progress=[])

    def event(name, *details):
        trace.append((state.iteration, name, *details))

    class Optimizer:
        def __init__(self, name):
            self.name = name
            self.pending = 0
            self.steps = []
            self.clears = 0

        def step(self):
            self.steps.append((state.iteration, self.pending))
            event(self.name + ".step", self.pending)

        def zero_grad(self, set_to_none=True):
            self.pending = 0
            self.clears += 1
            event(self.name + ".zero_grad")

    schedule_namespace = execute_definitions(
        "scene/gaussian_model.py", {"optimizer_step"}, {}, class_name="GaussianModel"
    )

    class Model:
        def __init__(self, *args):
            self.optimizer = Optimizer("main")
            self.shoptimizer = Optimizer("sh")
            self.ppisp = None
            self.get_xyz = [0, 1]
            self.dense_optimizer_updates = False
            self.optimizer_step = MethodType(schedule_namespace["optimizer_step"], self)
            state.models.append(self)

        def training_setup(self, opt):
            event("training_setup")

        def update_learning_rate(self, iteration):
            assert iteration == state.iteration
            event("learning_rate")

        def unlockSGdegree(self, degree):
            event("unlock_sg", degree)

        def oneupSHdegree(self):
            event("unlock_sh")

        def reset_3D_filter(self):
            event("filter")

        def compute_3D_filter(self, cameras):
            event("filter")

        def zero_grad(self, *args, **kwargs):
            raise AssertionError("The training loop must not clear accumulated model gradients")

    cameras = [
        SimpleNamespace(
            image_name=f"frame_{index}.png", nearest_id=[1 - index],
            has_depth_prior=True, has_normal_prior=True, depth_prior=None, normal_prior=None,
        ) for index in range(2)
    ]

    class Scene:
        def __init__(self, dataset, model):
            self.gaussians = model
            self.model_path = dataset.model_path
            self.ply_states = {}
            state.scenes.append(self)

        def getTrainCameras(self):
            return cameras

        def getTestCameras(self):
            return []

        def save(self, iteration):
            self.ply_states[iteration] = len(self.gaussians.optimizer.steps)
            event("save_ply")

    class Cache:
        def __init__(self, budget):
            self.calls = []
            state.caches.append(self)

        def ensure(self, camera, keep=()):
            self.calls.append((state.iteration, camera.image_name, tuple(item.image_name for item in keep)))
            event("cache", camera.image_name, tuple(item.image_name for item in keep))
            return camera

    class CUDAEvent:
        instances = 0

        def __init__(self, enable_timing):
            self.name = "start_event" if CUDAEvent.instances == 0 else "end_event"
            CUDAEvent.instances += 1
            self.recorded = False
            self.ready = False

        def record(self):
            if self.name == "start_event":
                state.iteration += 1
            self.recorded = True
            self.ready = False
            event(self.name + ".record")

        def synchronize(self):
            assert self.recorded
            self.ready = True
            event(self.name + ".synchronize")

        def elapsed_time(self, end):
            assert self.recorded and end.recorded and end.ready
            event("elapsed_time")
            return 1.25

    class Logger:
        def __init__(self, scene, opt, args, first_iteration, writer):
            self.stats = {"status": "running"}
            self.ema = {name: 0.0 for name in ("total", "depth_prior", "depth_normal", "normal_prior", "ncc")}
            self.flushes = []
            self.closed = False
            state.loggers.append(self)

        def record_loss(self, iteration, camera, losses, opt, schedule, elapsed, gaussians):
            assert elapsed == 1.25
            event("record_loss", camera.image_name)

        def flush_stats(self, model):
            self.flushes.append(dict(self.stats))
            event("flush_stats")

        def close(self):
            self.closed = True
            event("logger.close")

    class Progress:
        def __init__(self, **kwargs):
            self.closed = False
            self.updates = 0
            state.progress.append(self)

        def update(self, count):
            self.updates += count

        def set_postfix(self, value):
            pass

        def close(self):
            self.closed = True
            event("progress.close")

    class Loss:
        def backward(self):
            for optimizer in (state.models[0].optimizer, state.models[0].shoptimizer):
                optimizer.pending += 1
            event("backward")

    patchmatch_calls = []

    def patchmatch_factory(*args, **kwargs):
        assert multiview, "Disabled multi-view objectives must not construct PatchMatch"
        event("patchmatch.construct")
        return lambda camera, nearest: patchmatch_calls.append((camera.image_name, nearest.image_name if nearest else None))

    def compute_losses(buffers, camera, model, opt, schedule, *, patchmatch, nearest_camera, iteration):
        if failure_at == iteration:
            raise RuntimeError("synthetic training failure")
        if schedule.reg_enabled and (schedule.multi_view_ncc_weight > 0 or opt.lambda_multi_view_geo > 0):
            assert patchmatch is not None and nearest_camera is not None
            patchmatch(camera, nearest_camera)
        else:
            assert nearest_camera is None
            if not multiview:
                assert patchmatch is None
        event("loss", camera.image_name)
        return SimpleNamespace(total=Loss())

    checkpoint_states = {}

    def checkpoint(model, scene, iteration, args):
        checkpoint_states[iteration] = len(model.optimizer.steps)
        event("checkpoint")

    namespace = {
        "math": math, "dataclass": dataclass,
        "torch": SimpleNamespace(
            tensor=lambda *args, **kwargs: object(), float32="float32", no_grad=nullcontext,
            cuda=SimpleNamespace(Event=CUDAEvent),
        ),
        "load_checkpoint": lambda path: (None, start, None),
        "prepare_output": lambda *args: None,
        "restore_checkpoint": lambda *args: None,
        "GaussianModel": Model, "Scene": Scene, "FramePayloadCache": Cache,
        "PatchMatch": patchmatch_factory, "TrainingLogger": Logger, "tqdm": Progress,
        "randint": lambda lower, upper: 0, "sample": lambda candidates, count: candidates[:count],
        "render": lambda camera, *args, **kwargs: {"camera": camera},
        "compute_training_losses": compute_losses,
        "evaluate_views": lambda *args: event("evaluate"),
        "update_topology": lambda *args: event("topology"),
        "save_checkpoint": checkpoint,
    }
    execute_definitions("utils/training_losses.py", {"LossSchedule", "loss_schedule", "ramp_weight", "normal_prior_weight", "multi_view_ncc_weight"}, namespace)
    execute_definitions("train.py", {"validate_options", "validate_training_priors", "refresh_filter", "training"}, namespace)
    opt = SimpleNamespace(
        iterations=stop, densification_interval=100, opacity_reset_interval=3000,
        opacity_reset_min_remaining=0, depth_prior_ramp_iters=0, regularization_from_iter=0,
        normal_prior_from_iter=0, depth_prior_from_iter=0, lambda_depth_normal=0.0,
        lambda_normal_prior=0.0, lambda_depth_prior=0.0, lambda_multi_view_ncc=0.02 if multiview else 0.0,
        lambda_multi_view_geo=0.0, lambda_mask=0.0, lambda_dssim=0.2, real_depth_loss_mode="absolute",
        real_depth_relative_epsilon=0.01, real_depth_relative_floor=0.05,
        multi_view_patch_size=3, multi_view_pixel_noise_th=1.0,
    )
    args = SimpleNamespace(
        start_checkpoint=None, training_seed=0, debug_images=False, debug_from=-1, log_interval=1,
        test_iterations=[start + 1, stop], save_iterations=[start + 1, stop],
        checkpoint_iterations=[start + 1, stop],
    )
    dataset = SimpleNamespace(
        sh_degree=3, sg_degree=0, frame_gpu_budget=1, white_background=False,
        disable_filter3D=True, kernel_size=0.0, model_path="unused-output",
    )
    return SimpleNamespace(
        run=lambda: namespace["training"](dataset, opt, SimpleNamespace(debug=False), args),
        state=state, trace=trace, args=args, patchmatch_calls=patchmatch_calls,
        checkpoint_states=checkpoint_states, cameras=cameras,
    )


def test_lifecycle_preserves_fastgs_accumulation_save_order_and_final_skip():
    harness = make_harness(start=14999, stop=15040)
    harness.args.test_iterations = harness.args.save_iterations = harness.args.checkpoint_iterations = [15008, 15040]
    harness.run()
    model = harness.state.models[0]
    # The real FastGS method receives every non-final iteration. Only its own
    # update boundary clears gradients, including accumulation across 15k.
    assert model.optimizer.steps == [(15000, 1), (15008, 8)]
    assert model.shoptimizer.steps == [(15008, 9)]
    assert model.optimizer.clears == 2
    assert model.shoptimizer.clears == 1
    assert model.optimizer.pending == model.shoptimizer.pending == 32
    assert harness.state.scenes[0].ply_states == {15008: 1, 15040: 2}
    assert harness.checkpoint_states == {15008: 2, 15040: 2}

    events = [row[1] for row in harness.trace if row[0] == 15008]
    expected_order = [
        "backward", "end_event.record", "end_event.synchronize", "elapsed_time", "record_loss",
        "evaluate", "save_ply", "topology", "main.step", "checkpoint",
    ]
    assert [events.index(name) for name in expected_order] == sorted(events.index(name) for name in expected_order)
    assert harness.state.loggers[0].flushes[-1]["status"] == "completed"
    assert harness.state.loggers[0].closed and harness.state.progress[0].closed


def test_disabled_patchmatch_keeps_single_view_sampling_and_cache_calls():
    harness = make_harness()
    harness.run()
    assert harness.patchmatch_calls == []
    assert all(not row[2] for row in harness.state.caches[0].calls)
    sampled = [row[2] for row in harness.trace if row[1] == "loss"]
    assert sampled == ["frame_0.png", "frame_1.png", "frame_0.png"]


def test_neighbor_payload_load_keeps_current_camera_resident():
    harness = make_harness(start=7000, stop=7002, multiview=True)
    harness.run()
    assert harness.patchmatch_calls == [("frame_0.png", "frame_1.png"), ("frame_1.png", "frame_0.png")]
    assert harness.state.caches[0].calls == [
        (7001, "frame_0.png", ()), (7001, "frame_1.png", ("frame_0.png",)),
        (7002, "frame_1.png", ()), (7002, "frame_0.png", ("frame_1.png",)),
    ]


def test_zero_scheduled_ncc_does_not_sample_a_neighbor():
    harness = make_harness(multiview=True)
    harness.run()
    assert harness.patchmatch_calls == []
    assert all(not row[2] for row in harness.state.caches[0].calls)


@pytest.mark.parametrize("prior", ["normal", "depth"])
def test_missing_prior_fails_before_optimizer_even_when_weights_are_zero(prior):
    harness = make_harness()
    setattr(harness.cameras[1], f"has_{prior}_prior", False)
    with pytest.raises(ValueError, match="frame_1.png") as error:
        harness.run()
    assert prior in str(error.value)
    assert all(row[1] not in ("training_setup", "loss", "backward") for row in harness.trace)


def test_exception_marks_failed_and_closes_progress_and_logger():
    harness = make_harness(failure_at=2)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        harness.run()
    logger = harness.state.loggers[0]
    assert logger.flushes[-1]["status"] == "failed"
    assert "synthetic training failure" in logger.flushes[-1]["error"]
    assert logger.closed and harness.state.progress[0].closed
    assert 3 not in harness.checkpoint_states
