"""Budget, state identity, seed geometry and filter fallback contracts."""
import ast
from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def allocation():
    path = ROOT / "utils/geometry_topology.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "allocate_growth")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["allocate_growth"]


def pure_helper(name):
    path = ROOT / "utils/geometry_topology.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_source_quotas_borrow_without_inventing_candidates():
    allocate = allocation()
    assert allocate(100, 100, 100, 20) == (10, 5, 5)
    assert allocate(0, 100, 0, 20) == (0, 20, 0)
    assert allocate(100, 0, 0, 20) == (20, 0, 0)
    assert allocate(0, 0, 100, 20) == (0, 0, 20)


def test_hard_growth_budget_for_random_source_counts():
    allocate, rng = allocation(), np.random.default_rng(4)
    for _ in range(500):
        counts, available = rng.integers(0, 1000, size=3), int(rng.integers(0, 1500))
        chosen = allocate(*counts, available)
        assert all(0 <= x <= y for x, y in zip(chosen, counts))
        assert sum(chosen) == min(available, int(counts.sum()))
    assert allocate(100, 100, 100, -1) == (0, 0, 0)


def test_variable_spacing_dedup_uses_distance_not_incompatible_cell_numbers():
    dedup = pure_helper("spacing_deduplicate")
    xyz = np.array([[.99, 0., 0.], [1.01, 0., 0.], [9.9, 0., 0.], [10.5, 0., 0.]])
    spacing = np.array([1., .1, 10., .2])
    # First two lie almost on top of each other despite different x/h cells.
    # Point 2 is far enough, despite sharing the old floor(x/h)=0 with point 0.
    np.testing.assert_array_equal(dedup(xyz, spacing, 10), [0, 2])
    np.testing.assert_array_equal(dedup(xyz+[.4, .7, 100.], spacing, 10), [0, 2])
    np.testing.assert_array_equal(dedup(xyz, spacing, 1), [0])


def test_dedup_returned_pairs_obey_both_local_spacings():
    rng = np.random.default_rng(12)
    xyz, spacing = rng.normal(size=(200, 3)), rng.uniform(.01, 1., 200)
    chosen = pure_helper("spacing_deduplicate")(xyz, spacing, 80)
    assert len(chosen) <= 80
    for position, first in enumerate(chosen):
        for second in chosen[position+1:]:
            assert np.linalg.norm(xyz[first]-xyz[second]) >= .75*max(spacing[first], spacing[second])-1e-12


def test_seed_witnesses_need_distinct_recent_directions_not_just_frame_ids():
    choose = pure_helper("choose_seed_witnesses")
    target = np.array([0., 0., 2.])
    centers = np.array([[-.1, 0., 0.], [.1, 0., 0.], [-.1, 0., 0.]])
    assert choose(target, {0: 3, 2: 3}, centers, 3) == []
    assert choose(target, {0: 1, 1: 3}, centers, 3) == []
    assert choose(target, {0: 2, 1: 3}, centers, 3) == [0, 1]
    assert choose(target, {0: 2, 1: 3}, centers, 5) == []


def test_queue_expiry_releases_stale_capacity_without_discarding_recent_gaps():
    path = ROOT / "utils/geometry_topology.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "GeometryTopology")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_expire_seed_queue")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    recent = {"epoch": 3, "views": {0: 3}}
    active = {"epoch": 5, "views": {0: 4, 1: 5}}
    manager = SimpleNamespace(audit_epoch=5, queue={(0,): {"epoch": 2}, (1,): recent, (2,): active}, counts=Counter())
    expire = namespace["_expire_seed_queue"]
    assert expire(manager) == 1
    assert manager.queue == {(1,): recent, (2,): active}
    assert manager.queue[(1,)] is recent and manager.queue[(2,)] is active
    assert manager.counts["expired_seed_candidates"] == 1
    assert expire(manager) == 0
    manager.audit_epoch = 6
    assert expire(manager) == 1
    assert manager.queue == {(2,): active}
    assert manager.counts["expired_seed_candidates"] == 2


@pytest.fixture
def api():
    pytest.importorskip("torch", reason="Tensor contracts run in the PyTorch validation environment")
    spec = importlib.util.spec_from_file_location("topology_under_test", ROOT / "utils/geometry_topology.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model(api):
    torch = api.torch

    class Model:
        @property
        def get_xyz(self):
            return self._xyz

        @property
        def get_scaling(self):
            return self._scaling.exp()

        @property
        def get_opacity(self):
            return self._opacity.sigmoid()

        @property
        def get_opacity_with_3D_filter(self):
            return self.get_opacity*api.filter_coefficient(self.get_scaling, self.filter_3D)[:, None]

        def _iter_optimizers(self):
            return [self.optimizer, self.shoptimizer]

        def apply_topology_transaction(self, *args):
            return api.apply_gaussian_transaction(self, *args)

        def compute_joint_3D_filter(self, cameras, metadata, records):
            self.filter_3D = api.joint_filter_for_points(self.get_xyz, cameras, metadata["root_gen"],
                metadata["h_ref"], metadata["witness"], records, self.filter_3D)

    obj = Model()
    sizes = {"xyz": (3, 3), "f_dc": (3, 1, 3), "f_rest": (3, 15, 3), "opacity": (3, 1),
             "scaling": (3, 3), "rotation": (3, 4), "sg_axis": (3, 0, 3), "sg_sharpness": (3, 0), "sg_color": (3, 0, 3)}
    for name, field in api.PARAMETERS.items():
        values = torch.zeros(sizes[name])
        if name == "xyz":
            values[:, 0], values[:, 2] = torch.arange(3), 1
        if name == "rotation":
            values[:, 0] = 1
        if name == "scaling":
            values.fill_(np.log(.2))
        setattr(obj, field, torch.nn.Parameter(values))
    obj.appearance = torch.nn.Parameter(torch.zeros(2, 2))
    obj.app_model = SimpleNamespace(name="PGSR")
    groups = [{"params": [getattr(obj, field)], "name": name} for name, field in api.PARAMETERS.items() if name != "f_rest"]
    groups.append({"params": [obj.appearance], "name": "appearance_embeddings"})
    obj.optimizer = torch.optim.Adam(groups, lr=.001)
    obj.shoptimizer = torch.optim.Adam([{"params": [obj._features_rest], "name": "f_rest"}], lr=.001)
    sum(getattr(obj, field).square().sum() for field in api.PARAMETERS.values()).backward()
    for optimizer in obj._iter_optimizers():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    obj.filter_3D = torch.full((3, 1), .03)
    obj.xyz_gradient_accum = torch.arange(3).float()[:, None]
    obj.xyz_gradient_accum_abs = torch.arange(3).float()[:, None]+1
    obj.denom, obj.max_radii2D = torch.ones(3, 1), torch.arange(3).float()
    return obj


def payload(api, model, n):
    return {name: getattr(model, field).detach()[:n].clone() for name, field in api.PARAMETERS.items()}


def test_atomic_survivor_mapping_preserves_adam_and_camera_appearance(api, model):
    torch, old, appearance = api.torch, model._xyz, model.appearance
    old_state = {k: v.clone() for k, v in model.optimizer.state[old].items()}
    result = api.apply_gaussian_transaction(model, torch.tensor([2, 0]), payload(api, model, 1), torch.tensor([[.04]]),
        {"id": torch.tensor([10, 11, 12]), "grace": torch.tensor([100, 200, 300])},
        {"id": torch.tensor([13]), "grace": torch.tensor([500])}, 3)
    assert model._xyz is not old
    assert model.optimizer.param_groups[-1]["params"][0] is appearance
    torch.testing.assert_close(result["id"], torch.tensor([12, 10, 13]))
    torch.testing.assert_close(result["grace"], torch.tensor([300, 100, 500]))
    torch.testing.assert_close(model.filter_3D[:, 0], torch.tensor([.03, .03, .04]))
    state = model.optimizer.state[model._xyz]
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(state[key][:2], old_state[key][[2, 0]])
        assert not state[key][2].any()
    torch.testing.assert_close(state["step"], old_state["step"])
    assert old not in model.optimizer.state


def test_empty_transaction_is_identity_even_with_pending_gradients(api, model):
    torch, old = api.torch, model._xyz
    model._features_rest.grad = torch.ones_like(model._features_rest)
    metadata = {"id": torch.arange(3)}
    result = api.apply_gaussian_transaction(model, torch.arange(3), payload(api, model, 0), torch.empty(0, 1),
        metadata, {"id": torch.empty(0, dtype=torch.long)}, 3)
    assert result is metadata and model._xyz is old
    assert model._features_rest.grad is not None


def test_pending_sh_gradient_prevents_nonempty_transaction(api, model):
    torch, old = api.torch, model._xyz
    model._features_rest.grad = torch.ones_like(model._features_rest)
    with pytest.raises(RuntimeError, match="pending f_rest"):
        api.apply_gaussian_transaction(model, torch.tensor([0, 1]), payload(api, model, 0), torch.empty(0, 1),
            {"id": torch.arange(3)}, {"id": torch.empty(0, dtype=torch.long)}, 3)
    assert model._xyz is old


def test_invalid_payload_and_hard_cap_fail_before_commit(api, model):
    torch, old = api.torch, model._xyz
    meta, extra = {"id": torch.arange(3)}, {"id": torch.tensor([4])}
    with pytest.raises(ValueError, match="hard Gaussian"):
        api.apply_gaussian_transaction(model, torch.arange(3), payload(api, model, 1), torch.ones(1, 1), meta, extra, 3)
    bad = payload(api, model, 1)
    bad["rotation"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="rotation"):
        api.apply_gaussian_transaction(model, torch.tensor([0, 1]), bad, torch.ones(1, 1), meta, extra, 3)
    assert model._xyz is old


def test_seed_normal_axes_filtered_opacity_and_higher_sh(api, model):
    torch = api.torch
    normals, h, filtering = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.]]), torch.full((3,), .1), torch.full((3, 1), .01)
    seeds, valid = api.make_seed_payload(model, torch.zeros(3, 3), normals, h, filtering, torch.full((3, 3), .5))
    assert valid.all()
    rotation = api.rotations(seeds["rotation"])
    torch.testing.assert_close(rotation[:, :, 2], normals, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(3), atol=1e-6, rtol=0)
    scales = seeds["scaling"].exp()
    assert (scales[:, 2] < scales[:, 0]).all()
    torch.testing.assert_close(seeds["opacity"].sigmoid()[:, 0]*api.filter_coefficient(scales, filtering), torch.full((3,), .25))
    assert not seeds["f_dc"].any() and not seeds["f_rest"].any()
    assert seeds["sg_axis"].shape == (3, 0, 3)


def test_incompatible_seed_filter_is_rejected(api, model):
    torch = api.torch
    _, valid = api.make_seed_payload(model, torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]),
        torch.tensor([.01]), torch.tensor([[10.]]), torch.full((1, 3), .5))
    assert not valid[0]


def test_gen_only_filter_fallback_is_bounded_by_local_spacing(api):
    torch = api.torch
    camera = SimpleNamespace(R=torch.eye(3), T=torch.zeros(3), Cx=31.5, Cy=31.5, Fx=40., Fy=40., image_width=64, image_height=64)
    pose = np.eye(4)
    pose[0, 3] = 100
    virtual = {"K": [[40, 0, 32], [0, 40, 32], [0, 0, 1]], "c2w": pose.tolist(), "width": 64, "height": 64}
    result = api.joint_filter_for_points(torch.tensor([[0., 0., 1.], [100., 0., 1.], [100., 0., 1.]]), [camera],
        torch.tensor([False, True, False]), torch.tensor([0., .02, 0.]), torch.tensor([-1, 0, -1]), [virtual], torch.full((3, 1), 999.))
    assert float(result[0]) == pytest.approx(np.sqrt(.2)/40)
    assert float(result[1]) == pytest.approx(.002)
    assert float(result[2]) == pytest.approx(np.sqrt(.2)/40)
    with pytest.raises(ValueError, match="finite filter support"):
        api.joint_filter_for_points(torch.tensor([[100., 0., 1.]]), [camera], torch.tensor([True]),
            torch.tensor([.1]), torch.tensor([-1]), [], torch.full((1, 1), float("nan")))


def test_checkpoint_preserves_queue_grace_filter_and_rng(api, model):
    torch = api.torch
    teacher = SimpleNamespace(scene_scale=1., cameras=[], atlas={"xyz": np.array([[0., 0., 1.]])})
    scene = SimpleNamespace(getTrainCameras=lambda: [])
    manager = api.GeometryTopology(model, scene, teacher, {}, None, None, 0.)
    manager.meta["grace"][0] = 700
    manager.queue[(1, 2, 3, 4)] = {"views": {1: 2, 8: 3}, "checks": 2, "xyz": np.ones(3)}
    manager.pending = True
    manager.seeded_cells.add((5, 6, 7, 8))
    state = manager.state_dict()
    expected_random = manager.rng.random(4)
    restored = api.GeometryTopology(model, scene, teacher, {}, None, None, 0.)
    restored.load_state_dict(state)
    assert restored.pending and restored.queue[(1, 2, 3, 4)]["views"] == {1: 2, 8: 3}
    assert int(restored.meta["grace"][0]) == 700 and restored.seeded_cells == {(5, 6, 7, 8)}
    torch.testing.assert_close(restored.model.filter_3D, state["filter"])
    np.testing.assert_array_equal(restored.rng.random(4), expected_random)
    with pytest.raises(ValueError, match="Incompatible"):
        restored.load_state_dict({**state, "version": 1})


def topology_manager(api, model, monkeypatch, config=None):
    torch = api.torch
    stub = ModuleType("utils.vcd_utils")
    stub.compute_vcd_vcp_scores = lambda *args, **kwargs: (torch.zeros(3), torch.tensor([0., .2, .9]))
    monkeypatch.setitem(sys.modules, "utils.vcd_utils", stub)
    teacher = SimpleNamespace(scene_scale=3., cameras=[], real_cameras=[], real_depths=[],
        atlas={"xyz": np.array([[0., 0., 1.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
               "patch_id": np.array([0])},
        real_evidence=lambda points: {key: np.zeros(len(points), dtype=np.uint8)
            for key in ("evidence_class", "support_groups", "conflict_groups")})
    camera = SimpleNamespace(R=torch.eye(3), T=torch.zeros(3), Cx=4., Cy=4., Fx=4., Fy=4., image_width=9, image_height=9)
    scene = SimpleNamespace(getTrainCameras=lambda: [camera], cameras_extent=3.)
    manager = api.GeometryTopology(model, scene, teacher, config or {}, None, None, 0.,
                                   opt=SimpleNamespace(vcd_enable=True, vcp_enable=True))
    return manager


def test_geometry_candidate_grows_even_when_real_vcd_rejects_every_point(api, model, monkeypatch):
    torch = api.torch
    manager = topology_manager(api, model, monkeypatch, {"max_gaussians": 10, "max_growth_fraction": .9,
        "seed_from": 1, "seed_until": 1, "topology_until": 200})
    manager.meta["real_grad"][0] = .1
    manager.meta["real_seen"][0] = 1
    manager.meta["error_streak"][0] = 2
    manager.meta["geom_weight"][0] = 1
    manager.meta["geom_error"][0] = 2
    manager.meta["last_checked"][0] = 16
    manager.meta["grace"][0] = 7
    manager.last_audit = 16
    event = manager._commit(16)
    assert event["after"] == 4 and event["split_parents"] == 1
    assert manager.counts["real_selected"] == 0
    assert manager.counts["geometry_selected"] == 1
    children = manager.meta["parent_id"] == 0
    assert children.sum().item() == 2
    assert (manager.meta["source"][children] == 2).all()
    assert (manager.meta["grace"][children] == 7).all()
    assert manager.meta["root_geometry"][children].all()
    assert not manager.meta["root_gen"][children].any()
    assert torch.isfinite(model.get_xyz).all()


def test_vcp_ranks_only_unprotected_low_opacity_prune_candidates(api, model, monkeypatch):
    torch = api.torch
    manager = topology_manager(api, model, monkeypatch)
    with torch.no_grad():
        model._opacity.fill_(torch.logit(torch.tensor(.01)))
    manager.meta["real_seen"].fill_(2)
    event = manager._commit(16)
    assert event["pruned"] == 1
    # Equal low opacity; VCP .9 chooses ID 2 within the one-row deletion budget.
    torch.testing.assert_close(manager.meta["point_id"], torch.tensor([0, 1]))


def test_reset_recovery_window_prevents_immediate_low_opacity_deletion(api, model, monkeypatch):
    torch = api.torch
    manager = topology_manager(api, model, monkeypatch)
    with torch.no_grad():
        model._opacity.fill_(torch.logit(torch.tensor(.01)))
    manager.meta["real_seen"].fill_(2)
    event = manager._commit(3104)
    assert event["pruned"] == 0 and len(model.get_xyz) == 3


@pytest.mark.parametrize("checks, failures, expected_deleted", [(0, 0, 0), (2, 1, 0), (2, 2, 1)])
def test_gen_soft_prune_requires_actual_repeated_geometry_checks(api, model, monkeypatch, checks, failures, expected_deleted):
    torch = api.torch
    manager = topology_manager(api, model, monkeypatch)
    with torch.no_grad():
        model._opacity[2] = torch.logit(torch.tensor(.01))
    manager.meta["root_gen"][2] = True
    manager.meta["real_seen"][2] = 20  # Frustum exposure is not measured surface support.
    manager.meta["checks"][2] = checks
    manager.meta["unsupported_streak"][2] = failures
    manager.meta["last_checked"][2] = 16
    manager.meta["grace"][2] = 1
    event = manager._commit(16)
    assert event["pruned"] == expected_deleted
    assert (2 in manager.meta["point_id"].tolist()) == (expected_deleted == 0)


def test_real_free_space_conflict_still_overrides_gen_grace_without_geometry_checks(api, model, monkeypatch):
    manager = topology_manager(api, model, monkeypatch)
    manager.meta["root_gen"][2] = True
    manager.meta["grace"][2] = 9999
    manager.teacher.real_evidence = lambda points: {
        "evidence_class": np.full(len(points), 3, np.uint8),
        "support_groups": np.zeros(len(points), np.uint8), "conflict_groups": np.full(len(points), 2, np.uint8)}
    event = manager._commit(16)
    assert event["pruned"] == 1
    assert 2 not in manager.meta["point_id"].tolist()


@pytest.mark.parametrize("same_center", [False, True])
def test_persistent_world_gap_needs_translated_views_for_seed_transaction(api, model, monkeypatch, same_center):
    """A missing plane patch must produce an actual new Gaussian, not just queue logs."""
    torch = api.torch
    stub = ModuleType("utils.vcd_utils")
    stub.compute_vcd_vcp_scores = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "utils.vcd_utils", stub)
    records = []
    cameras = []
    for index, x in enumerate((-.1, .1)):
        pose = np.eye(4)
        pose[0, 3] = 0 if same_center else x
        if same_center and index == 1:
            angle = np.pi/4
            pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
        records.append({"K": [[4., 0., 4.5], [0., 4., 4.5], [0., 0., 1.]],
                        "c2w": pose.tolist(), "width": 9, "height": 9})
        normal = torch.zeros(3, 9, 9)
        normal[2] = -1
        target = {"depth": torch.full((1, 9, 9), 2.), "normal": normal,
                  "valid_depth": torch.ones(1, 9, 9, dtype=torch.bool), "weight_coverage": torch.ones(1, 9, 9),
                  "patch_id": torch.zeros(9, 9, dtype=torch.long)}
        cameras.append(SimpleNamespace(virtual_index=index, image_width=9, image_height=9, teacher=target))

    class Teacher:
        scene_scale = 3.
        cameras = records
        atlas = {"xyz": np.array([[0., 0., 2.]], np.float32), "normal_world": np.array([[0., 0., -1.]], np.float32),
                 "patch_id": np.array([0]), "face_id": np.array([0])}
        real_cameras, real_depths = [], []

        @staticmethod
        def real_evidence(points):
            return {key: np.zeros(len(points), dtype=np.uint8) for key in ("evidence_class", "support_groups", "conflict_groups")}

    real = SimpleNamespace(R=torch.eye(3), T=torch.zeros(3), Cx=4., Cy=4., Fx=4., Fy=4., image_width=9, image_height=9,
                           image_name="real", source_image_name="real.png", uid=0)
    scene = SimpleNamespace(getTrainCameras=lambda: [real], cameras_extent=3.)
    config = {"max_gaussians": 10, "max_growth": 10, "max_growth_fraction": .9, "seed_fraction": .5,
              "seed_from": 1, "seed_until": 100, "topology_until": 200, "seed_batch_limit": 3}
    manager = api.GeometryTopology(model, scene, Teacher(), config, None, None, 0.,
                                   opt=SimpleNamespace(vcd_enable=False, vcp_enable=False))
    with torch.no_grad():
        model._xyz[:, 0] = torch.tensor([10., 11., 12.])
        model._xyz[:, 2] = 2.
    missing = {"depth_status": torch.zeros(1, 9, 9, dtype=torch.int8)}
    for epoch in (1, 2):
        manager.audit_epoch = epoch
        for camera in cameras:
            manager._queue_missing(camera, missing)
    assert manager.queue[(0,)]["checks"] == 2
    assert manager.queue[(0,)]["views"] == {0: 2, 1: 2}
    manager.last_audit = 16
    event = manager._commit(16)
    if same_center:
        assert event["seeds"] == 0 and len(model.get_xyz) == 3
        return
    assert event["seeds"] == 1 and len(model.get_xyz) == 4
    torch.testing.assert_close(model.get_xyz[-1], torch.tensor([0., 0., 2.]))
    assert manager.seed_total == 1 and bool(manager.meta["root_gen"][-1])
    assert int(manager.meta["grace"][-1]) == 516
    assert manager.priority == {0, 1}
    assert (0,) in manager.seeded_cells and not manager.queue
    torch.testing.assert_close(model.get_opacity_with_3D_filter[-1], torch.tensor([.25]), atol=1e-6, rtol=0)
