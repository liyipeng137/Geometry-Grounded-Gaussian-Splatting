"""Physical CPU checks for progressive virtual-camera budgets and robust depth."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("virtual_scheduler_cpu", ROOT / "scene/virtual_geometry_priors.py")
virtual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(virtual)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, counts=(8, 8, 8), modify=None):
        depth, mask = self.root / "depth.npy", self.root / "mask.npy"
        np.save(depth, np.full((5, 7), 2., np.float32))
        np.save(mask, np.ones((5, 7), np.float32))
        views = []
        for group, count in enumerate(counts):
            for index in range(count):
                views.append({"image_name": f"group{group}_view{index:02d}", "target_id": str(group),
                              "spatial_group": str(group), "priority": 1., "loss_weight": 1., "target_priority": 1.,
                              "width": 7, "height": 5, "K": [[5, 0, 3.5], [0, 6, 2.5], [0, 0, 1]],
                              "c2w": np.eye(4).tolist(), "depth_path": str(depth), "mask_path": str(mask)})
        manifest = {"schema_version": 1, "depth_convention": "camera_z", "units": "scene",
                    "pixel_center_convention": "colmap_half_integer", "views": views}
        config = {"schema_version": 1, "manifest": str(self.root / "manifest.json"), "weight": .5,
                  "from_iter": 5, "ramp_iters": 0, "every_n_real_updates": 1,
                  "scheduler": {"real_camera_count": 9, "max_active": 999,
                                "activation_iterations": [5, 8, 11], "batch_sizes": [2, 3, 20],
                                "local_ramp_iters": 4, "probe_every": 2, "probe_views": 3}}
        if modify:
            modify(config, manifest)
        Path(config["manifest"]).write_text(json.dumps(manifest))
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return path

    @staticmethod
    def probe(view):
        return {"relative_abs_error": .1, "relative_camera_z_laplacian": .02,
                "zero_prediction_fraction": .1, "fixed_valid_pixels": 35}

    def test_later_batches_are_real_activations_and_never_exceed_real_camera_cap(self):
        priors = virtual.VirtualGeometryPriors(self.fixture())
        priors.maybe_update_schedule(4, self.probe)
        self.assertEqual(len(priors.active_indices), 0)
        self.assertFalse(priors.due(4, 4))
        priors.maybe_update_schedule(5, self.probe)
        self.assertEqual(len(priors.active_indices), 2)
        initial = set(priors.active_indices)
        view, _, _, _ = priors.draw(device="cpu")
        self.assertEqual(priors.weight_for(view, 5), .125)
        for iteration in range(6, 12):
            priors.maybe_update_schedule(iteration, self.probe)
        self.assertEqual(priors.active_cap, 6)  # floor(9*2/3), despite requested max 999
        self.assertEqual(len(priors.active_indices), 6)
        self.assertTrue(initial <= priors.active_indices)
        events = priors.record["scheduler"]["activation_events"]
        self.assertEqual([event["added_views"] for event in events], [2, 3, 1])
        newest = next(index for index, activated in priors.activation_iterations.items() if activated == 11)
        self.assertEqual(priors.weight_for(priors.views[newest], 11), .125)
        self.assertEqual(priors.weight_for(priors.views[next(iter(initial))], 11), .5)
        self.assertGreater(priors.record["scheduler"]["probe_render_samples"], 0)
        self.assertTrue(any(view["rank_score"] > view["priority"] for event in events[1:] for view in event["views"]))
        self.assertEqual(priors.record["scheduler"]["cumulative_activated_views"], 6)

    def test_checkpoint_resume_preserves_future_activation_probes_and_sampling(self):
        path = self.fixture()
        original = virtual.VirtualGeometryPriors(path)
        for iteration in range(1, 10):
            original.maybe_update_schedule(iteration, self.probe)
            if original.due(iteration, iteration):
                original.draw(device="cpu")
        state = json.loads(json.dumps(original.checkpoint_state()))
        restored = virtual.VirtualGeometryPriors(path, checkpoint_state=state)
        for iteration in range(10, 18):
            original.maybe_update_schedule(iteration, self.probe)
            restored.maybe_update_schedule(iteration, self.probe)
            left = original.draw(device="cpu")[0]
            right = restored.draw(device="cpu")[0]
            self.assertEqual(left["image_name"], right["image_name"])
            self.assertEqual(original.weight_for(left, iteration), restored.weight_for(right, iteration))
        self.assertEqual(json.loads(json.dumps(original.checkpoint_state())), json.loads(json.dumps(restored.checkpoint_state())))

    def test_same_patch_many_cameras_does_not_multiply_budget_and_confidence_survives_normalization(self):
        def modify(config, manifest):
            config["scheduler"].update(real_camera_count=100, activation_iterations=[5], batch_sizes=[21],
                                       local_ramp_iters=0, target_budget="bounded_sqrt_priority")
            for view in manifest["views"]:
                if view["target_id"] == "1":
                    view["loss_weight"] = .25
        priors = virtual.VirtualGeometryPriors(self.fixture(counts=(1, 20), modify=modify))
        priors.maybe_update_schedule(5, self.probe)
        counts, weights = {"0": 0, "1": 0}, {"0": 0., "1": 0.}
        for _ in range(200):
            view, _, _, _ = priors.draw(device="cpu")
            counts[view["target_id"]] += 1
            weights[view["target_id"]] += priors.weight_for(view, 5)
        self.assertEqual(counts, {"0": 100, "1": 100})
        self.assertEqual(weights["1"], weights["0"] * .25)

    def test_bounded_target_priority_budget_is_independent_of_number_of_views(self):
        def modify(config, manifest):
            config["scheduler"].update(real_camera_count=100, activation_iterations=[5], batch_sizes=[21],
                                       local_ramp_iters=0, target_budget="bounded_sqrt_priority")
            for view in manifest["views"]:
                view["target_priority"] = 100. if view["target_id"] == "0" else 1.
        priors = virtual.VirtualGeometryPriors(self.fixture(counts=(1, 20), modify=modify))
        priors.maybe_update_schedule(5, self.probe)
        tickets = priors.record["scheduler"]["target_budget_tickets"]
        self.assertEqual(tickets, {"0": 3, "1": 1})
        counts = {"0": 0, "1": 0}
        for _ in range(200):
            counts[priors.draw(device="cpu")[0]["target_id"]] += 1
        self.assertEqual(counts, {"0": 150, "1": 50})

    def test_spatial_fairness_prevents_one_high_priority_location_taking_initial_batch(self):
        def modify(config, manifest):
            config["scheduler"]["batch_sizes"][0] = 3
            for view in manifest["views"]:
                if view["target_id"] == "0":
                    view["priority"] = 1000.
        priors = virtual.VirtualGeometryPriors(self.fixture(modify=modify))
        priors.maybe_update_schedule(5, self.probe)
        self.assertEqual({priors.views[index]["spatial_group"] for index in priors.active_indices}, {"0", "1", "2"})

    def test_saturation_keeps_extreme_depth_loss_finite_and_downweights_conflict_gradient(self):
        config = {"mode": "relative_charbonnier", "epsilon": .005, "denominator_floor": .05,
                  "deadband_absolute": 0., "deadband_relative": 0., "robust_saturation": .05}
        prediction = torch.tensor([[[0., 20., 1e20]]], requires_grad=True)
        loss = virtual.virtual_depth_loss(prediction, torch.full_like(prediction, 2), torch.ones_like(prediction), config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(loss.detach()), 0)
        self.assertLessEqual(float(loss.detach()), .05)
        self.assertLess(float(prediction.grad[0, 0, 0]), 0)
        self.assertLess(float(prediction.grad[0, 0, 1]), .001)

    def test_scheduler_off_retains_original_draws_and_does_not_probe(self):
        def modify(config, manifest):
            del config["scheduler"]
        priors = virtual.VirtualGeometryPriors(self.fixture(modify=modify))
        priors.maybe_update_schedule(5, lambda view: self.fail("Disabled scheduler must not probe"))
        view = priors.draw(device="cpu")[0]
        self.assertEqual(priors.weight_for(view, 5), priors.weight_at(5))
        self.assertNotIn("scheduler_state", priors.checkpoint_state())


if __name__ == "__main__":
    unittest.main()
