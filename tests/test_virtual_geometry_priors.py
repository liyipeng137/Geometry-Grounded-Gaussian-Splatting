"""CPU checks for virtual-depth coordinates, fixed-mask loss and group budgets."""

import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("virtual_geometry_cpu", ROOT / "scene/virtual_geometry_priors.py")
virtual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(virtual)


class VirtualGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, counts=(1, 7)):
        views = []
        for group, count in enumerate(counts):
            for index in range(count):
                name = f"target{group}_view{index}"
                depth, mask = self.root / f"{name}_depth.npy", self.root / f"{name}_mask.npy"
                np.save(depth, np.full((5, 7), 2, dtype=np.float32))
                weights = np.ones((5, 7), dtype=np.float32)
                weights[0, 0] = 0
                np.save(mask, weights)
                views.append({"image_name": name, "target_id": str(group), "width": 7, "height": 5,
                              "K": [[5., 0, 3.2], [0, 6., 2.7], [0, 0, 1]], "c2w": np.eye(4).tolist(),
                              "depth_path": str(depth), "mask_path": str(mask),
                              "depth_sha256": virtual.sha256_file(depth), "mask_sha256": virtual.sha256_file(mask)})
        source = self.root / "source.ply"
        source.write_bytes(b"immutable source geometry")
        manifest = {"schema_version": 1, "depth_convention": "camera_z", "units": "scene",
                    "pixel_center_convention": "colmap_half_integer", "views": views,
                    "source_models": [{"path": str(source), "sha256": virtual.sha256_file(source)}]}
        mp = self.root / "manifest.json"
        mp.write_text(json.dumps(manifest))
        config = {"schema_version": 1, "manifest": str(mp), "weight": .5, "from_iter": 5000,
                  "ramp_iters": 2000, "every_n_real_updates": 2,
                  "loss": {"mode": "relative_charbonnier", "deadband_absolute": .02, "deadband_relative": .01}}
        cp = self.root / "config.json"
        cp.write_text(json.dumps(config))
        return cp, mp, config, manifest

    def test_balanced_targets_do_not_multiply_weight_with_view_count_or_change_real_rng(self):
        cp, _, _, _ = self.fixture()
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        priors = virtual.VirtualGeometryPriors(cp)
        names = [priors.draw(device="cpu")[0]["target_id"] for _ in range(200)]
        self.assertEqual(names.count("0"), 100)
        self.assertEqual(names.count("1"), 100)
        self.assertEqual(random.getstate(), python_state)
        for left, right in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(left, right)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(priors.weight_at(4999), 0)
        self.assertAlmostEqual(priors.weight_at(5000), .5 / 2000)
        self.assertAlmostEqual(priors.weight_at(6999), .5)
        self.assertFalse(priors.due(4999, 2))
        self.assertFalse(priors.due(5001, 3))
        self.assertTrue(priors.due(5001, 4))

    def test_zero_depth_is_penalized_with_gradient_and_fixed_mask_denominator(self):
        loss_config = {"mode": "relative_charbonnier", "epsilon": .01, "denominator_floor": .05,
                       "deadband_absolute": 0., "deadband_relative": 0.}
        pred = torch.tensor([[[0., 1.9, float("nan")]]], requires_grad=True)
        target = torch.tensor([[[2., 2., float("nan")]]])
        mask = torch.tensor([[[1., .5, 0.]]])
        loss = virtual.virtual_depth_loss(pred, target, mask, loss_config)
        loss.backward()
        self.assertGreater(float(loss.detach()), .6)
        self.assertLess(float(pred.grad[0, 0, 0]), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[0, 0, 2]), 0)
        duplicate = virtual.virtual_depth_loss(pred.repeat(1, 2, 1), target.repeat(1, 2, 1), mask.repeat(1, 2, 1), loss_config)
        self.assertAlmostEqual(float(loss.detach()), float(duplicate.detach()), places=6)
        loss_config["deadband_absolute"] = .2
        almost = torch.tensor([[[1.9, 2.1]]], requires_grad=True)
        dead = virtual.virtual_depth_loss(almost, torch.full_like(almost, 2), torch.ones_like(almost), loss_config)
        dead.backward()
        self.assertEqual(float(dead.detach()), 0)
        self.assertTrue((almost.grad == 0).all())

    def test_absolute_loss_and_nonfinite_prediction_failure(self):
        config = {"mode": "absolute_charbonnier", "epsilon": .01, "denominator_floor": .05,
                  "deadband_absolute": .1, "deadband_relative": .0}
        pred, truth, mask = torch.tensor([1.]), torch.tensor([2.]), torch.tensor([1.])
        expected = (.9 ** 2 + .01 ** 2) ** .5 - .01
        self.assertAlmostEqual(float(virtual.virtual_depth_loss(pred, truth, mask, config)), expected, places=6)
        with self.assertRaises(FloatingPointError):
            virtual.virtual_depth_loss(pred * float("nan"), truth, mask, config)

    def test_camera_projects_colmap_pixels_and_corrects_offcenter_depth(self):
        cp, _, _, _ = self.fixture((1,))
        priors = virtual.VirtualGeometryPriors(cp)
        view, camera, _, _ = priors.draw(device="cpu")
        K = np.asarray(view["K"])
        point = torch.tensor([.3, -.2, 2., 1.])
        projected = point @ camera.full_proj_transform
        ndc = projected[:2] / projected[3]
        actual = (ndc.numpy() + 1) * np.array([7, 5]) / 2
        expected = np.array([K[0, 0] * .3 / 2 + K[0, 2], K[1, 1] * -.2 / 2 + K[1, 2]])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        y, x = np.mgrid[:5, :7]
        raster_radius = np.sqrt(1 + ((x - 3) / 5) ** 2 + ((y - 2) / 6) ** 2)
        camera_radius = np.sqrt(1 + ((x - (3.2 - .5)) / 5) ** 2 + ((y - (2.7 - .5)) / 6) ** 2)
        # The native rasterizer uses centered rays; its reported depth needs this ratio.
        native = torch.as_tensor((2 * camera_radius / raster_radius)[None], dtype=torch.float32)
        self.assertTrue(torch.allclose(camera.camera_z_depth(native), torch.full_like(native, 2), atol=1e-6))
        self.assertFalse(hasattr(camera, "original_image"))

    def test_resume_preserves_sampling_and_rejects_changed_inputs(self):
        cp, _, _, manifest = self.fixture()
        priors = virtual.VirtualGeometryPriors(cp)
        for _ in range(11):
            priors.draw(device="cpu")
        state = json.loads(json.dumps(priors.checkpoint_state()))
        expected = [priors.draw(device="cpu")[0]["image_name"] for _ in range(40)]
        resumed = virtual.VirtualGeometryPriors(cp, checkpoint_state=state)
        self.assertEqual(expected, [resumed.draw(device="cpu")[0]["image_name"] for _ in range(40)])
        changed = Path(manifest["views"][0]["depth_path"])
        np.save(changed, np.full((5, 7), 2.1, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            virtual.VirtualGeometryPriors(cp, checkpoint_state=state)

    def test_invalid_convention_and_nonrigid_pose_are_rejected(self):
        cp, mp, _, manifest = self.fixture((1,))
        manifest["depth_convention"] = "ray_distance"
        mp.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            virtual.VirtualGeometryPriors(cp)
        manifest["depth_convention"] = "camera_z"
        manifest["views"][0]["c2w"][0][0] = 2
        mp.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not rigid"):
            virtual.VirtualGeometryPriors(cp)

    def test_gradient_proof_and_actual_optimizer_accounting(self):
        cp, _, _, _ = self.fixture((1,))
        priors = virtual.VirtualGeometryPriors(cp)
        class Gaussian:
            pass
        gaussians = Gaussian()
        for field in ("_xyz", "_scaling", "_rotation", "_opacity"):
            setattr(gaussians, field, torch.nn.Parameter(torch.ones(2, 3)))
        loss = gaussians._xyz.square().mean() + gaussians._scaling.square().mean()
        priors.diagnose_gradients(.5 * loss, gaussians, 5000)
        loss.backward()
        diagnostic = priors.record["gradient_diagnostics"][0]["weighted_virtual_only_gradient_norms"]
        self.assertGreater(diagnostic["xyz"], 0)
        self.assertIsNone(diagnostic["rotation"])
        view, _, target, mask = priors.draw(device="cpu")
        priors.record_loss(view, target, mask, loss, .5, 5000)
        self.assertEqual(priors.record["actual_optimizer_updates"], 0)
        priors.record_optimizer_step()
        self.assertEqual(priors.record["actual_optimizer_updates"], 1)
        priors.record_optimizer_step()
        self.assertEqual(priors.record["actual_optimizer_updates"], 1)
        self.assertEqual(priors.record["loss_samples_reaching_optimizer"], 1)


if __name__ == "__main__":
    unittest.main()
