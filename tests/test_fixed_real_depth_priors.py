"""CPU tests for immutable external depth identity and source-specific training."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_real_depth_cpu", ROOT / "scene/fixed_real_depth_priors.py")
fixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed)


class FakeCamera:
    def __init__(self, name, prior=False):
        self.image_name = name
        self.image_width, self.image_height = 7, 5
        self.source_K_colmap = np.array([[5, 0, 3.5], [0, 5, 2.5], [0, 0, 1]], dtype=np.float64)
        self.R = torch.eye(3)
        self.T = torch.tensor([.2, -.3, .4])
        self.original_image_cpu = torch.full((3, 5, 7), 120, dtype=torch.uint8)
        self.has_depth_prior = prior
        self.depth_prior_cpu = torch.full((1, 5, 7), 9.) if prior else None
        self.depth_prior = None

    def _store_payload(self, name, value):
        setattr(self, name + "_cpu", value)
        setattr(self, name, None)


class FixedDepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def write(self, path, document):
        path.write_text(json.dumps(document))

    def fixture(self, missing=2, original=1):
        cameras = [FakeCamera(f"trusted_{i}.png", True) for i in range(original)]
        targets = [FakeCamera(f"nested/frame_{i:06d}.png") for i in range(missing)]
        views, proofs = [], []
        for index, camera in enumerate(targets):
            paths = {field: self.root / f"{index}_{field}.npy" for field in ("depth", "mask")}
            depth = np.full((5, 7), 2.5, np.float32)
            mask = np.ones((5, 7), bool)
            mask[0, 0] = False
            np.save(paths["depth"], depth)
            np.save(paths["mask"], mask)
            rgb = self.root / f"{index}_rgb.png"
            Image.fromarray(camera.original_image_cpu.permute(1, 2, 0).numpy()).save(rgb)
            views.append({"image_name": camera.image_name, "width": 7, "height": 5,
                          **{field: str(path) for field, path in paths.items()},
                          **{field + "_sha256": fixed.sha256_file(path) for field, path in paths.items()}})
            pose = np.eye(4)
            pose[:3, 3] = camera.T.numpy()
            proofs.append({"image_name": camera.image_name, "width": 7, "height": 5,
                           "intrinsics_px": camera.source_K_colmap.tolist(), "world_to_camera": pose.tolist(),
                           "rgb": {"path": str(rgb), "sha256": fixed.sha256_file(rgb)}})
        source = self.root / "source_camera_model.bin"
        source.write_bytes(b"frozen camera and mesh source")
        manifest = {"schema_version": 1, "status": "complete", "kind": "genrecon_mesh_guided_lingbot",
                    "depth_convention": "camera_z", "depth_units": "scene", "views": views}
        mp = self.root / "producer.json"
        self.write(mp, manifest)
        identity = {"schema_version": 1, "kind": "fixed_real_depth_camera_identity",
                    "manifest": {"path": str(mp), "sha256": fixed.sha256_file(mp)},
                    "depth_convention": "camera_z", "depth_units": "scene",
                    "pixel_center_convention": "colmap_half_integer",
                    "source_sha256": {str(source): fixed.sha256_file(source)}, "views": proofs}
        ip = self.root / "identity.json"
        self.write(ip, identity)
        config = {"schema_version": 1, "manifest": str(mp), "identity_manifest": str(ip),
                  "weight": .1, "from_iter": 4001, "ramp_iters": 1000, "min_depth": .01}
        cp = self.root / "config.json"
        self.write(cp, config)
        return cameras + targets, cp, config, mp, manifest, ip, identity

    def test_all_131_attach_after_initialization_preserving_398_and_rng(self):
        cameras, cp, _, mp, _, ip, _ = self.fixture(131, 398)
        originals = [camera.depth_prior_cpu for camera in cameras[:398]]
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        hook = fixed.FixedRealDepthPriors(str(cp), cameras, recipe="real_only")
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(before, after)
        self.assertEqual(hook.record["completed_prior_count"], 131)
        self.assertEqual(hook.record["original_prior_count"], 398)
        self.assertEqual(hook.record["total_real_camera_count"], 529)
        self.assertEqual(hook.record["status"], "complete")
        self.assertEqual(hook.record["manifest_sha256"], fixed.sha256_file(mp))
        self.assertEqual(hook.record["identity_sha256"], fixed.sha256_file(ip))
        for camera, prior in zip(cameras[:398], originals):
            self.assertIs(camera.depth_prior_cpu, prior)
            self.assertFalse(hasattr(camera, "real_depth_fixed_prior"))
            self.assertEqual(hook.weight_for(camera, 4001, .4), .4)
        for camera in cameras[398:]:
            self.assertTrue(camera.has_depth_prior)
            self.assertEqual(camera.depth_prior_cpu[0, 0, 0], 0)
            self.assertTrue(torch.all(camera.depth_prior_cpu[0, 1:] == 2.5))
            self.assertFalse(hasattr(camera, "real_depth_completion"))
            self.assertEqual(hook.weight_for(camera, 4000, .4), 0)
            self.assertAlmostEqual(hook.weight_for(camera, 4001, .4), .0001)
            self.assertAlmostEqual(hook.weight_for(camera, 5000, .4), .1)
            hook.record_loss(camera, 3500, torch.tensor(0.), 0)
            hook.record_loss(camera, 4001, torch.tensor(2.), .0001)
        self.assertEqual(hook.record["samples"], 131)
        self.assertEqual(hook.record["inactive_samples"], 131)
        self.assertEqual(hook.record["sampled_camera_count"], 131)
        self.assertTrue(hook.record["all_targets_sampled"])
        self.assertAlmostEqual(hook.record["weighted_loss_sum"], 131 * .0002)

    def test_bad_depth_or_mask_hash_cannot_partially_mutate_cameras(self):
        cameras, cp, _, _, manifest, _, _ = self.fixture()
        np.save(manifest["views"][-1]["depth"], np.full((5, 7), 7, np.float32))
        with self.assertRaisesRegex(ValueError, "depth SHA256 mismatch"):
            fixed.FixedRealDepthPriors(str(cp), cameras, recipe="real_only")
        self.assertTrue(cameras[0].has_depth_prior)
        self.assertFalse(any(camera.has_depth_prior for camera in cameras[1:]))

    def test_undefined_target_subset_and_overwriting_original_are_rejected(self):
        cameras, cp, _, mp, manifest, ip, identity = self.fixture()
        manifest["views"] = manifest["views"][:-1]
        self.write(mp, manifest)
        identity["manifest"]["sha256"] = fixed.sha256_file(mp)
        self.write(ip, identity)
        with self.assertRaisesRegex(ValueError, "entire original missing-depth set"):
            fixed.FixedRealDepthPriors(str(cp), cameras, recipe="real_only")
        self.assertFalse(any(camera.has_depth_prior for camera in cameras[1:]))

    def test_calibration_pose_and_actual_rgb_identity_are_checked(self):
        for field, message in (("intrinsics_px", "intrinsic"), ("world_to_camera", "world pose"), ("rgb", "real training RGB")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                old_root = self.root
                self.root = Path(temporary)
                cameras, cp, _, _, _, ip, identity = self.fixture()
                if field == "intrinsics_px":
                    identity["views"][0][field][0][0] += .1
                elif field == "world_to_camera":
                    identity["views"][0][field][0][3] += .01
                else:
                    cameras[1].original_image_cpu[0, 0, 0] += 1
                self.write(ip, identity)
                with self.assertRaisesRegex(ValueError, message):
                    fixed.FixedRealDepthPriors(str(cp), cameras, recipe="real_only")
                self.assertFalse(any(camera.has_depth_prior for camera in cameras[1:]))
                self.root = old_root

    def test_source_and_manifest_identity_are_content_bound(self):
        cameras, cp, _, _, _, _, identity = self.fixture()
        Path(next(iter(identity["source_sha256"]))).write_bytes(b"source mutated")
        with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
            fixed.FixedRealDepthPriors(str(cp), cameras, recipe="real_only")
        self.assertFalse(any(camera.has_depth_prior for camera in cameras[1:]))

    def test_invalid_config_modes_and_resume_fail_before_gpu_scene(self):
        cameras, cp, config, _, _, _, _ = self.fixture()
        for kwargs, message in (({"recipe": "r2c"}, "real_only"),
                                ({"recipe": "real_only", "checkpoint": "old.pth"}, "checkpoint resume"),
                                ({"recipe": "real_only", "completion_config": "other.json"}, "mutually exclusive")):
            with self.assertRaisesRegex(ValueError, message):
                fixed.FixedRealDepthPriors(str(cp), cameras, **kwargs)
        for field, value in (("ramp_iters", -1), ("from_iter", True), ("weight", float("nan")), ("min_depth", 0), ("min_depth", .005)):
            invalid = copy.deepcopy(config)
            invalid[field] = value
            self.write(cp, invalid)
            with self.assertRaisesRegex(ValueError, field):
                fixed.load_fixed_prior_config(str(cp))
        fixed.validate_fixed_prior_options("", "legacy_config", "r2c", "old.pth")

    def test_training_guards_and_attachment_order_are_explicit(self):
        tree = ast.parse((ROOT / "train.py").read_text())
        training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training")
        calls = [(node.func.id, node.lineno) for node in ast.walk(training) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        line = lambda name: min(lineno for called, lineno in calls if called == name)
        self.assertLess(line("validate_fixed_prior_options"), line("Scene"))
        self.assertGreater(line("FixedRealDepthPriors"), line("Scene"))
        source = (ROOT / "train.py").read_text()
        self.assertIn('training_stats["real_depth_prior"] = fixed_depth_priors.record', source)
        self.assertIn('fixed_depth_priors.record_loss(viewpoint_cam, iteration, depth_prior_loss, real_depth_weight)', source)
        self.assertIn('parser.add_argument("--real_depth_prior_config", type=str, default="")', source)


if __name__ == "__main__":
    unittest.main()
