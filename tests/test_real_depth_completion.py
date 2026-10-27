"""CPU contracts for one-shot completion; the GPU renderer is replaced by a fake."""

import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scene/real_depth_completion.py"
SPEC = importlib.util.spec_from_file_location("real_depth_completion", MODULE_PATH)
completion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(completion)


class FakeCamera:
    def __init__(self, name, prior=False):
        self.image_name = name
        self.image_width, self.image_height = 7, 5
        self.source_K_colmap = np.array([[5, 0, 3.5], [0, 5, 2.5], [0, 0, 1]])
        self.original_image_cpu = torch.full((3, 5, 7), 120, dtype=torch.uint8)
        self.has_depth_prior = prior
        self.depth_prior_cpu = torch.full((1, 5, 7), 9.0) if prior else None
        self.depth_prior = None

    def _store_payload(self, name, value):
        setattr(self, name + "_cpu", value)
        setattr(self, name, None)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"schema_version": 1, "trigger_iteration": 3000, "weight": 0.1,
                       "ramp_iters": 1000, "output": str(self.root / "completion"),
                       "python": sys.executable, "script": str(self.root / "fake_infer.py"),
                       "model_name": "local.pt", "use_fp16": True, "alpha_threshold": 0.9,
                       "edge_relative_threshold": 0.05, "edge_dilation": 1, "min_depth": 0.01}
        Path(self.config["script"]).write_text('''import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--input-manifest'); a=p.parse_args()
c=json.loads(Path(a.config).read_text()); m=json.loads(Path(a.input_manifest).read_text()); root=Path(c['output'])
views=[]
for v in m['views']:
 d=np.load(v['depth']); mask=np.ones_like(d,dtype=bool); d=np.full_like(d,3)
 dp=root/(v['key']+'_result.npy'); mp=root/(v['key']+'_mask.npy'); np.save(dp,d); np.save(mp,mask)
 views.append({'image_name':v['image_name'],'depth':str(dp),'mask':str(mp)})
out={'schema_version':1,'status':'complete','views':views,'snapshot':m['snapshot'],
     'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
     'input_manifest_sha256':hashlib.sha256(Path(a.input_manifest).read_bytes()).hexdigest()}
(root/'manifest.json').write_text(json.dumps(out))
''')
        self.config_path = self.root / "config.json"
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def tearDown(self):
        self.temporary.cleanup()

    def test_confidence_mask_keeps_plane_and_excludes_edges_and_alpha(self):
        depth, alpha = torch.full((5, 7), 2.0), torch.ones(5, 7)
        self.assertTrue(completion.confident_depth_mask(depth, alpha, self.config).all())
        depth[2, 2] = 0
        mask = completion.confident_depth_mask(depth, alpha, self.config)
        self.assertEqual(int(mask.sum()), 34)
        depth[:, 4:] = 4
        alpha[0, 0] = 0.8
        mask = completion.confident_depth_mask(depth, alpha, self.config)
        self.assertFalse(mask[:, 2:6].any())
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[:, -1].all())

    def test_rng_restored_on_exception_without_initializing_cuda(self):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        with self.assertRaises(RuntimeError):
            with completion.preserve_rng_states():
                random.random(); np.random.random(); torch.rand(10)
                raise RuntimeError("test")
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(1), expected[2]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_hook_preserves_original_prior_and_rng_and_completes_only_once(self):
        original, missing = FakeCamera("trusted.png", True), FakeCamera("nested/missing.png")
        original_depth = original.depth_prior_cpu
        hook = completion.RealDepthCompletion(self.config_path, [original, missing], recipe="real_only")

        def save_ply(path):
            random.random(); np.random.random(); torch.rand(1)
            Path(path).write_bytes(b"snapshot")

        def render(camera, *args, **kwargs):
            random.random(); np.random.random(); torch.rand(1)
            depth = torch.full((1, 5, 7), 2.0)
            alpha = torch.ones_like(depth)
            alpha[:, 2, 3] = 0  # Output may complete this previously unanchored pixel.
            return {"median_depth": depth, "mask": alpha}

        cache = SimpleNamespace(clear=lambda: None)
        args = (SimpleNamespace(save_ply=save_ply), render, None, None, 1, cache)
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        hook.maybe_run(2999, 2999, *args)
        self.assertFalse(hook.done)
        hook.maybe_run(3000, 3000, *args)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(1), expected[2]))
        self.assertIs(original.depth_prior_cpu, original_depth)
        self.assertFalse(hasattr(original, "real_depth_completion"))
        self.assertTrue(missing.has_depth_prior)
        self.assertTrue((missing.depth_prior_cpu == 3).all())
        self.assertEqual(hook.weight_for(original, 3001, 0.4), 0.4)
        self.assertEqual(hook.weight_for(missing, 3000, 0.4), 0)
        self.assertAlmostEqual(hook.weight_for(missing, 3001, 0.4), 0.0001)
        self.assertAlmostEqual(hook.weight_for(missing, 4000, 0.4), 0.1)
        hook.record_loss(missing, 3001, torch.tensor(2.0), 0.0001)
        self.assertEqual(hook.record["samples"], 1)
        self.assertEqual(hook.record["by_image"][missing.image_name]["first_sample_iteration"], 3001)
        self.assertAlmostEqual(hook.record["weighted_loss_sum"], 0.0002)
        self.assertEqual(hook.record["actual_optimizer_updates_at_trigger"], 3000)
        input_manifest = json.loads((hook.output / "input_manifest.json").read_text())
        self.assertEqual(input_manifest["views"][0]["intrinsics_px"], missing.source_K_colmap.tolist())
        self.assertTrue(Path(input_manifest["views"][0]["raw_depth"]).exists())
        output_hash = hook.record["output_manifest_sha256"]
        hook.maybe_run(3000, 3000, *args)
        self.assertEqual(hook.record["output_manifest_sha256"], output_hash)
        self.assertFalse(torch.cuda.is_initialized())

    def test_output_validation_is_atomic_and_cannot_overwrite_trusted_depth(self):
        cameras = [FakeCamera("a.png"), FakeCamera("b.png")]
        views = []
        for index, camera in enumerate(cameras):
            depth_path, mask_path = self.root / f"d{index}.npy", self.root / f"m{index}.npy"
            np.save(depth_path, np.full((5, 7), 2, np.float32))
            np.save(mask_path, np.ones((5, 7), bool))
            views.append({"image_name": camera.image_name, "depth": str(depth_path), "mask": str(mask_path)})
        manifest = {"schema_version": 1, "views": views}
        np.save(views[1]["depth"], np.full((5, 7), np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "invalid or empty"):
            completion.attach_completed_priors(cameras, manifest, self.root, 0.01)
        self.assertFalse(any(camera.has_depth_prior for camera in cameras))
        np.save(views[1]["depth"], np.full((5, 7), 2, np.float32))
        cameras[1].has_depth_prior = True
        with self.assertRaisesRegex(ValueError, "existing prior"):
            completion.attach_completed_priors(cameras, manifest, self.root, 0.01)
        self.assertFalse(cameras[0].has_depth_prior)

    def test_hook_rejects_incomplete_or_mismatched_output_identity(self):
        script_path = Path(self.config["script"])
        valid_script = script_path.read_text()
        cases = [("status", "pilot_complete", "status=complete"),
                 ("config_sha256", None, "config identity"),
                 ("input_manifest_sha256", None, "input identity"),
                 ("snapshot", {"path": "wrong", "sha256": "wrong"}, "snapshot identity")]
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field):
                self.config["output"] = str(self.root / f"bad-output-{index}")
                self.write_config()
                script_path.write_text(valid_script.replace(
                    "(root/'manifest.json').write_text(json.dumps(out))",
                    f"out[{field!r}]={value!r}\n(root/'manifest.json').write_text(json.dumps(out))"))
                camera = FakeCamera("missing.png")
                hook = completion.RealDepthCompletion(self.config_path, [camera], recipe="real_only")
                gaussians = SimpleNamespace(save_ply=lambda path: Path(path).write_bytes(b"snapshot"))
                render = lambda *a, **kw: {"median_depth": torch.full((1, 5, 7), 2.0), "mask": torch.ones(1, 5, 7)}
                with self.assertRaisesRegex(ValueError, message):
                    hook.maybe_run(3000, 3000, gaussians, render, None, None, 1, None)
                self.assertFalse(camera.has_depth_prior)
                self.assertFalse(hook.done)

    def test_invalid_config_recipe_resume_and_trigger_count_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "continuous"):
            completion.RealDepthCompletion(self.config_path, [FakeCamera("a")], recipe="real_only", checkpoint="old.pth")
        with self.assertRaisesRegex(ValueError, "real_only"):
            completion.RealDepthCompletion(self.config_path, [FakeCamera("a")], recipe="r2c")
        hook = completion.RealDepthCompletion(self.config_path, [FakeCamera("a")], recipe="real_only")
        with self.assertRaisesRegex(ValueError, "optimizer updates"):
            hook.maybe_run(3000, 2999, None, None, None, None, None, None)
        self.config["alpha_threshold"] = 1.1
        self.write_config()
        with self.assertRaisesRegex(ValueError, "alpha_threshold"):
            completion.load_completion_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
