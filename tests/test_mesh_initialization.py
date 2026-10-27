"""CPU reader/provenance and sampling regressions for the real-only init comparison."""

import ast
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.dataset_readers import readColmapSceneInfo
from scene.mesh_initialization import sha256_file
from scene.train_split import validate_checkpoint_split, validate_initialization_proof
from scene.training_sampling import CameraSamplingSchedule


class MeshInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.names = ["frame_000001.png", "frame_000002.png"]
        model = self.dataset / "sparse/0"
        model.mkdir(parents=True)
        (model / "cameras.txt").write_text("1 SIMPLE_PINHOLE 8 6 5 4 3\n")
        (model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_000001.png\n\n2 1 0 0 0 .1 0 0 1 frame_000002.png\n\n")
        (self.dataset / "images").mkdir()
        (self.dataset / "depth").mkdir()
        for name in self.names:
            Image.new("RGB", (8, 6), (50, 80, 100)).save(self.dataset / "images" / name)
            np.save(self.dataset / "depth" / Path(name).with_suffix(".npy"), np.full((6, 8), 2, np.float32))
        self.ply = self.root / "surface.ply"
        self.source_mesh = self.root / "source_mesh.ply"
        self.manifest_path = self.root / "surface.ply.manifest.json"
        self.write_ply()
        self.source_mesh.write_bytes(self.ply.read_bytes())
        self.manifest = {
            "schema_version": 1, "kind": "mesh_surface", "dataset_root": str(self.dataset),
            "train_image_names": self.names,
            "initialization_ply": {"path": str(self.ply), "sha256": sha256_file(self.ply), "point_count": 3},
            "source_mesh": {"path": str(self.source_mesh), "sha256": sha256_file(self.source_mesh)},
            "sampling": {"method": "area_uniform", "seed": 42},
            "coloring": {"method": "visible_real_cameras"},
            "calibration_sha256": {str(model / name): sha256_file(model / name) for name in ("cameras.txt", "images.txt")},
            "views": [{"name": name, "image_path": str(self.dataset / "images" / name),
                       "image_sha256": sha256_file(self.dataset / "images" / name)} for name in self.names],
        }
        self.write_manifest()
        self.kwargs = dict(initialization_kind="mesh_surface", initialization_ply=str(self.ply),
                           initialization_manifest=str(self.manifest_path))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.temporary.cleanup()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def write_ply(self, invalid=None):
        dtype = [(name, "f4") for name in ("x", "y", "z")] + [(name, "u1") for name in ("red", "green", "blue")]
        if invalid == "gaussian":
            dtype.append(("opacity", "f4"))
        vertices = np.zeros(3, dtype=dtype)
        vertices["x"] = [0, .1, .2]
        vertices["z"] = 2
        vertices["red"], vertices["green"], vertices["blue"] = 50, 80, 100
        if invalid == "nonfinite":
            vertices["x"][0] = np.nan
        if invalid == "black":
            vertices["red"], vertices["green"], vertices["blue"] = 0, 0, 0
        elements = [PlyElement.describe(vertices, "vertex")]
        if invalid == "faces":
            face = np.empty(1, dtype=[("vertex_indices", "i4", (3,))])
            face["vertex_indices"][0] = [0, 1, 2]
            elements.append(PlyElement.describe(face, "face"))
        PlyData(elements, text=False).write(str(self.ply))

    def read(self, cache="run"):
        return readColmapSceneInfo(str(self.dataset), None, False, init_cache_dir=str(self.root / cache), **self.kwargs)

    def test_mesh_reader_and_same_source_resume_preserve_proof(self):
        scene = self.read()
        np.testing.assert_allclose(scene.point_cloud.points[:, 2], 2)
        np.testing.assert_allclose(scene.point_cloud.colors[0], np.array([50, 80, 100]) / 255)
        proof = json.loads((self.root / "run/camera_split.json").read_text())
        validate_initialization_proof(proof)
        self.assertEqual(proof["initialization"]["kind"], "mesh_surface")
        self.assertEqual(proof["initialization"]["point_count"], 3)
        self.read("resume")
        result = validate_checkpoint_split(self.root / "checkpoint.pth", self.root / "resume", previous_record=proof)
        self.assertEqual(result["initialization_kind"], "mesh_surface")
        self.assertFalse(list((self.root / "run").glob("rgbd_init*")))

    def test_changed_or_missing_preserved_assets_rejected(self):
        self.read()
        proof = json.loads((self.root / "run/camera_split.json").read_text())
        original = self.source_mesh.read_bytes()
        self.source_mesh.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "source mesh.*changed"):
            validate_initialization_proof(proof)
        self.source_mesh.write_bytes(original)
        self.ply.unlink()
        with self.assertRaisesRegex(ValueError, "PLY.*missing"):
            self.read()

    def test_manifest_mutation_cannot_resume_with_old_proof(self):
        self.read()
        proof = json.loads((self.root / "run/camera_split.json").read_text())
        self.manifest["sampling"]["seed"] = 99
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "manifest content changed"):
            validate_initialization_proof(proof)

    def test_wrong_camera_membership_or_root_rejected(self):
        self.manifest["train_image_names"] = self.names[:1]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "membership"):
            self.read()
        self.manifest["train_image_names"] = self.names
        self.manifest["dataset_root"] = str(self.root)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "dataset root"):
            self.read()

    def test_non_point_or_invalid_xyz_rgb_rejected(self):
        for invalid, expected in (("faces", "not a mesh"), ("gaussian", "Gaussian checkpoint"),
                                  ("nonfinite", "finite"), ("black", "random-color")):
            with self.subTest(invalid=invalid):
                self.write_ply(invalid)
                self.manifest["initialization_ply"]["sha256"] = sha256_file(self.ply)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, expected):
                    self.read()

    def test_point_count_and_alternate_rgbd_flags_rejected(self):
        self.manifest["initialization_ply"]["point_count"] = 4
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "point count disagrees"):
            self.read()
        self.kwargs["initialization_kind"] = "rgbd"
        with self.assertRaisesRegex(ValueError, "does not accept"):
            self.read()

    def test_changed_dataset_cannot_rebind_old_mesh_checkpoint(self):
        self.read()
        proof = json.loads((self.root / "run/camera_split.json").read_text())
        Image.new("RGB", (8, 6), (1, 2, 3)).save(self.dataset / "images" / self.names[0])
        with self.assertRaisesRegex(ValueError, "color RGB.*changed"):
            self.read("resume")
        with self.assertRaisesRegex(ValueError, "color RGB.*changed"):
            validate_initialization_proof(proof)


class SamplingScheduleTests(unittest.TestCase):
    def test_checkpoint_recipe_cannot_cross_rgbd_mesh_or_real_pseudo_contracts(self):
        path = Path(__file__).resolve().parents[1] / "train.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "require_r2c_checkpoint_state")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        validate = namespace["require_r2c_checkpoint_state"]
        for kind in ("train_only_rgbd", "mesh_surface"):
            state = {"schema_version": 1, "recipe": "real_only", "iteration": 20000,
                     "camera_split": {"initialization": {"kind": kind}}}
            self.assertIs(validate(state, 20000, expected_recipe="real_only"), state)
            with self.assertRaisesRegex(ValueError, "recipe differs"):
                validate(state, 20000, expected_recipe="r2c")
            if kind == "mesh_surface":
                with self.assertRaisesRegex(ValueError, "match its recipe"):
                    validate({**state, "recipe": "r2c"}, 20000)

    def test_preprocessing_randomness_and_camera_input_order_do_not_change_schedule(self):
        cameras = [SimpleNamespace(source_image_name=f"frame_{i}.png", nearest_id=[(i + 1) % 5, (i + 2) % 5]) for i in range(5)]
        first, second = CameraSamplingSchedule(cameras, 42), CameraSamplingSchedule(list(reversed(cameras)), 42)
        first_order, second_order = [], []
        for _ in range(20):
            first_order.append(first.draw_real().source_image_name)
            for _ in range(37):
                random.random()
                np.random.random()
            second_order.append(second.draw_real().source_image_name)
        self.assertEqual(first_order, second_order)
        for offset in range(0, 20, 5):
            self.assertEqual(len(set(first_order[offset:offset + 5])), 5)
        self.assertEqual(first.proof()["real_camera_order_sha256"], second.proof()["real_camera_order_sha256"])

    def test_neighbor_randomness_is_independent_of_real_draws_and_global_rng(self):
        cameras = [SimpleNamespace(source_image_name=f"frame_{i}.png", nearest_id=[0, 1, 2]) for i in range(3)]
        first, second = CameraSamplingSchedule(cameras, 42), CameraSamplingSchedule(cameras, 42)
        one, two = [], []
        for _ in range(30):
            first.draw_real()
            one.append(first.draw_neighbor(cameras[0]).source_image_name)
            random.random()
            two.append(second.draw_neighbor(cameras[0]).source_image_name)
        self.assertEqual(one, two)
        self.assertEqual(first.proof()["neighbor_camera_order_sha256"], second.proof()["neighbor_camera_order_sha256"])


if __name__ == "__main__":
    unittest.main()
