"""CPU adapter contract tests; fake ISP does not claim to test PPISP CUDA math."""

import copy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "ppisp_appearance_adapter",
    Path(__file__).resolve().parents[1] / "scene/ppisp_appearance.py",
)
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)
PPISPAppearance = adapter_module.PPISPAppearance


@dataclass
class Config:
    use_controller: bool = True
    controller_distillation: bool = True
    scheduler_decay_max_steps: int = 30000


class FakeISP(torch.nn.Module):
    def __init__(self, num_cameras, num_frames, config):
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.exposure = torch.nn.Parameter(torch.zeros(num_frames))
        self.camera_gain = torch.nn.Parameter(torch.ones(num_cameras))
        self.calls = []

    def forward(self, rgb, pixel_coords, resolution, camera_idx, frame_idx):
        self.calls.append((rgb.shape, pixel_coords, resolution, camera_idx, frame_idx))
        exposure = self.exposure[frame_idx].exp() if frame_idx >= 0 else 1.0
        return rgb * exposure * self.camera_gain[camera_idx]

    def get_regularization_loss(self):
        return self.exposure.square().mean() + self.camera_gain.square().mean()

    def create_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.01)]

    def create_schedulers(self, optimizers, max_optimization_iters):
        self.requested_iterations = max_optimization_iters
        optimizer = optimizers[0]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
            ],
            milestones=[2],
        )
        self._ppisp_scheduler = scheduler
        return [scheduler]


BACKEND = SimpleNamespace(PPISP=FakeISP, PPISPConfig=Config, __version__="1.3.0")


def cameras():
    return [
        SimpleNamespace(uid=1, colmap_id=2, image_name="frame", source_image_name="b/frame.png"),
        SimpleNamespace(uid=0, colmap_id=9, image_name="frame", source_image_name="a/frame.png"),
    ]


def make(cams=None):
    with patch.dict(sys.modules, {"ppisp": BACKEND}):
        return PPISPAppearance(cameras() if cams is None else cams, device="cpu")


def update(appearance, frame=0):
    image = torch.full((3, 2, 4), 0.2, requires_grad=True)
    loss = appearance.apply(image, frame).square().mean() + appearance.regularization_loss()
    loss.backward()
    assert image.grad is not None
    appearance.step()


class PPISPAppearanceTests(unittest.TestCase):
    def test_frames_and_physical_cameras_have_distinct_stable_indices(self):
        appearance = make()
        self.assertEqual(appearance.frame_keys, ["a/frame.png", "b/frame.png"])
        self.assertEqual(appearance.camera_ids, [2, 9])
        self.assertEqual(appearance.frame_to_camera, [1, 0])
        self.assertEqual(appearance.frame_index_by_name["b/frame.png"], 1)
        self.assertFalse(appearance.config.use_controller)
        self.assertFalse(appearance.config.controller_distillation)
        image = torch.rand(3, 2, 4, requires_grad=True)
        with torch.no_grad():
            appearance.model.exposure[1] = torch.log(torch.tensor(2.0))
            appearance.model.camera_gain[0] = 3.0
        corrected = appearance.apply(image, 1, camera_id=2)
        torch.testing.assert_close(corrected, image * 6)
        corrected.sum().backward()
        torch.testing.assert_close(image.grad, torch.full_like(image, 6.0))
        self.assertEqual(appearance.model.calls[-1], (torch.Size([2, 4, 3]), None, (4, 2), 0, 1))
        same_camera = cameras()
        same_camera[0].colmap_id = 9
        self.assertEqual(make(same_camera).model.num_cameras, 1)

    def test_unknown_frame_uses_explicit_camera_without_borrowing_training_exposure(self):
        appearance = make()
        with torch.no_grad():
            appearance.model.exposure.fill_(3)
            appearance.model.camera_gain[1] = 2
        image = torch.ones(3, 2, 4)
        torch.testing.assert_close(appearance.apply(image, -1, camera_id=9), image * 2)
        self.assertEqual(appearance.model.calls[-1][-2:], (1, -1))
        for frame, camera_id in ((-1, None), (-1, 100), (2, None), (0, 2)):
            with self.subTest(frame=frame, camera_id=camera_id), self.assertRaises(ValueError):
                appearance.apply(image, frame, camera_id)

    def test_optimizer_and_scheduler_resume_continue_exactly(self):
        first = make()
        first.setup_optimizers(20)
        for _ in range(3):
            update(first)
        saved = copy.deepcopy(first.capture())
        second = make()
        # The scheduler horizon belongs to Config, not the host's stop iteration.
        second.setup_optimizers(40)
        second.restore(saved)
        self.assertEqual(second.completed_updates, 3)
        self.assertEqual(second.schedulers[0].last_epoch, first.schedulers[0].last_epoch)
        self.assertEqual(second.optimizers[0].param_groups[0]["lr"], first.optimizers[0].param_groups[0]["lr"])
        update(first, frame=1)
        update(second, frame=1)
        for a, b in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertIsNone(a.grad)
            self.assertIsNone(b.grad)
        self.assertEqual(second.completed_updates, 4)

    def test_export_preserves_learned_mapping_without_optimizer_state(self):
        source = make()
        source.setup_optimizers(20)
        update(source)
        exported = copy.deepcopy(source.export_state())
        self.assertNotIn("optimizer_states", exported)
        self.assertNotIn("scheduler_states", exported)
        restored = make()
        restored.load_export_state(exported)
        image = torch.ones(3, 2, 4)
        torch.testing.assert_close(restored.apply(image, 0), source.apply(image, 0))
        self.assertEqual(restored.completed_updates, 1)
        self.assertEqual(restored.optimizers, [])
        with self.assertRaisesRegex(RuntimeError, "setup_optimizers"):
            restored.step()

    def test_restore_rejects_changed_frame_camera_config_or_dependency(self):
        source = make()
        source.setup_optimizers(20)
        saved = copy.deepcopy(source.capture())
        changes = {
            "frame_keys": ["other.png", "b/frame.png"],
            "camera_ids": [2, 10],
            "frame_to_camera": [0, 1],
            "config": {"use_controller": True},
            "dependency_commit": "different",
            "backend_version": "different",
        }
        for key, value in changes.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                source.restore({**saved, key: value})

    def test_bad_frame_identity_and_missing_package_fail_clearly(self):
        duplicate_names = cameras()
        duplicate_names[1].source_image_name = "b/frame.png"
        with self.assertRaisesRegex(ValueError, "unique"):
            make(duplicate_names)
        invalid_uid = cameras()
        invalid_uid[0].uid = 8
        with self.assertRaisesRegex(ValueError, "contiguous"):
            make(invalid_uid)
        with patch.dict(sys.modules, {"ppisp": None}):
            with self.assertRaisesRegex(ImportError, "pip install -e submodules/ppisp"):
                PPISPAppearance(cameras(), device="cpu")


if __name__ == "__main__":
    unittest.main()
