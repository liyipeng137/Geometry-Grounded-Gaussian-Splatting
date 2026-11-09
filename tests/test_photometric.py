"""CPU 验证光度组合数学；CUDA fused_ssim 内核由实际 GPU 验证覆盖。"""

import ast
from enum import Enum
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.photometric import get_photometric_backend, photometric_loss


class PhotometricTests(unittest.TestCase):
    def test_public_package_fallback_normalizes_uint8_and_uses_valid_nchw(self):
        prediction = torch.tensor(
            [[[0.2, 0.7]]], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([[[0, 255]]], dtype=torch.uint8)
        calls = []

        def ssim(pred, reference, padding):
            calls.append((pred, reference, padding))
            return pred.new_tensor(0.25)

        with patch.dict(sys.modules, {"fused_ssim": SimpleNamespace(fused_ssim=ssim)}):
            self.assertEqual(get_photometric_backend(), "torch_l1+fused_ssim(valid)")
            total, l1, similarity = photometric_loss(prediction, target, 0.2)
        self.assertAlmostEqual(l1.item(), 0.25)
        self.assertAlmostEqual(similarity.item(), 0.25)
        self.assertAlmostEqual(total.item(), 0.35)
        self.assertEqual(calls[0][0].shape, (1, 1, 1, 2))
        self.assertEqual(calls[0][2], "valid")
        torch.testing.assert_close(
            calls[0][1], torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
        )
        total.backward()
        torch.testing.assert_close(
            prediction.grad, torch.tensor([[[0.4, -0.4]]], dtype=torch.float64)
        )

    def test_float_target_scale_is_preserved_and_both_gradients_are_combined(self):
        prediction = torch.tensor(
            [[[0.7, 0.45]]], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([[[0.5, 0.75]]], dtype=torch.float64)

        def ssim(pred, reference, padding):
            # 用可解析导数的替身验证组合，不把这个函数当作真实 SSIM 算法。
            return 1 - (pred - reference).square().mean()

        with patch.dict(sys.modules, {"fused_ssim": SimpleNamespace(fused_ssim=ssim)}):
            total, l1, similarity = photometric_loss(prediction, target, 0.2)
        self.assertAlmostEqual(l1.item(), 0.25)
        self.assertAlmostEqual(similarity.item(), 0.935)
        self.assertAlmostEqual(total.item(), 0.213)
        total.backward()
        torch.testing.assert_close(
            prediction.grad, torch.tensor([[[0.44, -0.46]]], dtype=torch.float64)
        )

    def test_batched_input_does_not_gain_extra_dimension(self):
        prediction = torch.ones(2, 3, 12, 12)
        target = torch.zeros_like(prediction)

        def ssim(pred, reference, padding):
            self.assertEqual(pred.shape, prediction.shape)
            self.assertEqual(reference.shape, prediction.shape)
            self.assertEqual(padding, "valid")
            return pred.new_tensor(0.2)

        with patch.dict(sys.modules, {"fused_ssim": SimpleNamespace(fused_ssim=ssim)}):
            total, l1, _ = photometric_loss(prediction, target, 1.0)
        self.assertAlmostEqual(l1.item(), 1.0)
        self.assertAlmostEqual(total.item(), 0.8, places=6)

    def test_custom_extension_is_used_without_reinterpreting_its_contract(self):
        prediction = torch.zeros(3, 12, 12)
        target = torch.zeros(3, 12, 12, dtype=torch.uint8)
        expected = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))

        def custom(pred, reference, weight):
            self.assertIs(pred, prediction)
            self.assertIs(reference, target)
            self.assertEqual(weight, 0.2)
            return expected

        def unexpected_ssim(*args, **kwargs):
            raise AssertionError(
                "Standard fallback must not run when custom API exists"
            )

        module = SimpleNamespace(photometric_loss=custom, fused_ssim=unexpected_ssim)
        with patch.dict(sys.modules, {"fused_ssim": module}):
            self.assertEqual(
                get_photometric_backend(), "custom_fused_ssim.photometric_loss"
            )
            result = photometric_loss(prediction, target, 0.2)
        self.assertIs(result, expected)

    def test_custom_backend_failure_is_not_silently_replaced_by_fallback(self):
        def custom(*args):
            raise RuntimeError("custom kernel failed")

        with patch.dict(
            sys.modules, {"fused_ssim": SimpleNamespace(photometric_loss=custom)}
        ):
            with self.assertRaisesRegex(RuntimeError, "custom kernel failed"):
                photometric_loss(torch.ones(3, 12, 12), torch.zeros(3, 12, 12), 0.2)

    def test_fallback_rejects_shape_broadcasting(self):
        with patch.dict(sys.modules, {"fused_ssim": SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, "matching CHW or NCHW"):
                photometric_loss(torch.ones(3, 12, 12), torch.zeros(1, 12, 12), 0.2)

    def test_appearance_still_uses_corrected_l1_but_original_render_ssim(self):
        # 仅载入已有外观函数，避免导入 scene/CUDA/PatchMatch；测试函数本身未重写。
        path = Path(__file__).resolve().parents[1] / "utils" / "loss_utils.py"
        module_ast = ast.parse(path.read_text())
        function = next(
            node
            for node in module_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "photometric_rgb_loss"
        )
        models = Enum("Appearance", ["NO", "GS", "GOF", "PGSR", "PPISP"])
        namespace = {
            "torch": torch,
            "photometric_loss": photometric_loss,
            "GaussianModel": SimpleNamespace(App_model=models),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        image = torch.tensor([[[0.2, 0.4]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0.0, 0.8]]], dtype=torch.float64)
        exposure = torch.tensor(
            [math.log(2), 0.1], dtype=torch.float64, requires_grad=True
        )
        model = SimpleNamespace(
            app_model=models.PGSR, get_apperance_embedding=lambda _: exposure
        )

        def ssim(pred, reference, padding):
            return 1 - (pred - reference).square().mean()

        with patch.dict(sys.modules, {"fused_ssim": SimpleNamespace(fused_ssim=ssim)}):
            total, l1 = namespace["photometric_rgb_loss"](image, target, model, 0, 0.2)
        self.assertAlmostEqual(l1.item(), 0.3)
        self.assertAlmostEqual(total.item(), 0.26)
        total.backward()
        torch.testing.assert_close(
            image.grad, torch.tensor([[[0.84, 0.72]]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            exposure.grad, torch.tensor([0.48, 0.8], dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
