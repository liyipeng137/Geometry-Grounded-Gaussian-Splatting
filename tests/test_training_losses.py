"""CPU 数值/梯度回归；不导入 Gaussian CUDA 扩展或执行渲染。"""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils import training_losses as losses


def options(**overrides):
    values = dict(
        regularization_from_iter=10,
        lambda_normal_prior=0.0,
        normal_prior_from_iter=7,
        lambda_depth_prior=0.4,
        depth_prior_from_iter=5,
        depth_prior_ramp_iters=4,
        lambda_dssim=0.2,
        lambda_depth_normal=0.05,
        lambda_mask=0.0,
        lambda_multi_view_ncc=0.0,
        lambda_multi_view_geo=0.0,
        real_depth_loss_mode="absolute",
        real_depth_relative_epsilon=0.01,
        real_depth_relative_floor=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def camera(**overrides):
    values = dict(
        uid=0,
        original_image=torch.zeros(3, 2, 2),
        normal_prior=None,
        normal_prior_mask=None,
        depth_prior=None,
        gt_mask=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def zero_photo(image, target, gaussians, uid, lambda_dssim):
    zero = image.sum() * 0
    return zero, zero


class TrainingLossTests(unittest.TestCase):
    def test_schedule_obeys_start_frames_and_depth_ramp(self):
        cam = camera(depth_prior=torch.ones(1, 2, 2), normal_prior=torch.ones(3, 2, 2))
        opt = options(lambda_normal_prior=0.3)
        before = losses.loss_schedule(cam, opt, 4)
        self.assertFalse(before.needs_depth)
        self.assertEqual(before.depth_prior_weight, 0.0)
        start = losses.loss_schedule(cam, opt, 5)
        self.assertTrue(start.depth_prior_enabled)
        self.assertFalse(start.normal_prior_enabled)
        self.assertAlmostEqual(start.depth_prior_weight, 0.1)
        self.assertTrue(losses.loss_schedule(cam, opt, 7).normal_prior_enabled)
        self.assertAlmostEqual(
            losses.loss_schedule(cam, opt, 8).depth_prior_weight, 0.4
        )
        self.assertTrue(losses.loss_schedule(cam, opt, 10).reg_enabled)
        self.assertFalse(losses.loss_schedule(camera(), opt, 8).needs_depth)

    def test_no_ramp_enables_full_weight_at_start(self):
        self.assertEqual(losses.ramp_weight(4, 5, 0), 0)
        self.assertEqual(losses.ramp_weight(5, 5, 0), 1)

    def test_absolute_depth_counts_missing_prediction_and_excludes_invalid_prior(self):
        prediction = torch.tensor(
            [0.0, 3.0, 100.0], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([2.0, 2.0, 0.0], dtype=torch.float64)
        value = losses.masked_l1_depth_loss(prediction, target, target > 0.01)
        self.assertAlmostEqual(value.item(), 1.5)
        value.backward()
        torch.testing.assert_close(
            prediction.grad, torch.tensor([-0.5, 0.5, 0.0], dtype=torch.float64)
        )

    def test_relative_depth_matches_analytical_value_and_gradient_with_floor(self):
        prediction = torch.tensor(
            [1.2, 4.0, 0.03, 9.0], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([1.0, 2.0, 0.01, 0.0], dtype=torch.float64)
        value = losses.relative_charbonnier_depth_loss(
            prediction,
            target,
            torch.ones(4, dtype=torch.bool),
            eps=0.1,
            denominator_floor=0.05,
        )
        residuals = [0.2, 1.0, 0.4]
        expected = sum(math.sqrt(r * r + 0.01) for r in residuals) / 3 - 0.1
        self.assertAlmostEqual(value.item(), expected, places=12)
        value.backward()
        expected_grad = [
            r / math.sqrt(r * r + 0.01) / scale / 3
            for r, scale in zip(residuals, [1.0, 2.0, 0.05])
        ] + [0.0]
        torch.testing.assert_close(
            prediction.grad, torch.tensor(expected_grad, dtype=torch.float64)
        )

    def test_empty_and_nonfinite_depth_priors_return_finite_zero(self):
        prediction = torch.ones(3)
        target = torch.tensor([float("nan"), float("inf"), -1.0])
        value = losses.relative_charbonnier_depth_loss(
            prediction, target, torch.ones(3, dtype=torch.bool)
        )
        self.assertEqual(value.item(), 0)
        self.assertEqual(
            losses.masked_l1_depth_loss(
                prediction, target, torch.zeros(3, dtype=torch.bool)
            ).item(),
            0,
        )

    def test_disabled_geometry_needs_no_depth_buffer_and_does_not_double_count_l1(self):
        image = torch.full((3, 2, 2), 0.25, requires_grad=True)
        cam = camera(original_image=torch.zeros(3, 2, 2, dtype=torch.uint8))
        opt = options()

        def photo(prediction, target, *_):
            self.assertEqual(target.dtype, torch.uint8)
            return prediction.mean() * 2, prediction.mean()

        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=photo),
            patch.object(
                losses,
                "_depth_to_normal",
                side_effect=AssertionError("disabled normal path called"),
            ),
        ):
            result = losses.compute_training_losses(
                {"render": image}, cam, None, opt, losses.loss_schedule(cam, opt, 1)
            )
        self.assertAlmostEqual(result.total.item(), 0.5)
        self.assertIn("l1", result.raw_terms())
        self.assertNotIn(
            "l1", result.weighted_terms(opt, losses.loss_schedule(cam, opt, 1))
        )
        result.total.backward()
        torch.testing.assert_close(
            image.grad, torch.full_like(image, 2 / image.numel())
        )

    def test_normal_consistency_keeps_full_image_denominator_and_gradient(self):
        image = torch.zeros(3, 2, 2, requires_grad=True)
        rendered_normal = torch.zeros(3, 2, 2)
        rendered_normal[2] = 1
        rendered_normal.requires_grad_()
        depth_normal = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, -1.0], [0.0, 1.0]],
            ]
        )
        valid = torch.tensor([[[True, False], [True, False]]])
        cam = camera()
        opt = options(lambda_depth_normal=1.0)
        pkg = {
            "render": image,
            "median_depth": torch.ones(1, 2, 2),
            "normal": rendered_normal,
        }
        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo),
            patch.object(
                losses, "_depth_to_normal", return_value=(depth_normal, valid)
            ),
        ):
            result = losses.compute_training_losses(
                pkg, cam, None, opt, losses.loss_schedule(cam, opt, 10)
            )
        self.assertAlmostEqual(result.depth_normal.item(), 0.25)
        result.total.backward()
        expected = -depth_normal * valid / 4
        torch.testing.assert_close(rendered_normal.grad, expected)

    def test_depth_weight_ramp_applies_once_and_fixed_prior_mask_is_preserved(self):
        image = torch.full((3, 1, 3), 0.25, requires_grad=True)
        depth = torch.tensor([[[0.0, 3.0, 100.0]]], requires_grad=True)
        cam = camera(depth_prior=torch.tensor([[[2.0, 2.0, 0.0]]]))
        opt = options()
        schedule = losses.loss_schedule(cam, opt, 5)

        def photo(prediction, *_):
            return prediction.mean(), prediction.mean()

        with patch.object(losses, "_photometric_rgb_loss", side_effect=photo):
            result = losses.compute_training_losses(
                {"render": image, "median_depth": depth}, cam, None, opt, schedule
            )
        self.assertAlmostEqual(result.depth_prior.item(), 1.5)
        # RGB=.25，加上 depth=.4 * (1/4) * 1.5；不应重复乘 ramp 或丢弃深度0的预测。
        self.assertAlmostEqual(result.total.item(), 0.4, places=6)
        result.total.backward()
        torch.testing.assert_close(depth.grad, torch.tensor([[[-0.05, 0.05, 0.0]]]))

    def test_normal_prior_remains_unsigned_and_does_not_differentiate_alpha(self):
        image = torch.zeros(3, 2, 2, requires_grad=True)
        normal = torch.zeros(3, 2, 2)
        normal[2] = 1
        alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
        cam = camera(
            normal_prior=-normal,
            normal_prior_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        )
        opt = options(lambda_normal_prior=1.0, lambda_depth_normal=0.0)
        pkg = {
            "render": image,
            "median_depth": torch.ones(1, 2, 2),
            "normal": normal,
            "mask": alpha,
        }
        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo),
            patch.object(
                losses,
                "_depth_to_normal",
                return_value=(normal, torch.ones(1, 2, 2, dtype=torch.bool)),
            ),
        ):
            result = losses.compute_training_losses(
                pkg, cam, None, opt, losses.loss_schedule(cam, opt, 7)
            )
        self.assertAlmostEqual(result.normal_prior.item(), 0.0)
        result.total.backward()
        self.assertIsNone(alpha.grad)

    def test_mask_only_penalizes_background_pixels(self):
        image = torch.zeros(3, 1, 2, requires_grad=True)
        alpha = torch.tensor([[[0.8, 0.8]]], requires_grad=True)
        cam = camera(gt_mask=torch.tensor([[[1.0, 0.0]]]))
        opt = options(lambda_mask=1.0)
        with patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo):
            result = losses.compute_training_losses(
                {"render": image, "mask": alpha},
                cam,
                None,
                opt,
                losses.loss_schedule(cam, opt, 1),
            )
        self.assertAlmostEqual(result.mask.item(), -math.log(0.2) / 2, places=6)
        result.total.backward()
        torch.testing.assert_close(alpha.grad, torch.tensor([[[0.0, 2.5]]]))

    def test_patchmatch_uses_explicit_weights_and_requires_dependency_only_when_enabled(
        self,
    ):
        image = torch.zeros(3, 2, 2, requires_grad=True)
        cam = camera()
        opt = options(
            lambda_multi_view_ncc=0.2, lambda_multi_view_geo=0.3, lambda_depth_normal=0
        )
        pkg = {
            "render": image,
            "median_depth": torch.ones(1, 2, 2),
            "normal": torch.ones(3, 2, 2),
        }
        depth_normal = torch.zeros(3, 2, 2)
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        schedule = losses.loss_schedule(cam, opt, 10)
        calls = []

        def patchmatch(*args):
            calls.append(args)
            return image.new_tensor(2.0), image.new_tensor(3.0)

        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo),
            patch.object(
                losses, "_depth_to_normal", return_value=(depth_normal, valid)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "PatchMatch"):
                losses.compute_training_losses(pkg, cam, None, opt, schedule)
            result = losses.compute_training_losses(
                pkg, cam, None, opt, schedule, patchmatch=patchmatch, iteration=10
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][4], 10)
        self.assertIs(calls[0][5], depth_normal)
        self.assertAlmostEqual(result.total.item(), 1.3, places=6)


if __name__ == "__main__":
    unittest.main()
