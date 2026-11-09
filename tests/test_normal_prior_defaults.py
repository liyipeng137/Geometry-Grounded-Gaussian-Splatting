"""Normal-prior demo 日程及有向损失的 CPU 回归，不调用 CUDA renderer。"""

from argparse import ArgumentParser
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from arguments import OptimizationParams
from utils import training_losses as losses


def options(**overrides):
    values = dict(
        iterations=20000,
        regularization_from_iter=30000,
        lambda_normal_prior=0.1,
        normal_prior_from_iter=1,
        lambda_depth_prior=0.0,
        depth_prior_from_iter=3000,
        depth_prior_ramp_iters=0,
        lambda_dssim=0.2,
        lambda_depth_normal=0.05,
        lambda_mask=0.0,
        lambda_multi_view_ncc=0.0,
        lambda_multi_view_geo=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def camera(prior=None, mask=None):
    return SimpleNamespace(
        uid=0,
        original_image=torch.zeros(3, 2, 3, dtype=torch.float64),
        normal_prior=prior,
        normal_prior_mask=mask,
        depth_prior=None,
        gt_mask=None,
    )


def zero_photo(image, *_):
    zero = image.sum() * 0
    return zero, zero


class NormalPriorDefaultsTests(unittest.TestCase):
    def test_cli_defaults_enable_normal_from_first_iteration(self):
        parser = ArgumentParser()
        params = OptimizationParams(parser)
        opt = params.extract(parser.parse_args([]))
        self.assertEqual(opt.iterations, 20000)
        self.assertEqual(opt.lambda_normal_prior, 0.1)
        self.assertEqual(opt.normal_prior_from_iter, 1)

    def test_demo_boundaries_and_hold_after_15000(self):
        expected = {
            1: 0.1, 3000: 0.1, 3001: 0.15, 5000: 0.15,
            5001: 0.2, 7000: 0.2, 7001: 0.2, 11000: 0.2,
            11001: 0.19998749843730468,
            15000: 0.1, 15001: 0.1, 20000: 0.1,
        }
        for iteration, value in expected.items():
            with self.subTest(iteration=iteration):
                self.assertAlmostEqual(
                    losses.normal_prior_weight(iteration, options()), value, places=12
                )
        self.assertAlmostEqual(
            losses.normal_prior_weight(6000, options(lambda_normal_prior=0.2)), 0.4
        )

    def test_short_budget_truncates_geometry_but_keeps_init_thresholds(self):
        for iteration, value in (
            (7000, 0.2), (7001, 0.2), (8500, 0.2),
            (8501, 0.19996665555185064), (10000, 0.1),
        ):
            with self.subTest(iterations=10000, iteration=iteration):
                self.assertAlmostEqual(
                    losses.normal_prior_weight(iteration, options(iterations=10000)),
                    value, places=12,
                )
        for budget, iteration, value in (
            (200, 200, 0.1), (4000, 3000, 0.1), (4000, 3001, 0.15),
            (4000, 4000, 0.15), (7000, 5000, 0.15), (7000, 5001, 0.2),
            (7000, 7000, 0.2),
        ):
            with self.subTest(iterations=budget, iteration=iteration):
                self.assertAlmostEqual(
                    losses.normal_prior_weight(iteration, options(iterations=budget)),
                    value,
                )

    def test_zero_weight_missing_frame_and_explicit_start(self):
        cam = camera(torch.ones(3, 2, 3))
        enabled = losses.loss_schedule(cam, options(), 1)
        self.assertTrue(enabled.normal_prior_enabled)
        self.assertTrue(enabled.needs_depth)
        self.assertAlmostEqual(enabled.normal_prior_weight, 0.1)
        for current, opt in (
            (cam, options(lambda_normal_prior=0)),
            (camera(), options()),
            (cam, options(normal_prior_from_iter=7)),
        ):
            schedule = losses.loss_schedule(current, opt, 1)
            self.assertFalse(schedule.normal_prior_enabled)
            self.assertFalse(schedule.needs_depth)
        self.assertTrue(
            losses.loss_schedule(cam, options(normal_prior_from_iter=7), 7).normal_prior_enabled
        )

    def test_signed_cosine_alpha_threshold_and_mask_intersection(self):
        prior = torch.zeros(3, 2, 3, dtype=torch.float64)
        prior[2] = 1
        # 仅像素 0/1 满足三种 mask；像素 2 恰好 alpha=.5，必须排除。
        prior_mask = torch.tensor([[[True, True, True], [False, True, True]]])
        valid = torch.tensor([[[True, True, True], [True, False, True]]])
        alpha = torch.tensor([[[0.8, 0.9, 0.5], [0.8, 0.8, 0.49]]], dtype=torch.float64, requires_grad=True)
        rendered = -prior.clone()
        derived = -prior.clone()
        rendered[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        derived[:, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
        rendered.requires_grad_()
        derived.requires_grad_()
        image = torch.zeros_like(prior, requires_grad=True)
        cam = camera(prior, prior_mask)
        opt = options()
        schedule = losses.loss_schedule(cam, opt, 6000)
        pkg = {
            "render": image, "median_depth": torch.ones(1, 2, 3),
            "normal": rendered, "mask": alpha,
        }
        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo),
            patch.object(losses, "_depth_to_normal", return_value=(derived, valid)),
        ):
            result = losses.compute_training_losses(pkg, cam, None, opt, schedule)
        # 正交像素贡献 1+1，反向像素贡献 2+2；只对这两个像素平均。
        self.assertAlmostEqual(result.normal_prior.item(), 3.0)
        self.assertAlmostEqual(result.weighted_terms(opt, schedule)["normal_prior"].item(), 0.6)
        self.assertAlmostEqual(result.total.item(), 0.6)
        result.total.backward()
        expected_gradient = torch.zeros_like(prior)
        expected_gradient[2, 0, 0] = -0.1  # .2 权重 / 2 有效像素。
        torch.testing.assert_close(rendered.grad, expected_gradient)
        torch.testing.assert_close(derived.grad, expected_gradient)
        self.assertIsNone(alpha.grad)

    def test_empty_qualified_normal_domain_is_finite_zero(self):
        prior = torch.zeros(3, 2, 3)
        prior[2] = 1
        cam = camera(prior, torch.ones(1, 2, 3, dtype=torch.bool))
        opt = options()
        schedule = losses.loss_schedule(cam, opt, 1)
        image = torch.zeros_like(prior, requires_grad=True)
        pkg = {
            "render": image, "median_depth": torch.ones(1, 2, 3),
            "normal": -prior, "mask": torch.full((1, 2, 3), 0.5),
        }
        with (
            patch.object(losses, "_photometric_rgb_loss", side_effect=zero_photo),
            patch.object(losses, "_depth_to_normal", return_value=(prior, torch.ones(1, 2, 3, dtype=torch.bool))),
        ):
            result = losses.compute_training_losses(pkg, cam, None, opt, schedule)
        self.assertEqual(result.normal_prior.item(), 0)
        self.assertEqual(result.total.item(), 0)
        result.total.backward()
        torch.testing.assert_close(image.grad, torch.zeros_like(image))


if __name__ == "__main__":
    unittest.main()
