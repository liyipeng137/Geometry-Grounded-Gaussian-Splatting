"""CPU physical checks for conservative interval repair, independent of any GS."""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("intervals_cpu", ROOT / "scene/virtual_geometry_intervals.py")
intervals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(intervals)


class IntervalTests(unittest.TestCase):
    config = {"epsilon": .005, "robust_saturation": .05, "weak_deadband_relative": .002}

    def targets(self):
        target = torch.full((1, 2, 3), .8)
        weights = torch.ones_like(target)
        repair = torch.ones_like(target, dtype=torch.bool)
        return target, weights, repair, target * .98, target * 1.02

    def test_stable_including_interval_boundaries_has_exact_zero_loss_and_gradient(self):
        target, weights, repair, lo, hi = self.targets()
        pred = target.clone()
        pred[0, 0] = lo[0, 0]
        pred[0, 0, 0] = hi[0, 0, 0]
        pred.requires_grad_()
        loss = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue((pred.grad == 0).all())

    def test_near_wall_four_percent_fold_is_penalized_but_two_percent_is_free(self):
        target, weights, repair, lo, hi = self.targets()
        pred = (target * 1.04).requires_grad_()
        loss = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertTrue((pred.grad > 0).all())
        free = (target * 1.01).requires_grad_()
        zero = intervals.virtual_interval_depth_loss(free, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        zero.backward()
        self.assertTrue((free.grad == 0).all())

    def test_zero_prediction_cannot_remove_potential_or_escape_loss(self):
        target, weights, repair, lo, hi = self.targets()
        pred = torch.zeros_like(target, requires_grad=True)
        loss = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertTrue((pred.grad < 0).all())
        report = intervals.interval_excess_statistics(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        self.assertEqual(report["potential_pixels"], 6)
        self.assertEqual(report["outside_interval_pixels"], 6)
        self.assertEqual(report["zero_prediction_fraction"], 1.)

    def test_near_wall_fold_moves_both_sides_toward_tube_without_touching_stable_centers(self):
        target, weights, repair, lo, hi = self.targets()
        factors = torch.tensor([[[.96, 1., 1.04], [1.04, 1., .96]]])
        prediction = (target * factors).requires_grad_()
        loss = intervals.virtual_interval_depth_loss(prediction, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertTrue((prediction.grad[factors < 1] < 0).all())
        self.assertTrue((prediction.grad[factors > 1] > 0).all())
        self.assertTrue((prediction.grad[factors == 1] == 0).all())

    def test_fixed_potential_denominator_and_external_view_confidence(self):
        target, weights, repair, lo, hi = self.targets()
        pred = target.clone(); pred[0, 0, 0] = 0.
        one_bad = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        all_bad = intervals.virtual_interval_depth_loss(torch.zeros_like(pred), target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        quarter = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi, view_loss_weight=.25)
        self.assertAlmostEqual(float(one_bad * 6), float(all_bad), places=7)
        self.assertEqual(float(quarter), float(one_bad * .25))

    def test_mixed_classes_keep_weak_point_two_percent_band_distinct(self):
        target, weights, repair, lo, hi = self.targets()
        repair[0, 0, 0] = False
        pred = (target * 1.01).requires_grad_()
        loss = intervals.virtual_interval_depth_loss(pred, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertGreater(float(pred.grad[0, 0, 0]), 0.)
        self.assertTrue((pred.grad[repair] == 0).all())

    def test_nonfinite_prediction_and_zero_containing_interval_fail(self):
        target, weights, repair, lo, hi = self.targets()
        with self.assertRaises(FloatingPointError):
            intervals.virtual_interval_depth_loss(target * float('nan'), target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        with self.assertRaises(ValueError):
            intervals.virtual_interval_depth_loss(target, target, weights, self.config, repair_mask=repair, lower=torch.zeros_like(lo), upper=hi)

    def test_finite_extreme_conflict_has_bounded_finite_loss_and_gradient(self):
        target, weights, repair, lo, hi = self.targets()
        prediction = torch.full_like(target, torch.finfo(torch.float32).max).requires_grad_()
        loss = intervals.virtual_interval_depth_loss(prediction, target, weights, self.config, repair_mask=repair, lower=lo, upper=hi)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLessEqual(float(loss.detach()), .050001)


if __name__ == '__main__':
    unittest.main()
