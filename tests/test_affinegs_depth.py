import unittest

import torch

from utils.affinegs_depth import affinegs_depth_loss


class AffineGSDepthTests(unittest.TestCase):
    def test_prior_scale_invariance_in_value_and_gradient(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 51), torch.linspace(-1, 1, 67), indexing="ij")
        p = (2 + 0.4 * x + 0.3 * y + (x > 0.3)).unsqueeze(0)
        z = (p + 0.12 * (5 * x + y).sin()).requires_grad_()
        first = affinegs_depth_loss(z, p)
        second = affinegs_depth_loss(z, p * 4)
        grad, = torch.autograd.grad(first, z)
        other, = torch.autograd.grad(second, z)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(grad, other)
        self.assertGreater(first.item(), 0)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_invalid_depths_do_not_poison_guided_fit_or_backward(self):
        x = torch.linspace(1, 2, 32).expand(1, 24, -1).clone()
        z = x.flip(-1).contiguous()
        z[:, :5] = float("nan")
        x[:, -5:] = 0
        z.requires_grad_()
        loss = affinegs_depth_loss(z, x)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue((z.grad[:, :5] == 0).all())
        self.assertTrue((z.grad[:, -5:] == 0).all())

    def test_flat_prior_empty_support_and_constant_render(self):
        z = torch.ones(1, 10, 17, requires_grad=True)
        for p in (torch.ones_like(z), torch.zeros_like(z), torch.full_like(z, float("nan"))):
            loss = affinegs_depth_loss(z, p)
            self.assertEqual(loss.item(), 0)
            self.assertTrue((torch.autograd.grad(loss, z)[0] == 0).all())
        prior = torch.linspace(1, 2, 17).expand_as(z)
        loss = affinegs_depth_loss(z, prior)
        grad, = torch.autograd.grad(loss, z)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
