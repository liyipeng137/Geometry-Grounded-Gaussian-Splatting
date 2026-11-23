"""AffineGS-style real-view depth supervision for a GGGS median camera-Z render.

The positive-scale-aligned monocular prior supplies relative disparity only.
The guided affine fields and re-alignment target are estimated without autograd.
Equations 1–5 and 9–10: https://arxiv.org/abs/2611.07709
"""

import math

import torch
import torch.nn.functional as F


# With the existing outer depth weight of 0.4 these give 0.1 for the
# disparity gradient loss and 0.4 for the re-aligned depth loss.
GRADIENT_RATIO = 0.25
MAX_LOSS_EDGE = 960
GUIDE_EPSILON = 0.01  # guide is normalized to [0, 1]
GUIDE_RADIUS_FRACTION = 0.5  # fraction of the shorter image dimension


def _quantiles(values, fractions):
    # Raster-order stratification bounds sorting cost without touching training RNG.
    sample = values[::max(1, math.ceil(values.numel() / 16384))]
    return torch.quantile(sample, sample.new_tensor(fractions))


def _box_sum(values, radius):
    """Clipped square windows via two prefix sums (work independent of radius)."""
    for dim in (-1, -2):
        size = values.size(dim)
        coords = torch.arange(size, device=values.device)
        zero = torch.zeros_like(values.narrow(dim, 0, 1))
        prefix = torch.cat((zero, values.cumsum(dim)), dim=dim)
        values = (prefix.index_select(dim, (coords + radius + 1).clamp_max(size))
                  - prefix.index_select(dim, (coords - radius).clamp_min(0)))
    return values


def _guided_fields(guide, fields, valid):
    """Fast mask-aware guided filter. Fit on coarse moments, apply the full guide."""
    h, w = guide.shape
    stride = max(1, math.ceil(max(h, w) / 96))
    size = (math.ceil(h / stride), math.ceil(w / stride))
    moments = torch.cat((torch.ones_like(guide)[None], guide[None], guide.square()[None],
                         fields, guide * fields)) * valid
    moments = F.interpolate(moments[None], size=size, mode="area")[0]
    radius = max(1, round(min(size) * GUIDE_RADIUS_FRACTION))
    sums = _box_sum(moments, radius)
    count = sums[0].clamp_min(1e-8)
    mean_i, mean_ii = sums[1] / count, sums[2] / count
    mean_p, mean_ip = sums[3:5] / count, sums[5:7] / count
    slope = (mean_ip - mean_i * mean_p) / ((mean_ii - mean_i.square()).clamp_min(0) + GUIDE_EPSILON)
    intercept = mean_p - slope * mean_i
    supported = (sums[0] > 0).to(guide.dtype)
    fit_sum = _box_sum(torch.cat((slope, intercept, supported[None])) * supported, radius)
    fit = fit_sum[:4] / fit_sum[4].clamp_min(1)
    fit = F.interpolate(fit[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return fit[:2] * guide + fit[2:]


def _gradient(field):
    dx = (field[1:-1, 2:] - field[1:-1, :-2]) * 0.5
    dy = (field[2:, 1:-1] - field[:-2, 1:-1]) * 0.5
    return torch.stack((dx, dy))


def affinegs_depth_loss(render_depth, prior_depth):
    """Return gradient matching + dynamically re-aligned L1 as a single depth term."""
    stride = max(1, math.ceil(max(render_depth.shape[-2:]) / MAX_LOSS_EDGE))
    z = render_depth[0, ::stride, ::stride].float()
    p = prior_depth[0, ::stride, ::stride].detach().float()
    good_prior = torch.isfinite(p) & (p > 0)
    valid = good_prior & torch.isfinite(z.detach()) & (z.detach() > 0)
    z = torch.where(valid, z, 0.0)
    if min(z.shape) < 3:
        return z.sum() * 0
    interior = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
                & valid[:-2, 1:-1] & valid[2:, 1:-1])
    if not interior.any():
        return z.sum() * 0
    prior_inv = torch.where(good_prior, p, 1.0).reciprocal()
    p_lo, p_mid, p_hi = _quantiles(prior_inv[good_prior], [0.01, 0.5, 0.99])
    # A map with no relative disparity range supplies no calibratable signal.
    if p_hi - p_lo <= torch.finfo(p.dtype).eps * p_hi:
        return z.sum() * 0
    dz = _gradient(z)
    with torch.no_grad():
        z_lo, z_hi = _quantiles(z[valid].reciprocal(), [0.01, 0.99])
        constant_render = z_hi - z_lo <= torch.finfo(z.dtype).eps * z_hi
        z_lo, z_hi = (torch.where(constant_render, z_lo * (p_lo / p_mid), z_lo),
                      torch.where(constant_render, z_hi * (p_hi / p_mid), z_hi))
        guide = ((prior_inv - p_lo) / (p_hi - p_lo)).clamp(0, 1)
        d = z_lo + (z_hi - z_lo) * guide
        dd = _gradient(d)
        target_mag = torch.linalg.vector_norm(dd, dim=0)
        depth_mag = torch.linalg.vector_norm(dz.detach(), dim=0)
        dc = d[1:-1, 1:-1]
        # Flat stencils cannot determine a local scale; identity is well-defined.
        raw_scale = target_mag / (dc.square() * depth_mag + 1e-5)
        observable = interior & (depth_mag > 1e-5) & (target_mag > 1e-5)
        raw_scale = torch.where(observable, raw_scale, 1.0)
        lo, hi = _quantiles(raw_scale[interior], [0.05, 0.95]).clamp_min(1e-3)
        scale = F.pad(raw_scale.clamp(lo, hi), (1, 1, 1, 1), value=1.0)
        shift = d.reciprocal() - scale * z
        support = F.pad(interior, (1, 1, 1, 1))
        scale, shift = _guided_fields(guide, torch.stack((scale, shift)), support)
        scale = scale.clamp(lo, hi)
        corrected = ((d.reciprocal() - shift) / scale)[valid]
        zv = z[valid]
        cx = corrected - corrected.mean()
        cy = zv - zv.mean()
        var = cx.square().mean()
        gain = (cx * cy).mean() / var.clamp_min(torch.finfo(z.dtype).tiny)
        fit_ok = (var > 1e-8 * corrected.square().mean()) & (gain > 0) & torch.isfinite(gain)
        target = zv.mean() + torch.where(fit_ok, gain, 1.0) * cx
        factor = -scale[1:-1, 1:-1] * dc.square()
        mag_lo, mag_hi = _quantiles(target_mag[interior], [0.05, 0.95])

    predicted = factor * dz
    predicted_mag = torch.linalg.vector_norm(predicted, dim=0)
    with torch.no_grad():
        pred_lo, pred_hi = _quantiles(predicted_mag[interior], [0.05, 0.95])
    magnitude_loss = (predicted_mag.clamp(pred_lo, pred_hi) - target_mag.clamp(mag_lo, mag_hi)).square()
    # At zero gradient an angle is undefined; the magnitude term still applies.
    directional = interior & (target_mag > 1e-4) & (predicted_mag.detach() > 1e-4)
    cosine = F.cosine_similarity(predicted, dd, dim=0, eps=1e-4).clamp(-1, 1)
    angle_loss = torch.where(directional, 1 - cosine, 0.0).sum() / interior.sum()
    gradient_loss = magnitude_loss[interior].mean() + angle_loss
    return (z[valid] - target).abs().mean() + GRADIENT_RATIO * gradient_loss
