"""Fixed-evidence virtual-depth intervals for conservative strong-region repair.

No observed GS or baseline model selects the potential masks or interval bounds.
Those are offline source/real-evidence inputs. In-band predictions have exactly
zero auxiliary loss and gradient; out-of-band and zero-depth predictions retain
the same fixed potential denominator. Alpha is deliberately not an input.
"""

import math
import torch


def validate_interval_targets(target, potential_weight, repair_mask, lower, upper):
    shape = target.shape
    if any(value.shape != shape for value in (potential_weight, repair_mask, lower, upper)):
        raise ValueError("Interval targets, weights, class masks and bounds must share one camera grid")
    if repair_mask.dtype != torch.bool:
        raise ValueError("repair_mask must be boolean")
    if not torch.isfinite(potential_weight).all() or (potential_weight < 0).any() or (potential_weight > 1).any():
        raise ValueError("Fixed potential weights must be finite in [0,1]")
    potential = potential_weight > 0
    if not potential.any() or (repair_mask & ~potential).any():
        raise ValueError("Potential mask must be nonempty and contain every repair pixel")
    if not torch.isfinite(target[potential]).all() or not (target[potential] > 0).all():
        raise ValueError("Fixed source depth must be finite and positive on every potential pixel")
    if repair_mask.any():
        lo, hi, center = lower[repair_mask], upper[repair_mask], target[repair_mask]
        if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (lo > 0).all():
            raise ValueError("Repair intervals require finite positive lower bounds")
        if (lo > center).any() or (hi < center).any():
            raise ValueError("Repair interval must contain the fixed source depth")
    return potential


def interval_bounds(target, potential_weight, *, repair_mask=None, lower=None, upper=None,
                    weak_deadband_relative=.002, weak_deadband_absolute=0.):
    """Combine weak-source bands and fixed per-pixel repair intervals."""
    for value in (weak_deadband_relative, weak_deadband_absolute):
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Weak depth bands must be finite and nonnegative")
    if repair_mask is None:
        repair_mask = torch.zeros_like(target, dtype=torch.bool)
    if lower is None:
        if repair_mask.any():
            raise ValueError("Repair potential requires explicit offline lower/upper bounds")
        lower = torch.zeros_like(target)
    if upper is None:
        if repair_mask.any():
            raise ValueError("Repair potential requires explicit offline lower/upper bounds")
        upper = torch.zeros_like(target)
    potential = validate_interval_targets(target, potential_weight, repair_mask, lower, upper)
    target = target.detach()
    half_width = weak_deadband_absolute + weak_deadband_relative * target
    weak_lower, weak_upper = target - half_width, target + half_width
    weak = potential & ~repair_mask
    if (weak_lower[weak] <= 0).any():
        raise ValueError("A weak interval may not contain zero depth; reduce its absolute band")
    lo = torch.where(repair_mask, lower.detach(), weak_lower)
    hi = torch.where(repair_mask, upper.detach(), weak_upper)
    return potential, repair_mask, lo, hi


def virtual_interval_depth_loss(prediction, target, potential_weight, config, *,
                                repair_mask=None, lower=None, upper=None, view_loss_weight=1.):
    """One fixed-denominator auxiliary objective; confidence multiplies outside.

    The loss contains both pixel classes, so adding repair never creates a
    second optimizer update or a second independently normalized view budget.
    """
    if prediction.shape != target.shape:
        raise ValueError("Interval prediction and fixed target grids differ")
    if isinstance(view_loss_weight, bool) or not isinstance(view_loss_weight, (float, int)) or not math.isfinite(view_loss_weight) or not 0 <= view_loss_weight <= 1:
        raise ValueError("Offline view confidence must lie in [0,1]")
    epsilon = config.get("epsilon", .005)
    floor = config.get("denominator_floor", .05)
    saturation = config.get("robust_saturation", .05)
    strength = config.get("repair_strength", 1.)
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
           for value in (epsilon, floor, saturation)) or isinstance(strength, bool) or not isinstance(strength, (float, int)) or not 0 <= strength <= 1:
        raise ValueError("Interval loss requires positive scales/saturation and repair_strength in [0,1]")
    potential, repair, lo, hi = interval_bounds(
        target, potential_weight, repair_mask=repair_mask, lower=lower, upper=upper,
        weak_deadband_relative=config.get("weak_deadband_relative", config.get("deadband_relative", .002)),
        weak_deadband_absolute=config.get("weak_deadband_absolute", config.get("deadband_absolute", 0.)),
    )
    pred = prediction[potential]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite prediction in fixed interval potential")
    # ReLU and the Charbonnier have zero derivative at zero, including the
    # interval boundary. No model-dependent mask enters the denominator.
    # The selected potential is small. FP64 here prevents finite extreme FP32
    # predictions overflowing during normalization; the final scalar/gradient
    # return to the prediction dtype. It does not change any interval or mask.
    work = pred.to(torch.float64)
    excess = torch.maximum((lo[potential].to(torch.float64) - work).clamp_min(0), (work - hi[potential].to(torch.float64)).clamp_min(0))
    mode = config.get("mode", "relative_charbonnier")
    if mode == "relative_charbonnier":
        excess = excess / target[potential].detach().to(torch.float64).clamp_min(floor)
    elif mode != "absolute_charbonnier":
        raise ValueError("Unsupported interval depth loss units")
    charb = (torch.hypot(excess, excess.new_tensor(epsilon)) - epsilon).clamp_min(0)
    # Stable bounded value and influence. At charb==0 this is exactly zero.
    values = saturation * (charb / (saturation + charb))
    class_weight = torch.where(repair[potential], values.new_tensor(strength), values.new_tensor(1.))
    fixed_weights = potential_weight[potential].detach()
    result = view_loss_weight * (values * class_weight * fixed_weights).sum() / fixed_weights.sum()
    return result.to(prediction.dtype)


@torch.no_grad()
def interval_excess_statistics(prediction, target, potential_weight, config, *, repair_mask=None, lower=None, upper=None):
    potential, repair, lo, hi = interval_bounds(
        target, potential_weight, repair_mask=repair_mask, lower=lower, upper=upper,
        weak_deadband_relative=config.get("weak_deadband_relative", config.get("deadband_relative", .002)),
        weak_deadband_absolute=config.get("weak_deadband_absolute", config.get("deadband_absolute", 0.)),
    )
    if not torch.isfinite(prediction[potential]).all():
        raise FloatingPointError("Nonfinite prediction in interval probe potential")
    excess = torch.maximum((lo - prediction).clamp_min(0), (prediction - hi).clamp_min(0))
    relative = excess[potential] / target[potential].clamp_min(config.get("denominator_floor", .05))
    weights = potential_weight[potential]
    return {"relative_interval_excess": float((relative * weights).sum().item() / weights.sum().item()),
            "potential_pixels": int(potential.sum().item()), "repair_potential_pixels": int((potential & repair).sum().item()),
            "outside_interval_pixels": int(((excess > 0) & potential).sum().item()),
            "zero_prediction_fraction": float((prediction[potential] <= 0).float().mean().item())}
