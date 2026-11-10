"""虚拟 D/N loss：teacher 固定分母，可靠性只乘分子，不制造 RGB。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


DEFAULT_WEIGHTS = {"depth": .20, "normal_render": .05, "normal_depth": .05, "coverage": .05}


@dataclass
class VirtualGeometryLosses:
    total: torch.Tensor
    raw: dict
    weighted: dict
    metrics: dict
    maps: dict


def depth_normal(depth, valid, camera):
    """camera-Z 导出相机空间朝相机法线；无效 stencil 在叉乘前清理。"""
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    height, width = depth.shape[-2:]
    x = (torch.arange(width, device=depth.device, dtype=depth.dtype) - camera.Cx) / camera.Fx
    y = (torch.arange(height, device=depth.device, dtype=depth.dtype) - camera.Cy) / camera.Fy
    points = torch.cat((depth * x[None, None], depth * y[None, :, None], depth), dim=0)
    dy, dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1], points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    cross = torch.cross(dy, dx, dim=0)
    lengths = torch.linalg.vector_norm(cross, dim=0, keepdim=True)
    interior = (valid[:, 1:-1, 1:-1] & valid[:, 2:, 1:-1] & valid[:, :-2, 1:-1]
                & valid[:, 1:-1, 2:] & valid[:, 1:-1, :-2] & (lengths > 1e-12))
    normal = F.pad(F.normalize(cross, dim=0, eps=1e-12), (1, 1, 1, 1))
    return normal, F.pad(interior, (1, 1, 1, 1), value=False)


def compute_virtual_geometry_losses(render_pkg, camera, weights=None, ramp=1.0):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights) != set(DEFAULT_WEIGHTS) or any(float(value) < 0 for value in weights.values()):
        raise ValueError("Virtual loss requires four nonnegative configured weights")
    teacher = camera.teacher
    prediction, alpha, rendered_normal = render_pkg["median_depth"], render_pkg["mask"], render_pkg["normal"]
    if prediction.ndim == 2:
        prediction = prediction[None]
    if alpha.ndim == 2:
        alpha = alpha[None]
    target, target_normal = teacher["depth"], teacher["normal"]
    if prediction.shape != target.shape or rendered_normal.shape != target_normal.shape:
        raise ValueError("Student and teacher grids differ")
    teacher_valid, normal_teacher_valid = teacher["valid_depth"].bool(), teacher["valid_normal"].bool()
    # 学生近处错误表面仍需 D 梯度；.2 只限制 teacher 的相机/网格准备。
    finite_depth = torch.isfinite(prediction) & (prediction > 0)
    finite_alpha = torch.isfinite(alpha)
    safe_depth = torch.where(finite_depth, prediction, torch.zeros_like(prediction))
    safe_alpha = torch.where(finite_alpha, alpha, torch.zeros_like(alpha))
    normal_finite = torch.isfinite(rendered_normal).all(dim=0, keepdim=True)
    safe_normal = torch.where(torch.isfinite(rendered_normal), rendered_normal, torch.zeros_like(rendered_normal))
    normal_length = torch.linalg.vector_norm(safe_normal, dim=0, keepdim=True)
    normalized = F.normalize(safe_normal, dim=0, eps=1e-6)
    student_valid = finite_depth & finite_alpha & (safe_alpha >= .55)
    if render_pkg.get("depth_status") is not None:
        status = render_pkg["depth_status"].reshape_as(prediction)
        student_valid &= status == 3
    else:
        status = torch.where(student_valid, 3, torch.where(finite_alpha & (safe_alpha >= .55), 2, 1))
    student_valid = student_valid.detach()
    normal_valid = (student_valid & normal_finite & (normal_length > 1e-6)).detach()
    derived_normal, stencil_valid = depth_normal(safe_depth, student_valid, camera)
    stencil_valid = stencil_valid.detach()
    relative = (safe_depth - target) / target.clamp_min(.2)
    depth_error = torch.sqrt(relative.square() + .01 ** 2) - .01
    render_error = (1 - (normalized * target_normal).sum(dim=0, keepdim=True)).clamp(0, 2)
    derived_error = (1 - (derived_normal * target_normal).sum(dim=0, keepdim=True)).clamp(0, 2)
    coverage_error = F.relu(.8 - safe_alpha).square()
    q = teacher["q"]

    def reduce(error, valid, reliability, normalizer):
        # Z 由全池 q 和 teacher 域决定；学生缺面及 unknown 降权都不能缩小分母。
        numerator = (torch.where(valid, error, torch.zeros_like(error)) * q * reliability).sum()
        return numerator / max(float(camera.normalizers[normalizer]), 1e-12)

    raw = {
        "depth": reduce(depth_error, teacher_valid & student_valid, teacher["weight_depth"], "depth"),
        "normal_render": reduce(render_error, normal_teacher_valid & normal_valid, teacher["weight_normal"], "normal"),
        "normal_depth": reduce(derived_error, normal_teacher_valid & stencil_valid, teacher["weight_normal"], "normal"),
        "coverage": reduce(coverage_error, teacher_valid, teacher["weight_coverage"], "coverage"),
    }
    weighted = {key: float(ramp) * float(weights[key]) * value for key, value in raw.items()}
    total = sum(weighted.values())
    with torch.no_grad():
        count = teacher_valid.sum().clamp_min(1)
        normal_count = normal_teacher_valid.sum().clamp_min(1)
        missing = teacher_valid & ~student_valid
        fixed_depth = torch.where(student_valid, relative.abs(), torch.ones_like(relative))
        fixed_normal = torch.where(normal_valid, render_error, torch.full_like(render_error, 2.))
        metrics = {
            "teacher_pixels": int(teacher_valid.sum()),
            "teacher_normal_pixels": int(normal_teacher_valid.sum()),
            "missing_fraction": float(missing.sum() / count),
            "fixed_depth_relative_error": float(torch.where(teacher_valid, fixed_depth, 0).sum() / count),
            "fixed_normal_cosine_error": float(torch.where(normal_teacher_valid, fixed_normal, 0).sum() / normal_count),
            "root_search_failed_fraction": float(((status == 2) & teacher_valid).sum() / count),
            "low_coverage_fraction": float(((status == 1) & teacher_valid).sum() / count),
            "no_contributor_fraction": float(((status == 0) & teacher_valid).sum() / count),
        }
        maps = {"depth_valid": student_valid, "normal_valid": normal_valid, "normal_depth_valid": stencil_valid,
                "teacher_valid": teacher_valid, "teacher_normal_valid": normal_teacher_valid,
                "relative_depth_error": relative.abs().detach(), "normal_error": render_error.detach(),
                "normal_depth_error": derived_error.detach(), "missing": missing, "depth_status": status.detach()}
    return VirtualGeometryLosses(total, raw, weighted, metrics, maps)
