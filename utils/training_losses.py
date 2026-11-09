#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

"""真实相机的训练目标：集中管理启用时机、原始损失与加权贡献。

normal prior 与 NCC 沿用 train_demo 的权重日程；其他真实视角算式保持不变。
RGB 的曝光/外观分支继续由 photometric_rgb_loss 负责；这里只组织各目标。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossSchedule:
    """一次真实相机更新需要哪些渲染缓冲，以及 depth prior 的当前权重。"""

    reg_enabled: bool
    normal_prior_enabled: bool
    depth_prior_enabled: bool
    needs_depth: bool
    depth_prior_weight: float
    normal_prior_weight: float
    multi_view_ncc_weight: float


@dataclass
class LossTerms:
    """l1 仅用于诊断；rgb 已含 L1/DSSIM 和当前外观模型处理，不能重复加权。"""

    total: torch.Tensor
    rgb: torch.Tensor
    l1: torch.Tensor
    depth_normal: torch.Tensor
    normal_prior: torch.Tensor
    depth_prior: torch.Tensor
    ncc: torch.Tensor
    geo: torch.Tensor
    mask: torch.Tensor
    appearance_regularization: torch.Tensor

    def raw_terms(self):
        """返回逐项原始值；不 detach，调用者可自行决定记录或反传。"""
        return {
            name: getattr(self, name)
            for name in (
                "rgb",
                "l1",
                "depth_normal",
                "normal_prior",
                "depth_prior",
                "ncc",
                "geo",
                "mask",
                "appearance_regularization",
            )
        }

    def weighted_terms(self, opt, schedule):
        """这些项相加得到 total；l1 是 rgb 内部诊断，因此不单独计入。"""
        return {
            "rgb": self.rgb,
            "mask": opt.lambda_mask * self.mask,
            "depth_normal": opt.lambda_depth_normal * self.depth_normal,
            "normal_prior": schedule.normal_prior_weight * self.normal_prior,
            "depth_prior": schedule.depth_prior_weight * self.depth_prior,
            "ncc": schedule.multi_view_ncc_weight * self.ncc,
            "geo": opt.lambda_multi_view_geo * self.geo,
            "appearance_regularization": self.appearance_regularization,
        }


def ramp_weight(iteration, start_iteration, ramp_iterations):
    """起始步也参与 ramp：第一个启用步为 1/ramp_iterations；0 表示立即启用。"""
    if iteration < start_iteration:
        return 0.0
    if ramp_iterations <= 0:
        return 1.0
    return min((iteration - start_iteration + 1) / ramp_iterations, 1.0)


def normal_prior_weight(iteration, opt):
    """train_demo 的 7k init / 15k geometry 日程；15k 后保持尾段权重。

    lambda_normal_prior 是初始/尾段基准，默认 .1；中段依次为 1.5 倍和 2 倍。
    短训按 demo 截断 geometry 终点；长训截断进度，避免原 demo 外推成负权重。
    """
    if iteration < opt.normal_prior_from_iter:
        return 0.0
    if iteration <= 3000:
        multiplier = 1.0
    elif iteration <= 5000:
        multiplier = 1.5
    elif iteration <= 7000:
        multiplier = 2.0
    else:
        geometry_end = min(15000, opt.iterations)
        progress = min((iteration - 7001) / max(geometry_end - 7001, 1), 1.0)
        multiplier = 2.0 if progress < 0.5 else 2.0 - (progress - 0.5) / 0.5
    return opt.lambda_normal_prior * multiplier


def multi_view_ncc_weight(iteration, opt):
    """有 normal prior 的 demo NCC 日程；15k 后保持尾值，不采用无 prior 分支。

    lambda_multi_view_ncc 默认 .2，geometry 前半程降至其一半，后半程降至四分之一。
    短训的 geometry 终点与 normal prior 一样截到总 iterations。
    """
    if iteration <= 7000:
        return 0.0
    geometry_end = min(15000, opt.iterations)
    progress = min((iteration - 7001) / max(geometry_end - 7001, 1), 1.0)
    if progress < 0.5:
        multiplier = 1.0 + (0.5 - 1.0) * (progress / 0.5)
    else:
        multiplier = 0.5 + (0.25 - 0.5) * ((progress - 0.5) / 0.5)
    return opt.lambda_multi_view_ncc * multiplier


def loss_schedule(camera, opt, iteration):
    """根据实际参数和当前帧 prior 决定启用项，不再由目录名称猜测场景类型。"""
    reg_enabled = iteration >= opt.regularization_from_iter
    normal_weight = normal_prior_weight(iteration, opt) if camera.normal_prior is not None else 0.0
    normal_enabled = normal_weight > 0
    depth_enabled = (
        opt.lambda_depth_prior > 0
        and iteration >= opt.depth_prior_from_iter
        and camera.depth_prior is not None
    )
    return LossSchedule(
        reg_enabled=reg_enabled,
        normal_prior_enabled=normal_enabled,
        depth_prior_enabled=depth_enabled,
        needs_depth=reg_enabled or normal_enabled or depth_enabled,
        depth_prior_weight=opt.lambda_depth_prior
        * ramp_weight(iteration, opt.depth_prior_from_iter, opt.depth_prior_ramp_iters),
        normal_prior_weight=normal_weight,
        multi_view_ncc_weight=multi_view_ncc_weight(iteration, opt) if reg_enabled else 0.0,
    )


def masked_l1_depth_loss(pred_depth, gt_depth, valid_mask):
    """在 prior 指定的固定有效域上比较 camera-Z；不按预测是否命中缩小分母。"""
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        return pred_depth.new_zeros(())
    return (pred_depth[valid_mask] - gt_depth[valid_mask]).abs().mean()


def relative_charbonnier_depth_loss(
    pred_depth, gt_depth, valid_mask, eps=0.01, denominator_floor=0.05
):
    """保留原尺度的相对鲁棒误差，不拟合自由的 scale/shift。

    分母由固定 prior 决定并截断下限，避免近零深度获得无限权重。
    按原实现先分别归一化预测与目标，再取差；保留浮点计算次序。
    """
    valid_mask = valid_mask.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
    if not valid_mask.any():
        return pred_depth.new_zeros(())
    scale = gt_depth[valid_mask].detach().clamp_min(denominator_floor)
    prediction = pred_depth[valid_mask] / scale
    target = gt_depth[valid_mask] / scale
    return torch.sqrt((prediction - target).square() + eps * eps).mean() - eps


def _photometric_rgb_loss(*args):
    # 延迟导入 CUDA/外观模型依赖，让纯 depth 算式可以在 CPU 独立验证。
    from utils.loss_utils import photometric_rgb_loss

    return photometric_rgb_loss(*args)


def _depth_to_normal(*args):
    from utils.graphics_utils import depth_to_normal

    return depth_to_normal(*args)


def compute_training_losses(
    render_pkg,
    camera,
    gaussians,
    opt,
    schedule,
    patchmatch=None,
    nearest_camera=None,
    iteration=0,
):
    """计算一个真实视角的损失；不执行 backward、更新参数或改变几何拓扑。

    保留旧管道的两种不同分母：normal consistency 的无效像素置零后按
    全图平均；外部 prior 按其有效像素平均。不能在重构中无意统一二者。
    """
    image = render_pkg["render"]
    zero = image.new_zeros(())
    # 保留输入 dtype（可能为 uint8）；fused photometric loss 负责其解释。
    target_image = camera.original_image.to(device=image.device)
    rgb, l1 = _photometric_rgb_loss(
        image, target_image, gaussians, camera.uid, opt.lambda_dssim
    )
    terms = LossTerms(zero, rgb, l1, zero, zero, zero, zero, zero, zero, zero)
    if getattr(gaussians, "ppisp", None) is not None:
        # 官方正则包含其自身权重，防止场景颜色与 ISP 参数任意相互补偿。
        terms.appearance_regularization = gaussians.ppisp.regularization_loss()
    depth = render_pkg["median_depth"] if schedule.needs_depth else None
    depth_normal = valid_points = None

    # 自一致性约束比较当前 Gaussian normal 与当前 median depth 导出的 normal。
    # PatchMatch 同样需要 depth_normal，因此即便其损失权重为零也保留计算。
    if schedule.reg_enabled or schedule.normal_prior_enabled:
        depth_normal, valid_points = _depth_to_normal(camera, depth)
        if schedule.reg_enabled and opt.lambda_depth_normal > 0:
            error = 1 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)
            terms.depth_normal = torch.where(
                valid_points.squeeze(), error, torch.zeros_like(error)
            ).mean()

    # 与 train_demo 一致：有向 cosine，只监督 alpha>.5 的有效几何像素。
    # alpha 仅用于布尔筛选，不缩放 prior，也不接收本项的梯度。
    if schedule.normal_prior_enabled:
        prior = camera.normal_prior
        mask = camera.normal_prior_mask.squeeze(0) & (render_pkg["mask"].squeeze(0) > 0.5) & valid_points.squeeze()
        if mask.any():
            render_error = (
                1 - F.cosine_similarity(render_pkg["normal"], prior, dim=0)
            )
            depth_error = 1 - F.cosine_similarity(depth_normal, prior, dim=0)
            terms.normal_prior = (render_error + depth_error)[mask].mean()

    if schedule.depth_prior_enabled:
        prior = camera.depth_prior
        mask = prior > 0.01  # 原管道的有效深度阈值；不是渲染结果产生的 mask。
        if opt.real_depth_loss_mode == "relative_charbonnier":
            terms.depth_prior = relative_charbonnier_depth_loss(
                depth,
                prior,
                mask,
                eps=opt.real_depth_relative_epsilon,
                denominator_floor=opt.real_depth_relative_floor,
            )
        else:
            terms.depth_prior = masked_l1_depth_loss(depth, prior, mask)

    if schedule.reg_enabled and (
        schedule.multi_view_ncc_weight > 0 or opt.lambda_multi_view_geo > 0
    ):
        if patchmatch is None:
            raise ValueError("Enabled multi-view losses require a PatchMatch instance")
        terms.ncc, terms.geo = patchmatch(
            gaussians, render_pkg, camera, nearest_camera, iteration, depth_normal
        )

    # 原 mask 项只惩罚背景被占据，并非前景/背景两侧的完整二元交叉熵。
    if opt.lambda_mask > 0 and camera.gt_mask is not None:
        background_probability = 1 - render_pkg["mask"].clamp(1e-6, 1 - 1e-6)
        terms.mask = (-(1 - camera.gt_mask) * torch.log(background_probability)).mean()

    contributions = terms.weighted_terms(opt, schedule)
    # 保留旧管道的相加顺序，减少仅由重排运算带来的浮点差异。
    terms.total = contributions["rgb"]
    for name in ("mask", "depth_normal", "normal_prior", "depth_prior", "ncc", "geo"):
        terms.total = terms.total + contributions[name]
    if getattr(gaussians, "ppisp", None) is not None:
        terms.total = terms.total + contributions["appearance_regularization"]
    return terms
