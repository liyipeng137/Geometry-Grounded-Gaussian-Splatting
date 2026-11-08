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

"""光度损失兼容层：支持扩展版 fused_ssim，也支持官方公开版本。

部分历史环境额外提供 fused_ssim.photometric_loss；公开包只有 fused_ssim。
缺少前者时恢复原显式数学定义，不依赖一个仓库未提供的自定义入口。
两个后端的浮点归约和 kernel 可以不同，因此不承诺逐位一致或相同耗时。
"""

from importlib import import_module

import torch


def get_photometric_backend():
    """返回实际选择的后端名称，供 training_config 记录复现条件。"""
    module = import_module("fused_ssim")
    if callable(getattr(module, "photometric_loss", None)):
        return "custom_fused_ssim.photometric_loss"
    return "torch_l1+fused_ssim(valid)"


def photometric_loss(prediction, target, lambda_dssim):
    """兼容现有三元调用接口；公开包回退返回 (total, L1, SSIM)。

    自定义后端存在时原样委托；公开包路径显式计算：
    total = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)。
    uint8 target 转成 [0,1]；浮点 target 保持已有尺度，不猜测、不 clamp。
    输入可以是 CHW 或 NCHW，标准 fused_ssim 始终收到 NCHW 与 valid padding。
    """
    module = import_module("fused_ssim")
    custom_loss = getattr(module, "photometric_loss", None)
    if callable(custom_loss):
        return custom_loss(prediction, target, lambda_dssim)

    if prediction.ndim not in (3, 4) or target.shape != prediction.shape:
        raise ValueError("Photometric inputs must have matching CHW or NCHW shapes")
    if not prediction.is_floating_point():
        raise TypeError("Photometric prediction must be floating point")

    target_float = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.dtype == torch.uint8:
        target_float = target_float * (1.0 / 255.0)
    l1 = (prediction - target_float).abs().mean()
    prediction_batch = prediction.unsqueeze(0) if prediction.ndim == 3 else prediction
    target_batch = target_float.unsqueeze(0) if target_float.ndim == 3 else target_float
    similarity = module.fused_ssim(
        prediction_batch.contiguous(), target_batch.contiguous(), padding="valid"
    )
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - similarity)
    return total, l1, similarity
