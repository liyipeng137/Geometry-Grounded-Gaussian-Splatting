import random

import torch

from gaussian_renderer import render


def sample_vcd_cameras(cameras, num_cams):
    if len(cameras) == 0:
        return []
    k = min(num_cams, len(cameras))
    return random.sample(cameras, k)


def apply_appearance_correction(image, gaussians, view_uid):
    app_model = gaussians.app_model
    if app_model is gaussians.App_model.NO:
        return image

    appearance_embedding = gaussians.get_apperance_embedding(view_uid)

    if app_model is gaussians.App_model.GS:
        return torch.addmm(
            appearance_embedding[:3, 3, None],
            appearance_embedding[:3, :3],
            image.reshape(3, -1),
        ).reshape_as(image)

    if app_model is gaussians.App_model.PGSR:
        return torch.addcmul(appearance_embedding[1], torch.exp(appearance_embedding[0]), image)

    if app_model is gaussians.App_model.GOF:
        # Keep GOF behavior aligned with training crop logic as much as possible.
        # Outside the valid crop we keep the original render.
        orig_h, orig_w = image.shape[1:]
        h, w = orig_h // 32 * 32, orig_w // 32 * 32
        top, left = (orig_h - h) // 2, (orig_w - w) // 2
        if h == 0 or w == 0:
            return image
        crop = image[:, top : top + h, left : left + w]
        down = torch.nn.functional.interpolate(crop[None], size=(h // 32, w // 32), mode="bilinear", align_corners=True)[0]
        embedding_map = appearance_embedding[None].repeat(h // 32, w // 32, 1).permute(2, 0, 1)
        net_in = torch.cat([down, embedding_map], dim=0)[None]
        mapping = gaussians.appearance_network(net_in)
        corrected = image.clone()
        corrected[:, top : top + h, left : left + w] = mapping * crop
        return corrected

    return image


def build_metric_map(corrected_image, gt_image, gt_mask, loss_thresh):
    per_pixel_l1 = torch.mean(torch.abs(corrected_image - gt_image), dim=0)
    if gt_mask is not None:
        valid_mask = gt_mask.to(device=per_pixel_l1.device).squeeze(0) > 0.5
    else:
        valid_mask = torch.ones_like(per_pixel_l1, dtype=torch.bool)

    metric_map = torch.zeros_like(per_pixel_l1, dtype=torch.bool)
    if valid_mask.any():
        valid_values = per_pixel_l1[valid_mask]
        denom = (valid_values.max() - valid_values.min()).clamp_min(1e-8)
        norm_values = (valid_values - valid_values.min()) / denom
        metric_map[valid_mask] = norm_values > loss_thresh
    return metric_map.reshape(-1).to(torch.int32)


def _compute_masked_photometric_loss(corrected_image, gt_image, gt_mask):
    per_pixel_l1 = torch.mean(torch.abs(corrected_image - gt_image), dim=0)
    if gt_mask is not None:
        valid_mask = gt_mask.to(device=per_pixel_l1.device).squeeze(0) > 0.5
    else:
        valid_mask = torch.ones_like(per_pixel_l1, dtype=torch.bool)
    if not valid_mask.any():
        return torch.tensor(0.0, device=per_pixel_l1.device)
    return per_pixel_l1[valid_mask].mean()


def _normalize_tensor(values):
    if values is None:
        return None
    if values.numel() == 0:
        return values
    valid = torch.isfinite(values)
    if not valid.any():
        return torch.zeros_like(values)
    finite_vals = values[valid]
    vmin = finite_vals.min()
    vmax = finite_vals.max()
    if torch.abs(vmax - vmin) < 1e-8:
        out = torch.zeros_like(values)
        out[valid] = 0.0
        return out
    out = torch.zeros_like(values)
    out[valid] = (values[valid] - vmin) / (vmax - vmin)
    return out


def compute_vcd_vcp_scores(
    camlist,
    gaussians,
    pipe,
    background,
    kernel_size,
    loss_thresh,
    need_vcd=True,
    need_vcp=True,
):
    if len(camlist) == 0 or (not need_vcd and not need_vcp):
        return None, None

    full_metric_counts = None
    full_metric_score = None
    with torch.no_grad():
        for cam in camlist:
            render_pkg = render(
                cam,
                gaussians,
                pipe,
                background,
                kernel_size,
                require_depth=False,
            )
            corrected_image = apply_appearance_correction(render_pkg["render"], gaussians, cam.uid)
            gt_image = cam.original_image.cuda()
            photometric_loss = _compute_masked_photometric_loss(corrected_image, gt_image, cam.gt_mask)
            metric_map = build_metric_map(corrected_image, gt_image, cam.gt_mask, loss_thresh)

            metric_pkg = render(
                cam,
                gaussians,
                pipe,
                background,
                kernel_size,
                require_depth=False,
                get_flag=True,
                metric_map=metric_map,
            )
            counts = metric_pkg["accum_metric_counts"].to(torch.float32)
            if need_vcd:
                if full_metric_counts is None:
                    full_metric_counts = counts.clone()
                else:
                    full_metric_counts += counts
            if need_vcp:
                weighted = photometric_loss * counts
                if full_metric_score is None:
                    full_metric_score = weighted
                else:
                    full_metric_score += weighted

    importance_score = None
    pruning_score = None
    if need_vcd and full_metric_counts is not None:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode="floor")
    if need_vcp and full_metric_score is not None:
        pruning_score = _normalize_tensor(full_metric_score)
    return importance_score, pruning_score
