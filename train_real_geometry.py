"""Train a colour-free Gaussian surface from grayscale stereo, aligned depths and normals."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import (
    REAL_GEOMETRY_DEPTH_LOSS_MODES,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from gaussian_renderer import render
from scene import Scene
from scene.geometry_gaussians import GeometryGaussians
from scene.cameras import FramePayloadCache
from train import validate_training_priors
from utils.depth_normal_prior import NORMAL_DIR, depth_normal_preparation
from utils.general_utils import safe_state
from utils.loss_utils import PatchMatch
from utils.real_geometry_topology import RealGeometryTopology, TopologyConfig
from utils.training_io import write_json
from utils.training_losses import compute_training_losses, loss_schedule


def refinement_policy(step, total_steps, base_ncc=0.1):
    """Return the NCC weight, learning-rate multiplier and pruning audit eligibility."""
    if total_steps <= 7000:
        topo = step <= total_steps - min(400, total_steps // 10)
        return base_ncc, 1.0, topo
    end = max(7000, round(0.75 * total_steps))
    if step <= 7000:
        return base_ncc, 1.0, step <= 6600
    if step <= end:
        ramp = min(1.0, (step - 7000) / max(1, min(2000, end - 7000)))
        return base_ncc * (1 + 0.5 * ramp), 1.0, True
    progress = (step - end) / (total_steps - end)
    return base_ncc * (1.5 - 0.9 * progress), 1.0 - 0.75 * progress, False


def cadence_windows(total_steps, cfg):
    """Pair the growth and pruning audits for the accelerated real-view recipe."""

    def pairs(steps):
        return list(zip(steps[::2], steps[1::2]))

    prune_audits = [
        s
        for s in range(cfg.first, total_steps + 1, cfg.interval)
        if refinement_policy(s, total_steps)[2]
    ]
    middle = round(0.15 * total_steps)
    late = round(0.75 * total_steps)
    stop = round(0.90 * total_steps)
    fast = max(1, cfg.interval // 2)
    growth = []
    for start, end, interval in (
        (cfg.first, middle, cfg.interval),
        (max(cfg.first, middle + fast), late - 1, fast),
        (max(cfg.first, late), stop, cfg.interval),
    ):
        growth.extend(pairs(range(start, end + 1, interval)))
    return growth, pairs(prune_audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args, optim_args, pipe_args = (
        ModelParams(parser, fixed_normal_dir=NORMAL_DIR),
        OptimizationParams(parser),
        PipelineParams(parser),
    )
    parser.set_defaults(
        use_decoupled_appearance=0,
        sh_degree=0,
        iterations=20000,
        init_num_points=250000,
        frame_gpu_budget=4,
        depth_prior_from_iter=1,
        regularization_from_iter=401,
        lambda_multi_view_ncc=0.1,
        depth_prior_ramp_iters=0,
    )
    parser.add_argument("--training_seed", type=int, default=0)
    parser.add_argument("--audit_from", type=int, default=800)
    parser.add_argument("--audit_interval", type=int, default=200)
    parser.add_argument("--audit_views", type=int, default=12)
    parser.add_argument(
        "--densification_schedule",
        choices=("accelerated",),
        default="accelerated",
        help=argparse.SUPPRESS,  # Accept the existing accelerated invocation.
    )
    parser.add_argument("--coverage_weight", type=float, default=0.05)
    args = parser.parse_args()
    dataset, opt, pipe = (
        model_args.extract(args),
        optim_args.extract(args),
        pipe_args.extract(args),
    )
    if opt.real_depth_loss_mode not in REAL_GEOMETRY_DEPTH_LOSS_MODES:
        parser.error(f"--real_depth_loss_mode must be one of {REAL_GEOMETRY_DEPTH_LOSS_MODES}")
    if (
        not args.model_path
        or args.init_mesh
        or not args.source_path
        or args.iterations < 1
        or args.use_decoupled_appearance != 0
        or args.sh_degree != 0
        or args.sg_degree != 0
        or args.disable_filter3D
        or args.frame_gpu_budget < 2
        or not math.isfinite(args.coverage_weight)
        or args.coverage_weight < 0
        or min(args.audit_from, args.audit_interval, args.audit_views) < 1
    ):
        parser.error(
            "Use a new -m, -s, TSDF or --init_ply, positive iteration/audit settings, real filtered Gaussians, appearance 0, SG 0 and a frame cache >=2."
        )
    dataset.gray_observations = True
    out = Path(dataset.model_path)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cfg_args").write_text(str(argparse.Namespace(**vars(dataset))))
    cfg = TopologyConfig(
        interval=args.audit_interval, first=args.audit_from, views=args.audit_views
    )
    growth_windows, prune_windows = cadence_windows(opt.iterations, cfg)
    write_json(
        out / "geometry_config.json",
        {
            "recipe": "gray_geometry_v1",
            "argv": sys.argv,
            "args": vars(args),
            "topology": asdict(cfg),
            "rgb_weight": 0,
            "photometric_vcd_vcp": False,
            "virtual_training_cameras": 0,
            "appearance": "no per-point colour, SH, SG, exposure or RGB output",
            "gray": "cached single-channel float32 luma at training resolution",
            "schedule": "0.10 NCC up to 7k, ramp to 0.15 by 9k, taper to 0.06 over final quarter; geometry rates taper to 0.25x",
            "geometry_parameters": ["xyz", "scaling", "rotation", "opacity"],
            "opacity_resets": [],
            "ncc_from": opt.regularization_from_iter,
            "ncc_weight": opt.lambda_multi_view_ncc,
            "topology_until": growth_windows[-1][1] if growth_windows else 0,
            "growth_audit_pairs": growth_windows,
            "prune_audit_pairs": prune_windows,
        },
    )
    safe_state(False, seed=args.training_seed)
    model = GeometryGaussians()
    with depth_normal_preparation(dataset) as prepare_priors:
        scene = Scene(
            dataset, model, initialization_seed=args.training_seed,
            prepare_camera_priors=prepare_priors,
        )
    write_json(out / "depth_normal_prior.json", scene.prior_preparation)
    cameras = scene.getTrainCameras()
    validate_training_priors(cameras)
    model.training_setup(opt)
    geometry = model.point_parameters
    if any(
        c.original_image is not None
        or c.original_image_cpu is not None
        or c.gray_observation_cpu.shape[0] != 1
        for c in cameras
    ):
        raise RuntimeError("Training cameras must retain only one observation channel")
    model.compute_3D_filter(cameras=cameras)
    model.save_ply(out / "point_cloud/iteration_0/point_cloud.ply")
    cache = FramePayloadCache(dataset.frame_gpu_budget)
    bg = torch.full((3,), float(dataset.white_background), device="cuda")
    topo = RealGeometryTopology(
        model, scene, pipe, bg, dataset.kernel_size, cache, args.training_seed, cfg
    )
    topo.configure_cadence(growth_windows, prune_windows)
    pm = PatchMatch(
        opt.multi_view_patch_size,
        opt.multi_view_pixel_noise_th,
        dataset.kernel_size,
        pipe,
        debug=False,
    )
    rng, stack = random.Random(args.training_seed), []
    ncc_steps = torch.zeros((), device="cuda", dtype=torch.long)
    probes, visits = [], Counter()
    checkpoints = sorted(
        {i for i in (7000, 15000, opt.iterations) if i <= opt.iterations}
    )
    start = time.monotonic()
    with (out / "geometry_loss.jsonl").open("x", buffering=1) as log:
        for it in tqdm(range(1, opt.iterations + 1), mininterval=3):
            if not stack:
                stack = cameras.copy()
                rng.shuffle(stack)
            cam = cache.ensure(stack.pop())
            visits[cam.image_name] += 1
            ncc_weight, lr_multiplier, _ = refinement_policy(
                it, opt.iterations, opt.lambda_multi_view_ncc
            )
            model.set_rates(it, lr_multiplier)
            buf = render(
                cam,
                model,
                pipe,
                bg,
                dataset.kernel_size,
                require_depth=True,
                depth_search_expansions=2,
            )
            sched = loss_schedule(cam, opt, it)
            sched = replace(
                sched,
                multi_view_ncc_weight=ncc_weight if sched.reg_enabled else 0.0,
            )
            peer = None
            if sched.reg_enabled and cam.nearest_id:
                peer = cache.ensure(
                    cameras[rng.choice(cam.nearest_id)],
                    keep=(cam,),
                    fields=("gray_observation",),
                )
            terms = compute_training_losses(
                buf, cam, model, opt, sched, pm, peer, it, supervise_rgb=False
            )
            valid = torch.isfinite(cam.depth_prior) & (cam.depth_prior > 0.01)
            coverage = torch.where(
                valid, (0.8 - buf["mask"]).clamp_min(0).square(), 0.0
            ).sum() / valid.sum().clamp_min(1)
            total = terms.total + args.coverage_weight * coverage
            if not torch.isfinite(total).all():
                raise FloatingPointError(f"Nonfinite objective at {it}")
            ncc_steps += (terms.ncc.detach().abs().sum() > 0).long()
            if (
                sched.reg_enabled
                and terms.ncc.requires_grad
                and (it == opt.regularization_from_iter or it % 500 == 0)
            ):
                dg, ng = torch.autograd.grad(
                    sched.multi_view_ncc_weight * terms.ncc.sum(),
                    (buf["median_depth"], buf["normal"]),
                    retain_graph=True,
                    allow_unused=True,
                )
                probes.append(
                    {
                        "step": it,
                        "depth_grad_l1": float(dg.abs().sum())
                        if dg is not None
                        else 0.0,
                        "normal_grad_l1": float(ng.abs().sum())
                        if ng is not None
                        else 0.0,
                    }
                )
            total.backward()
            if it == 1 or it % 200 == 0 or it == opt.iterations:
                gradients = {
                    k: float(getattr(model, geometry[k]).grad.abs().sum())
                    for k in sorted(geometry)
                }
                if not all(torch.isfinite(torch.tensor(list(gradients.values())))):
                    raise FloatingPointError(f"Nonfinite geometry gradients at {it}")
                record = {
                    "step": it,
                    "camera": cam.image_name,
                    "points": len(model.get_xyz),
                    "ncc_weight": sched.multi_view_ncc_weight,
                    "lr_multiplier": lr_multiplier,
                    "total": float(total.detach()),
                    "depth": float(terms.depth_prior.detach()),
                    "normal": float(terms.normal_prior.detach()),
                    "geo": float(terms.geo.detach()),
                    "ncc": float(terms.ncc.detach()),
                    "coverage": float(coverage.detach()),
                    "gradients_l1": gradients,
                    "ncc_nonzero_steps": int(ncc_steps),
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            del buf, terms, coverage, total
            topo.step(it)
            if it % 400 == 0 and (not topo.events or topo.events[-1]["step"] != it):
                model.compute_3D_filter(cameras=cameras)
            if it in checkpoints:
                model.compute_3D_filter(cameras=cameras)
                model.save_ply(
                    out / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
                )
                write_json(
                    out / f"stage_{it}.json",
                    {
                        "step": it,
                        "points": len(model.get_xyz),
                        "ncc_nonzero_steps": int(ncc_steps),
                        "topology": dict(topo.counts),
                        "lr_multiplier": lr_multiplier,
                    },
                )
    if set(g["name"] for g in model.optimizer.param_groups) != set(geometry) or not all(
        torch.isfinite(getattr(model, k)).all() for k in geometry.values()
    ):
        raise RuntimeError("Invalid final geometry or unexpected optimizer groups")
    model.export_neutral_viewer(
        out / "point_cloud" / f"iteration_{opt.iterations}" / "neutral_3dgsviewer.ply"
    )
    write_json(
        out / "geometry_stats.json",
        {
            "status": "complete",
            "optimizer_steps": opt.iterations,
            "points": len(model.get_xyz),
            "point_parameters": list(geometry),
            "observation_channels": 1,
            "appearance_fields": [],
            "checkpoints": checkpoints,
            "ncc_nonzero_steps": int(ncc_steps),
            "ncc_gradient_probes": probes,
            "observed_cameras": len(visits),
            "min_visits": min(visits.values()),
            "max_visits": max(visits.values()),
            "sampling": dict(visits),
            "topology": dict(topo.counts),
            "events": topo.events,
            "seconds": time.monotonic() - start,
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        },
    )


if __name__ == "__main__":
    main()
