"""Experimental real-view geometry-only Gaussian training; native GGGS PLY output."""

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

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.cameras import FramePayloadCache
from train import validate_training_priors
from utils.general_utils import safe_state
from utils.geometry_topology import PARAMETERS
from utils.loss_utils import PatchMatch
from utils.real_geometry_topology import RealGeometryTopology, TopologyConfig
from utils.training_io import write_json
from utils.training_losses import compute_training_losses, loss_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args, optim_args, pipe_args = (
        ModelParams(parser),
        OptimizationParams(parser),
        PipelineParams(parser),
    )
    parser.set_defaults(
        use_decoupled_appearance=0,
        iterations=7000,
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
    parser.add_argument("--coverage_weight", type=float, default=0.05)
    args = parser.parse_args()
    dataset, opt, pipe = (
        model_args.extract(args),
        optim_args.extract(args),
        pipe_args.extract(args),
    )
    if (
        not args.model_path
        or not args.init_ply
        or not args.source_path
        or args.iterations < 1
        or args.use_decoupled_appearance != 0
        or args.sg_degree != 0
        or args.disable_filter3D
        or args.frame_gpu_budget < 2
        or args.real_depth_loss_mode != "absolute"
        or not math.isfinite(args.coverage_weight)
        or args.coverage_weight < 0
        or min(args.audit_from, args.audit_interval, args.audit_views) < 1
    ):
        parser.error(
            "Use a new -m, -s, --init_ply, positive iteration/audit settings, real filtered Gaussians, appearance 0, SG 0 and a frame cache >=2."
        )
    out = Path(dataset.model_path)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cfg_args").write_text(str(argparse.Namespace(**vars(dataset))))
    cfg = TopologyConfig(
        interval=args.audit_interval, first=args.audit_from, views=args.audit_views
    )
    write_json(
        out / "geometry_config.json",
        {
            "recipe": "real_geometry",
            "argv": sys.argv,
            "args": vars(args),
            "topology": asdict(cfg),
            "rgb_weight": 0,
            "photometric_vcd_vcp": False,
            "virtual_training_cameras": 0,
            "color": "frozen initialization DC; split inherits, stereo seed samples its reference photo once",
            "geometry_parameters": ["xyz", "scaling", "rotation", "opacity"],
            "opacity_resets": [],
            "ncc_from": opt.regularization_from_iter,
            "ncc_weight": opt.lambda_multi_view_ncc,
            "topology_until": opt.iterations - min(400, opt.iterations // 10),
        },
    )
    safe_state(False, seed=args.training_seed)
    model = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, model, initialization_seed=args.training_seed)
    cameras = scene.getTrainCameras()
    validate_training_priors(cameras)
    model.training_setup(opt)
    geometry = {"xyz", "scaling", "rotation", "opacity"}
    for key, attr in PARAMETERS.items():
        getattr(model, attr).requires_grad_(key in geometry)
    model.compute_3D_filter(cameras=cameras)
    scene.save(0, save_viewer=False)
    cache = FramePayloadCache(dataset.frame_gpu_budget)
    bg = torch.full((3,), float(dataset.white_background), device="cuda")
    topo = RealGeometryTopology(
        model, scene, pipe, bg, dataset.kernel_size, cache, args.training_seed, cfg
    )
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
    start = time.monotonic()
    with (out / "geometry_loss.jsonl").open("x", buffering=1) as log:
        for it in tqdm(range(1, opt.iterations + 1), mininterval=3):
            if not stack:
                stack = cameras.copy()
                rng.shuffle(stack)
            cam = cache.ensure(stack.pop())
            visits[cam.image_name] += 1
            model.update_learning_rate(it)
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
                multi_view_ncc_weight=opt.lambda_multi_view_ncc
                if sched.reg_enabled
                else 0.0,
            )
            peer = None
            if sched.reg_enabled and cam.nearest_id:
                peer = cache.ensure(
                    cameras[rng.choice(cam.nearest_id)],
                    keep=(cam,),
                    fields=("original_image",),
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
                    k: float(getattr(model, PARAMETERS[k]).grad.abs().sum())
                    for k in sorted(geometry)
                }
                if not all(torch.isfinite(torch.tensor(list(gradients.values())))):
                    raise FloatingPointError(f"Nonfinite geometry gradients at {it}")
                record = {
                    "step": it,
                    "camera": cam.image_name,
                    "points": len(model.get_xyz),
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
            if it <= opt.iterations - min(400, opt.iterations // 10):
                topo.step(it)
            if it % 400 == 0 and (not topo.events or topo.events[-1]["step"] != it):
                model.compute_3D_filter(cameras=cameras)
    model.compute_3D_filter(cameras=cameras)
    frozen = all(
        not getattr(model, a).requires_grad and getattr(model, a).grad is None
        for k, a in PARAMETERS.items()
        if k not in geometry
    )
    dc_equal = torch.equal(model._features_dc.detach(), topo.meta["frozen_dc"])
    if (
        not frozen
        or not dc_equal
        or model._features_rest.count_nonzero()
        or not torch.isfinite(model.get_xyz).all()
    ):
        raise RuntimeError("Final geometry/frozen appearance invariant failed")
    scene.save(opt.iterations, save_viewer=False)
    viewer = (
        out
        / "point_cloud"
        / f"iteration_{opt.iterations}"
        / "point_cloud_3dgsviewer.ply"
    )
    model.save_3dgsviewer_ply(str(viewer))
    write_json(
        out / "geometry_stats.json",
        {
            "status": "complete",
            "optimizer_steps": opt.iterations,
            "points": len(model.get_xyz),
            "active_sh_degree": model.active_sh_degree,
            "colors_frozen_verified": frozen and dc_equal,
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
