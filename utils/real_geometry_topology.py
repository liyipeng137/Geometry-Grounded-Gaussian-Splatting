"""Real-camera topology driven by surface evidence, visibility and binocular matching."""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from utils.geometry_topology import (
    apply_gaussian_transaction,
    make_seed_payload,
    real_filter_for_points,
    rotations,
    spacing_deduplicate,
)


@dataclass(frozen=True)
class TopologyConfig:
    interval: int = 200
    first: int = 800
    views: int = 12
    stride: int = 4
    max_points: int = 2_500_000
    max_splits: int = 6000
    max_seeds: int = 1500
    max_prune: int = 1200
    grace: int = 600
    parallax_degrees: float = 2.0


def camera_points(camera, xy, depth):
    return torch.stack(
        (
            (xy[..., 0] - camera.Cx) / camera.Fx * depth,
            (xy[..., 1] - camera.Cy) / camera.Fy * depth,
            depth,
        ),
        -1,
    )


def project(camera, xyz):
    q = xyz @ camera.R + camera.T
    z = q[..., 2]
    xy = q[..., :2] / z[..., None].clamp_min(1e-6)
    uv = (
        (
            xy * xy.new_tensor([camera.Fx, camera.Fy])
            + xy.new_tensor([camera.Cx, camera.Cy])
        )
        .round()
        .long()
    )
    u, v = uv.unbind(-1)
    inside = (
        (z > 0)
        & (u >= 0)
        & (u < camera.image_width)
        & (v >= 0)
        & (v < camera.image_height)
    )
    return (
        u.clamp(0, camera.image_width - 1),
        v.clamp(0, camera.image_height - 1),
        z,
        inside,
    )


def structural_variation(points, normals, valid, tau):
    """Adjacent same-plane samples have small point-to-plane distance even on a slanted wall."""
    structure = torch.zeros_like(valid)
    edge = torch.zeros_like(valid)
    for axis in (0, 1):
        for sign in (-1, 1):
            peer_p = torch.roll(points, sign, axis)
            peer_n = torch.roll(normals, sign, axis)
            both = valid & torch.roll(valid, sign, axis)
            both.select(axis, 0 if sign == 1 else -1).fill_(False)
            jump = ((peer_p - points) * normals).sum(-1).abs() > tau
            edge |= both & jump
            structure |= both & (jump | ((normals * peer_n).sum(-1) < 0.975))
    return structure, edge


def stereo_seed_match(camera, peer, xy, z, normals, tau):
    """Test actual gray patches at the proposed surface and at two alternative ranges."""
    import warp_patch_ncc

    rot = peer.world_view_transform[:3, :3].T @ camera.world_view_transform[:3, :3]
    trans = -rot @ camera.world_view_transform[3, :3] + peer.world_view_transform[3, :3]
    images = (
        camera.gray_image.squeeze().contiguous(),
        peer.gray_image.squeeze().contiguous(),
    )
    intr = (
        camera.Fx,
        camera.Fy,
        camera.Cx,
        camera.Cy,
        peer.Fx,
        peer.Fy,
        peer.Cx,
        peer.Cy,
    )

    def score(depth):
        cc, valid = warp_patch_ncc.warp_patch_ncc(
            depth.contiguous(),
            normals.contiguous(),
            xy.int().contiguous(),
            rot.T,
            trans,
            *images,
            *intr,
            False,
        )
        return cc.reshape(-1), valid.reshape(-1).bool()

    center, usable = score(z)
    offset = torch.maximum(3 * tau, 0.08 * z)
    low, vl = score((z - offset).clamp_min(0.01))
    high, vh = score(z + offset)
    # A flat, unobservable or out-of-frame alternative cannot certify a new point.
    return (
        usable & vl & vh & (center > 0.65) & (center > torch.maximum(low, high) + 0.025)
    )


class RealGeometryTopology:
    def __init__(
        self, model, scene, pipe, background, kernel_size, cache, seed, config=None
    ):
        self.model, self.scene, self.pipe, self.bg = model, scene, pipe, background
        self.kernel, self.cache = kernel_size, cache
        self.cfg = config or TopologyConfig()
        self.cams = scene.getTrainCameras()
        self.extent = scene.cameras_extent
        self.rng = np.random.default_rng(seed)
        self.pairs = []
        self.remaining = []
        self.previous = None
        dev = model.get_xyz.device
        self.meta = {
            "birth": torch.zeros(len(model.get_xyz), dtype=torch.long, device=dev),
        }
        self.counts = Counter()
        self.events = []

    def render(self, cam):
        from gaussian_renderer import render

        return render(
            cam,
            self.model,
            self.pipe,
            self.bg,
            self.kernel,
            require_depth=True,
            depth_search_expansions=2,
        )

    def choose_pairs(self):
        eligible = [i for i, c in enumerate(self.cams) if c.nearest_id]
        if not eligible:
            return []
        if not self.remaining:
            self.remaining = self.rng.permutation(eligible).tolist()
        ref = self.remaining[: self.cfg.views]
        self.remaining = self.remaining[len(ref) :]
        return [(i, int(self.rng.choice(self.cams[i].nearest_id))) for i in ref]

    def inspect(self, cam, peer, buf, peer_buf):
        from gaussian_renderer import geometry_score

        cfg, model = self.cfg, self.model
        s = cfg.stride
        sl = (slice(s // 2, None, s), slice(s // 2, None, s))
        d = cam.depth_prior[0][sl]
        v, u = torch.meshgrid(
            torch.arange(s // 2, cam.image_height, s, device=d.device),
            torch.arange(s // 2, cam.image_width, s, device=d.device),
            indexing="ij",
        )
        xy = torch.stack((u, v), -1)
        n = F.normalize(cam.normal_prior[:, sl[0], sl[1]].permute(1, 2, 0), dim=-1)
        valid = torch.isfinite(d) & (d > 0.01) & cam.normal_prior_mask[0][sl].bool()
        dc = torch.where(valid, d, 0.0)
        p = camera_points(cam, xy, dc)
        world = (p - cam.T) @ cam.R.T
        pu, pv, pz, inside = project(peer, world)
        pd = peer.depth_prior[0, pv, pu]
        pn = F.normalize(peer.normal_prior[:, pv, pu].permute(1, 2, 0), dim=-1)
        nw = n @ cam.R.T
        tau = torch.maximum(0.025 * dc, d.new_tensor(self.extent / 512))
        view1 = F.normalize(cam.camera_center - world, dim=-1)
        view2 = F.normalize(peer.camera_center - world, dim=-1)
        cosine = (view1 * view2).sum(-1)
        angle_ok = (cosine < math.cos(math.radians(cfg.parallax_degrees))) & (
            cosine > 0.45
        )
        match = (
            valid
            & inside
            & angle_ok
            & torch.isfinite(pd)
            & (pd > 0.01)
            & peer.normal_prior_mask[0, pv, pu].bool()
            & ((pz - pd).abs() <= torch.maximum(0.04 * pd, 1.5 * tau))
            & ((nw * (pn @ peer.R.T)).sum(-1) > 0.82)
            & ((nw * view1).sum(-1).abs() > 0.2)
        )
        structure, edge = structural_variation(p, n, valid, tau)
        z = buf["median_depth"][0][sl]
        status = buf["depth_status"][0][sl]
        ng = F.normalize(buf["normal"][:, sl[0], sl[1]].permute(1, 2, 0), dim=-1)
        residual = (z - dc).abs() / tau
        normal_error = (1 - (ng * n).sum(-1).clamp(-1, 1)) / 0.06
        near = (status == 3) & (residual < 3)
        err = torch.where(
            near & structure, torch.maximum(residual, normal_error).clamp(max=8), 0.0
        )
        good = near & (residual <= 1) & (normal_error <= 1)
        maps = torch.zeros(3, cam.image_height, cam.image_width, device=d.device)
        maps[0][sl], maps[1][sl], maps[2][sl] = err, good.float(), match.float()
        full_d = cam.depth_prior[0]
        full_tau = torch.maximum(0.025 * full_d, d.new_tensor(self.extent / 512))
        score = geometry_score(
            cam, model, self.pipe, self.bg, self.kernel, full_d, 3 * full_tau, *maps
        )

        # Signed free-space checks use an actually visible near-side contributor.
        # Exclude depth boundaries and leave everything behind the first surface alone.
        gu, gv, gz, visible = project(cam, model.get_xyz)
        gx = (gu // s).clamp(0, d.shape[1] - 1)
        gy = (gv // s).clamp(0, d.shape[0] - 1)
        gz_prior = d[gy, gx]
        margin = 3 * tau[gy, gx] + model.get_scaling_with_3D_filter.max(1).values
        in_front = (
            visible
            & match[gy, gx]
            & ~edge[gy, gx]
            & (status[gy, gx] == 3)
            & (gz + margin < gz_prior)
            & ((gz - z[gy, gx]).abs() < margin)
            & (buf["radii"] > 0)
        )
        peer_status = peer_buf["depth_status"][0, pv, pu]
        peer_z = peer_buf["median_depth"][0, pv, pu]
        missing = (status == 0) | (status == 1) | ((status == 3) & (z > dc + 3 * tau))
        peer_missing = (
            (peer_status == 0)
            | (peer_status == 1)
            | ((peer_status == 3) & (peer_z > pz + 3 * tau))
        )
        eligible = torch.where((match & missing & peer_missing).flatten())[0]
        seeds = {}
        if len(eligible):
            order = ((eligible * 2654435761) % 2147483647).argsort()[:384]
            ids = eligible[order]
            fy, fx = torch.unravel_index(ids, d.shape)
            verified = stereo_seed_match(
                cam, peer, xy[fy, fx], dc[fy, fx], n[fy, fx], tau[fy, fx]
            )
            fy, fx = fy[verified], fx[verified]
            key = (v[fy, fx] * cam.image_width + u[fy, fx]).cpu().tolist()
            xyz, normal = world[fy, fx].cpu().numpy(), nw[fy, fx].cpu().numpy()
            spacing = (
                (4 * dc[fy, fx] / min(cam.Fx, cam.Fy))
                .clamp_min(self.extent / 1400)
                .cpu()
                .numpy()
            )
            seeds = {
                (cam.image_name, k): (pos, norm, h)
                for k, pos, norm, h in zip(key, xyz, normal, spacing)
            }
        return score, in_front, seeds

    def seed_rows(self, seeds, cap):
        empty = {
            k: getattr(self.model, a)[:0].detach()
            for k, a in self.model.point_parameters.items()
        }
        if not seeds or cap <= 0:
            return empty, self.model.filter_3D[:0]
        xyz, norm, h = (np.asarray(x) for x in zip(*seeds))
        ids = spacing_deduplicate(xyz, h, 4 * cap)
        xyz, norm, h = (x[ids] for x in (xyz, norm, h))
        model = self.model
        tree = cKDTree(model.get_xyz.detach().cpu().numpy())
        _, neighbours = tree.query(xyz, k=min(4, len(model.get_xyz)), workers=1)
        ix = torch.as_tensor(neighbours, device=model.get_xyz.device).reshape(
            len(xyz), -1
        )
        pos = model.get_xyz.new_tensor(xyz)
        basis = rotations(model._rotation[ix].reshape(-1, 4)).reshape(*ix.shape, 3, 3)
        local = torch.einsum("nki,nkij->nkj", pos[:, None] - model.get_xyz[ix], basis)
        nearby = (
            local / model.get_scaling_with_3D_filter[ix].clamp_min(1e-8)
        ).square().sum(-1) < 1.5**2
        keep = ~(nearby & (model.get_opacity_with_3D_filter[ix, 0] > 0.08)).any(1)
        filt, seen = real_filter_for_points(pos, self.cams)
        payload, ok = make_seed_payload(model, pos, norm, h, filt)
        take = torch.where(keep & seen & ok)[0][:cap]
        return {k: v[take] for k, v in payload.items()}, filt[take]

    def safe_removals(self, candidates, witnesses):
        if not len(candidates):
            return candidates
        model = self.model

        # Probe the actual joint effect of removing the whole batch on the same rays.
        def probe(c):
            b = self.render(c)
            z = b["median_depth"][0, 2::5, 2::5]
            valid = b["depth_status"][0, 2::5, 2::5] == 3
            d = c.depth_prior[0, 2::5, 2::5]
            dom = torch.isfinite(d) & (d > 0.01)
            return valid & dom, valid & dom & ((z - d).abs() < 0.05 * d)

        before = []
        for i in witnesses:
            before.append(probe(self.cache.ensure(self.cams[i])))
        opacity = model._opacity[candidates].clone()
        try:
            model._opacity[candidates] = -24.0
            for i, (hit, good) in zip(witnesses, before):
                after, ok = probe(self.cache.ensure(self.cams[i]))
                if (hit & ~after).sum() > max(2, int(hit.sum()) // 20000) or (
                    good & ~ok
                ).sum() > max(2, int(good.sum()) // 10000):
                    self.counts["vetoed_removals"] += len(candidates)
                    return candidates[:0]
        finally:
            model._opacity[candidates] = opacity
        return candidates

    @torch.no_grad()
    def step(self, iteration):
        cfg, model = self.cfg, self.model
        if iteration < cfg.first or (iteration - cfg.first) % cfg.interval:
            return
        if self.previous is None:
            self.pairs = self.choose_pairs()
        n = len(model.get_xyz)
        dev = model.get_xyz.device
        error = torch.zeros(n, device=dev)
        support, near, opportunity = (torch.zeros_like(error) for _ in range(3))
        free = torch.zeros(n, dtype=torch.bool, device=dev)
        separate_free = torch.zeros_like(free)
        first_ray = torch.zeros_like(model.get_xyz)
        seeds = {}
        for i, j in self.pairs:
            cam = self.cache.ensure(self.cams[i])
            peer = self.cache.ensure(self.cams[j], keep=(cam,))
            buf = self.render(cam)
            scores, conflict, proposals = self.inspect(
                cam, peer, buf, self.render(peer)
            )
            error += scores["error_sum"]
            near += scores["near_weight_sum"]
            support += scores["support_sum"]
            opportunity += scores["weight_sum"]
            ray = F.normalize(cam.camera_center - model.get_xyz, dim=-1)
            separate_free |= (
                conflict
                & free
                & (
                    (first_ray * ray).sum(-1)
                    < math.cos(math.radians(cfg.parallax_degrees))
                )
            )
            first_ray[conflict & ~free] = ray[conflict & ~free]
            free |= conflict
            seeds.update(proposals)
        age = iteration - self.meta["birth"] >= cfg.grace
        rank = error / near.clamp_min(1e-8)
        split = (
            (rank > 0.75)
            & (near > 0.015)
            & age
            & (model.get_scaling.max(1).values > self.extent / 1200)
        )
        protected = support > 0.015
        low = (model.get_opacity_with_3D_filter[:, 0] < 0.03) & (opportunity > 0.02)
        prune = ~protected & age & (separate_free | low)
        self.counts.update(
            audits=1,
            seed_proposals=len(seeds),
            free_space_multiview=int(separate_free.sum()),
        )
        if self.previous is None:
            self.previous = (split, prune, seeds)
            return
        prev_split, prev_prune, prev_seeds = self.previous
        self.previous = None
        ids = torch.where(prune & prev_prune)[0]
        ids = ids[
            (
                separate_free[ids].float() * 2
                + 1
                - model.get_opacity_with_3D_filter[ids, 0]
            ).argsort(descending=True)[: cfg.max_prune]
        ]
        removed = self.safe_removals(ids, sorted({i for i, _ in self.pairs}))
        drop = torch.zeros(n, dtype=torch.bool, device=dev)
        drop[removed] = True
        budget = max(0, min(12000, n // 20, cfg.max_points - n + len(removed)))
        eligible = torch.where(split & prev_split & ~drop)[0]
        parents = eligible[
            rank[eligible].argsort(descending=True)[: min(cfg.max_splits, budget)]
        ]
        new_seeds = [value for key, value in seeds.items() if key in prev_seeds]
        additions, filt = self.seed_rows(
            new_seeds, min(budget - len(parents), cfg.max_seeds)
        )
        seed_count = len(filt)
        drop[parents] = True
        owners = parents.repeat(2)
        children = {
            k: getattr(model, a)[owners].clone()
            for k, a in self.model.point_parameters.items()
        }
        if len(parents):
            axes = rotations(model._rotation[parents])
            scales = model.get_scaling[parents]
            dim = scales.argmax(1)
            tangent = axes[torch.arange(len(parents), device=dev), :, dim]
            offset = (
                tangent
                * (0.45 * scales.max(1).values.clamp(max=self.extent / 80))[:, None]
            )
            children["xyz"] += torch.cat((offset, -offset))
            children["scaling"].scatter_add_(
                1,
                dim.repeat(2)[:, None],
                torch.full_like(owners[:, None], math.log(0.65), dtype=scales.dtype),
            )
        combined = {
            k: torch.cat((children[k], additions[k]))
            for k in self.model.point_parameters
        }
        extra_meta = {
            "birth": torch.full(
                (len(combined["xyz"]),), iteration, device=dev, dtype=torch.long
            ),
        }
        self.meta = apply_gaussian_transaction(
            model,
            torch.where(~drop)[0],
            combined,
            torch.cat((model.filter_3D[owners], filt)),
            self.meta,
            extra_meta,
            cfg.max_points,
        )
        if len(owners) or seed_count:
            model.compute_3D_filter(cameras=self.cams)
        self.events.append(
            {
                "step": iteration,
                "parents_split": len(parents),
                "seeds": seed_count,
                "prune_candidates": len(ids),
                "pruned": len(removed),
                "points": len(model.get_xyz),
            }
        )
        self.counts.update(
            transactions=1,
            parents_split=len(parents),
            seeds=seed_count,
            pruned=len(removed),
        )
