"""Joint-only topology: independent geometry evidence and atomic point changes.

No training renderer is imported until an audit runs. Pure tensor operations are
therefore testable on CPU; the baseline Gaussian lifecycle remains untouched.
"""

from collections import Counter
import copy
import math

import numpy as np
from scipy.spatial import cKDTree
import torch


PARAMETERS = {
    "xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
    "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation",
    "sg_axis": "_sg_axis", "sg_sharpness": "_sg_sharpness", "sg_color": "_sg_color",
}
DEFAULTS = {
    "max_gaussians": 2_500_000, "topology_interval": 100, "audit_interval": 500,
    "audit_views": 16, "priority_views": 8, "topology_until": 15000,
    "seed_from": 5001, "seed_until": 14000, "seed_batch_limit": 5000,
    "seed_total_limit": 150000, "grace_updates": 500, "geometry_reserve": .1,
    "max_growth": 20000, "max_growth_fraction": .05, "seed_fraction": .01,
    "geometry_reserve_release": 12000, "patch_prune_fraction": .1,
    "seed_pixels_per_view": 1024, "seed_queue_limit": 50000,
}


def rotations(quaternions):
    """World rotation matrices for wxyz quaternions, on the input device."""
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(-1, 3, 3)


def normal_quaternions(normals):
    n = torch.nn.functional.normalize(normals, dim=-1)
    q = torch.stack((1+n[:, 2], -n[:, 1], n[:, 0], torch.zeros_like(n[:, 0])), -1)
    reverse = n[:, 2] < -1 + 1e-6
    q[reverse] = q.new_tensor([0, 1, 0, 0])
    return torch.nn.functional.normalize(q, dim=-1)


def filter_coefficient(scales, filtering):
    squared = scales.square()
    return (squared.prod(-1) / (squared + filtering.square()).prod(-1)).sqrt()


@torch.no_grad()
def apply_gaussian_transaction(model, keep, additions, addition_filter, metadata,
                               addition_metadata, max_points=2_500_000):
    """Build every replacement first, then commit one row mapping to all state.

    Only consumed-gradient boundaries are accepted. Surviving Adam moments and
    group step counters remain intact; new rows receive zero moments.
    """
    count = len(model.get_xyz)
    device = model.get_xyz.device
    keep = torch.as_tensor(keep, dtype=torch.long, device=device)
    added = len(additions["xyz"])
    if len(keep) + added > max_points:
        raise ValueError("Topology transaction exceeds the hard Gaussian budget")
    if keep.ndim != 1 or (len(keep) and (keep.min() < 0 or keep.max() >= count)):
        raise ValueError("Invalid survivor index mapping")
    if len(torch.unique(keep)) != len(keep):
        raise ValueError("A survivor cannot appear twice")
    if added == 0 and torch.equal(keep, torch.arange(count, device=device)):
        return metadata
    if set(additions) != set(PARAMETERS) or set(metadata) != set(addition_metadata):
        raise ValueError("Incomplete topology parameter or metadata payload")
    for key, values in metadata.items():
        if len(values) != count or len(addition_metadata[key]) != added:
            raise ValueError(f"Metadata row mismatch: {key}")
    if addition_filter.shape != (added, 1) or model.filter_3D.shape != (count, 1):
        raise ValueError("Filter rows must match their Gaussian rows")
    replacements, optimizer_updates = {}, []
    for optimizer in model._iter_optimizers():
        for group in optimizer.param_groups:
            name = group["name"]
            if name not in PARAMETERS:
                continue
            old = group["params"][0]
            if old.grad is not None:
                raise RuntimeError(f"Topology before pending {name} gradient was consumed")
            extra = additions[name].to(device=old.device, dtype=old.dtype)
            if extra.shape != (added, *old.shape[1:]) or not torch.isfinite(extra).all():
                raise ValueError(f"Invalid new Gaussian parameter: {name}")
            new = torch.nn.Parameter(torch.cat((old[keep], extra), 0))
            previous = optimizer.state.get(old)
            state = None
            if previous is not None:
                state = {}
                for key, value in previous.items():
                    if torch.is_tensor(value) and value.shape == old.shape:
                        state[key] = torch.cat((value[keep], torch.zeros_like(extra)), 0)
                    else:
                        state[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            replacements[name] = new
            optimizer_updates.append((optimizer, group, old, new, state))
    if set(replacements) != set(PARAMETERS):
        raise ValueError("Gaussian optimizers do not own all required point tensors")
    mapped = {key: torch.cat((value[keep], addition_metadata[key].to(value)), 0)
              for key, value in metadata.items()}
    filtering = torch.cat((model.filter_3D[keep], addition_filter.to(model.filter_3D)), 0)
    if not torch.isfinite(filtering).all() or (filtering < 0).any():
        raise ValueError("Invalid topology filter")
    for optimizer, group, old, new, state in optimizer_updates:
        optimizer.state.pop(old, None)
        group["params"][0] = new
        if state is not None:
            optimizer.state[new] = state
    for name, parameter in replacements.items():
        setattr(model, PARAMETERS[name], parameter)
    model.filter_3D = filtering
    for name in ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom", "max_radii2D"):
        value = getattr(model, name)
        setattr(model, name, torch.cat((value[keep], value.new_zeros((added, *value.shape[1:]))), 0))
    return mapped


@torch.no_grad()
def real_filter_for_points(points, cameras):
    """Match baseline real-camera frusta, without its unsupported-point fallback."""
    distance = points.new_full((len(points),), torch.inf)
    focal = 0.0
    for camera in cameras:
        xyz = points @ camera.R.to(points) + camera.T.to(points)
        z = xyz[:, 2]
        uv = xyz[:, :2] / z.clamp_min(1e-8)[:, None]
        uv[:, 0] += (camera.Cx - (camera.image_width-1)/2) / camera.Fx
        uv[:, 1] += (camera.Cy - (camera.image_height-1)/2) / camera.Fy
        valid = ((z > .2) & (uv[:, 0].abs() <= camera.image_width/camera.Fx*.575)
                 & (uv[:, 1].abs() <= camera.image_height/camera.Fy*.575))
        distance = torch.where(valid, torch.minimum(distance, z), distance)
        focal = max(focal, camera.Fx)
    if focal <= 0:
        raise ValueError("At least one positive-focal real camera is required")
    valid = torch.isfinite(distance)
    return (distance / focal * math.sqrt(.2))[:, None], valid


@torch.no_grad()
def joint_filter_for_points(points, cameras, root_gen, h_ref, witness, virtual_records, previous):
    result, real_valid = real_filter_for_points(points, cameras)
    # Preserve baseline behavior for original real points, including unsupported ones.
    legacy = ~real_valid & ~root_gen
    if legacy.any():
        if real_valid.any():
            result[legacy] = result[real_valid].max()
        else:
            result[legacy] = previous[legacy]
    fallback = ~real_valid & root_gen
    result[fallback] = previous[fallback]
    for view in torch.unique(witness[fallback]).tolist():
        if view < 0 or view >= len(virtual_records):
            continue
        ids = torch.where(fallback & (witness == view))[0]
        record = virtual_records[view]
        pose = points.new_tensor(record["c2w"])
        intrinsic = points.new_tensor(record["K"])
        q = (points[ids] - pose[:3, 3]) @ pose[:3, :3]
        projected = q @ intrinsic.T
        uv = projected[:, :2] / q[:, 2:3].clamp_min(1e-8)
        valid = ((q[:, 2] > .2) & (uv[:, 0] >= 0) & (uv[:, 0] < record["width"])
                 & (uv[:, 1] >= 0) & (uv[:, 1] < record["height"]))
        footprint = q[:, 2] / max(float(intrinsic[0, 0]), float(intrinsic[1, 1])) * math.sqrt(.2)
        result[ids[valid], 0] = torch.minimum(footprint[valid], .1*h_ref[ids[valid]])
    if not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("A joint point has no finite filter support")
    return result


def allocate_growth(real_count, geometry_count, seed_count, available):
    """Disjoint source quotas, borrowing unused slots without crossing the cap."""
    available = max(0, int(available))
    counts = [int(real_count), int(geometry_count), int(seed_count)]
    initial = [available//2, available//4, available - available//2 - available//4]
    assigned = [min(count, quota) for count, quota in zip(counts, initial)]
    remaining = available - sum(assigned)
    # Unserved geometry/holes borrow before already well-supported real regions.
    for index in (2, 1, 0):
        extra = min(remaining, counts[index] - assigned[index])
        assigned[index] += extra
        remaining -= extra
    return tuple(assigned)


def spacing_deduplicate(points, spacing, limit):
    """Stable variable-radius rejection in one fixed world-coordinate grid."""
    points, spacing = np.asarray(points), np.asarray(spacing)
    if not len(points) or limit <= 0:
        return np.empty(0, np.int64)
    if (points.shape != (len(spacing), 3) or not np.isfinite(points).all()
            or not np.isfinite(spacing).all() or (spacing <= 0).any()):
        raise ValueError("Deduplication needs finite points and positive local spacing")
    cell_size = .75 * float(spacing.max())
    cells = np.floor(points/cell_size).astype(np.int64)
    buckets, chosen = {}, []
    for index, cell in enumerate(cells):
        nearby = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby.extend(buckets.get((int(cell[0])+dx, int(cell[1])+dy, int(cell[2])+dz), ()))
        if nearby:
            distance2 = np.square(points[nearby]-points[index]).sum(1)
            required = .75*np.maximum(spacing[nearby], spacing[index])
            if (distance2 < required**2).any():
                continue
        key = tuple(map(int, cell))
        buckets.setdefault(key, []).append(index)
        chosen.append(index)
        if len(chosen) >= limit:
            break
    return np.asarray(chosen, np.int64)


def choose_seed_witnesses(xyz, view_epochs, camera_centers, epoch, minimum_angle=3., maximum_age=1):
    """Two recent geometric directions, not two frame IDs or camera rotations."""
    ids = sorted(int(view) for view, checked in view_epochs.items() if epoch-maximum_age <= checked <= epoch)
    if len(ids) < 2:
        return []
    directions = np.asarray(camera_centers)[ids]-np.asarray(xyz)
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1e-8
    ids = np.asarray(ids, np.int64)[valid]
    directions = directions[valid]/lengths[valid, None]
    if len(ids) < 2:
        return []
    cosine = np.clip(directions @ directions.T, -1., 1.)
    separated = np.triu(cosine <= math.cos(math.radians(minimum_angle)), k=1)
    if not separated.any():
        return []
    first, second = np.unravel_index(np.argmin(np.where(separated, cosine, np.inf)), cosine.shape)
    return [int(ids[first]), int(ids[second])]


def make_seed_payload(model, xyz, normals, spacing, filtering, rgb):
    xyz = torch.as_tensor(xyz, dtype=model.get_xyz.dtype, device=model.get_xyz.device)
    spacing = torch.as_tensor(spacing, dtype=xyz.dtype, device=xyz.device)
    normals = torch.as_tensor(normals, dtype=xyz.dtype, device=xyz.device)
    rgb = torch.as_tensor(rgb, dtype=xyz.dtype, device=xyz.device)
    scales = spacing[:, None] * xyz.new_tensor([.7, .7, .2])
    coefficient = filter_coefficient(scales, filtering)
    probability = .25 / coefficient.clamp_min(1e-12)
    valid = ((probability < .95) & torch.isfinite(probability) & (spacing > 0)
             & torch.isfinite(xyz).all(1) & torch.isfinite(normals).all(1) & (normals.norm(dim=1) > .5))
    payload = {name: getattr(model, field).new_zeros((len(xyz), *getattr(model, field).shape[1:]))
               for name, field in PARAMETERS.items()}
    payload.update(xyz=xyz, scaling=scales.log(), rotation=normal_quaternions(normals),
                   opacity=torch.logit(probability.clamp(1e-6, 1-1e-6))[:, None])
    payload["f_dc"][:, 0, :] = (rgb.clamp(0, 1) - .5) / .28209479177387814
    if payload["sg_axis"].numel():
        payload["sg_axis"][..., 2] = 1
    return payload, valid


class GeometryTopology:
    def __init__(self, gaussians, scene, teacher, config, pipe, background,
                 kernel_size, frame_cache=None, opt=None):
        self.model, self.scene, self.teacher = gaussians, scene, teacher
        self.config = {**DEFAULTS, **config}
        self.pipe, self.background, self.kernel_size = pipe, background, kernel_size
        self.frame_cache, self.opt = frame_cache, opt
        self.real_cameras = scene.getTrainCameras()
        self.records = teacher.cameras
        self.virtual_centers = np.asarray([np.asarray(record["c2w"])[:3, 3] for record in self.records]).reshape(-1, 3)
        self.scale = float(teacher.scene_scale)
        self.device = gaussians.get_xyz.device
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        count = len(gaussians.get_xyz)
        self.meta = self._empty_metadata(count)
        self.meta["point_id"] = torch.arange(count, device=self.device)
        self.next_id, self.seed_total = count, 0
        self.audit_cursor, self.audit_epoch, self.pending = 0, 0, False
        self.last_audit, self.iteration = 0, 0
        self.queue, self.seeded_cells = {}, set()
        self.priority = set()
        self.counts = Counter()
        self.last_event = {}
        self.atlas_tree = cKDTree(np.asarray(teacher.atlas["xyz"]))
        self.atlas_xyz = torch.as_tensor(np.asarray(teacher.atlas["xyz"]), device=self.device, dtype=torch.float32)

    def _empty_metadata(self, n):
        integer = ("point_id", "parent_id", "source", "birth", "grace", "patch_id", "witness",
                   "witness2", "checks", "error_streak", "unsupported_streak", "last_checked")
        floating = ("h_ref", "real_grad", "real_abs", "real_seen", "virtual_grad", "virtual_abs",
                    "virtual_seen", "geom_error", "geom_support", "geom_weight", "real_support")
        out = {key: torch.zeros(n, dtype=torch.long, device=self.device) for key in integer}
        out.update({key: torch.zeros(n, device=self.device) for key in floating})
        for key in ("parent_id", "patch_id", "witness", "witness2"):
            out[key].fill_(-1)
        out["root_gen"] = torch.zeros(n, dtype=torch.bool, device=self.device)
        out["root_geometry"] = torch.zeros(n, dtype=torch.bool, device=self.device)
        return out

    def _opt(self, key, default):
        return getattr(self.opt, key, self.config.get(key, default))

    @torch.no_grad()
    def _collect(self, buffers, prefix):
        visible = buffers["visibility_filter"]
        gradient = buffers["viewspace_points"].grad
        if gradient is None:
            raise RuntimeError("Topology statistics must be collected after backward")
        finite = visible & torch.isfinite(gradient).all(-1)
        self.meta[prefix+"_grad"][finite] += gradient[finite, :2].norm(dim=-1)
        self.meta[prefix+"_abs"][finite] += gradient[finite, 2:].norm(dim=-1)
        self.meta[prefix+"_seen"][finite] += 1
        self.model.max_radii2D[visible] = torch.maximum(self.model.max_radii2D[visible], buffers["radii"][visible])

    def collect_real(self, buffers):
        self._collect(buffers, "real")

    def collect_virtual(self, buffers, camera):
        del camera
        self._collect(buffers, "virtual")

    def _view(self, index):
        return self.teacher.view(int(index))

    def _audit_indices(self):
        length = len(self.teacher)
        regular = [(self.audit_cursor+i) % length for i in range(min(length, self.config["audit_views"]))]
        self.audit_cursor = (self.audit_cursor+len(regular)) % length
        priority = sorted(self.priority)[:self.config["priority_views"]]
        return list(dict.fromkeys(regular+priority))

    def _expire_seed_queue(self):
        expired = [key for key, item in self.queue.items() if item["epoch"] < self.audit_epoch-2]
        for key in expired:
            del self.queue[key]
        self.counts["expired_seed_candidates"] += len(expired)
        return len(expired)

    @torch.no_grad()
    def _queue_missing(self, camera, buffers):
        """Revisit fixed world samples; random image pixels cannot earn persistence."""
        target = camera.teacher
        status = buffers["depth_status"].squeeze()
        record = self.records[camera.virtual_index]
        intrinsic = self.atlas_xyz.new_tensor(record["K"])
        intrinsic[0] *= camera.image_width / record["width"]
        intrinsic[1] *= camera.image_height / record["height"]
        pose = self.atlas_xyz.new_tensor(record["c2w"])
        q = (self.atlas_xyz-pose[:3, 3]) @ pose[:3, :3]
        uv = q @ intrinsic.T
        uv = uv[:, :2]/q[:, 2:3].clamp_min(1e-8)
        xx, yy = uv.floor().long().unbind(1)
        inside = ((q[:, 2] > .2) & (xx >= 0) & (yy >= 0)
                  & (xx < camera.image_width) & (yy < camera.image_height))
        ids = torch.where(inside)[0]
        if not len(ids):
            return
        x, y = xx[ids], yy[ids]
        depth = target["depth"].squeeze()[y, x]
        footprint = 2*q[ids, 2]/max(float(intrinsic[0, 0]), float(intrinsic[1, 1]))
        # Subpixel projection may shift the sampled plane depth. Keep this narrow
        # pixel-footprint band, plus patch identity, rather than a scene-sized NN gate.
        atlas_patch = torch.as_tensor(self.teacher.atlas["patch_id"], device=self.device)[ids]
        same_surface = ((q[ids, 2]-depth).abs() <= torch.maximum(.25*footprint, depth.new_tensor(2e-5*self.scale)))
        same_surface &= target["patch_id"].squeeze()[y, x] == atlas_patch
        sample_normals = self.atlas_xyz.new_tensor(self.teacher.atlas["normal_world"])[ids]
        hit_normals = target["normal"].permute(1, 2, 0)[y, x] @ pose[:3, :3].T
        same_surface &= (sample_normals*hit_normals).sum(1).abs() > .8
        eligible = (same_surface & target["valid_depth"].squeeze()[y, x].bool()
                    & (target["weight_coverage"].squeeze()[y, x] > 0)
                    & ((status[y, x] == 0) | (status[y, x] == 1)))
        ids = ids[eligible].cpu().numpy()
        # Fixed world priority yields repeatable cross-view opportunities. Existing
        # queue entries are always revisited before admitting bounded new samples.
        previous = [int(i) for i in ids if (int(i),) in self.queue]
        new = [int(i) for i in ids if (int(i),) not in self.queue]
        new.sort(key=lambda i: ((i*2654435761) & 0xffffffff, i))
        ids = previous + new[:self.config["seed_pixels_per_view"]]
        selected = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        sizes = (2*q[selected, 2]/max(float(intrinsic[0, 0]), float(intrinsic[1, 1]))).cpu().numpy()
        weights = target["weight_coverage"].squeeze()[yy[selected], xx[selected]].cpu().numpy()
        for index, h, weight in zip(ids, sizes, weights):
            key = (index,)
            xyz = self.teacher.atlas["xyz"][index]
            normal = self.teacher.atlas["normal_world"][index]
            patch = int(self.teacher.atlas["patch_id"][index])
            if key in self.seeded_cells or not np.isfinite(normal).all() or np.linalg.norm(normal) < .5:
                continue
            item = self.queue.get(key)
            if item is None:
                if len(self.queue) >= self.config["seed_queue_limit"]:
                    continue
                item = {"xyz": xyz.copy(), "normal": normal.copy(), "patch": patch, "h": float(h),
                        "weight": float(weight), "views": {}, "checks": 0, "epoch": -1}
                self.queue[key] = item
            item["views"] = {view: epoch for view, epoch in item["views"].items()
                             if epoch >= self.audit_epoch-1}
            item["views"][int(camera.virtual_index)] = self.audit_epoch
            if item["epoch"] != self.audit_epoch:
                item["checks"] = item["checks"]+1 if item["epoch"] >= self.audit_epoch-2 else 1
                item["epoch"] = self.audit_epoch

    @torch.no_grad()
    def _audit(self, iteration):
        from gaussian_renderer import geometry_score, render
        self.audit_epoch += 1
        self._expire_seed_queue()
        n = len(self.model.get_xyz)
        errors = torch.zeros(n, device=self.device)
        support, near, opportunities = errors.clone(), errors.clone(), errors.clone()
        views = torch.zeros(n, dtype=torch.long, device=self.device)
        priority_checked = set()
        for index in self._audit_indices():
            camera = self._view(index)
            target = camera.teacher
            buffers = render(camera, self.model, self.pipe, self.background, self.kernel_size,
                             require_depth=True, depth_search_expansions=2)
            d, pred = target["depth"].squeeze(), buffers["median_depth"].squeeze()
            valid = (buffers["depth_status"].squeeze() == 3) & torch.isfinite(pred)
            tau = torch.maximum(d*.02, d.new_tensor(self.scale/64))
            distance = (pred-d).abs()
            normals = torch.nn.functional.normalize(buffers["normal"], dim=0)
            cosine = (normals*target["normal"]).sum(0).clamp(-1, 1)
            teacher_normal_valid = target["valid_normal"].squeeze().bool()
            angle = torch.where(teacher_normal_valid, torch.acos(cosine), torch.zeros_like(cosine))
            same_layer = valid & (distance <= 3*tau)
            error = torch.maximum(distance/tau, angle/math.radians(15)).clamp(max=10)
            error = torch.where(same_layer & (error > 1), error, torch.zeros_like(error))
            good = same_layer & (distance <= tau) & (angle <= math.radians(20))
            weight = target["weight_depth"].squeeze()*target["q"].squeeze()*target["valid_depth"].squeeze()
            scores = geometry_score(camera, self.model, self.pipe, self.background, self.kernel_size,
                                    d, 3*tau, error, good.float(), weight)
            errors += scores["error_sum"]
            support += scores["support_sum"]
            near += scores["near_weight_sum"]
            opportunities += scores["weight_sum"]
            seen = scores["near_weight_sum"] > 1e-7
            views += seen
            # Gen fallback is tied to its fixed birth witness, including descendants.
            self.meta["witness"][seen & ~self.meta["root_gen"]] = index
            if self.config["seed_from"]-1000 <= iteration <= self.config["seed_until"]:
                self._queue_missing(camera, buffers)
            priority_checked.add(index)
        # A Gen point merely behind some other teacher surface has no negative
        # witness. Its own surface band must actually receive scoring opportunity.
        checked = torch.where(self.meta["root_gen"], near > 1e-7, opportunities > 1e-7)
        self.meta["geom_error"][checked] = errors[checked]/near[checked].clamp_min(1e-7)
        self.meta["geom_support"][checked] = support[checked]/near[checked].clamp_min(1e-7)
        self.meta["geom_weight"][checked] = near[checked]
        self.meta["checks"][checked] += 1
        self.meta["last_checked"][checked] = iteration
        wrong = checked & (views >= 2) & (self.meta["geom_error"] > 1)
        self.meta["error_streak"][checked] = torch.where(wrong[checked], self.meta["error_streak"][checked]+1, 0)
        unsupported = checked & (self.meta["geom_support"] < .05)
        self.meta["unsupported_streak"][checked] = torch.where(unsupported[checked], self.meta["unsupported_streak"][checked]+1, 0)
        # True-depth support is independent of photo-error VCD and protects real detail.
        real_support = torch.zeros(n, device=self.device)
        selected = self.rng.choice(len(self.real_cameras), min(10, len(self.real_cameras)), replace=False)
        for index in selected:
            camera = self.real_cameras[index]
            if self.frame_cache is not None:
                camera = self.frame_cache.ensure(camera)
            depth = camera.depth_prior.squeeze()
            valid = torch.isfinite(depth) & (depth > .2)
            depth = torch.where(valid, depth, torch.zeros_like(depth))
            tau = torch.maximum(depth*.02, depth.new_tensor(self.scale/64))
            score = geometry_score(camera, self.model, self.pipe, self.background, self.kernel_size,
                                   depth, tau, torch.zeros_like(depth), valid.float(), valid.float())
            real_support += score["support_sum"]
        self.meta["real_support"] = real_support
        self.last_audit = iteration
        self.priority -= priority_checked
        self.counts["audits"] += 1

    def _canonical_colors(self, xyz):
        from utils.virtual_camera_selection import depth_evidence
        colors = []
        model_cameras = {getattr(c, "source_image_name", None) or c.image_name: c for c in self.real_cameras}
        for record, depth in zip(self.teacher.real_cameras, self.teacher.real_depths):
            camera = model_cameras.get(record["name"])
            if camera is None:
                continue
            evidence = depth_evidence(xyz, record, depth, self.scale)["support"]
            ids = np.flatnonzero(evidence)
            if not len(ids):
                continue
            pose, intrinsic = np.asarray(record["c2w"]), np.asarray(record["K"])
            q = (xyz[ids]-pose[:3, 3]) @ pose[:3, :3]
            uv = q @ intrinsic.T
            uv = uv[:, :2]/uv[:, 2:3]
            x = np.clip((uv[:, 0]*camera.image_width/record["width"]).astype(int), 0, camera.image_width-1)
            y = np.clip((uv[:, 1]*camera.image_height/record["height"]).astype(int), 0, camera.image_height-1)
            image = getattr(camera, "original_image_cpu", None)
            if image is None:
                image = camera.original_image
            if image is None:
                raise RuntimeError("Real camera has neither a CPU nor device RGB payload")
            rgb = image[:, torch.as_tensor(y, device=image.device), torch.as_tensor(x, device=image.device)].T.float()
            if image.dtype == torch.uint8:
                rgb /= 255
            if self.model.app_model.name == "PGSR":
                appearance = self.model.get_apperance_embedding(camera.uid).detach().to(rgb)
                rgb = (rgb-appearance[1])/torch.exp(appearance[0])
            row = np.full((len(xyz), 3), np.nan, np.float32)
            row[ids] = rgb.clamp(0, 1).cpu().numpy()
            colors.append(row)
        result = np.full((len(xyz), 3), .5, np.float32)
        if colors:
            stack = np.stack(colors)
            known = np.isfinite(stack[..., 0]).any(0)
            result[known] = np.nanmedian(stack[:, known], axis=0)
        return result

    @torch.no_grad()
    def _seeds(self, iteration, limit):
        empty = {name: getattr(self.model, attr).new_empty((0, *getattr(self.model, attr).shape[1:]))
                 for name, attr in PARAMETERS.items()}
        if limit <= 0 or not self.config["seed_from"] <= iteration <= self.config["seed_until"]:
            return empty, self._empty_metadata(0), self.model.filter_3D[:0], []
        eligible = [(key, item) for key, item in self.queue.items()
                    if item["checks"] >= 2 and len(item["views"]) >= 2 and item["epoch"] == self.audit_epoch]
        eligible.sort(key=lambda pair: (-pair[1]["weight"], pair[0]))
        reserved = set(self.priority)
        selected = []
        for key, item in eligible:
            witnesses = choose_seed_witnesses(item["xyz"], item["views"], self.virtual_centers, self.audit_epoch)
            if not witnesses:
                continue
            if len(reserved | set(witnesses)) > self.config["priority_views"]:
                continue
            selected.append((key, item, witnesses))
            reserved.update(witnesses)
            if len(selected) >= 4*limit:
                break
        if not selected:
            return empty, self._empty_metadata(0), self.model.filter_3D[:0], []
        xyz = np.asarray([item["xyz"] for _, item, _ in selected])
        h = np.asarray([item["h"] for _, item, _ in selected])
        existing = self.model.get_xyz.detach().cpu().numpy()
        tree = cKDTree(existing)
        distances, neighbors = tree.query(xyz, k=min(4, len(existing)), workers=1)
        if distances.ndim == 1:
            distances, neighbors = distances[:, None], neighbors[:, None]
        close = distances[:, 0] < 4*h
        h[close] = np.clip(np.median(distances[close], axis=1), .5*h[close], 2*h[close])
        nearest = torch.as_tensor(neighbors, device=self.device)
        delta = self.model.get_xyz.new_tensor(xyz)[:, None, :]-self.model.get_xyz[nearest]
        bases = rotations(self.model._rotation[nearest].reshape(-1, 4)).reshape(*nearest.shape, 3, 3)
        local = torch.einsum("nki,nkij->nkj", delta, bases)
        scales = (self.model.get_scaling[nearest].square()+self.model.filter_3D[nearest].square()).sqrt()
        mahalanobis = (local/scales.clamp_min(1e-8)).square().sum(-1).cpu().numpy()
        opacity = self.model.get_opacity_with_3D_filter[nearest, 0].cpu().numpy()
        evidence = self.teacher.real_evidence(xyz)
        covered = ((mahalanobis < 9) & (opacity > .01)).any(axis=1)
        valid = (distances[:, 0] > .75*h) & ~covered & (evidence["evidence_class"] != 3)
        valid_ids = np.flatnonzero(valid)
        chosen = valid_ids[spacing_deduplicate(xyz[valid_ids], h[valid_ids], limit)]
        if not len(chosen):
            return empty, self._empty_metadata(0), self.model.filter_3D[:0], []
        selected = [selected[i] for i in chosen]
        xyz, h = xyz[chosen], h[chosen]
        meta = self._empty_metadata(len(xyz))
        meta["source"].fill_(3)
        meta["root_gen"].fill_(True)
        meta["root_geometry"].fill_(True)
        meta["birth"].fill_(iteration)
        meta["grace"].fill_(iteration+self.config["grace_updates"])
        meta["h_ref"] = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        meta["patch_id"] = torch.tensor([item["patch"] for _, item, _ in selected], device=self.device)
        meta["witness"] = torch.tensor([views[0] for _, _, views in selected], device=self.device)
        meta["witness2"] = torch.tensor([views[1] for _, _, views in selected], device=self.device)
        points = self.model.get_xyz.new_tensor(xyz)
        filtering = joint_filter_for_points(points, self.real_cameras, meta["root_gen"], meta["h_ref"],
            meta["witness"], self.records, points.new_full((len(points), 1), float("nan")))
        payload, valid = make_seed_payload(self.model, xyz, np.asarray([item["normal"] for _, item, _ in selected]),
                                           h, filtering, self._canonical_colors(xyz))
        keys = [key for (key, _, _), okay in zip(selected, valid.tolist()) if okay]
        return {key: value[valid] for key, value in payload.items()}, {key: value[valid] for key, value in meta.items()}, filtering[valid], keys

    @torch.no_grad()
    def _children(self, parents, source, iteration):
        n = len(parents)
        if not n:
            return {}, self._empty_metadata(0), [], torch.empty(0, dtype=torch.long, device=self.device)
        scale = self.model.get_scaling[parents]
        basis = rotations(self.model._rotation[parents])
        largest = scale.argmax(1)
        tangent = basis[torch.arange(n, device=self.device), :, largest]
        if source == 2:
            _, atlas_ids = self.atlas_tree.query(self.model.get_xyz[parents].detach().cpu().numpy())
            normals = self.model.get_xyz.new_tensor(self.teacher.atlas["normal_world"][atlas_ids])
            tangent -= (tangent*normals).sum(1, keepdim=True)*normals
            fallback = rotations(normal_quaternions(normals))[:, :, 0]
            tangent = torch.where((tangent.norm(dim=1) < .1)[:, None], fallback, tangent)
            tangent = torch.nn.functional.normalize(tangent, dim=1)
            self.meta["patch_id"][parents] = torch.as_tensor(self.teacher.atlas["patch_id"][atlas_ids], device=self.device)
        offset = .5*scale.max(1).values.clamp(max=.02*self.scale)[:, None]*tangent
        split = scale.max(1).values > self._opt("percent_dense", .01)*self.scene.cameras_extent
        if source == 2:
            split |= scale.max(1).values > self.scale/256
        # One row for clones, two for splits. Both use bounded, symmetric offsets.
        owners = torch.cat((parents, parents[split]))
        xyz = torch.cat((self.model.get_xyz[parents]+offset, self.model.get_xyz[parents[split]]-offset[split]))
        evidence = self.teacher.real_evidence(xyz.cpu().numpy())["evidence_class"]
        bad_owners = owners[torch.as_tensor(evidence == 3, device=self.device)]
        valid = ~torch.isin(owners, bad_owners)
        owners, xyz = owners[valid], xyz[valid]
        split_parents = parents[split & ~torch.isin(parents, bad_owners)]
        payload = {name: getattr(self.model, field)[owners].detach().clone() for name, field in PARAMETERS.items()}
        payload["xyz"] = xyz
        is_split = torch.isin(owners, split_parents)
        payload["scaling"][is_split] -= math.log(1.6)
        meta = {key: value[owners].clone() for key, value in self.meta.items()}
        meta["h_ref"][is_split] /= 1.6
        for key in ("real_grad", "real_abs", "real_seen", "virtual_grad", "virtual_abs", "virtual_seen", "geom_error",
                    "geom_support", "geom_weight", "real_support", "checks", "error_streak", "unsupported_streak", "last_checked"):
            meta[key].zero_()
        meta["parent_id"] = self.meta["point_id"][owners].clone()
        meta["birth"].fill_(iteration)
        meta["source"].fill_(source)
        if source == 2:
            meta["root_geometry"].fill_(True)
        return payload, meta, self.model.filter_3D[owners].clone(), split_parents

    @torch.no_grad()
    def _commit(self, iteration):
        from utils.vcd_utils import compute_vcd_vcp_scores
        n = len(self.model.get_xyz)
        finite = torch.ones(n, dtype=torch.bool, device=self.device)
        for field in PARAMETERS.values():
            finite &= torch.isfinite(getattr(self.model, field)).flatten(1).all(1)
        if not finite.all():
            if not finite.any():
                raise FloatingPointError("Every Gaussian is nonfinite; refusing an empty reconstruction")
            empty = {name: getattr(self.model, field)[:0] for name, field in PARAMETERS.items()}
            self.meta = self.model.apply_topology_transaction(torch.where(finite)[0], empty,
                self.model.filter_3D[:0], self.meta, {key: value[:0] for key, value in self.meta.items()},
                self.config["max_gaussians"])
            self.refresh_filter()
            self.counts["nonfinite_pruned"] += int((~finite).sum().item())
            self.last_event = {"iteration": iteration, "before": n, "after": len(self.model.get_xyz),
                               "nonfinite_pruned": int((~finite).sum().item()), "seeds": 0,
                               "last_seed_audit": self.last_event.get("last_seed_audit", -1)}
            return self.last_event
        visible = self.meta["real_seen"].clamp_min(1)
        grad = self.meta["real_grad"]/visible
        real = grad >= self._opt("densify_grad_threshold", .0002)
        importance = pruning = None
        if self._opt("vcd_enable", True) or self._opt("vcp_enable", True):
            ids = self.rng.choice(len(self.real_cameras), min(self._opt("vcd_num_cams", 10), len(self.real_cameras)), replace=False)
            importance, pruning = compute_vcd_vcp_scores([self.real_cameras[i] for i in ids], self.model,
                self.pipe, self.background, self.kernel_size, self._opt("vcd_loss_thresh", .1),
                need_vcd=self._opt("vcd_enable", True), need_vcp=self._opt("vcp_enable", True), frame_cache=self.frame_cache)
        if importance is not None:
            real &= importance > self._opt("vcd_importance_thresh", 5)
        geometry = ((self.meta["error_streak"] >= 2) & (self.meta["geom_weight"] > 1e-7)
                    & (self.meta["last_checked"] >= iteration-2*self.config["audit_interval"]))
        if iteration < self.config["seed_from"]:
            geometry.zero_()
        real &= ~geometry
        good = (self.meta["geom_support"] >= .25) & (self.meta["last_checked"] >= iteration-2*self.config["audit_interval"])
        good |= self.meta["real_support"] > 1e-6
        low = self.model.get_opacity.squeeze() < .05
        recent_check = self.meta["last_checked"] >= iteration-2*self.config["audit_interval"]
        checked_without_support = ((self.meta["checks"] >= 2) & (self.meta["unsupported_streak"] >= 2) & recent_check)
        ordinary_observed = (self.meta["checks"] >= 2) | (self.meta["real_seen"] >= 2)
        observed = torch.where(self.meta["root_gen"], checked_without_support, ordinary_observed)
        prune = low & observed & ~good & (self.meta["grace"] < iteration)
        reset = self.config.get("last_opacity_reset", 3000)
        recovery_end = self.config.get("start_iteration", 3501)
        if reset <= iteration < recovery_end:
            prune.zero_()
        # A hard measured-free-space conflict overrides grace for all checked suspects.
        suspects = torch.where(prune | self.meta["root_gen"])[0]
        if len(suspects) > 50000:
            suspects = suspects[torch.as_tensor(self.rng.choice(len(suspects), 50000, replace=False), device=self.device)]
        hard = torch.zeros(n, dtype=torch.bool, device=self.device)
        reviewed = torch.zeros_like(hard)
        if len(suspects):
            reviewed[suspects] = True
            evidence = self.teacher.real_evidence(self.model.get_xyz[suspects].detach().cpu().numpy())
            hard[suspects] = torch.as_tensor(evidence["evidence_class"] == 3, device=self.device)
            # Real geometric support protects thin structure independently of appearance.
            prune[suspects[torch.as_tensor(evidence["support_groups"] > 0, device=self.device)]] = False
        # Budgeted checks never turn an unreviewed point into a negative witness.
        self.counts["deferred_prune_reviews"] += int((prune & ~reviewed).sum().item())
        prune &= reviewed
        prune |= hard
        # VCP ranks only otherwise removable rows; it cannot veto valid geometry.
        prune_rank = (1-self.model.get_opacity.squeeze()).clone()
        if pruning is not None:
            prune_rank += pruning.reshape(-1).nan_to_num()
        # Limit non-conflict removal per source patch, so a batch cannot empty a surface.
        for patch in torch.unique(self.meta["patch_id"][prune & ~hard]).tolist():
            if patch < 0:
                continue
            members = self.meta["patch_id"] == patch
            ids = torch.where(prune & ~hard & members)[0]
            limit = max(1, int(members.sum().item()*self.config["patch_prune_fraction"]))
            if len(ids) > limit:
                order = prune_rank[ids].argsort(descending=True)
                prune[ids[order[limit:]]] = False
        soft_ids = torch.where(prune & ~hard)[0]
        soft_budget = max(1, int(n*.01))
        if len(soft_ids) > soft_budget:
            order = prune_rank[soft_ids].argsort(descending=True)
            prune[soft_ids[order[soft_budget:]]] = False
        real &= ~prune
        geometry &= ~prune
        real_ids = torch.where(real)[0]
        real_ids = real_ids[grad[real_ids].argsort(descending=True)]
        geometry_ids = torch.where(geometry)[0]
        geometry_ids = geometry_ids[self.meta["geom_error"][geometry_ids].argsort(descending=True)]
        available = min(self.config["max_growth"], int(n*self.config["max_growth_fraction"]),
                        self.config["max_gaussians"]-n+int(prune.sum().item()))
        available = max(0, available)
        seed_limit = min(self.config["seed_batch_limit"], int(n*self.config["seed_fraction"]),
                         self.config["seed_total_limit"]-self.seed_total, available)
        if self.last_audit <= self.last_event.get("last_seed_audit", -1):
            seed_limit = 0
        seeds, seed_meta, seed_filter, seed_keys = self._seeds(iteration, seed_limit)
        real_slots, geo_slots, seed_slots = allocate_growth(len(real_ids), len(geometry_ids), len(seeds["xyz"]), available)
        if iteration < self.config["geometry_reserve_release"] or len(geometry_ids) or len(seeds["xyz"]):
            ordinary = int((~self.meta["root_geometry"]).sum().item())
            real_slots = min(real_slots, max(0, int(self.config["max_gaussians"]*(1-self.config["geometry_reserve"]))-ordinary))
        payloads, metadata, filters, removed = [], [], [], [torch.where(prune)[0]]
        for parents, source in ((real_ids[:real_slots], 1), (geometry_ids[:geo_slots], 2)):
            payload, meta, filtering, split = self._children(parents, source, iteration)
            if payload:
                payloads.append(payload)
                metadata.append(meta)
                filters.append(filtering)
                removed.append(split)
        if seed_slots:
            payloads.append({k: v[:seed_slots] for k, v in seeds.items()})
            metadata.append({k: v[:seed_slots] for k, v in seed_meta.items()})
            filters.append(seed_filter[:seed_slots])
        additions = {name: torch.cat([p[name] for p in payloads]) if payloads else getattr(self.model, field)[:0]
                     for name, field in PARAMETERS.items()}
        new_meta = {name: torch.cat([m[name] for m in metadata]) if metadata else values[:0]
                    for name, values in self.meta.items()}
        added = len(additions["xyz"])
        new_meta["point_id"] = torch.arange(self.next_id, self.next_id+added, device=self.device)
        keep_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        keep_mask[torch.cat(removed)] = False
        keep = torch.where(keep_mask)[0]
        filtering = torch.cat(filters) if filters else self.model.filter_3D[:0]
        self.meta = self.model.apply_topology_transaction(keep, additions, filtering, self.meta, new_meta,
                                                         self.config["max_gaussians"])
        self.next_id += added
        self.seed_total += seed_slots
        for index, key in enumerate(seed_keys[:seed_slots]):
            self.queue.pop(key)
            self.seeded_cells.add(key)
            self.priority.update((int(seed_meta["witness"][index]), int(seed_meta["witness2"][index])))
        self.refresh_filter()
        for key in ("real_grad", "real_abs", "real_seen", "virtual_grad", "virtual_abs", "virtual_seen"):
            self.meta[key].zero_()
        self.last_event = {"iteration": iteration, "before": n, "after": len(self.model.get_xyz), "added": added,
                           "pruned": int(prune.sum()), "split_parents": sum(len(ids) for ids in removed[1:]),
                           "seeds": seed_slots, "last_seed_audit": self.last_audit if seed_slots else self.last_event.get("last_seed_audit", -1)}
        self.counts["transactions"] += 1
        self.counts["seeded"] += seed_slots
        self.counts["real_selected"] += real_slots
        self.counts["geometry_selected"] += geo_slots
        self.counts["budget_deferred_real"] += len(real_ids)-real_slots
        self.counts["budget_deferred_geometry"] += len(geometry_ids)-geo_slots
        self.counts["budget_deferred_seed"] += len(seeds["xyz"])-seed_slots
        return self.last_event

    @torch.no_grad()
    def refresh_filter(self):
        self.model.compute_joint_3D_filter(self.real_cameras, self.meta, self.records)

    @torch.no_grad()
    def step(self, iteration):
        self.iteration = iteration
        if iteration >= self.config["topology_until"]:
            self.pending = False
            if iteration % self.config["topology_interval"] == 0:
                self.refresh_filter()
            return None
        if iteration >= self.config.get("start_iteration", 3501)-1 and iteration % self.config["audit_interval"] == 0:
            self._audit(iteration)
        if iteration > self._opt("densify_from_iter", 500) and iteration % self.config["topology_interval"] == 0:
            self.pending = True
        if not self.pending or iteration % 16:
            if not self.pending and iteration % self.config["topology_interval"] == 0:
                self.refresh_filter()
            return None
        for optimizer in self.model._iter_optimizers():
            for group in optimizer.param_groups:
                if group["name"] in PARAMETERS and group["params"][0].grad is not None:
                    raise RuntimeError("Joint topology requires consumed main and SH gradients")
        event = self._commit(iteration)
        self.pending = False
        return event

    def state_dict(self):
        return {"version": 2, "meta": {k: v.detach().cpu() for k, v in self.meta.items()},
                "filter": self.model.filter_3D.detach().cpu(), "next_id": self.next_id, "seed_total": self.seed_total,
                "audit_cursor": self.audit_cursor, "audit_epoch": self.audit_epoch, "last_audit": self.last_audit,
                "iteration": self.iteration, "pending": self.pending, "queue": copy.deepcopy(self.queue),
                "seeded_cells": sorted(self.seeded_cells), "priority": sorted(self.priority),
                "counts": dict(self.counts), "last_event": self.last_event,
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") != 2 or set(state["meta"]) != set(self.meta):
            raise ValueError("Incompatible geometry topology checkpoint")
        if any(not isinstance(item.get("views"), dict) for item in state["queue"].values()):
            raise ValueError("Geometry checkpoint needs per-view missing epochs")
        n = len(self.model.get_xyz)
        if any(len(v) != n for v in state["meta"].values()) or state["filter"].shape != (n, 1):
            raise ValueError("Geometry checkpoint point rows do not match model")
        self.meta = {k: v.to(self.device) for k, v in state["meta"].items()}
        self.model.filter_3D = state["filter"].to(self.device)
        for key in ("next_id", "seed_total", "audit_cursor", "audit_epoch", "last_audit", "iteration", "pending"):
            setattr(self, key, state[key])
        self.queue, self.seeded_cells = copy.deepcopy(state["queue"]), set(map(tuple, state["seeded_cells"]))
        self.priority, self.counts = set(state["priority"]), Counter(state["counts"])
        self.last_event = state["last_event"]
        self.rng.bit_generator.state = state["rng"]

    def summary(self):
        return {"points": len(self.model.get_xyz), "seed_total": self.seed_total, "queued_cells": len(self.queue),
                "priority_views": sorted(self.priority), "pending": self.pending,
                "score_source": "native_W_alphaT_teacher_ray_band_v1",
                "normal_error_degrees": 15, "depth_tolerance_relative": .02,
                "depth_tolerance_scene_fraction": 1/64, "hard_conflict_review_limit": 50000,
                "counts": dict(self.counts), "last_event": self.last_event}
