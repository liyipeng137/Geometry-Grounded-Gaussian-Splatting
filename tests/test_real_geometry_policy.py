from collections import Counter
from types import SimpleNamespace

import torch

from utils.real_geometry_topology import (
    RealGeometryTopology,
    camera_points,
    project,
    structural_variation,
)


def test_off_center_camera_projection_round_trip():
    r = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cam = SimpleNamespace(
        Cx=19.7,
        Cy=15.3,
        Fx=35.0,
        Fy=38.0,
        image_width=40,
        image_height=32,
        R=r,
        T=torch.tensor([0.7, -0.6, 0.1]),
    )
    uv = torch.tensor([[3, 12], [35, 2]])
    depth = torch.tensor([2.1, 0.8])
    world = (camera_points(cam, uv, depth) - cam.T) @ r.T
    x, y, z, inside = project(cam, world)
    assert inside.all()
    torch.testing.assert_close(torch.stack((x, y), 1), uv)
    torch.testing.assert_close(z, depth)


def test_slanted_plane_is_not_a_depth_break():
    y, x = torch.meshgrid(torch.arange(9.0), torch.arange(11.0), indexing="ij")
    z = 3 + 0.2 * x - 0.3 * y
    points = torch.stack((x, y, z), -1)
    normals = torch.nn.functional.normalize(
        torch.tensor([0.2, -0.3, -1.0]), dim=0
    ).expand_as(points)
    valid = torch.ones_like(x, dtype=torch.bool)
    tau = torch.full_like(x, 0.05)
    varied, edge = structural_variation(points, normals, valid, tau)
    assert not varied.any() and not edge.any()
    points[:, 6:, 2] += 1
    varied, edge = structural_variation(points, normals, valid, tau)
    assert edge[:, 5:7].all() and not edge[:, :5].any() and not edge[:, 7:].any()
    assert torch.equal(varied, edge)


def test_prune_probe_vetoes_new_holes_and_restores_logits():
    topo = object.__new__(RealGeometryTopology)
    topo.model = SimpleNamespace(_opacity=torch.tensor([[0.3], [0.7]]))
    topo.cams = [SimpleNamespace(depth_prior=torch.ones(1, 32, 32))]
    topo.cache = SimpleNamespace(ensure=lambda cam: cam)
    topo.counts = Counter()

    def draw(cam):
        present = bool(topo.model._opacity[0, 0] > -20)
        return {
            "median_depth": torch.ones_like(cam.depth_prior),
            "depth_status": torch.full_like(cam.depth_prior, 3 if present else 0),
        }

    topo.render = draw
    before = topo.model._opacity.clone()
    assert len(topo.safe_removals(torch.tensor([0]), [0])) == 0
    assert topo.counts["vetoed_removals"] == 1
    torch.testing.assert_close(topo.model._opacity, before)
    assert topo.safe_removals(torch.tensor([1]), [0]).tolist() == [1]
    torch.testing.assert_close(topo.model._opacity, before)
