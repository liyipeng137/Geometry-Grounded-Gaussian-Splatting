"""Native Gaussian surface model. All four Adam groups describe geometry."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import torch
from torch import nn

from scene.gaussian_model import GaussianModel, _ply_elements
from utils.general_utils import get_expon_lr_func


class GeometryGaussians(GaussianModel):
    geometry_only = True
    point_parameters = {
        "xyz": "_xyz",
        "scaling": "_scaling",
        "rotation": "_rotation",
        "opacity": "_opacity",
    }
    columns = {
        "xyz": ["x", "y", "z"],
        "scaling": [f"scale_{i}" for i in range(3)],
        "rotation": [f"rot_{i}" for i in range(4)],
        "opacity": ["opacity"],
    }

    def __init__(self):
        # Share geometric projection, 3D filtering and tetrahedron sampling with GGGS.
        # The appearance-bearing parent constructor is deliberately not used.
        self.setup_functions()
        self.optimizer = None

    def create_from_pcd(self, pcd, spatial_lr_scale):
        xyz = torch.as_tensor(
            np.asarray(pcd.points if hasattr(pcd, "points") else pcd._xyz),
            dtype=torch.float32,
            device="cuda",
        ).contiguous()
        self.spatial_lr_scale = spatial_lr_scale
        self._xyz = nn.Parameter(xyz)
        distance = distCUDA2(xyz.detach()).clamp_min(1e-7).sqrt()
        self._scaling = nn.Parameter(distance.log()[:, None].repeat(1, 3))
        q = xyz.new_zeros((len(xyz), 4))
        q[:, 0] = 1
        self._rotation = nn.Parameter(q)
        self._opacity = nn.Parameter(torch.logit(xyz.new_full((len(xyz), 1), 0.1)))
        self.reset_3D_filter()

    def training_setup(self, opt):
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt.position_lr_init * self.spatial_lr_scale,
            lr_final=opt.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.position_lr_max_steps,
        )
        rates = {
            "xyz": opt.position_lr_init * self.spatial_lr_scale,
            "scaling": opt.scaling_lr,
            "rotation": opt.rotation_lr,
            "opacity": opt.opacity_lr,
        }
        self.optimizer = torch.optim.Adam(
            [
                {
                    "name": key,
                    "params": [getattr(self, field)],
                    "lr": rates[key],
                    "base_lr": rates[key],
                }
                for key, field in self.point_parameters.items()
            ],
            lr=0.0,
            eps=1e-15,
        )

    def set_rates(self, step, multiplier):
        for group in self.optimizer.param_groups:
            base = (
                self.xyz_scheduler_args(step)
                if group["name"] == "xyz"
                else group["base_lr"]
            )
            group["lr"] = base * multiplier

    def _iter_optimizers(self):
        yield self.optimizer

    def save_ply(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names, arrays = [], []
        for key, field in self.point_parameters.items():
            names.extend(self.columns[key])
            arrays.append(getattr(self, field).detach().cpu().numpy())
        names.append("filter_3D")
        arrays.append(self.filter_3D.detach().cpu().numpy())
        vertices = _ply_elements(arrays, [(key, "f4") for key in names])
        tmp = path.with_suffix(".ply.tmp")
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            comments=["GGGS geometry v1: log scales, wxyz quaternion, opacity logit"],
        ).write(str(tmp))
        tmp.replace(path)

    def load_ply(self, path):
        v = PlyData.read(str(path), mmap="r")["vertex"].data
        for key, field in self.point_parameters.items():
            values = np.column_stack([v[k] for k in self.columns[key]])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {key} in geometry PLY")
            setattr(
                self,
                field,
                nn.Parameter(torch.tensor(values, dtype=torch.float32, device="cuda")),
            )
        filt = np.asarray(v["filter_3D"])
        if not np.isfinite(filt).all() or (filt < 0).any():
            raise ValueError("Invalid native 3D filter")
        self.filter_3D = torch.tensor(filt.copy(), dtype=torch.float32, device="cuda")[
            :, None
        ]

    @torch.no_grad()
    def export_neutral_viewer(self, path):
        """A constant neutral DC is an export adapter for conventional SH0 viewers."""
        xyz = self.get_xyz.detach().cpu().numpy()
        zero = np.zeros_like(xyz)
        scales, alpha = self.get_scaling_n_opacity_with_3D_filter
        arrays = [
            xyz,
            zero,
            zero,
            torch.logit(alpha.clamp(1e-6, 1 - 1e-6)).cpu().numpy(),
            scales.clamp_min(1e-12).log().cpu().numpy(),
            self._rotation.cpu().numpy(),
        ]
        names = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            *self.columns["scaling"],
            *self.columns["rotation"],
        ]
        vertices = _ply_elements(arrays, [(k, "f4") for k in names])
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            comments=["Neutral SH0 viewer export; native model has no appearance"],
        ).write(str(path))
