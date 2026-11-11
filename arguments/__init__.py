#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t is bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t is bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.sg_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._dataset = ""
        self._resolution = -1
        self.max_image_width = 0
        self._white_background = False
        self.data_device = "cuda"
        self.frame_gpu_budget = -1
        self.eval = False
        self.use_decoupled_appearance = 3  # 0: NO, 1: GS, 2: GOF, 3: PGSR, 4: PPISP
        self.disable_filter3D = False
        self.kernel_size = 0.0  # Size of 2D filter in mip-splatting

        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5
        self.mask_dir = "masks"
        self.mask_format = "png"
        self.normal_prior_dir = "normals"
        self.normal_prior_format = "png"
        self.depth_prior_dir = "depth"
        self.depth_prior_format = "png"
        self.depth_prior_scale = 1000.0
        self.depth_confidence_dir = "confidence"
        self.depth_confidence_format = "png"
        self.train_exclude_images = ""
        self.init_ply = ""  # Optional XYZ/RGB initialization; training/filter policy is unchanged.
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.init_ply = os.path.abspath(os.path.expanduser(g.init_ply)) if getattr(g, "init_ply", "") else ""
        g.max_image_width = getattr(g, "max_image_width", 0) or 0
        if g.max_image_width < 0:
            raise ValueError("max_image_width must be nonnegative")
        g.train_exclude_images = (
            os.path.abspath(g.train_exclude_images)
            if getattr(g, "train_exclude_images", "")
            else ""
        )
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.dense_optimizer_updates = False
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # self.feature_dc_lr = 0.0025
        self.feature_dc_lr = 0.0013
        self.feature_rest_lr = 0.00011
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.sg_axis_lr = 0.002
        self.sg_sharpness_lr = 0.095
        self.sg_color = 0.00064
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.pgsr_appearance_lr = 0.001
        self.gs_appearance_lr_init = 0.01
        self.gs_appearance_lr_final = 0.001
        self.gs_appearance_lr_delay_steps = 0
        self.gs_appearance_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth_normal = 0.05
        self.lambda_mask = 0.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.opacity_reset_min_remaining = 0
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000

        self.regularization_from_iter = 7000  # if scene case change to 3000
        self.densify_grad_threshold = 0.0002

        self.lambda_multi_view_geo = 0.02
        # train_demo NCC 日程的起始权重：.2 → .1 → .05。
        self.lambda_multi_view_ncc = 0.2

        # train_demo 的 normal prior 日程：初始/尾段基准 .1，中段最高 2 倍。
        self.lambda_normal_prior = 0.1
        self.normal_prior_from_iter = 1
        self.lambda_depth_prior = 0.4
        self.real_depth_loss_mode = "absolute"
        self.real_depth_relative_epsilon = 0.01
        self.real_depth_relative_floor = 0.05
        self.depth_prior_from_iter = 3000
        self.depth_prior_ramp_iters = 0
        self.vcd_enable = True
        self.vcd_from_iter = 500
        self.vcd_num_cams = 10  # if scene case change to 15
        self.vcd_loss_thresh = 0.1
        self.vcd_importance_thresh = 5
        self.vcp_enable = True
        self.vcp_from_iter = 500
        self.vcp_remove_ratio = 0.9
        self.multi_view_patch_size = 3
        self.multi_view_pixel_noise_th = 1.0

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
