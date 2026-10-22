import argparse
import os

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import readColmapCameras
from utils.pd_utils import generate_ply_from_rgbd


def load_colmap(source, images):
    sparse = os.path.join(source, "sparse", "0")
    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
        intrinsics = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    cameras = readColmapCameras(extrinsics, intrinsics, os.path.join(source, images))
    return cameras, intrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--images", default="images")
    parser.add_argument("--num_points", type=int, default=200000)
    parser.add_argument("--voxel_length", type=float, default=0.02)
    parser.add_argument("--sdf_trunc", type=float, default=0.06)
    parser.add_argument("--depth_min", type=float, default=0.05)
    parser.add_argument("--depth_max", type=float, default=50.0)
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = args.output or os.path.join(source, "tsdf_test.ply")
    cameras, intrinsics = load_colmap(source, args.images)
    print(f"cameras: {len(cameras)}, sparse_cameras: {len(intrinsics)}")
    generate_ply_from_rgbd(
        train_cam_infos=cameras,
        num_points=args.num_points,
        ply_path=output,
        cam_intrinsics=intrinsics,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )


if __name__ == "__main__":
    main()
