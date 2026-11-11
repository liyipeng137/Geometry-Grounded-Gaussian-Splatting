#!/usr/bin/env python3
"""Select and preview virtual cameras without importing CUDA training modules."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_camera_selection import run_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=160)
    parser.add_argument("--candidate-limit", type=int, default=1200)
    parser.add_argument("--trajectory-offset-fraction", type=float, default=.08,
                        help="Lateral/vertical trajectory offset fraction; does not change surface/interpolation poses")
    parser.add_argument("--long-edge", type=int, default=256)
    parser.add_argument("--preview-long-edge", type=int, default=512)
    parser.add_argument("--surface-samples", type=int, default=120000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_selection(args.dataset, args.mesh, args.output, args.target_count, args.candidate_limit,
                  args.long_edge, args.surface_samples, args.threads, args.seed, args.preview_long_edge,
                  trajectory_offset_fraction=args.trajectory_offset_fraction)


if __name__ == "__main__":
    main()
