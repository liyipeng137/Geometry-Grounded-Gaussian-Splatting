#!/usr/bin/env python3
"""从已认可的虚拟机位生成冻结的 D/N teacher；不选择新机位、不训练。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_geometry_data import prepare_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    prepare_teacher(args.approved_manifest, args.output, args.dataset, args.long_edge, args.threads)


if __name__ == "__main__":
    main()
