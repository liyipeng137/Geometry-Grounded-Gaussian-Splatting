#!/usr/bin/env python3
"""One delivery entry: cameras, geometry teacher, initialization and GGGS training."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Matching checkpoint; use a new output directory"
    )
    args = parser.parse_args()
    # Read the thread budget before importing NumPy/Open3D/Torch.
    import yaml

    base = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    user = yaml.safe_load(args.config.read_text()) if args.config else {}
    if not isinstance(user, dict) or not isinstance(user.get("runtime", {}), dict):
        parser.error("config and runtime must be mappings")
    threads = user.get("runtime", {}).get("cpu_threads", base["runtime"]["cpu_threads"])
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        parser.error("runtime.cpu_threads must be a positive integer")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from utils.delivery_pipeline import run

    run(args.dataset, args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
