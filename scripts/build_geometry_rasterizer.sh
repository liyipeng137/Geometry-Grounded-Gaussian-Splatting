#!/usr/bin/env bash
# Build under a private prefix; use its lib directory first on PYTHONPATH.
set -euo pipefail
if [[ $# != 1 ]]; then echo "usage: bash scripts/build_geometry_rasterizer.sh BUILD_DIRECTORY" >&2; exit 2; fi
mkdir -p "$1"
build_dir=$(cd "$1" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo/submodules/diff-gaussian-rasterization"
mkdir -p "$build_dir/lib/diff_gaussian_rasterization"
python setup.py build_ext --build-lib "$build_dir/lib" --build-temp "$build_dir/obj"
cp diff_gaussian_rasterization/__init__.py "$build_dir/lib/diff_gaussian_rasterization/"
cd "$build_dir"
PYTHONPATH="$build_dir/lib:${PYTHONPATH:-}" python -c 'import torch, diff_gaussian_rasterization as r; assert r._C.geometry_only_channels; print(r._C.__file__)'
