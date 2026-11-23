# Grayscale, appearance-free surface training

This recipe uses one observed luminance channel and trains position, log scale,
quaternion and opacity. The native point PLY has 12 float32 fields: the eleven
parameter values plus the derived 3D filter. The CUDA specialization renders
camera-Z D50, oriented normal and alpha with zero colour channels. Reference
photos are decoded and resized on CPU, masked in the original convention, then
converted once to float32 luminance with weights 0.299, 0.587 and 0.114. Only
the single luminance plane is kept in the training frame cache. Original RGB
photos and the supplied coloured TSDF input PLY stay available separately.

## Run on UKD-4090

```bash
source /usr/local/miniconda3/bin/activate py312
cd /workspace/Geometry-Grounded-Gaussian-Splatting
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=5
data=/workspace/333_data/333_for_train
build="$data/gggs/experiments/gray_geometry_20000"
bash scripts/build_geometry_rasterizer.sh "$build"
export PYTHONPATH="$build/lib:${PYTHONPATH:-}"
run="$data/gggs/runs/train_gray_geometry_20000"
python train_real_geometry.py -s "$data" -m "$run" --iterations 20000 \
  --init_ply "$data/gggs/cache/initialization/8022bbdf96d390bfa125/point_cloud.ply" \
  --init_num_points 250000 --training_seed 0 --frame_gpu_budget 4
python render_gray_geometry.py -m "$run" --output "$run/render_bundle" \
  --steps 7000 15000 20000 --interpolate frame_000266.png frame_000019.png
```

The trainer requires a new run directory. The standard photographic training
entry point still accepts its original RGB/SH models. A historical 7k
frozen-colour run can be viewed with `render_real_geometry.py`. The mini
extractor now reads just the geometry columns from either native PLY schema.
Use separate model directories for comparing multiple mini extractions since
the extractor uses fixed mesh filenames.

## 20,000 updates

| Updates | Matching and topology |
|---|---|
| 1–400 | TSDF initialization, absolute camera-Z L1, normal prior and alpha coverage. |
| 401–7,000 | Real-camera patch NCC 0.10, round-trip geometry and depth/normal consistency. Paired topology audits from 800 through 6,600, followed by a fixed-point interval. |
| 7,001–15,000 | NCC ramps from 0.10 to 0.15 by 9,000, then holds. Geometry audits resume at 7,200, using the same split, stereo seed and support-aware prune policy. |
| 15,001–20,000 | Fixed point set. NCC tapers from 0.15 to 0.06, and the geometry learning rates taper from 1.0 to 0.25 times the otherwise scheduled rates. |

Depth L1 weight is 0.4, coverage weight 0.05 on valid prior pixels (target
alpha 0.8), cross-view geometry 0.02 and internal depth/normal consistency
0.05 when matching is enabled. The existing external-normal schedule uses a
0.1 base. There is no opacity reset. The default 20k schedule saves native
states at 0, 7k, 15k and 20k. The optional display adapter stored next to the
final native PLY adds a constant neutral DC and bakes the filter for a generic
SH0 viewer. The shaded previews use the native rendered normal, not appearance
estimated from photographs.
