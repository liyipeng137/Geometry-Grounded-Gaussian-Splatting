# Real-view geometry-only Gaussian experiment

`train_real_geometry.py` optimizes Gaussian centres, scales, rotations and opacity.
Real RGB contributes through calibrated grey-patch NCC between two photographs.
The native viewer colour is fixed DC from the TSDF point cloud. A split inherits
its parent's colour and a new stereo-verified point samples one reference pixel
at birth. Higher-order SH stays zero, and per-frame appearance is disabled.

## Reproduce the 333-camera trial

Run in the existing GGGS CUDA environment on UKD-4090. Select a new output directory.

```bash
source /usr/local/miniconda3/bin/activate py312
cd /workspace/Geometry-Grounded-Gaussian-Splatting
data=/workspace/333_data/333_for_train
run="$data/gggs/runs/train_real_geometry_7000"
python train_real_geometry.py -s "$data" -m "$run" \
  --init_ply "$data/gggs/cache/initialization/8022bbdf96d390bfa125/point_cloud.ply" \
  --init_num_points 250000 --iterations 7000 --training_seed 0 --frame_gpu_budget 4 \
  --use_decoupled_appearance 0
python render_real_geometry.py -m "$run" --output "$run/render_bundle" \
  --interpolate frame_000266.png frame_000019.png
```

## Supervision and topology

- Real camera-Z L1 and the existing external normal supervision start at step 1.
  The coverage term encourages alpha above 0.8 on valid prior pixels. At step
  401, real-photo patch NCC, geometric round-trip error and the existing
  rendered-depth/normal consistency term become active. Each step updates the
  geometry once. The starting opacities come from the native TSDF-point
  initialisation and there are no opacity resets.
- Audits start at step 800 and repeat every 200 updates. The second audit of
  each pair revisits the same real camera pairs before changing the points.
  Native per-Gaussian attribution uses opacity, transmittance and a target
  depth band on a strided image grid. Nearby real-view depths, normals and a
  minimum parallax of two degrees determine voting opportunities.
- Local point-to-plane deviations and normal changes distinguish structural
  detail from a uniformly displaced plane. Persistent in-band errors on
  sufficiently large Gaussians permit long-axis splitting. Unsupported
  foreground proposals are backprojected only when both cameras lack the
  target layer and real NCC favours its depth over two alternative depths.
  Seeds are deduplicated, checked against existing ellipsoids, sized by local
  pixel footprint, and oriented by the measured normal.
- Pruning considers repeated observed low-opacity candidates or projected
  near-side conflicts in distinct directions. Real geometric support protects
  an existing surface. Each candidate batch is temporarily suppressed and
  re-rendered on its reference cameras. A batch that loses too many previously
  valid or depth-supported hits is vetoed. Survivor optimizer state and the
  3D filter follow the point mapping. New points receive a grace interval.

## Outputs

Native PLYs are saved at initialization and after the final optimizer step.
The final standard-viewer PLY bakes the 3D filter. `geometry_stats.json` records
optimizer steps, camera visits, NCC gradient probes, colour-freeze checks and
accepted/vetoed topology changes. `geometry_loss.jsonl` records sampled raw
losses and parameter gradients.

The render bundle contains native frozen-colour RGB, directional shading from
the Gaussian normal, camera-space normals, fixed-range camera-Z and alpha. Each
frame is rendered at the source resolution before making previews. A selected
frame includes float geometry and camera calibration. `walkthrough.mp4`
follows the observed camera order. Optional `interpolated.mp4` uses 73
interpolated poses between two explicitly selected cameras. Colour and shading
are separate visualizations. Quantitative depth and normal comparisons use real
training priors and report coverage separately from common-hit errors.
