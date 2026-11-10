# Virtual camera selection and pose review

This is the first stage of GenRecon geometry guidance. It selects virtual
cameras and exports review assets. It does not start GGGS training, modify the
input mesh, or produce a training D/N cache.

The implementation is based on `codex/gggs-baseline-cleanup-20261109` at
`8af420e82b1a43e3b8da6c124a288dd86c7ac2e5`.

## Run on UKD-4090

The existing `py312` environment has NumPy, SciPy, Pillow, plyfile, Open3D 0.19
and Plotly. Plotly is only needed for the interactive HTML export.

```bash
cd /workspace/Geometry-Grounded-Gaussian-Splatting
source /usr/local/miniconda3/bin/activate py312

python scripts/select_virtual_cameras.py \
  --dataset /workspace/dataset/245_for_train \
  --mesh /workspace/dataset/245_for_train/genrecon_mesh.ply \
  --output /workspace/outputs/gggs_virtual_cameras_245_20261110/attempt_01 \
  --target-count 160 --candidate-limit 1200 --surface-samples 120000 \
  --long-edge 256 --preview-long-edge 512 --threads 8 --seed 0

python scripts/export_virtual_camera_preview.py \
  --manifest /workspace/outputs/gggs_virtual_cameras_245_20261110/attempt_01/manifest.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_virtual_camera_selection.py
```

Use a new output directory for each attempt. The selector refuses a nonempty
directory so that a changed selection cannot silently overwrite an earlier one.
The exporter can regenerate the derived HTML/contact sheets beside a completed
manifest. The full directory is portable: HTML contains Plotly itself; PNGs and
JSONs remain relative files. Open `preview.html` or serve that directory with a
local HTTP server. Do not move only the HTML away from its sibling artifacts.

## Selection policy

1. Read the complete registered COLMAP pose/K set and corresponding camera-Z
   depth priors. Preserve COLMAP half-integer pixel centers and world axes.
2. Use all original mesh triangles for area-weighted surface sampling and exact
   ray intersection. No mesh simplification, patch translation, or per-view
   depth alignment enters selection.
3. Aggregate surface evidence across camera position groups. A surface behind
   a real first hit is occluded/unknown, not a free-space conflict. Unknown
   surfaces remain eligible. Multiple independent positions are needed for a
   hard conflict; mixed evidence is shown separately.
4. Propose local offsets and trajectory interpolations, plus cameras looking
   toward sampled surface regions. Roll follows a nearby real camera rather
   than a hardcoded world-up axis.
5. Check anchor-to-candidate paths directly against real depth free space,
   original-mesh clearance, and exact segment intersections. These checks
   concern the camera path, not whether every target ray is already observed.
6. Score actual first-hit coverage and viewing directions. Pixel hits are
   associated with surface samples, then candidate samples receive a second
   exact first-hit check to exclude nearby occluded layers.
7. Greedily select views with new surface/direction coverage and diminishing
   returns for repeated observations. `--target-count` is an upper target,
   not a reason to weaken checks when fewer useful views survive.

The current surface-region representation is a spatial/unsigned-normal proxy
over area samples, not connected-component segmentation of every mesh face.
Coverage numbers are area-sample estimates, not exact percentages of all mesh
triangles. Free-space tests use depth priors, so they are evidence checks rather
than a guarantee about physical clearance. Every candidate keeps its measured
checks and rejection reason for review.

## Review assets

- `preview.html`: all real and virtual centers, exact-K frusta, group toggles,
  checked anchor links, camera selection, and original-mesh PNG inspection.
- `contact_sheet*.jpg`: all selected virtual cameras, family-specific sheets,
  and all real cameras.
- `real_cameras.json`, `virtual_cameras.json`: complete c2w/K records. Pose and
  camera intrinsics use the scoring resolution; `preview.width/height` separately
  records the review PNG resolution with an equivalent field of view.
- `candidates.json`, `candidate_audit.jsonl`: all proposals and rejection reasons.
- `report.json`: candidate counts, evidence categories, estimated coverage,
  runtime and explicit approximation boundaries.
- `sample_atlas.npz`, `coverage.npz`: sampled surface provenance and exposure.
- `scene_preview.npz`: sampled scene context colored by evidence, not texture.
- `source_fingerprints.json`, `receipt.json`, `state.json`: input/code hashes and
  selector-produced artifact integrity. HTML and contact sheets are subsequent
  derived exports and are not claimed to be in the selector receipt.

The interactive scene uses surface points for responsiveness. The per-camera
shading/normal/depth PNGs use the complete original mesh. Camera ordering is
not a travel path; optional connection lines show only checked anchor links.

For pose approval, inspect overall coverage, unusual camera directions, low
clearance paths and the per-camera exact-mesh images. Final training integration
is intentionally deferred until this selection is reviewed.
