# 245-frame virtual camera review, 2026-11-10

First-stage selection completed on `UKD-4090`; no GGGS training was started.

- Dataset: `/workspace/dataset/245_for_train`.
- Immutable teacher: `genrecon_mesh.ply`, 15,281,042 vertices / 33,020,336 faces.
- Output: `/workspace/outputs/gggs_virtual_cameras_245_20261110/attempt_01`.
- 245 real cameras; 1,200 proposals; 1,128 passed safety/coverage checks.
- 160 selected virtual cameras: 77 trajectory-neighborhood, 83 surface-oriented.
- 72 proposals rejected because their checked path was not entirely in observed
  free space. Conditions were not loosened to fill the requested count.
- 160 unique full poses; 152 distinct virtual camera centers. 56 views reuse a
  real camera center with a different rotation; 104 change the center.
- All selected rotations are orthonormal (maximum error `1.55e-15`), with no
  full pose equal to an existing real pose.
- Selected minimum mesh clearance: `0.25827` scene units versus required
  `0.02379`. Selected near-clip pixel fraction is zero.
- Median usable teacher pixel fraction: 96.64%; minimum: 58.43%.
- Area-sample coverage of non-conflicting mesh surface: 46.835%; two-direction
  coverage: 34.807%. The denominator includes surfaces that cannot be reached
  from the admitted camera space; these are not whole-scene completeness claims.
- Selection and all 405-camera image exports: 144.34 seconds, excluding HTML
  export and local transfer.

Every real and selected virtual camera has 384×512 original-mesh shading,
normal and depth PNGs. The interactive scene uses 40,000 surface samples for
responsiveness, not a simplified teacher for selection.

## Validation

30 geometry/selection tests are included for the remote `py312` environment, including
Open3D triangle-mesh camera-Z, normal and occlusion fixtures. Local validation
passed 28 tests with only those two Open3D cases skipped. Ruff and Python syntax
checks passed.

Independent pose audit found no duplicate poses, no selected failed path and
no selected near-clip issue. The local copy matched all 1,225 selector receipt
hashes. The first real camera's gray render was visually checked against its
original photo for coordinate/frame alignment.

The browser preview supports family toggles, unknown filtering, top/side views,
exact-mesh image switching, and original-center-versus-translated view badges.
These interactions were checked in a real browser; the final page had no console
errors. Plotly emitted only a Canvas2D readback performance warning.
Review `preview.html` and its contact sheets before proceeding with training
cache generation or joint loss/topology implementation.

See [selection policy and commands](virtual_camera_selection.md).
