# Direct LK panorama sweeps

The rigid and depth-aware methods share direct photometric alignment. The depth-aware extension permits local image regions to respond differently to camera translation, while estimating one shared translation and in-plane roll per observation.

## Rigid method

```text
select source frames at the requested interval
normalize the scan direction
for each adjacent pair:
    run centered pyramidal LK for dx, dy and roll
accumulate rigid transforms to the middle selected frame
warp images and masks into a common canvas
cache seam costs that do not depend on the requested view
for each view:
    choose coordinated vertical seams by the original DP objective
    compose strips with the original feathering
```

Caching changes when invariant quantities are calculated, retaining the seam objective and composition. `lk-original` exposes this route through the CLI.

## Depth-aware method

For a point `p` in reference image `j`, predict its location in image `i` as:

```text
p_i = R(roll_i - roll_j) (p - center) + center
      + local_slope_j(p) * (translation_i - translation_j)
```

The scalar local slope describes relative inverse depth under lateral motion. Its scale is fixed by the initial rigid LK translation. The translation coordinates and slopes are not a metric 3D reconstruction. This simplified relation assumes a predominantly lateral scan, limited roll, and no large changes of camera intrinsics or forward motion.

```text
normalize the scan direction
initialize the first motion with centered pyramidal rigid LK
use grayscale images at half output resolution for estimation
for each candidate image:
    predict its camera position from the preceding displacement
    align its intensities against up to three preceding observations
        use their local slopes to predict pixel locations
        solve robust weighted Gauss–Newton normal equations
        accept an update only when the fixed-sample objective decreases
    if the estimated displacement is too small or reverses:
        retain the preceding observation and inspect the next candidate
        stop if twelve candidates fail to provide forward support
    estimate local slopes using up to five preceding images
    refresh the preceding image's slopes using the new observation
interpolate reliable slope samples into maps for rendering
for each requested view:
    select the corresponding slit through camera-position space
    compensate local displacement in adjacent source observations
    interpolate supported colors
    use nearby supported observations of the same model ray at borders
    leave uncovered output samples empty
```

Intensity patches are 3×3 samples on a six-pixel estimation grid. The motion solver uses at most ten iterations, shape estimation eight, and the preceding-shape refresh four. Photometric robustness downweights large residuals; low-gradient or poorly fitted shape samples receive low confidence. Relative slope bounds are 0.2–3.0. These limits bound the implementation, and can also limit the depth range it represents correctly.

## Output and configuration

`--method auto` is the CLI default. It identifies reviewed example inputs by SHA256 and restores their [selected configurations](selected-results.md): depth LK for Boat, House, Garden and Kessaria; original LK for the other five. These are manually reviewed presets, not automatic visual-quality estimates. Unrecognized input bytes use original LK with maximum source dimension 640 and stride 10. Renaming a file does not change selection; re-encoding changes its hash and uses the unknown-input policy.

Known auto examples use maximum source dimension 800. Original presets retain exact archived frame-index lists; depth presets use stride 3. Presets restore explicit active-area crops, including House's encoded side bars. Explicit `--method lk-original`, `--method lk-depth` and `--method rectified` bypass presets and use maximum dimension 640 and strides 10, 3 and 1 respectively. `--max-side`, `--frame-stride` and `--active-area` override corresponding defaults; an explicit stride replaces an archived frame-index list.

`--max-side` is applied before the declared crop. Direction normalization restores vertical scans to their input orientation after rendering.

`--video --views 101 --fps 15` streams a 101-view H.264 file through FFmpeg. Rendering reuses one registration pass. The output width is fixed for the sweep; it is not independently stretched to fit each view. Alphas 0 and 1 can have weaker coverage and larger distortions. There is no interpolation between completed output views.

`run.json` records the requested and resolved method, preset, selection reason, applied defaults, source indices, skipped candidates, direction, configuration, registration diagnostics, normalized canvas placement, per-view coverage, and preparation/render/encoding times. Timing excludes evidence and image writes. `geometry.npz` contains the estimated camera positions and local slope maps when retained. Depth interpolation can combine observations, so this mode does not claim that every pixel has one owning source image.

The Python API prepares registration once and renders views on demand:

```python
from video_mosaicing.io import iter_frames
from video_mosaicing.lk_pipeline import DirectConfig, prepare_direct

scan = prepare_direct(iter_frames("scan.mp4", max_side=800, stride=3),
                      DirectConfig(method="lk-depth"))
image, coverage = scan.render(0.5)
```

The separate `video_mosaicing.pipeline` API retains the feature-based rectified method for existing callers.

## Limits

Local slopes reduce the restriction of a single rigid warp, but do not resolve every occlusion or moving object. Foliage, water, repeated textures and exposure changes can violate the intensity and locally constant-motion assumptions. The slope interpolation can smear depth boundaries. Dense sampling does not by itself establish better quality, and exact border views may require more distant source observations. Camera reversal beyond the bounded selection window remains an explicit error.

This implementation adapts sections 3–5 of Rav-Acha, Shor and Peleg, *Mosaicing with Parallax using Time Warping*, CVPR Workshops 2004. It supports lateral/vertical displacement and in-plane roll; full planar and spherical camera-motion models are outside this implementation.
