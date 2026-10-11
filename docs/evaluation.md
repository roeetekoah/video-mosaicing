# Evaluation status

11 October 2026. First integrated functionality revision; further parallax work remains.

The automated suite covers calibrated pinhole motion, projective boundary correspondence, nonfolding geometry, exact translation reduction, all four scan orientations, source-coordinate provenance, degenerate input rejection, gain/bias correction, direct-solver Jacobians, and the original LK regression cases. The ray/plane scene generator is independent of the strip renderer. Tests use fixed development seeds; broader held-out capture evaluation is still required.

## Local real-video comparisons

Four supplied course clips were used locally at maximum decoded side 640. They are development data, not cleared public demo assets. Kessaria and Iguazu contain encoded side bars; both methods use the same explicitly declared active bounds `(0.125, 0, 0.875, 1)`, giving 480×360 scene images without rescaling again. Boat and Trees use their full images. The baseline uses uniform sampling of 10% of frames, clamped to 40–120, four pyramid levels, 30 LK iterations, border 8, blur 1.5, margin 50, the original seam DP, and feather half-width 40. The revised pipeline decodes every frame, selects useful displacement, and renders hard rectified strips.

The table compares five distributed frame pairs at the **same legacy spacing**, using the same reserved mutually consistent SIFT correspondences for both estimators. The entries are medians of pair medians, in pixels. They measure correspondence agreement, not ground-truth motion or panorama quality.

| Clip | Rigid LK baseline (px) | Robust selected model (px) | Lower median error | Accepted pairs | Complete revised scan |
| --- | ---: | ---: | ---: | ---: | --- |
| Kessaria | 1.74 | 0.63 | 63.8% | 5/5 | Yes; five views |
| Iguazu | 0.88 | 0.36 | 59.3% | 5/5 | Yes; five views |
| Trees | 0.59 | 0.58 | 2.3% | 5/5 | Yes; visible parallax defects |
| Boat | 36.26 | 0.30 | Not an end-to-end gain | 4/5 | No; unsupported strip geometry |

The [machine-readable comparison](development-comparison.json) retains per-pair errors, source indices, accepted/rejected status, and both medians. Percentage reduction is `100 * (1 - revised / baseline)` on the comparable pairs. Boat is deliberately excluded from percentage claims because its failed pair and failed full scan would make such a claim misleading.

Boat's medians cover only its four commonly scored pairs; the fifth was rejected and remains in the benchmark record. Its complete scan also fails the strip-support gate. Good scores on the accepted pairs do not turn that failure into a successful panorama.

The residual tails give a useful additional check. Below is the **median of each pair's 90th-percentile error**, using the same comparable pairs. It is not the pooled 90th percentile across the video.

| Clip | Baseline pair-p90 median (px) | Revised pair-p90 median (px) |
| --- | ---: | ---: |
| Kessaria | 3.14 | 2.00 |
| Iguazu | 1.55 | 1.08 |
| Trees | 1.48 | 1.52 |
| Boat, accepted pairs only | 37.19 | 35.34 |

In particular, Boat's low revised median hides a large group of badly aligned correspondences. Trees also has no improvement in the residual tail. These are reasons to retain the failure and parallax limitations despite the stronger central scores.

Kessaria and Iguazu produce straighter, more coherent full panoramas than the reproduced rigid baseline, and complete five-view sweeps at anchors 0.1, 0.3, 0.5, 0.7, and 0.9. Kessaria retains its arch and wall structure; Iguazu avoids the pronounced curling seen in the baseline. The comparable registration medians improve by about 64% and 59%, respectively. Exposure bands, temporal changes in water, and local alignment defects remain visible. These are useful first results, not a claim of artifact-free reconstruction.

The active-area comparison matters. Leaving the encoded bars in Kessaria inflated the baseline median to 37.32 pixels, against 0.58 for the robust estimator. Excluding the same padding in both methods greatly improves the baseline too. The 1.74-versus-0.63 comparison above isolates the remaining registration benefit more fairly. Padding removal and the new estimator should not be credited as one undifferentiated gain.

Holding frames and geometry fixed, hard strips and two-pixel feathering complete on both active-area scans. Robust gain/bias correction completes on Kessaria. On Iguazu its accumulated correction exceeds the admissible range and is rejected; the unstable lighting/water fit is retained as a failure. Exposure correction remains disabled by default.

Trees remains close to its baseline registration accuracy. Its panorama completes, but depth-dependent contour discontinuities remain. Boat contains mixed motions for which the accepted plane approximation can reverse the strip interval at some rows. The pipeline reports the unsupported pair rather than emitting a folded result.

## What these results establish

### Garden: current renderer versus the historical showcase

Garden was rerun from the original 488-frame video with the current implementation at source revision `2dbad10a35fc5a1c32616611a6187acd2c95c831`. Processing uses maximum side 800 (800×450 frames), every decoded frame, 6-pixel motion spacing, automatic model selection, seed 47, full input bounds, exposure correction off, and blending off. It selects 260 frames and completes all 21 anchors from 0.1 to 0.9. The middle panorama is 2909×526 with 85.3% observed coverage; black areas remain visible. The H.264 sweep contains 40 independently rendered-view frames at 8 fps, traversing the 21 views forward and back without duplicating endpoints.

For the registration comparison, five distributed adjacent pairs are drawn from the historical **80 uniformly sampled frames**, at the same 800-pixel processing resolution for both estimators. The original four-level, 30-iteration rigid LK solver uses border 8 and blur 1.5. All five pairs pass the robust model checks. Median pair-median correspondence error falls from **1.7124 to 0.5095 pixels (70.2% lower)**. These reserved feature correspondences also inform model selection; this is a development proxy, not independent ground truth. Garden's 800-pixel figures should not be pooled directly with the 640-pixel table above.

The [full uncropped visual comparison](../assets/garden/comparison.png) shows the historical saved LK output and the current renderer output at the same pixel scale. The revised result has less accumulated tilt and a more level building façade, while retaining visible breaks around nearby railings. The two methods' alpha values have different meanings, and the original uses 80 frames with feathering while the current run uses 260 with hard strips. This is an end-to-end comparison, not an ablation attributing every change to registration alone.

The [run record](garden-run.json) retains input and output hashes, source revision, configuration, selected frame indices and model counts. Verification decoded every exported video frame and reproduced 2,000 sampled panorama pixels exactly from their recorded source coordinates. [Per-pair errors](garden-registration.json) are also retained. The historical panorama is preserved byte-for-byte; its view is 51/100, at alpha 50/99.

Reproduce the current showcase from a local copy of Garden.mp4:

```sh
python experiments/export_sweep_video.py Garden.mp4 --max-side 800 --steps 21 --fps 8 --output garden-current
python experiments/registration_benchmark.py Garden.mp4 --max-side 800 --samples 80 --pairs 5 --output garden-registration.json
```

No crop, horizon leveling, inpainting, exposure adjustment, or feathering is applied to the showcased current panorama.

This is a usable portfolio release for continuous, textured scans with manageable depth variation. It has a reproducible generated demo, explicit failure behavior, source-pixel provenance, and 53 passing automated checks, including independent calibrated geometry fixtures and the retained LK regression cases. The original coursework received 100/100; that grade is academic context, not a quality score for this extension.

It is not yet a general-purpose solution for arbitrary handheld footage. Five development pairs per clip give a small, selected view of registration accuracy. The validation points also influence model selection. There is no untouched real-video test set or scalar ground-truth score for complete panoramas, and three successful development scans out of four are not an estimated deployment success rate.

The next substantive improvement should address parallax in Boat and Trees: compare denser observed strips with bounded flow-assisted correspondence and two-dimensional seams. Accept it only if an independently generated depth/occlusion set and new real captures show fewer contour breaks without more holes, local stretch, or hidden failures. Cosmetic blending alone would not resolve that limitation.

## Reproduce and inspect

```sh
python experiments/compare.py YOUR_CLIP.mp4 --output comparisons
python experiments/registration_benchmark.py YOUR_CLIP.mp4 --output benchmark.json
video-mosaic YOUR_CLIP.mp4 --sweep 0.1 0.3 0.5 0.7 0.9 --output views
```

For the padded clips, add `--active-area 0.125 0 0.875 1` to each command. The comparison scripts accept the same bounds.

Keep full canvases, masks, parameters, failures, and source indices. Inspect fixed crops at the same scale in addition to the full images. A sparse correspondence score can miss parallax, repeated structures, a wrong dominant plane, or photometric seams; visual review and geometry evidence must accompany it.

The generated harbor demo checks basic transform and sampling behavior. The calibrated planar fixtures exercise perspective. The layered fixture includes occlusion, depth variation, and optional independent motion, but an improved parallax renderer has not yet been accepted. The next evaluation should compare dense observed strips with bounded flow-assisted strips and two-dimensional seams, measuring contour continuity, holes, and local stretch together.
