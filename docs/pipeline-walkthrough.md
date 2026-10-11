# Pipeline walkthrough: rigid LK to rectified manifold mosaicing

This walkthrough distinguishes the original coursework algorithm, the current default execution path, and optional experiments. The original project already implemented a complete mosaicing system; the extension changes its estimator, projection, and sampling policy rather than supplying previously missing basic stages.

## 1. The original pipeline

The coursework flow in `main.py::generate_panorama_from_frames` was:

```text
Read and uniformly sample frames
  → grayscale / blur / normalize scan direction
  → global pyramidal rigid Lucas–Kanade on consecutive selected frames
  → accumulate pairwise transforms into a middle reference frame
  → compute a canvas and warp every full frame and validity mask onto it
  → choose one seam column per adjacent overlap using dynamic programming
  → copy strips and feather the boundaries
  → repeat seam selection for different panorama views
```

The executable counterpart remains in `legacy_pipeline.py`. The centered rigid LK solver, transform utilities, canvas logic, original seam optimizer, strip compositor, explanatory comments, and regression cases remain available.

### The motion model

There are three parameters: horizontal translation `dx`, vertical translation `dy`, and in-plane rotation `theta`. With image center `c`, the forward transform is:

```text
W(x; p) = R(theta) (x - c) + c + (dx, dy)
```

Centering reduces the numerical coupling between translation and rotation. A small rotation gives `u ≈ dx - theta*y` and `v ≈ dy + theta*x` in centered coordinates.

The original code estimates these parameters directly from pixel intensities. This is **global direct image alignment**, not sparse LK tracking of Harris corners. Each permitted image pixel contributes to the same three-parameter fit.

### Why the pyramid and Jacobian are there

LK linearizes brightness changes around the current warp. A Gaussian pyramid makes a large full-resolution displacement smaller at the coarsest level. The estimate is refined from coarse to fine; translations double between levels, while rotation stays an angle.

The original Jacobian construction is:

```python
J_dx = Ix
J_dy = Iy
J_dtheta = -ys * Ix + xs * Iy
```

`Ix` and `Iy` are Sobel gradients of the fixed image; `xs` and `ys` are centered coordinates. Translation changes intensities through their spatial gradients. Rotation moves a point by approximately `(-y, x)`, giving the third column. The implementation forms the damped 3×3 normal matrix once per level, repeatedly warps the other image, calculates the intensity residual, solves the normal equations, and adds the parameter increments. Its parameter names make the Gauss–Newton structure explicit:

```python
minus_JT_r = _build_rhs(j_dx, j_dy, j_dtheta, residual_t, mask)
dp = np.linalg.solve(JT_J, minus_JT_r)
dx, dy, theta = (dx + float(dp[0]), dy + float(dp[1]), theta + float(dp[2]))
```

The fixed-gradient approximation and warp convention matter when interpreting this update. It should not be described as an exact forward-additive residual Jacobian at every iteration. The separate direct solver discussed below makes that consistency explicit.

### Original rendering and seams

Pairwise motions are composed toward a middle reference image. A middle reference shortens the longest accumulation path; it does not remove accumulated estimation errors or make a single projection suitable for every wide scan.

The warped frame corners determine a common canvas. The seam DP chooses **one vertical seam column per neighboring frame pair**, jointly across the sequence of overlaps. Its costs include visual disagreement, distance from the requested seam position, and changes between successive seam columns, subject to ordering. It is not a row-by-row curved seam around an object. Feathering blends around the chosen columns; an 81-pixel default band can soften exposure transitions but also double misaligned structures.

## 2. The default estimator now uses SIFT and RANSAC

Harris is a corner detector. LK is an intensity-based alignment/tracking method. They are not interchangeable alternatives: a sparse tracker can detect Harris corners and then track those corners with LK. The coursework implementation instead fitted a global rigid motion directly to image intensities.

The current default uses OpenCV SIFT keypoints and descriptors:

```python
sift = cv2.SIFT_create(nfeatures=cfg.features)
keypoints, descriptors = sift.detectAndCompute(gray, None)
```

SIFT searches for scale-space extrema using differences of Gaussian-blurred images, estimates a local orientation, and describes the surrounding gradient distribution with a 128-component descriptor. It provides both point detection and a representation for recognizing a point in another image. See [OpenCV's SIFT explanation](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html).

The implementation requests up to 2,400 features per image, matches descriptors in both directions, applies the 0.75 nearest/second-nearest distance-ratio test, and keeps only mutually consistent matches. These filters reduce ambiguous correspondences; they do not prove that all retained points belong to one stationary scene surface.

Matched coordinates then support geometric fitting. RANSAC repeatedly fits from small subsets and seeks a consistent set of correspondences under a two-pixel reprojection threshold. The rigid candidate uses similarity-RANSAC to choose inliers, followed by an SVD fit constrained to rotation and translation. The similarity, affine and homography candidates use OpenCV's robust fitting routines.

| Candidate | Parameters | Permitted deformation |
| --- | ---: | --- |
| Rigid | 3 | Translation and in-plane rotation |
| Similarity | 4 | Rigid motion plus uniform scale |
| Affine | 6 | Translation, rotation, unequal scale and shear |
| Homography | 8 | A projective plane-to-plane mapping |

One homography describes a plane under perspective, or a pure-rotation view under suitable camera assumptions. It does not align several depths under camera translation. More parameters are not automatically better.

The deterministic 70/30 correspondence split separates fitting from validation. Acceptance requires enough inliers, support across at least four cells of a 4×3 grid, horizontal/vertical spread, a minimum convex-hull area, acceptable validation residuals, plausible local magnification, and compatible strip geometry where evaluated. Selection chooses the simplest accepted model within 0.12 pixel of the best validation median. The validation points influence selection, so they are not an untouched final test set.

**There is no default SIFT→LK refinement step.** The default motion comes from the accepted feature-based model. The original LK remains the comparison baseline; the separate revised direct solver remains opt-in. The visible default results must not be attributed to that optional solver.

## 3. Sampling follows movement instead of a fixed frame count

The comparison baseline uniformly samples 10% of input frames, clamped to 40–120. The new selector decodes the video sequentially and registers candidates against the last retained frame. Near-identical frames are skipped. Small movements accumulate until the advance along the scan reaches the default six pixels. Unsupported gaps, reversal and unobservable motion produce failures.

This matters because a short frame interval can have almost no useful movement, while a long interval can leave the local convergence range of direct LK or lose overlap. Frame selection is now part of the geometric problem. The current accepted scans retain 90 frames for Kessaria and 135 for Iguazu.

## 4. The renderer changes the projection, not just the seam

The rectified renderer follows the symmetric strip construction of [Peleg et al., PAMI 2000, Appendix A.2](https://www.cs.huji.ac.il/~peleg/papers/pami00-manifold.pdf). SIFT is the implementation's motion-estimation front end; it is not the defining contribution of the manifold construction.

For each neighboring pair, `H` maps earlier-frame coordinates forward into the later frame. Choose a vertical anchor at `alpha*(width-1)`. Shift the later anchor vertically using the measured motion, pull that anchor into the earlier frame with `inverse(H)`, and construct the quadrilateral between the two anchors. A projectively defined middle boundary splits that interval into contributions from the two images.

Both sides share the same boundary correspondence. In `StripPair.sample`, the later frame uses `project(self.matrix, edge)` for that common boundary. The renderer interpolates from each anchor to the boundary to construct an inverse sampling map, then samples the original observations with `cv2.remap`.

The crucial change is that **full frames are no longer projected onto one accumulated reference plane before strip extraction**. Narrow local strips can accommodate a tilted scan without the same global reference-plane curling. Strip widths and vertical placements come from neighboring motions. Shared-boundary consistency is exact for the supplied pair model; it cannot correct a wrong model or eliminate parallax within a strip.

The renderer checks for folds, projective horizons and singular geometry. First/last caps retain the ends of the scan. Missing observations remain missing. There is no fitted straightening trend, inpainting or hidden scene crop.

The view parameter also has a more specific meaning now: it selects an anchor within the active input image. Originally it targeted a seam position in a warped overlap. Equal numerical alpha values across the two renderers are not a calibrated match between physical viewpoints.

## 5. Borders, exposure and blending

Kessaria and Iguazu encode black side bars occupying one-eighth of each side. The declared active bounds `(0.125, 0, 0.875, 1)` exclude those bars in memory and preserve all scene pixels. Both methods receive exactly the same active area in the fair comparison. Removing padding improved the original Kessaria baseline substantially too; that gain must not be credited exclusively to the replacement estimator.

The default revised compositor uses hard strips. Exposure correction and narrow feathering are separately selectable, disabled by default. The current showcase uses neither. A broad blend can conceal a seam while retaining doubled objects; that is why geometry is inspected before smoothing.

The added direct rigid solver in `refinement.py` is a separate experiment. It keeps the three-parameter centered model and Gauss–Newton approach, while differentiating the actual bilinear sampler and exact rotation, considering only genuine overlap, using robust residual weights and a line search, and consistently transferring transforms between pyramid levels. Optional gain/bias parameters model exposure. Its Jacobian is checked numerically. It is not the source of the current default improvements.

## 6. What the measured gains mean

The real-video score compares both estimators on the same reserved feature correspondences at the same sampled frame spacing. It measures the pixel distance between a projected source feature and its matched target. It is not the brightness objective optimized by LK and not ground-truth panorama quality. See [the evaluation](evaluation.md) for per-pair data, residual tails, identical active bounds, and failures.

Kessaria and Iguazu improve clearly. Trees remains nearly unchanged and has depth-dependent contour defects. Boat has strong central scores on accepted pairs but large residual tails and an explicitly rejected full scan. These limits remain visible in the evaluation.

The method therefore comprises a substantial estimator/renderer/sampling overhaul with the original system retained as an executable foundation and comparison. It is not merely a Harris detector added ahead of the existing LK, and it is not a complete reproduction of every Peleg system: there is no accepted dense-flow parallax renderer, GIST blending, MAD depth system, calibrated stereo, or 3D reconstruction.
