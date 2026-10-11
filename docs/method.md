# Geometry and implementation

The goal of this project is to generate panorama images from a single video that scans a scene, by aligning multiple frames into a common mosaic and producing different panorama views through strip placement. The central difficulty is global consistency: small inaccuracies in rotation and vertical translation can dominate the result over a long sequence.

## Rigid motion and the reference-canvas baseline

Input: two consecutive grayscale frames. Output: a rigid transformation consisting of horizontal translation, vertical translation, and in-plane rotation. The pyramidal Lucas–Kanade implementation uses the center of the frame as the origin of the axes. The image pyramid handles larger apparent motion by solving first at coarse scales and refining at finer scales.

For an image center c, the centered rigid warp is `W(x) = R(theta)(x-c) + c + t`. Positive rotation is clockwise in image coordinates, where y increases downward. Centering improves the conditioning of translation and rotation; it does not eliminate accumulated error.

The baseline accumulates pairwise transformations into a middle reference frame. Each frame's four corners are warped into the reference coordinate system. The minimum and maximum extents across all warped corners define the panorama canvas, with a margin for rounding and interpolation. Each warped frame also supplies a binary validity mask.

Dynamic programming selects one seam column for each adjacent overlap, balancing visual disagreement, the requested seam location, and movement between successive seam columns. Each panorama is composed by copying vertical strips from the warped frames. Only pixels marked valid by the mask are copied. Feathering blends neighboring observations around the seam; increasing its width can introduce ghosting and reduce sharpness.

These components remain in the code as a complete baseline. Choosing a central reference balances the accumulation paths, but a reference-plane projection still distorts wide tilted scans. Detrending the path can conceal this projection problem rather than solve it.

## Rectified manifold construction

The renderer follows the symmetric anchor construction in [Peleg et al., PAMI 2000](https://www.cs.huji.ac.il/~peleg/papers/pami00-manifold.pdf), Appendix A.2. It implements local rectified strips, not the paper's full ego-motion or pipe-projection systems.

Throughout the implementation, `H[k]` maps pixels of frame k **forward** to frame k+1. Frames are oriented so a rightward scene scan produces leftward image motion. Let P and Q be the top and bottom endpoints of the vertical anchor. For a pair, compute `d = (H(center)-center).y` and shift the later anchor by d. Pull that shifted anchor into the earlier image using `inverse(H)`.

The quadrilateral between the earlier anchor and the pulled-back later anchor defines a projective mapping from a unit rectangle. Its middle column supplies the symmetric shared boundary. Each interval comprises the earlier frame's right half strip and the later frame's left half strip. Anchors are placed at horizontal distances equal to the average center displacement measured in both images. The next anchor's vertical placement changes by `-d`.

The shared boundary is parameterized once in the earlier image, and all its samples are mapped through H into the later image. This is a small implementation extension to endpoint-only bilinear interpolation: correspondence is exact along the entire projective boundary. Within each half strip, the inverse sampling map interpolates smoothly between the anchor and that boundary. Anchor row coordinates remain linear, preserving continuity under the prescribed vertical offsets.

The renderer checks quadrilateral orientation, projective horizons, strip width, and sampled Jacobian determinants before allocating a canvas. It renders from original observations in one interpolation pass. Full first and last caps preserve the ends of the scan. Interior strip observations, output coverage, source index, and source coordinates are retained separately. No fitted vertical trend, inpainting, or crop is used to straighten a result.

## Estimation and frame selection

SIFT descriptors are matched with a ratio test and mutual consistency. A fixed 70/30 split reserves correspondences from fitting. RANSAC initializes rigid, similarity, affine, and homography candidates; the rigid candidate uses a constrained orthogonal fit. Candidates must pass spatial support, residual, magnification, and nonfolding-strip checks. Selection favors the simplest supported model within 0.12 pixel of the best validation median. Validation affects model selection, so it is not a final untouched test set.

The video is decoded sequentially. Nearly identical observations are skipped, while small motions are accumulated against the last selected frame. A useful frame normally advances the anchor by at least six pixels at the processing resolution. Orthogonal jitter does not create a new strip. Large unsupported jumps, genuine reversals, and scene cuts stop processing rather than silently dropping a difficult section.

Encoded matte needs separate handling. Kessaria and Iguazu have black side bars occupying one-eighth of the decoded width on each side. Their declared active area excludes those bars in memory, while preserving every scene pixel and mapping provenance back into the resized input. The same area is used for both methods in the comparison. No automatic scene crop is inferred from black pixels: dark scene content can be legitimate. An anchor that consistently falls inside a constant black column is rejected with a request to declare the active area.

The original LK implementation remains unchanged numerically as a comparison path. The additional direct rigid solver uses the residual `r(x) = gain * target(W(x)) + bias - source(x)`. It differentiates the actual bilinear sampler and exact centered rotation, scales the rotation parameter by image size, and performs robust Gauss–Newton updates with a line search. Pyramid transforms are conjugated by the coordinate scaling matrix, including the image-center convention. Only genuine overlapping samples contribute. This solver is tested separately and is not enabled in the default feature pipeline.

## Photometry and unresolved depth variation

Optional exposure fitting estimates gain and bias from low-frequency, unclipped overlap observations using robust orthogonal regression. Redundant temporal overlaps constrain a global gain/bias system whose middle-frame gauge is fixed. Implausible or poorly supported corrections are rejected. Narrow feathering is a separate option, and its additional source dependencies are recorded.

Local rectification removes the need for one global reference plane; it does not make a homography depth-aware. Dense observed frames reduce strip width, but moving water, independently moving objects, and nearby trees can still disagree with the fitted plane. Optical-flow interpolation, bounded two-dimensional seams, gradient-domain reconstruction, and minimal-aspect-distortion mosaicing are further experiments, not features claimed by this implementation.
