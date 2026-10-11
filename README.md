# Video Mosaicing

Generate wide panoramas and controlled strip-view sweeps from a continuous camera scan. The renderer uses symmetric rectified strips based on Peleg and colleagues' *Mosaicing on Adaptive Manifolds*, rather than projecting an entire sequence onto one reference image.

https://github.com/user-attachments/assets/e78cf42f-f200-46f8-8add-4d20f2d5fe9e

[Download the MP4 demo](assets/demo.mp4) · [Animated preview](assets/demo.gif) · [Full-resolution panorama](assets/harbor.png) · [Per-video comparison](docs/evaluation.md)

The animation shows a generated camera scan and the actual reconstructed panorama, revealed according to each pixel's source frame. It uses the fixed final geometry; it does not imply online reconstruction. Rebuild it with `python experiments/make_showcase.py` (requires FFmpeg). The [evaluation notes](docs/evaluation.md) separately describe real-video results and remaining failures.

Originally developed for Image Processing coursework at the Hebrew University of Jerusalem, **graded 100/100**. This release extends that foundation with robust registration, rectified manifold geometry, motion-based frame selection, and reproducible evaluation.

## Run

Python 3.10 or later:

```sh
python -m venv .venv
# Activate the environment using your platform's normal command.
python -m pip install ".[test]"
python -m video_mosaicing.demo --output demo-output/generated
python -m pytest -q
```

For a video or a folder of lexically ordered images:

```sh
video-mosaic scan.mp4 --output demo-output/scan
video-mosaic scan.mp4 --sweep 0.1 0.3 0.5 0.7 0.9 --output demo-output/views
```

For footage with encoded side bars, declare the scene bounds explicitly. For example, `--active-area 0.125 0 0.875 1` excludes the outer one-eighth on each side. Bounds are fractions of the input, applied after resizing; every run records them. The original footage remains unchanged. Anchors refer to this active scene area, and provenance maps back before its extraction.

Use a new output directory for each run. Every result includes the full panorama, an observation mask, configuration and motion evidence, and a compressed map from output pixels to source frames and coordinates in the resized input. The CLI processes all decoded frames, retaining useful views according to measured displacement. `--max-side` controls resolution; `--spacing` controls minimum strip spacing.

`--exposure` enables optional robust overlap gain/bias correction. `--blend 2` enables narrow feathering. Both are experimental comparisons, disabled by default. Wide blending can obscure misalignment, so geometry is evaluated with hard strips first.

## Method

SIFT correspondences initialize robust rigid, similarity, affine, and homography candidates. Reserved correspondences, spatial coverage, local magnification, and strip geometry determine whether a candidate is usable. The simplest model close to the best validation error is selected. Neighboring anchors define locally rectified strips, sampled once from the source images. Missing observations remain visible in the mask.

The original centered rigid Lucas–Kanade pipeline remains executable in `legacy_pipeline.py`. Its motion model, transform accumulation, seam optimization, and explanations are retained. `refinement.py` supplies a separate direct rigid solver with an analytically checked Jacobian, true overlap support, robust residuals, and a consistent update objective.

See [geometry and implementation](docs/method.md), [evaluation and limits](docs/evaluation.md), and the reproducible scripts under `experiments/`.

## Capture limits

Use a continuous, mostly monotone scan with substantial overlap and textured stationary content. A scene cut, insufficient support, reversal, or folding strip produces an explicit failure. A single homography cannot align every depth under translation. Strong parallax and moving objects can still cause broken contours or rejection. A strip-view sweep is a multiperspective visualization; it is not calibrated stereo or a reconstructed 3D scene.

Anchors near 0.1–0.9 are normally useful. An anchor exactly on an image border lacks symmetric support and is rejected. Increasing temporal density can support narrower strips, but cannot create unseen observations.

## References and data

Peleg, Rousso, Rav-Acha and Zomet, [*Mosaicing on Adaptive Manifolds*](https://www.cs.huji.ac.il/~peleg/papers/pami00-manifold.pdf), PAMI 2000, Section 3.4 and Appendix A.2; Rousso, Peleg and Finci, [*Video Mosaicing using Manifold Projection*](https://bmva-archive.org.uk/bmvc/1997/papers/peleg/paper.htm), BMVC 1997.

The repository's demo scenes are procedural. Supplied coursework videos, reference outputs, papers, personal recordings, and original report documents are excluded. Local development comparisons do not establish redistribution rights for their input footage.
