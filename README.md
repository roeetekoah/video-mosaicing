# Video Mosaicing

**Wide panoramas and moving viewpoints from a continuous camera scan.**

<p><strong><a href="https://github.com/roeetekoah">Roee Tekoah</a></strong> &nbsp;·&nbsp; <img src="https://img.shields.io/badge/Project_grade-96%2F100-24705a?style=flat-square" alt="Project grade: 96/100" height="24" align="absmiddle"></p>

Direct Lucas–Kanade alignment, coordinated panorama strips, and a local-depth extension for scenes with parallax. Developed for Image Processing at the **Hebrew University of Jerusalem**, graded **96/100**.

[Run the project](#run-the-project) · [Method walkthrough](docs/direct-lk.md) · [Selected results](docs/selected-results.md) · [Full evaluation](docs/direct-lk-evaluation.md)

## Showcase

https://github.com/user-attachments/assets/8782f787-a685-4c61-8468-2dfbb1f24fcc

**House · 101 rendered viewpoints from real footage.** Direct multi-frame LK estimates camera motion and local relative-depth slopes; the renderer samples overlapping observations to move through the panorama. The preview is resized from the actual output movie, with every rendered view retained.

[Watch the full-resolution movie](assets/showcase/house-sweep.mp4) · [Open the central panorama](assets/showcase/house-panorama.png) · [Recorded configuration](assets/showcase/showcase.json)

The recorded House run uses 144 source observations and an explicit crop of its encoded side bars. It takes 36.8 seconds for preparation and all 101 views at maximum source dimension 800, excluding encoding and evidence writes. It retains some local distortion; the showcase is a multiperspective panorama, not a calibrated 3D reconstruction.

## How it works

1. **Align overlapping images.** The original pyramidal LK solver estimates horizontal and vertical translation plus in-plane rotation directly from image intensities.
2. **Compose a panorama.** The original route accumulates rigid transforms, chooses coordinated strip boundaries with dynamic programming, and feathers their joins. Invariant seam costs are cached across viewpoints.
3. **Account for local parallax.** The depth-aware route extends direct LK to several observations and estimates local relative-depth slopes. Its renderer compensates local displacement before combining adjacent observations.
4. **Sweep the viewpoint.** Render different slit positions using one registration pass. Every movie frame is an actual rendered view; completed panorama images are not interpolated to manufacture motion.

Both direct LK routes remain available. The coursework source and report are preserved.

## Select the result that works for the scene

The default **`auto`** mode uses reviewed configurations for the nine supplied examples:

| Local-depth LK | Original rigid LK |
|---|---|
| Boat, House, Garden, Kessaria | Iguazu, Trees, Shinkansen, good_input, good_raw |

These are explicit visual selections for known inputs, identified by SHA256 rather than filenames. **Unknown inputs use the original LK route.** This is a reproducible preset policy, not a general automatic quality estimator. Explicit method and configuration options remain available.

The extension reduces some large discontinuities, but can introduce waviness or unstable local motion. The selected release retains the original result where that tradeoff is unfavorable. [Per-input choices and limitations →](docs/selected-results.md)

## Run the project

Use **Python 3.10 or later**. Install from the repository root:

```sh
python -m venv .venv
# Activate the environment for your platform.
python -m pip install .
```

Input can be a video or a folder of lexically ordered images. Choose a new output directory for each run:

```sh
# Reviewed configuration for a known example; original LK for a new input.
video-mosaic scan.mp4 --output output/panorama

# A complete movie of 101 rendered viewpoints; requires FFmpeg on PATH.
video-mosaic scan.mp4 --video --views 101 --fps 15 --output output/sweep

# Select a method explicitly.
video-mosaic scan.mp4 --method lk-original --output output/rigid
video-mosaic scan.mp4 --method lk-depth --output output/depth
```

Known `auto` examples restore their reviewed maximum dimension of 800, frame selection and active-area crop. Other inputs and explicit methods default to maximum dimension 640. `--max-side`, `--frame-stride` and `--active-area LEFT TOP RIGHT BOTTOM` override the corresponding defaults. Explicit methods use their own defaults rather than applying example presets.

`run.json` records the requested and selected methods, preset, source indices, motion, configuration, coverage and timing. Depth runs also save estimated geometry unless `--no-provenance` is supplied. [Configuration, output formats and Python API →](docs/direct-lk.md)

## Evaluation

Both direct LK methods were evaluated on **nine valid inputs and 101 views per input**. With the archived frame selections and active areas, the cached original route reproduces **all 909 archived panorama pixel hashes**. Intentional failure examples are excluded.

The depth-aware route takes approximately **1.2–2.2×** the original pipeline time in the recorded runs. Source density and canvas dimensions differ, so this measures the complete pipeline tradeoff. Visual selection considers the full sweep, local distortion and motion as well as overall panorama shape; it does not claim uniformly lower reconstruction error.

[Per-video timings and limitations](docs/direct-lk-evaluation.md) · [Selected release configurations](docs/selected-results.json) · [Full recorded runs](docs/direct-lk-runs.json)

```sh
python -m pip install ".[test]"
python -m pytest -q
```

Checks cover analytic depth recovery and rendering, source selection, output masks, video encoding, curated configuration overrides and the original pipeline. Real-input parity additionally checks that reviewed presets restore their saved outputs.

## Research foundations

- Peleg, Rousso, Rav-Acha and Zomet, [*Mosaicing on Adaptive Manifolds*](https://www.cs.huji.ac.il/~peleg/papers/pami00-manifold.pdf), PAMI 2000.
- Rousso, Peleg and Finci, [*Video Mosaicing using Manifold Projection*](https://bmva-archive.org.uk/bmvc/1997/papers/peleg/paper.htm), BMVC 1997.
- Rav-Acha, Shor and Peleg, [*Mosaicing with Parallax using Time Warping*](https://doi.org/10.1109/CVPR.2004.400), CVPR Workshops 2004, Sections 3–5. The local-depth route adapts lateral/vertical translation and in-plane roll; it does not reproduce the full planar and spherical formulations.

<details>
<summary>Additional feature-based pipeline</summary>

`--method rectified` retains a separate SIFT-based registration and rectified-strip implementation. It is independent of the direct LK showcase and selected configurations above. See its [walkthrough](docs/pipeline-walkthrough.md), [evaluation](docs/evaluation.md) and [Garden sweep](assets/garden/sweep.mp4). Procedural fixtures support reproducibility and are not real-footage quality evidence.

</details>

---

<sub>Selected rendered examples accompany the release. Source videos, supplied reference outputs, papers, personal recordings and historical report files are excluded.</sub>
