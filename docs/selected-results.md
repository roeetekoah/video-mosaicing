# Selected direct LK results

The release combines the two existing direct LK routes using a fixed, visually reviewed configuration for each supplied valid input. It adds no further motion solver or rendering experiment.

| Input | Selected method | Reason |
|---|---|---|
| Boat | Local-depth LK | Less accumulated curvature and improved overall scene layout; some local distortion remains. |
| House | Local-depth LK | Clearer building and garden continuity. Its explicit encoded-side-bar crop is part of this configuration. Main showcase. |
| Garden | Local-depth LK | More level panorama and fence layout; residual strip transitions remain. |
| Kessaria | Local-depth LK | Improved overall panorama shape. Moving transitions remain visible and are not claimed solved. |
| Iguazu | Original rigid LK | Retains the stronger visual result; the extension regresses. |
| Trees | Original rigid LK | Avoids the extension's strong local strip artifacts. |
| Shinkansen | Original rigid LK | Conservative motion/runtime choice: straightening in the extension comes with local rail and train waviness. |
| good_input | Original rigid LK | Avoids the extension's motion-estimation failure and shelf distortion. |
| good_raw | Original rigid LK | Avoids additional shelf waviness. |

The selections use full-sweep comparisons, five distributed detailed views, and additional sampled views across the candidate sweeps. They are qualitative curation of these inputs, not scores against a ground-truth panorama. The original and depth methods can sample different 3D rays at the same normalized sweep progress.

`auto` applies these settings only when the SHA256 of the input bytes matches a reviewed example. It restores resolution, source selection and crop. Unknown inputs use original LK; users may explicitly choose either method. Renaming does not affect selection. Re-encoding changes the content hash and uses the unknown-input policy.

The [machine-readable configuration](selected-results.json) and [package presets](../src/video_mosaicing/presets.py) record the choices. `run.json` explains the actual selection and records explicit overrides. `bad_input` and `bad_raw` remain intentional failure examples and are excluded.

The main [House movie](../assets/showcase/house-sweep.mp4) contains the 101 actual rendered views from the recorded depth-LK run, at 15 fps. The animated preview retains those same views at reduced spatial resolution. No intermediate panorama views are synthesized for presentation. Source footage and private bookshelf outputs are not published.

See the [full two-method evaluation](direct-lk-evaluation.md) for runtime and limitations. The selected release does not claim that depth LK is universally more accurate, faster or preferable.
