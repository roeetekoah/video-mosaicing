# Direct LK full-video evaluation

The two direct LK routes each produced all 101 requested views for every one of the nine valid inputs. `bad_input` and `bad_raw` are intentional failure examples and are excluded. This confirms output completeness; it does not establish that every video is visually improved or that the extension dominates every quality/runtime tradeoff.

The original route reproduces all 909 panorama pixel hashes from the archived controlled baseline. The coursework source and report remain preserved. The depth-aware route uses the original rigid solver for initialization, direct multi-frame photometric alignment with local slopes, and camera-position slit rendering.

## Runtime

One sequential run per method, at maximum input side 800, with 101 views spanning alpha 0–1. Times include decoding/preprocessing, registration, preparation and rendering. Encoding, image writes, checksums and report writes are excluded. Source sampling differs by method, and output canvas dimensions can differ; this is a pipeline tradeoff, not an isolated solver speed comparison. House depth mode also explicitly excludes its encoded side bars; the archived original retains them for exact baseline reproduction. Values are preliminary single-run measurements, not repeated-run confidence intervals.

| Input | Original sources | Depth sources | Original seconds | Depth seconds | Time ratio |
|---|---:|---:|---:|---:|---:|
| boat | 45 | 150 | 21.0 | 37.8 | 1.80× |
| House | 43 | 144 | 19.6 | 36.8 | 1.88× |
| Garden | 80 | 163 | 39.5 | 57.2 | 1.45× |
| Iguazu | 49 | 166 | 21.3 | 41.9 | 1.96× |
| Kessaria | 40 | 90 | 18.7 | 27.9 | 1.50× |
| Shinkansen | 40 | 77 | 10.7 | 23.3 | 2.17× |
| Trees | 40 | 128 | 26.4 | 40.9 | 1.55× |
| good_input | 40 | 67 | 26.3 | 31.4 | 1.19× |
| good_raw | 41 | 132 | 25.1 | 41.2 | 1.64× |

The depth-aware route takes about 1.2–2.2 times the original pipeline on these full inputs. It retains more observations and estimates more than a rigid transform. The original route remains useful when speed matters. Higher source density alone is not a quality guarantee.

## Visual interpretation

The review movies contain actual rendered outputs, with all nine depth-aware outputs also shown together. Pair comparisons use a shared display scale, retain unsupported canvas regions, and include native-scale reference-region details. They are synchronized by normalized sweep progress. Different rendering models need not follow identical 3D rays at the same alpha; source selection can also change the reference frame slightly. This is not a ground-truth reconstruction accuracy benchmark.

Distributed views show reduced large strip discontinuities in some scenes, while vegetation, masonry and depth boundaries retain waviness, ghosting or stretching. The bookshelf inputs introduce visible shelf waviness; Trees has strong artifacts at extreme viewpoints. The [selected release](selected-results.md) therefore retains original LK for five examples and uses the extension for four; unknown inputs also use original LK. Moving water and other dynamic content violate the stationary-intensity model. Acceptance of motion quality requires viewing the sweeps, not only still images or registration residuals. The extension is not a claim to reproduce the supplied reference outputs.

## Checks and reproduction

The 76 passing tests include known-depth recovery, analytic ray rendering at both endpoints and the center, duplicate-frame selection, oversized sparse sampling, original resize semantics, output masks, real video encoding and curated preset selection/overrides, in addition to the existing tests. Default-auto House and Iguazu runs additionally reproduce the reviewed central images pixel-for-pixel with identical source selection. Lossless rendering changes are checked against stored experiment frame hashes. The optional feature-based renderer remains covered by its existing tests.

[The recorded protocol](direct-lk-runs.json) contains input hashes, exact original source indices, depth sampling and skipped indices, canvas sizes, full timing components, and pixel hashes for all outputs. Input footage is required separately.

```sh
python experiments/evaluate_direct_lk.py /path/to/boat.mp4 --clip boat --method lk-original --output benchmark/original
python experiments/evaluate_direct_lk.py /path/to/boat.mp4 --clip boat --method lk-depth --output benchmark/depth
```

These commands render and hash all 101 views, saving five distributed panoramas and masks. For a movie, use the CLI's `--video` option described in the README. The exact recorded original source list is used by the benchmark script; the interactive original CLI otherwise uses its configured stride.
