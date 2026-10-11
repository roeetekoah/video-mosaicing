"""Multiple strip views in one rectified coordinate atlas.

Independent manifold panoramas have different gauges: their accumulated width
and vertical origin depend on the anchor. They must not be concatenated as a
viewpoint video. Here one reference manifold defines the output coordinates.
Other views select different source observations and transport those pixels
into that same atlas using the measured pairwise homographies.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import GeometryError, RenderResult, checked_homography, render_rectified


def _project_valid(matrix, xy):
    homogeneous = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1) @ matrix.T
    denominator = homogeneous[..., 2]
    valid = np.isfinite(homogeneous).all(axis=-1) & (np.abs(denominator) > 1e-9)
    result = np.full_like(xy, np.nan, dtype=float)
    np.divide(homogeneous[..., :2], denominator[..., None], out=result, where=valid[..., None])
    return result, valid


@dataclass
class SweepAtlas:
    frames: list
    reference: RenderResult
    reference_alpha: float
    cumulative: np.ndarray
    column_owner: np.ndarray
    column_points: np.ndarray
    candidate_xy: np.ndarray
    candidate_valid: np.ndarray


def prepare_atlas(frames, homographies, *, reference_alpha=.5):
    """Fix the geometry once; precompute admissible observations per column."""
    reference = render_rectified(frames, homographies, alpha=reference_alpha)
    h, w = frames[0].shape[:2]
    cumulative = [np.eye(3)]
    for matrix in homographies:
        product = checked_homography(matrix, (h, w)) @ cumulative[-1]
        product /= np.linalg.norm(product)
        cumulative.append(product)
    cumulative = np.array(cumulative)
    out_h, out_w = reference.coverage.shape
    owners = np.full(out_w, -1, np.int32)
    points = np.full((out_w, 2), np.nan)
    for x in range(out_w):
        rows = np.flatnonzero(reference.coverage[:, x])
        if len(rows):
            y = rows[len(rows)//2]
            owners[x] = reference.source_index[y, x]
            points[x] = reference.source_xy[y, x]
    candidate_xy = np.full((len(frames), out_w, 2), np.nan)
    valid = np.zeros((len(frames), out_w), bool)
    for owner in np.unique(owners[owners >= 0]):
        columns = np.flatnonzero(owners == owner)
        inverse = np.linalg.inv(cumulative[owner])
        for target in range(len(frames)):
            xy, finite = _project_valid(cumulative[target] @ inverse, points[columns])
            candidate_xy[target, columns] = xy
            valid[target, columns] = finite & (xy[:, 0] >= 0) & (xy[:, 0] <= w-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h-1)
    return SweepAtlas(frames, reference, float(reference_alpha), cumulative, owners, points, candidate_xy, valid)


def render_atlas_view(atlas, *, alpha=.5):
    """Select observed strips without changing the reference projection.

One source frame owns each output column. Selection minimizes the distance of
the column's central observed ray to the requested source-image anchor. At the
scan ends, the closest available observation supplies the cap. Unsupported
pixels remain explicitly unobserved; there is no image-to-image stabilization,
crossfade, optical-flow interpolation, or inpainting.
"""
    if not 0 < alpha < 1:
        raise ValueError('Sweep alpha must be strictly between 0 and 1')
    reference = atlas.reference
    h, w = atlas.frames[0].shape[:2]
    distance = np.abs(atlas.candidate_xy[..., 0] - alpha*(w-1))
    distance[~atlas.candidate_valid] = np.inf
    selected = np.argmin(distance, axis=0).astype(np.int32)
    selected[~np.isfinite(distance.min(axis=0))] = -1
    # Keep the reference observation at its own anchor, including both caps.
    if abs(alpha-atlas.reference_alpha) < 1e-12:
        selected = atlas.column_owner.copy()
    image = np.zeros_like(reference.image)
    source_index = np.full_like(reference.source_index, -1)
    source_xy = np.full_like(reference.source_xy, np.nan)
    for owner in np.unique(atlas.column_owner[atlas.column_owner >= 0]):
        for target in np.unique(selected[atlas.column_owner == owner]):
            if target < 0: continue
            columns = np.flatnonzero((atlas.column_owner == owner) & (selected == target))
            rows, offsets = np.nonzero(reference.coverage[:, columns])
            xx = columns[offsets]
            xy = reference.source_xy[rows, xx].astype(float)
            transform = atlas.cumulative[target] @ np.linalg.inv(atlas.cumulative[owner])
            mapped, finite = _project_valid(transform, xy)
            valid = finite & (mapped[:, 0] >= 0) & (mapped[:, 0] <= w-1) & (mapped[:, 1] >= 0) & (mapped[:, 1] <= h-1)
            rows, xx, mapped = rows[valid], xx[valid], mapped[valid]
            if not len(rows): continue
            # OpenCV limits each remap dimension to < 32767, including a flat
            # coordinate list for the wide first/last cap.
            for start in range(0, len(rows), 30000):
                take = slice(start, start+30000)
                pixels = cv2.remap(atlas.frames[target], mapped[take, 0, None].astype(np.float32),
                                   mapped[take, 1, None].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
                image[rows[take], xx[take]] = pixels
            source_index[rows, xx] = target
            source_xy[rows, xx] = mapped
    coverage = source_index >= 0
    if coverage.sum() < .5*reference.coverage.sum():
        raise GeometryError('Sweep view has insufficient observed support', dict(alpha=float(alpha)))
    evidence = dict(method='fixed_rectified_atlas_sweep', alpha=float(alpha),
                    reference_alpha=atlas.reference_alpha, canvas_hw=list(coverage.shape),
                    coverage_fraction=float(coverage.mean()),
                    reference_coverage_fraction=float(reference.coverage.mean()),
                    selected_frame_per_column=selected.tolist(),
                    reference_geometry=reference.evidence,
                    observation_policy='One measured source frame per column; nearest supported anchor; missing pixels remain unobserved')
    return RenderResult(image, coverage, source_index, source_xy, evidence)
