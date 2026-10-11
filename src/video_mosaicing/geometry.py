"""Symmetric rectified strips, following Peleg et al., PAMI 2000, Appendix A.2.

All H matrices map pixels in frame k to pixels in frame k+1. Coordinates are
pixel centers, x right and y down. The normalized scan moves right in the
scene, so scene points move left in successive frames. No cumulative image
homography is used to render the mosaic.
"""
from dataclasses import dataclass

import cv2
import numpy as np


class GeometryError(ValueError):
    """Motion cannot support a nonfolding strip in the requested direction."""
    def __init__(self,message,evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def project(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    shape = points.shape
    p = np.column_stack((points.reshape(-1, 2), np.ones(points.size // 2))) @ np.asarray(matrix).T
    if not np.isfinite(p).all() or np.any(np.abs(p[:, 2]) < 1e-9):
        raise GeometryError("Projection reaches the projective horizon")
    return (p[:, :2] / p[:, 2:]).reshape(shape)


def checked_homography(matrix, hw):
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all() or abs(np.linalg.det(m)) < 1e-10:
        raise GeometryError("Singular or nonfinite homography")
    m = m / np.linalg.norm(m)
    h, w = hw
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float)
    denominator = np.column_stack((corners, np.ones(4))) @ m[2]
    if np.min(denominator) * np.max(denominator) <= 0:
        raise GeometryError("Homography crosses the projective horizon")
    if abs(m[2, 2]) > 1e-9: m = m / m[2, 2]
    return m


@dataclass
class StripPair:
    matrix: np.ndarray
    anchor: float
    height: float
    dy: float
    width: float
    boundary: np.ndarray  # shared boundary's endpoints in the earlier frame

    def boundary_points(self, v):
        v = np.asarray(v)
        return self.boundary[0] + v[..., None] * (self.boundary[1] - self.boundary[0])

    def sample(self, half, u, v):
        """Inverse map for either half of this inter-anchor interval.

        The common boundary is parameterized once and mapped through H. This
        retains exact projective correspondence along the entire seam, rather
        than approximating it by independently interpolating its endpoints.
        Anchor edges remain linear in row, allowing vertical-offset placement.
        """
        u, v = np.broadcast_arrays(u, v)
        edge = self.boundary_points(v)
        anchor = np.stack((np.full_like(v, self.anchor), self.height*v), axis=-1)
        if half == "right":
            return anchor*(1-u[..., None]) + edge*u[..., None]
        if half == "left":
            anchor[..., 1] += self.dy
            return project(self.matrix, edge)*(1-u[..., None]) + anchor*u[..., None]
        raise ValueError("half must be left or right")


def symmetric_pair(matrix, hw, alpha=.5):
    h, w = hw
    if not 0 <= alpha <= 1: raise ValueError("alpha must be in [0,1]")
    matrix = checked_homography(matrix, hw)
    anchor = alpha*(w-1)
    center = np.array([anchor, (h-1)/2])
    dy = float(project(matrix, center)[1] - center[1])
    a = np.array([[anchor, 0], [anchor, h-1]], float)
    shifted = a + [0, dy]
    b = project(np.linalg.inv(matrix), shifted)
    # Rectangle -> [earlier anchor, later anchor pulled back], in cyclic order.
    quad = np.array([a[0], b[0], b[1], a[1]], np.float32)
    e1 = np.roll(quad, -1, axis=0)-quad
    e2 = np.roll(quad, -2, axis=0)-np.roll(quad, -1, axis=0)
    cross = e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
    if np.any(cross <= 1e-5):
        raise GeometryError("Anchor interval reverses or folds; split the scan")
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    rect_to_source = cv2.getPerspectiveTransform(unit, quad)
    boundary = project(rect_to_source, [[.5, 0], [.5, 1]])
    midpoint = boundary.mean(axis=0)
    other_midpoint = project(matrix, midpoint)
    if not (0 <= midpoint[0] <= w-1 and 0 <= other_midpoint[0] <= w-1):
        raise GeometryError("Requested anchor lacks symmetric strip support; move it inward or use denser frames")
    delta_a = float(project(np.linalg.inv(matrix), center+[0, dy])[0] - anchor)
    delta_b = float(anchor - project(matrix, center)[0])
    width = (delta_a + delta_b)/2
    if not np.isfinite(width) or width < .35 or width > .8*w:
        raise GeometryError(f"Unsupported inter-anchor displacement: {width:.3f}px")
    pair = StripPair(matrix, anchor, h-1, dy, width, boundary)
    # Smooth inverse maps must preserve orientation over each half strip.
    us, vs = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 17))
    for half in ("left", "right"):
        eps = 1e-4
        du = (pair.sample(half, us+eps, vs)-pair.sample(half, us-eps, vs))/(2*eps)
        dv = (pair.sample(half, us, vs+eps)-pair.sample(half, us, vs-eps))/(2*eps)
        det = du[..., 0]*dv[..., 1]-du[..., 1]*dv[..., 0]
        if not np.isfinite(det).all() or det.min() <= 1e-5:
            raise GeometryError("Rectified strip has a fold or singular Jacobian")
    return pair


@dataclass
class RenderResult:
    image: np.ndarray
    coverage: np.ndarray
    source_index: np.ndarray
    source_xy: np.ndarray
    evidence: dict


def render_rectified(frames, homographies, *, alpha=.5, gains=None):
    """Render once from original pixels; retain full canvas and provenance.

    Full first/last image caps preserve the ends of the scan. Interior frames
    contribute narrow symmetric strips. Black output pixels with coverage=False
    are missing observations, not invented or inpainted scene content.
    """
    if len(frames) < 2 or len(homographies) != len(frames)-1:
        raise ValueError("Need N images and N-1 forward homographies")
    h, w = frames[0].shape[:2]
    if any(f.shape != (h, w, 3) or f.dtype != np.uint8 for f in frames):
        raise ValueError("Frames must have equal HxWx3 uint8 shape")
    pairs = []
    for i,m in enumerate(homographies):
        try: pairs.append(symmetric_pair(m,(h,w),alpha))
        except GeometryError as error:
            error.evidence.update(pair=i,matrix=np.asarray(m).tolist(),alpha=alpha)
            raise
    anchor = alpha*(w-1)
    xs = anchor + np.r_[0, np.cumsum([p.width for p in pairs])]
    ys = np.r_[0, -np.cumsum([p.dy for p in pairs])]
    y0 = np.floor(ys.min())
    out_h = int(np.ceil(ys.max()+h-1)-y0)+1
    out_w = int(np.ceil(xs[-1]+w-1-anchor))+1
    if out_h*out_w > 80_000_000 or out_h > 4*h:
        raise GeometryError("Implausibly large canvas; check motion or split the shot")
    image = np.zeros((out_h, out_w, 3), np.uint8)
    source_index = np.full((out_h, out_w), -1, np.int32)
    source_xy = np.full((out_h, out_w, 2), np.nan, np.float32)
    seams, segments, stretches = [], [], []

    def paint(index, left, right, ybase, mapper, kind):
        if right-left < 1e-9: return
        columns = np.arange(max(0, int(np.ceil(left))), min(out_w, int(np.ceil(right))))
        # Last pixel of the final cap belongs to it.
        if kind == "last_cap" and right == out_w-1:
            columns = np.r_[columns, out_w-1]
        if not len(columns): return
        rows = np.arange(out_h)
        xx, yy = np.meshgrid(columns.astype(float), rows.astype(float)+y0)
        u = (xx-left)/(right-left)
        v = (yy-ybase)/(h-1)
        xy = mapper(u, v)
        valid = (v >= 0) & (v <= 1) & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)
        sampled = cv2.remap(frames[index], xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if gains is not None:
            sampled = np.clip(sampled.astype(float)*gains[index,0]+gains[index,1],0,255).astype(np.uint8)
        ri, ci = np.nonzero(valid)
        image[ri, columns[ci]] = sampled[ri, ci]
        source_xy[ri, columns[ci]] = xy[ri, ci]
        source_index[ri, columns[ci]] = index
        # Sample the inverse-map Jacobian; values are source pixels/output pixel.
        uu, vv = np.meshgrid(np.linspace(.05, .95, 5), np.linspace(.05, .95, 9))
        e = 1e-4
        du = (mapper(uu+e, vv)-mapper(uu-e, vv))/(2*e*(right-left))
        dv = (mapper(uu, vv+e)-mapper(uu, vv-e))/(2*e*(h-1))
        singular = np.linalg.svd(np.stack((du, dv), axis=-1), compute_uv=False)
        stretches.extend(singular.reshape(-1).tolist())
        segments.append(dict(frame=index, kind=kind, x=[float(left), float(right)], y_offset=float(ybase), valid_fraction=float(valid.mean())))

    paint(0, 0, anchor, ys[0], lambda u,v: np.stack((u*anchor, v*(h-1)), -1), "first_cap")
    boundary_errors = []
    for i, pair in enumerate(pairs):
        seam = xs[i]+pair.width/2
        paint(i, xs[i], seam, ys[i], lambda u,v,p=pair: p.sample("right",u,v), "right")
        paint(i+1, seam, xs[i+1], ys[i], lambda u,v,p=pair: p.sample("left",u,v), "left")
        seams.append(float(seam))
        v = np.linspace(0, 1, 101)
        error = np.linalg.norm(project(pair.matrix, pair.sample("right", np.ones_like(v),v))-pair.sample("left",np.zeros_like(v),v),axis=-1)
        boundary_errors.append(float(error.max()))
    paint(len(frames)-1, xs[-1], xs[-1]+w-1-anchor+1e-7, ys[-1],
          lambda u,v: np.stack((anchor+u*(w-1-anchor), v*(h-1)), -1), "last_cap")
    coverage = source_index >= 0
    evidence = dict(method="symmetric_rectified_manifold", alpha=float(alpha),
                    canvas_hw=[out_h,out_w], anchor_x=xs.tolist(), anchor_y=ys.tolist(),
                    y_origin=float(y0), seams=seams, segments=segments,
                    pair_widths=[p.width for p in pairs], vertical_offsets=[p.dy for p in pairs],
                    boundary_correspondence_max_px=max(boundary_errors),
                    inverse_stretch_range=[min(stretches),max(stretches)],
                    coverage_fraction=float(coverage.mean()))
    return RenderResult(image, coverage, source_index, source_xy, evidence)
