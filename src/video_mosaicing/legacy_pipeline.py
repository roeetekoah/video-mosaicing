"""The selected modular coursework pipeline, with explicit evidence and checks."""
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .canvas import compute_canvas
from .data_structures import LKConfig
from .diagnostics import DiagnosticConfig, RegistrationError, inspect_registration
from .direction_normalization import normalize_scan_direction
from .lucas_kanade import estimate_rigid_lk_pyramids
from .preprocess import preprocess_frame
from .rigid_transformations import accumulate_to_ref
from .seams import compute_frame_x_ranges, seams_for_view, seams_for_view_scored_dp
from .strips import compose_strips_feathered
from .warping import warp_rgb_and_mask


@dataclass(frozen=True)
class MosaicConfig:
    lk: LKConfig = field(default_factory=lambda: LKConfig(num_iters=60, border=8))
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    blur_sigma: float = 1.0
    margin: int = 12
    alpha: float = 0.5
    coverage_threshold: float = 0.5
    pad_fraction: float = 0.10
    seam_lambda: float = 5e-4
    seam_mu: float = 2e-3
    seam_stride: int = 2
    max_seam_jump: int = 250
    feather_half_width: int = 40

    def __post_init__(self):
        if not 0 <= self.alpha <= 1: raise ValueError("alpha must be in [0,1]")
        if self.margin < 0 or self.seam_stride < 1: raise ValueError("Invalid canvas/seam settings")


def mosaic(frames, config=MosaicConfig(), *, allow_unreliable=False):
    """Return full, uncropped BGR canvas, coverage, and JSON-safe evidence.

    All motions in evidence refer to the direction-normalized sequence. The
    diagnostics gate is the sole added robustness feature. Bypassing it is an
    explicit option for investigating failures, not a fallback or repair.
    """
    if len(frames) < 2: raise ValueError("Need at least two frames")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 32:
        raise ValueError("Frames must be HxWx3 BGR images, at least 32 pixels per side")
    if any(f.shape != shape or f.dtype != np.uint8 for f in frames):
        raise ValueError("All frames must have equal shape and uint8 BGR dtype")
    gray = [preprocess_frame(f, blur_sigma=config.blur_sigma) for f in frames]
    frames, gray, direction = normalize_scan_direction(frames, gray, config.lk, blur_sigma=config.blur_sigma)
    pairwise, records = [], []
    for i, (a, b) in enumerate(zip(gray, gray[1:])):
        motion = estimate_rigid_lk_pyramids(a, b, config.lk)
        diagnostic = inspect_registration(a, b, motion, config.lk.border, config.diagnostics)
        records.append(dict(pair=i, motion=asdict(motion), diagnostic=diagnostic))
        if not diagnostic['accepted'] and not allow_unreliable:
            raise RegistrationError(i, diagnostic)
        pairwise.append(motion)
    ref = len(frames) // 2
    cumulative = accumulate_to_ref(pairwise, ref)
    h, w = gray[0].shape
    canvas_hw, shift = compute_canvas(cumulative, (h, w), config.margin)
    warped, masks = [], []
    for f, t in zip(frames, cumulative):
        im, mask = warp_rgb_and_mask(f, shift @ t.as_matrix(h, w), canvas_hw)
        warped.append(im); masks.append(mask)
    ranges = compute_frame_x_ranges(masks, config.coverage_threshold)
    seams = seams_for_view_scored_dp(warped, masks, ranges, config.alpha, canvas_hw[1],
                                   int(np.ceil(config.pad_fraction*w)), lam=config.seam_lambda,
                                   mu=config.seam_mu, stride=config.seam_stride)
    fallback = len(seams) > 1 and int(np.abs(np.diff(seams)).max()) > config.max_seam_jump
    if fallback: seams = seams_for_view(ranges, config.alpha)
    composite = compose_strips_feathered(warped, masks, seams, canvas_hw, config.feather_half_width)
    # Coverage of the selected hard strips (not just union-of-frames coverage).
    coverage = np.zeros(canvas_hw, bool)
    left = 0
    for i, mask in enumerate(masks):
        right = seams[i] if i < len(seams) else canvas_hw[1]
        if right > left: coverage[:, left:right] = mask[:, left:right]
        left = right
    evidence = dict(model="centered_2d_rigid", config=asdict(config), direction=direction,
                    reference_index=ref, pairs=records, cumulative=[asdict(t) for t in cumulative],
                    canvas_hw=list(canvas_hw), shift=shift.tolist(), seams=seams,
                    seam_fallback=bool(fallback), allow_unreliable=allow_unreliable,
                    coverage_fraction=float(coverage.mean()))
    if direction in ('TB','BT'):
        code = cv2.ROTATE_90_CLOCKWISE if direction == 'TB' else cv2.ROTATE_90_COUNTERCLOCKWISE
        composite = cv2.rotate(composite, code)
        coverage = cv2.rotate(coverage.astype(np.uint8), code).astype(bool)
    return composite, coverage, evidence
