import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
from .data_structures import RigidTransform, LKConfig
from .preprocess import preprocess_frame
from .rigid_transformations import estimate_pairwise

def _rotate_bgr_90(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """
    Rotate frames by k*90 degrees CCW.
    k in {0,1,2,3}
    """
    k %= 4
    if k == 0:
        return frames
    code = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}[k]
    return [cv2.rotate(f, code) for f in frames]

def dominant_scan_direction(pairwise: List[RigidTransform], h: int, w: int, dominance_ratio: float=1.2, eps: float=0.001) -> str:
    """
    Infer scan direction robustly using the displacement of the image center
    under each pairwise rigid transform matrix.

    This avoids being fooled when dy is inflated by rotation/parallax.
    """
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    p = np.array([cx, cy, 1.0], dtype=np.float64)
    deltas = []
    for t in pairwise:
        M = t.as_matrix(h, w).astype(np.float64)
        q = M @ p
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        deltas.append((dx, dy))
    deltas = np.array(deltas, dtype=np.float64)
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    score_x = float(np.median(np.abs(dx)))
    score_y = float(np.median(np.abs(dy)))
    if score_x < eps and score_y < eps:
        return 'LR'
    if score_x >= dominance_ratio * score_y:
        return 'LR' if np.median(dx) <= 0 else 'RL'
    if score_y >= dominance_ratio * score_x:
        return 'TB' if np.median(dy) <= 0 else 'BT'
    if score_x >= score_y:
        return 'LR' if np.median(dx) <= 0 else 'RL'
    else:
        return 'TB' if np.median(dy) <= 0 else 'BT'

def dominant_scan_direction_phase(frames_gray: List[np.ndarray], *, max_probe_pairs: int=12, downscale: float=0.25, dominance_ratio: float=1.3, eps: float=0.2) -> str:
    """
    Robust scan direction inference using phase correlation on grayscale frames.

    Returns one of: "LR", "RL", "TB", "BT"

    Interpretation:
      - phaseCorrelate(a,b) returns shift of b relative to a (content motion).
      - If content moves LEFT (dx < 0), camera panned RIGHT => "LR"
      - If content moves RIGHT (dx > 0), camera panned LEFT  => "RL"
      Similarly for dy.
    """
    if len(frames_gray) < 2:
        return 'LR'
    n = len(frames_gray)
    k = min(max_probe_pairs, n - 1)
    idx = np.linspace(0, n - 2, k).astype(int)
    shifts = []
    for i in idx:
        a = frames_gray[i]
        b = frames_gray[i + 1]
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 1.5)
        b = cv2.GaussianBlur(b, (0, 0), 1.5)
        if downscale != 1.0:
            a = cv2.resize(a, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
        if resp >= 0.2 and np.isfinite(dx) and np.isfinite(dy):
            shifts.append((dx, dy))
    if len(shifts) == 0:
        return 'LR'
    shifts = np.array(shifts, dtype=np.float64)
    dx = shifts[:, 0]
    dy = shifts[:, 1]
    score_x = float(np.median(np.abs(dx)))
    score_y = float(np.median(np.abs(dy)))
    if score_x < eps and score_y < eps:
        return 'LR'
    if score_x >= dominance_ratio * score_y:
        return 'LR' if np.median(dx) < 0 else 'RL'
    if score_y >= dominance_ratio * score_x:
        return 'TB' if np.median(dy) < 0 else 'BT'
    if score_x >= score_y:
        return 'LR' if np.median(dx) < 0 else 'RL'
    else:
        return 'TB' if np.median(dy) < 0 else 'BT'

def normalize_scan_direction(frames_bgr: List[np.ndarray], frames_gray: List[np.ndarray], cfg: LKConfig, max_probe_pairs: int=8, *, blur_sigma: float) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    """
    Normalize input frames so that dominant scan direction is Left -> Right.

    Returns:
        frames_bgr_norm
        frames_gray_norm
        direction   ∈ {'LR', 'RL', 'TB', 'BT'}
    """
    d1 = dominant_scan_direction_phase(frames_gray, max_probe_pairs=12, downscale=0.25)
    d2 = dominant_scan_direction_phase(frames_gray, max_probe_pairs=12, downscale=0.5)
    if d1 == d2:
        direction = d1
    else:
        K = min(max_probe_pairs, len(frames_gray) - 1)
        probe_pairwise = estimate_pairwise(frames_gray[:K + 1], cfg)
        h0, w0 = frames_gray[0].shape[:2]
        direction = dominant_scan_direction(probe_pairwise, h=h0, w=w0)
    if direction == 'RL':
        frames_bgr = frames_bgr[::-1]
        frames_gray = frames_gray[::-1]
    elif direction == 'TB':
        frames_bgr = _rotate_bgr_90(frames_bgr, k=1)
        frames_gray = [preprocess_frame(f, blur_sigma=blur_sigma) for f in frames_bgr]
    elif direction == 'BT':
        frames_bgr = _rotate_bgr_90(frames_bgr, k=3)
        frames_gray = [preprocess_frame(f, blur_sigma=blur_sigma) for f in frames_bgr]
    return (frames_bgr, frames_gray, direction)

def undo_scan_normalization(panoramas: List[Image.Image], direction: str) -> List[Image.Image]:
    """
    Undo frame normalization for output panoramas.
    """
    if direction == 'TB':
        panoramas = [im.rotate(-90, expand=True) for im in panoramas]
    elif direction == 'BT':
        panoramas = [im.rotate(90, expand=True) for im in panoramas]
    return panoramas
