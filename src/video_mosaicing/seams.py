import cv2
import numpy as np
from typing import List, Tuple

def compute_frame_x_ranges(mask_list, coverage_thresh=0.7):
    """
    For each warped frame mask, compute the leftmost and rightmost x-columns that contain valid pixels.

    We prefer "usable" columns (enough vertical coverage), but in real videos the very first / last frames
    or hard warps can create sparse masks that would otherwise yield an empty range and break seams.

    Strategy:
      1) Try the requested coverage_thresh.
      2) If no columns pass, progressively relax the threshold down to 0.
      3) If still empty (all-false mask), return (0, -1).

    Args:
        mask_list: list of (H,W) boolean masks.
        coverage_thresh: initial fraction-of-rows threshold.

    Returns:
        List of (x_min, x_max) per frame. If a frame has no valid pixels at all, returns (0, -1).
    """
    x_ranges = []
    ladder = [coverage_thresh, 0.5, 0.3, 0.15, 0.05, 0.0]
    for m in mask_list:
        H, W = m.shape
        if not np.any(m):
            x_ranges.append((0, -1))
            continue
        col_cov = m.sum(axis=0) / float(H)
        xs = None
        for th in ladder:
            good = col_cov >= th
            cand = np.where(good)[0]
            if cand.size > 0:
                xs = cand
                break
        if xs is None or xs.size == 0:
            cols = np.where(m.any(axis=0))[0]
            if cols.size == 0:
                x_ranges.append((0, -1))
            else:
                x_ranges.append((int(cols[0]), int(cols[-1])))
        else:
            x_ranges.append((int(xs[0]), int(xs[-1])))
    return x_ranges

def seams_for_view(x_ranges: List[Tuple[int, int]], alpha: float) -> List[int]:
    """
    Compute seam x-positions between each consecutive pair of frames for a given "view" parameter alpha.

    For frames i and i+1, define their x-coverage on the canvas as:
        [a0, a1] and [b0, b1]

    Their overlap (if any) is:
        [ov0, ov1] where ov0=max(a0,b0), ov1=min(a1,b1)

    The seam is chosen within the overlap:
        seam = ov0 + alpha*(ov1-ov0), alpha in [0,1]

    Interpretation:
      - alpha=0 picks left edge of overlap (more "left" view)
      - alpha=1 picks right edge of overlap (more "right" view)
      - alpha=0.5 picks middle overlap

    If there is no overlap, we fall back to a reasonable midpoint between extents.

    Args:
        x_ranges: List of (x_min, x_max) coverage for each frame on the canvas.
        alpha: View parameter in [0,1].

    Returns:
        seams: List of length (n_frames-1), where seams[i] is the seam between frame i and i+1.
    """
    seams: List[int] = []
    for i in range(len(x_ranges) - 1):
        a0, a1 = x_ranges[i]
        b0, b1 = x_ranges[i + 1]
        ov0 = max(a0, b0)
        ov1 = min(a1, b1)
        if ov0 <= ov1:
            s = int(round(ov0 + alpha * (ov1 - ov0)))
            pad = 20
            if ov1 - ov0 >= 2 * pad:
                s = int(round(np.clip(s, ov0 + pad, ov1 - pad)))
        else:
            s = int(round(0.5 * (a1 + b0)))
        seams.append(s)
    return seams

def _column_visual_cost(wi: np.ndarray, wj: np.ndarray, mi: np.ndarray, mj: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Per-column visual cost between two warped BGR frames over jointly-valid pixels.

    Returns an array of length (x1-x0+1) where each entry is the mean absolute RGB difference in that column
    averaged over rows where both masks are True. If a column has no jointly-valid pixels, cost is +inf.
    """
    x0 = int(x0)
    x1 = int(x1)
    assert wi.shape[:2] == mi.shape and wj.shape[:2] == mj.shape
    H = mi.shape[0]
    costs = np.full((x1 - x0 + 1,), np.inf, dtype=np.float64)
    for x in range(x0, x1 + 1):
        v = mi[:, x] & mj[:, x]
        if not np.any(v):
            continue
        a = wi[:, x, :].astype(np.float32)
        b = wj[:, x, :].astype(np.float32)
        diff = np.abs(a - b).mean(axis=1)
        costs[x - x0] = float(diff[v].mean())
    return costs

def seams_for_view_scored_dp(warped_bgr: List[np.ndarray], masks: List[np.ndarray], x_ranges: List[Tuple[int, int]], alpha: float, out_w: int, pad_x: int, *, lam: float=0.0005, mu: float=0.002, stride: int=2) -> List[int]:
    """
    DP / shortest-path over seam candidates across overlaps (frame index dimension).

    Returns seams list length n-1, one x per adjacent pair, compatible with strip compositing.

    Endpoint policy:
      - alpha == 0  -> return leftmost safe seams (lo)
      - alpha == 1  -> return rightmost safe seams (hi)
      This preserves the assignment's "extremes" requirement exactly.

    For 0 < alpha < 1:
      - build candidate x's per overlap (strided)
      - cost_i(x) = visual_disagreement_i(x) + lam*(x-target_i)^2
      - transition cost = mu*(x_next - x_cur)^2 with constraint x_next >= x_cur (monotone)

    Notes:
      - visual_disagreement is computed column-wise over the valid overlap mask,
        similar to seams_for_view_scored in your current file.
    """
    n = len(x_ranges)
    if n < 2:
        return []
    warped_gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for im in warped_bgr]
    cand_xs: List[np.ndarray] = []
    cand_costs: List[np.ndarray] = []
    safe_lo_hi: List[Tuple[int, int]] = []
    for i in range(n - 1):
        a0, a1 = x_ranges[i]
        b0, b1 = x_ranges[i + 1]
        ov0 = max(a0, b0)
        ov1 = min(a1, b1)
        if ov0 > ov1:
            gap_l = int(np.clip(min(a1, b0), 0, out_w - 1))
            gap_r = int(np.clip(max(a1, b0), 0, out_w - 1))
            if gap_l == gap_r:
                xs = np.array([gap_l], dtype=np.int32)
            else:
                k = max(3, int(np.ceil((gap_r - gap_l) / max(1, 10 * stride))))
                xs = np.linspace(gap_l, gap_r, num=k, dtype=np.float32)
                xs = np.unique(np.round(xs).astype(np.int32))
            target = int(round(gap_l + alpha * (gap_r - gap_l)))
            unary = lam * ((xs - target) ** 2).astype(np.float64)
            unary += 1.0
            cand_xs.append(xs)
            cand_costs.append(unary)
            safe_lo_hi.append((int(xs.min()), int(xs.max())))
            continue
        lo = ov0 + pad_x
        hi = ov1 - pad_x
        if lo > hi:
            lo2 = int(np.clip(ov0, 0, out_w - 1))
            hi2 = int(np.clip(ov1, 0, out_w - 1))
            if lo2 > hi2:
                s = int(np.clip(round(0.5 * (a1 + b0)), 0, out_w - 1))
                xs = np.array([s], dtype=np.int32)
                unary = np.array([1.0], dtype=np.float64)
            else:
                xs = np.arange(lo2, hi2 + 1, stride, dtype=np.int32)
                if xs.size == 0:
                    xs = np.array([int(round(0.5 * (lo2 + hi2)))], dtype=np.int32)
                vcost = _column_visual_cost(warped_bgr[i], warped_bgr[i + 1], masks[i], masks[i + 1], lo2, hi2)
                vcost_x = vcost[xs - lo2]
                target = int(round(lo2 + alpha * (hi2 - lo2)))
                unary = vcost_x + lam * ((xs - target) ** 2).astype(np.float64)
                if not np.isfinite(unary).any():
                    unary = lam * ((xs - target) ** 2).astype(np.float64) + 1.0
            cand_xs.append(xs)
            cand_costs.append(unary)
            safe_lo_hi.append((int(xs.min()), int(xs.max())))
            continue
        lo = int(np.clip(lo, 0, out_w - 1))
        hi = int(np.clip(hi, 0, out_w - 1))
        safe_lo_hi.append((lo, hi))
        xs = np.arange(lo, hi + 1, max(1, int(stride)), dtype=np.int32)
        target = float(lo + alpha * (hi - lo))
        g1 = warped_gray[i][:, lo:hi + 1]
        g2 = warped_gray[i + 1][:, lo:hi + 1]
        m = masks[i][:, lo:hi + 1] & masks[i + 1][:, lo:hi + 1]
        diff = np.abs(g1 - g2) * m.astype(np.float32)
        denom = m.sum(axis=0).astype(np.float32)
        full_cost = np.where(denom > 0, diff.sum(axis=0) / denom, np.inf).astype(np.float64)
        full_cost_sub = full_cost[::max(1, int(stride))][:xs.shape[0]]
        unary = full_cost_sub + lam * (xs.astype(np.float64) - target) ** 2
        cand_xs.append(xs)
        cand_costs.append(unary)
    if alpha <= 0.0:
        return [lo for lo, hi in safe_lo_hi]
    if alpha >= 1.0:
        return [hi for lo, hi in safe_lo_hi]
    K0 = cand_xs[0].shape[0]
    dp_prev = cand_costs[0].copy()
    back_ptrs: List[np.ndarray] = [np.full(K0, -1, dtype=np.int32)]
    for i in range(1, len(cand_xs)):
        xs_prev = cand_xs[i - 1].astype(np.float64)
        xs_cur = cand_xs[i].astype(np.float64)
        Kp, Kc = (xs_prev.shape[0], xs_cur.shape[0])
        dx = xs_cur[None, :] - xs_prev[:, None]
        trans = mu * dx ** 2
        trans = np.where(dx >= 0.0, trans, np.inf)
        total = dp_prev[:, None] + trans
        if not np.isfinite(total).any():
            dx = xs_cur[None, :] - xs_prev[:, None]
            trans = mu * dx ** 2
            total = dp_prev[:, None] + trans
        argmin = np.argmin(total, axis=0).astype(np.int32)
        dp_cur = cand_costs[i] + total[argmin, np.arange(Kc)]
        back_ptrs.append(argmin)
        dp_prev = dp_cur
    seams: List[int] = [0] * len(cand_xs)
    k = int(np.argmin(dp_prev))
    seams[-1] = int(cand_xs[-1][k])
    for i in range(len(cand_xs) - 1, 0, -1):
        k = int(back_ptrs[i][k])
        seams[i - 1] = int(cand_xs[i - 1][k])
    return seams
