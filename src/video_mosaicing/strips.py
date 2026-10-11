import numpy as np
from typing import List, Tuple

def compose_strips(warped: List[np.ndarray], masks: List[np.ndarray], seams: List[int], out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Build one panorama by strip compositing.

    We paste vertical strips in order:
      frame 0: x in [0, seams[0])
      frame i: x in [seams[i-1], seams[i]) for i=1..n-2
      last   : x in [seams[-1], W)

    Within each strip, we only copy pixels where that frame is valid (mask True).

    Args:
        warped: List of warped BGR frames on the common canvas, each (H,W,3) uint8.
        masks: List of boolean masks corresponding to warped frames, each (H,W).
        seams: List of seam x positions between consecutive frames, length n_frames-1.
        :param seams:
        :param warped:
        :param masks:
        :param out_hw:
    Returns:
        pano_bgr: (H,W,3) uint8 panorama image (BGR).
    """
    out_h, out_w = out_hw
    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    left = 0
    for i in range(len(warped)):
        right = seams[i] if i < len(seams) else out_w
        if right <= left:
            left = right
            continue
        strip = warped[i][:, left:right, :]
        m = masks[i][:, left:right]
        pano_slice = pano[:, left:right, :]
        pano_slice[m] = strip[m]
        pano[:, left:right, :] = pano_slice
        left = right
    return pano

def compose_strips_feathered(warped: List[np.ndarray], masks: List[np.ndarray], seams: List[int], out_hw: Tuple[int, int], blend_width: int=40) -> np.ndarray:
    """
    Strip compositing with optional feather blending around seams.

    - First, paste hard strips exactly like your current implementation.
    - Then, for each seam between frame i and i+1, blend in a vertical band
      [s-blend_width, s+blend_width].

    Args:
        warped: list of (H,W,3) uint8 warped frames (BGR)
        masks:  list of (H,W) bool masks
        seams:  list of seam x positions, length n-1
        out_hw: (H,W)
        blend_width: half-width of feather band in pixels. 0 disables blending.

    Returns:
        pano: (H,W,3) uint8
    """
    out_h, out_w = out_hw
    n = len(warped)
    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    left = 0
    for i in range(n):
        right = seams[i] if i < len(seams) else out_w
        if right <= left:
            left = right
            continue
        strip = warped[i][:, left:right, :]
        m = masks[i][:, left:right]
        pano_slice = pano[:, left:right, :]
        pano_slice[m] = strip[m]
        pano[:, left:right, :] = pano_slice
        left = right
    if blend_width <= 0 or len(seams) == 0:
        return pano
    pano_f = pano.astype(np.float32)
    for i, s in enumerate(seams):
        j = i + 1
        if j >= n:
            break
        x0 = max(0, s - blend_width)
        x1 = min(out_w, s + blend_width + 1)
        if x1 <= x0:
            continue
        A = warped[i][:, x0:x1, :].astype(np.float32)
        B = warped[j][:, x0:x1, :].astype(np.float32)
        mA = masks[i][:, x0:x1]
        mB = masks[j][:, x0:x1]
        both = mA & mB
        if not np.any(both):
            continue
        band_w = x1 - x0
        alpha = np.linspace(1.0, 0.0, band_w, dtype=np.float32)[None, :, None]
        blend = alpha * A + (1.0 - alpha) * B
        pano_band = pano_f[:, x0:x1, :]
        pano_band[both] = blend[both]
        pano_f[:, x0:x1, :] = pano_band
    return np.clip(pano_f, 0, 255).astype(np.uint8)
