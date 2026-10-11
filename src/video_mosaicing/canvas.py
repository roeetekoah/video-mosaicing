import numpy as np
from typing import Tuple, List
from .data_structures import RigidTransform
from .warping import warp_corners

def compute_canvas(cum_T: List[RigidTransform], frame_hw: Tuple[int, int], margin: int=50) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Compute a panorama canvas size that fits all warped frames, plus a translation ("shift") to make coords positive.

    Steps:
      1) Warp each frame's corners using its transform to the reference frame.
      2) Take min/max extents over all corners.
      3) Expand by `margin`.
      4) Build a shift transform that translates min_x/min_y to 0.

    Args:
        cum_T: List of 3x3 transforms mapping frame i -> reference frame.
        frame_hw: (height, width) of the original frames (assumed all equal).
        margin: Extra pixels around the bounds to avoid tight cropping.

    Returns:
        out_hw: (out_h, out_w) of the panorama canvas.
        shift: 3x3 transform to translate reference coords into canvas coords.
               If a point in ref coords is (x,y), canvas coords are shift @ [x,y,1]^T.
    """
    h, w = frame_hw
    all_xy = []
    for T in cum_T:
        M = T.as_matrix(h, w)
        pts = warp_corners(M, h, w)
        all_xy.append(pts)
    all_xy = np.vstack(all_xy)
    min_x, min_y = (float(np.min(all_xy[:, 0])), float(np.min(all_xy[:, 1])))
    max_x, max_y = (float(np.max(all_xy[:, 0])), float(np.max(all_xy[:, 1])))
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    out_w = int(np.ceil(max_x - min_x + 1))
    out_h = int(np.ceil(max_y - min_y + 1))
    shift = np.eye(3, dtype=np.float64)
    shift[0, 2] = -min_x
    shift[1, 2] = -min_y
    return ((out_h, out_w), shift)
