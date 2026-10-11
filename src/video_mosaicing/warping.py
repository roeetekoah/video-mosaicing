import cv2
import numpy as np
from typing import Tuple

def warp_corners(T: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Warp the four image corners through a 3x3 homogeneous transform.

    Args:
        T: 3x3 homogeneous transform.
        h: Image height.
        w: Image width.

    Returns:
        Array of shape (4,2) containing warped (x,y) of corners:
        (0,0), (w-1,0), (w-1,h-1), (0,h-1).
    """
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64).T
    warped = T @ corners
    warped = warped[:2, :] / warped[2:3, :]
    return warped.T

def warp_rgb_and_mask(img_bgr: np.ndarray, T_canvas_from_img: np.ndarray, out_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp an RGB frame (BGR in OpenCV) into the panorama canvas and also compute a validity mask.

    We warp:
      - the BGR image using bilinear interpolation
      - a constant-255 mask using nearest neighbor (to preserve crisp validity)

    Args:
        img_bgr: Input image (H,W,3) uint8.
        T_canvas_from_img: 3x3 homography mapping input image coords -> canvas coords.
                           (Even though motion is rigid, homography warpPerspective is convenient.)
        out_hw: (out_h, out_w) canvas size.

    Returns:
        warped_bgr: (out_h, out_w, 3) uint8 warped frame.
        valid_mask: (out_h, out_w) boolean mask where warped pixels are valid (came from the frame).
    """
    out_h, out_w = out_hw
    warped = cv2.warpPerspective(img_bgr, T_canvas_from_img, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    mask = cv2.warpPerspective(ones, T_canvas_from_img, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped, mask > 0)
