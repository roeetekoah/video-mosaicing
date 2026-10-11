import cv2
import numpy as np
from typing import Tuple
from .data_structures import LKConfig, RigidTransform
from .preprocess import build_gaussian_pyramid
'\nLK Stage: data flow and call graph\nestimate_rigid_lk_pyramids(img0, img1, cfg)\n  ├─ build_gaussian_pyramid(img0, cfg.num_levels, sigma=cfg.pyr_sigma) -> pyr0[level]\n  ├─ build_gaussian_pyramid(img1, cfg.num_levels, sigma=cfg.pyr_sigma) -> pyr1[level]\n  └─ for level = L..0:\n       ├─ (dx, dy) *= 2  [between pyramid levels]\n       └─ _lk_level_update(Im1=pyr0[level], Im2=pyr1[level], dx, dy, theta, cfg)\n            ├─ _centered_coords(h, w) -> xs, ys\n            ├─ _interior_mask(h, w, cfg.border) -> mask\n            ├─ _sobel_grads(Im2) -> Ix, Iy\n            ├─ _jacobians_rigid(Ix, Iy, xs, ys) -> J_dx, J_dy, J_dtheta\n            ├─ _stack_jacobian_cols(J_*, mask) -> j0, j1, j2\n            ├─ _build_normal_matrix(j0, j1, j2, cfg.damping) -> A  [constant per level]\n            └─ for iter = 1..cfg.num_iters:\n                 ├─ _rigid_affine(dx, dy, theta) -> M (2x3)\n                 ├─ _warp_affine(Im1, M, w, h) -> Im3  [warped Im1]\n                 ├─ It = Im2 - Im3\n                 ├─ _build_rhs(j0, j1, j2, It, mask) -> b\n                 ├─ dp = solve(A, b)\n                 └─ update: dx += dp[0], dy += dp[1], theta += dp[2]\n'

def _centered_coords(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create centered image coordinate grids.

    Coordinates are defined so that (0,0) is at the image center,
    which is required for numerical stability when estimating rotation
    in Lucas–Kanade (course recommendation).

    Args:
        h: image height
        w: image width

    Returns:
        xs, ys: float32 arrays of shape (h, w) with centered x and y coordinates
    """
    xs = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0)[None, :].repeat(h, axis=0)
    ys = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0)[:, None].repeat(w, axis=1)
    return (xs, ys)

def _interior_mask(h: int, w: int, border: int) -> np.ndarray:
    """
    Create a mask that excludes image borders.

    Pixels near image borders have unreliable gradient estimates and
    may not overlap after warping. The LK reminder explicitly advises
    ignoring them.

    Args:
        h: image height
        w: image width
        border: number of pixels to exclude on each side

    Returns:
        mask: boolean array of shape (h, w), True for valid pixels
    """
    mask = np.zeros((h, w), dtype=bool)
    if h > 2 * border and w > 2 * border:
        mask[border:h - border, border:w - border] = True
    return mask

def _sobel_grads(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial image gradients using Sobel filters.

    Gradients are computed on the *fixed* image (Im2) once per pyramid level,
    as specified in the LK reminder. This is what allows us to compute them
    only once - not computing it over the warped image.

    Args:
        img: grayscale float32 image

    Returns:
        Ix, Iy: horizontal and vertical gradients
    """
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return (Ix, Iy)

def _rigid_affine_centered(dx: float, dy: float, theta: float, h: int, w: int) -> np.ndarray:
    """
    Construct a 2x3 affine matrix for a rigid transform that rotates about
    the image center and then translates.

    This function makes the warping step consistent with the Lucas–Kanade
    Jacobian, which is derived using *centered image coordinates*.

    Mathematical model:
        Let c = (cx, cy) be the image center.
        The desired warp is:
            x' = R(theta) * (x - c) + c + t
        where:
            R(theta) is the 2D rotation matrix,
            t = (dx, dy) is the translation.

        Expanding:
            x' = R(theta) * x + (t + c - R(theta) * c)

        Therefore the affine matrix has:
            linear part      = R(theta)
            translation part = t + c - R(theta) * c

    Coordinate conventions:
        - Input pixel coordinates (x, y) are in standard image coordinates
          with origin at the top-left.
        - Rotation is performed about the *image center*, not the origin.
        - Translation (dx, dy) is applied after rotation.
        - This matches the centered-coordinate Jacobian used in LK.

    Args:
        dx: horizontal translation (pixels)
        dy: vertical translation (pixels)
        theta: rotation angle in radians (positive = clockwise in image coordinates (y downward))
        h: image height
        w: image width

    Returns:
        M: 2x3 float32 affine matrix suitable for cv2.warpAffine,
           mapping Im1 coordinates toward Im2 coordinates.
    """
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    tx = dx + cx - (c * cx - s * cy)
    ty = dy + cy - (s * cx + c * cy)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

def _warp_affine(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Warp an image using a rigid affine transform.

    Warping uses reflection padding to avoid introducing artificial
    black regions near boundaries.

    Args:
        img: input image to warp
        M: 2x3 affine matrix
        out_w: output width
        out_h: output height

    Returns:
        warped image of shape (out_h, out_w)
    """
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def _jacobians_rigid(Ix: np.ndarray, Iy: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Jacobian images for rigid motion parameters.

    Motion model (small angle approximation):
        u(x,y) = -theta * y + dx
        v(x,y) =  theta * x + dy

    Args:
        Ix, Iy: image gradients of Im2
        xs, ys: centered coordinate grids

    Returns:
        J_dx, J_dy, J_dtheta: Jacobian images for dx, dy, theta
    """
    J_dx = Ix
    J_dy = Iy
    J_dtheta = -ys * Ix + xs * Iy
    return (J_dx, J_dy, J_dtheta)

def _flatten_jacobians(J_dx: np.ndarray, J_dy: np.ndarray, J_dtheta: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and flatten Jacobian columns over valid pixels.

    Args:
        J_dx, J_dy, J_dtheta: Jacobian images
        mask: boolean mask of valid pixels

    Returns:
        j_dx, j_dy, j_dtheta: 1D arrays used to construct normal equations
    """
    j_dx = J_dx[mask].reshape(-1)
    j_dy = J_dy[mask].reshape(-1)
    j_dtheta = J_dtheta[mask].reshape(-1)
    return (j_dx, j_dy, j_dtheta)

def _build_normal_matrix(j_dx: np.ndarray, j_dy: np.ndarray, j_dtheta: np.ndarray, damping: float) -> np.ndarray:
    """
    Build the normal matrix A = J^T J for rigid LK.

    A is constant per pyramid level and reused across iterations.

    Based on my personal GN algorithm "crash course":
    Gauss-Newton normal equations

    Args:
        j_dx, j_dy, j_dtheta: flattened Jacobian columns
        damping: small Tikhonov regularization (IML) term for numerical stability

    Returns:
        A: 3x3 normal matrix
    """
    A = np.array([[np.dot(j_dx, j_dx), np.dot(j_dx, j_dy), np.dot(j_dx, j_dtheta)], [np.dot(j_dy, j_dx), np.dot(j_dy, j_dy), np.dot(j_dy, j_dtheta)], [np.dot(j_dtheta, j_dx), np.dot(j_dtheta, j_dy), np.dot(j_dtheta, j_dtheta)]], dtype=np.float64)
    if damping > 0:
        A += damping * np.eye(3, dtype=np.float64)
    return A

def _build_rhs(j_dx: np.ndarray, j_dy: np.ndarray, j_dtheta: np.ndarray, residual_t: np.ndarray, border_mask: np.ndarray) -> np.ndarray:
    """
    Build the right-hand side vector b = -J^T It.
    Keeping in mind that we solve: JTJΔθ=−JTr
    While J is the jacobian of every pixel's residual error,
    in relation to theta=(dx, dy, alpha). r(dx, dy, alpha) is the residual:
    here called It.

    Args:
        j_dx, j_dy, j_dtheta: flattened Jacobian columns
        residual_t: temporal difference image (Im2 - Im3)
        border_mask: valid-pixel mask

    Returns:
        b: 3x1 RHS vector
    """
    residual_t = residual_t[border_mask].reshape(-1).astype(np.float64)
    return np.array([-np.dot(j_dx, residual_t), -np.dot(j_dy, residual_t), -np.dot(j_dtheta, residual_t)], dtype=np.float64)

def _lk_level_update(Im1: np.ndarray, Im2: np.ndarray, dx: float, dy: float, theta: float, cfg: LKConfig) -> Tuple[float, float, float]:
    """
    Perform iterative global LK updates at a single pyramid level.

    Args:
        Im1: source image at this pyramid level
        Im2: target image at this pyramid level
        dx, dy, theta: initial motion estimate for this level
        cfg: LK configuration parameters

    Returns:
        Updated (dx, dy, theta) after convergence or max iterations
    """
    h, w = Im2.shape
    xs, ys = _centered_coords(h, w)
    mask = _interior_mask(h, w, cfg.border)
    if not np.any(mask):
        return (dx, dy, theta)
    Ix, Iy = _sobel_grads(Im2)
    J_dx, J_dy, J_dtheta = _jacobians_rigid(Ix, Iy, xs, ys)
    j_dx, j_dy, j_dtheta = _flatten_jacobians(J_dx, J_dy, J_dtheta, mask)
    JT_J = _build_normal_matrix(j_dx, j_dy, j_dtheta, cfg.damping)
    for _ in range(cfg.num_iters):
        motion = _rigid_affine_centered(dx, dy, theta, h, w)
        warped_I1 = _warp_affine(Im1, motion, w, h)
        residual_t = (Im2 - warped_I1).astype(np.float32)
        minus_JT_r = _build_rhs(j_dx, j_dy, j_dtheta, residual_t, mask)
        dp = np.linalg.solve(JT_J, minus_JT_r)
        dx, dy, theta = (dx + float(dp[0]), dy + float(dp[1]), theta + float(dp[2]))
        if float(dp @ dp) < cfg.eps2:
            break
    return (dx, dy, theta)

def _upscale_motion(dx: float, dy: float, theta: float, level: int, top_level: int) -> Tuple[float, float, float]:
    """When moving from coarser->finer: translations double, theta stays."""
    if level == top_level:
        return (dx, dy, theta)
    return (2.0 * dx, 2.0 * dy, theta)

def estimate_rigid_lk_pyramids(img0: np.ndarray, img1: np.ndarray, cfg: LKConfig=LKConfig()) -> RigidTransform:
    """
    Estimate rigid motion between two images using global, iterative,
    pyramidal Lucas–Kanade.

    This is the canonical LK implementation aligned with the course
    reminder and assignment constraints.

    Args:
        :param img0: reference grayscale image (float32, [0,1])
        :param img1: target grayscale image (float32, [0,1])
        :param cfg: LK configuration
    Returns:
        RigidTransform mapping img0 -> img1

    """
    if img0.ndim != 2 or img1.ndim != 2:
        raise ValueError('estimate_rigid_lk_pyramids expects 2D grayscale images.')
    if img0.shape != img1.shape:
        raise ValueError('img0 and img1 must have the same shape.')
    if img0.dtype != np.float32 or img1.dtype != np.float32:
        raise ValueError("LK expects float32 grayscale images in [0,1]")
    if not np.isfinite(img0).all() or not np.isfinite(img1).all():
        raise ValueError("Images must be finite")
    if min(img0.min(), img1.min()) < 0 or max(img0.max(), img1.max()) > 1:
        raise ValueError("Images must be in [0,1]")
    pyr0 = build_gaussian_pyramid(img0, cfg.num_levels, sigma=cfg.pyr_sigma)
    pyr1 = build_gaussian_pyramid(img1, cfg.num_levels, sigma=cfg.pyr_sigma)
    dx, dy, theta = (0.0, 0.0, 0.0)
    for level in range(cfg.num_levels - 1, -1, -1):
        dx, dy, theta = _upscale_motion(dx, dy, theta, level, cfg.num_levels - 1)
        dx, dy, theta = _lk_level_update(pyr0[level], pyr1[level], dx, dy, theta, cfg)
    return RigidTransform(dx=dx, dy=dy, theta=theta)
