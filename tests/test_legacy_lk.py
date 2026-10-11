import numpy as np
import cv2
import pytest

from video_mosaicing.data_structures import LKConfig
from video_mosaicing.lucas_kanade import _rigid_affine_centered, _warp_affine, estimate_rigid_lk_pyramids


def _make_test_image(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)
    cv2.circle(img, (w // 3, h // 2), 12, 1.0, -1)
    cv2.rectangle(img, (w // 2, h // 3), (w // 2 + 18, h // 3 + 25), 0.7, -1)
    cv2.line(img, (10, 10), (w - 10, h - 20), 0.5, 2)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return img


def _warp_forward(img: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    h, w = img.shape
    M = _rigid_affine_centered(dx, dy, theta, h, w)
    return _warp_affine(img, M, w, h)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


@pytest.fixture
def cfg() -> LKConfig:
    return LKConfig(num_levels=4, num_iters=30, pyr_sigma=1.0, border=8, damping=1e-6, eps2=1e-12)


def test_rigid_affine_centered_keeps_center_fixed() -> None:
    h, w = 100, 140
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = _rigid_affine_centered(0.0, 0.0, np.deg2rad(10.0), h, w)
    x2 = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    y2 = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    assert abs(x2 - cx) < 1e-4
    assert abs(y2 - cy) < 1e-4


@pytest.mark.parametrize("h,w,dx_gt,dy_gt,th_gt,dx_tol,dy_tol,th_tol", [
    (120, 160, 5.0, -3.0, 0.0, 0.5, 0.5, 2e-3),
    (160, 160, 2.0,  1.0, np.deg2rad(2.0), 0.7, 0.7, np.deg2rad(0.3)),
])
def test_lk_recovers_motion(cfg: LKConfig, h: int, w: int, dx_gt: float, dy_gt: float, th_gt: float, dx_tol: float, dy_tol: float, th_tol: float) -> None:
    Im1 = _make_test_image(h, w)
    Im2 = _warp_forward(Im1, dx_gt, dy_gt, th_gt)
    motion = estimate_rigid_lk_pyramids(Im1, Im2, cfg)
    dx, dy, th = motion.dx, motion.dy, motion.theta
    assert abs(dx - dx_gt) < dx_tol
    assert abs(dy - dy_gt) < dy_tol
    assert abs(th - th_gt) < th_tol


def test_lk_identity(cfg: LKConfig) -> None:
    Im1 = _make_test_image(120, 160)
    motion = estimate_rigid_lk_pyramids(Im1, Im1.copy(), cfg)
    dx, dy, th = motion.dx, motion.dy, motion.theta
    assert abs(dx) < 1e-3
    assert abs(dy) < 1e-3
    assert abs(th) < 1e-6


def test_lk_reduces_error(cfg: LKConfig) -> None:
    Im1 = _make_test_image(120, 160)
    Im2 = _warp_forward(Im1, 6.0, -4.0, 0.0)
    e0 = _mse(Im2, _warp_forward(Im1, 0.0, 0.0, 0.0))
    motion = estimate_rigid_lk_pyramids(Im1, Im2, cfg)
    dx, dy, th = motion.dx, motion.dy, motion.theta
    e1 = _mse(Im2, _warp_forward(Im1, dx, dy, th))
    assert e1 < 0.2 * e0

