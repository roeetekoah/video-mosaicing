from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class RigidTransform:
    """2D rigid transform: rotation theta + translation (dx, dy)."""
    dx: float
    dy: float
    theta: float

    def as_matrix(self, h: int, w: int) -> np.ndarray:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        c, s = (np.cos(self.theta), np.sin(self.theta))
        tx = self.dx + cx - (c * cx - s * cy)
        ty = self.dy + cy - (s * cx + c * cy)
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]])

@dataclass(frozen=True)
class LKConfig:
    num_levels: int = 4
    num_iters: int = 30
    border: int = 5
    damping: float = 1e-06
    eps2: float = 1e-10
    pyr_sigma: float = 1.0

    def __post_init__(self):
        if self.num_levels < 1 or self.num_iters < 1 or self.border < 0:
            raise ValueError("levels/iterations must be positive; border nonnegative")
        if self.damping < 0 or self.eps2 <= 0 or self.pyr_sigma < 0:
            raise ValueError("Invalid numerical LK settings")
