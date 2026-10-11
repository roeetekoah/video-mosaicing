"""Heuristic observability and residual checks; these are not probability estimates."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .data_structures import RigidTransform


@dataclass(frozen=True)
class DiagnosticConfig:
    gradient_threshold: float = 0.005
    min_texture_fraction: float = 0.01
    min_eigenvalue_ratio: float = 1e-5
    min_overlap_fraction: float = 0.5
    max_mae: float = 0.06


def inspect_registration(source, target, motion: RigidTransform, border=8,
                         config=DiagnosticConfig()):
    """Measure undamped, radius-normalized J^T J and true valid-overlap MAE.

    Sobel / 8 approximates intensity change per pixel for the 3x3 kernel.
    Dividing the rotation column by image radius compares its parameter in
    approximate edge-displacement pixels, rather than mixing pixels and radians.
    Sparse support is checked separately: one bright pixel can have full rank.
    """
    h, w = target.shape
    interior = np.zeros((h, w), bool)
    if h > 2 * border and w > 2 * border:
        interior[border:h-border if border else h, border:w-border if border else w] = True
    ix = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3) / 8
    iy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3) / 8
    y, x = np.indices((h, w), dtype=np.float32)
    x -= (w - 1) / 2
    y -= (h - 1) / 2
    j = np.stack([ix, iy, (-y * ix + x * iy) / max(h, w)], axis=-1)[interior].astype(float)
    eigenvalues = np.linalg.eigvalsh(j.T @ j / max(1, len(j)))
    ratio = max(0., float(eigenvalues[0])) / max(float(eigenvalues[-1]), 1e-20)
    texture = float(np.mean(np.hypot(ix, iy)[interior] > config.gradient_threshold)) if len(j) else 0.
    matrix = motion.as_matrix(h, w)[:2]
    warped = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_LINEAR)
    # Bilinear interpolation must draw only from actual source pixels.
    support = cv2.warpAffine(np.ones_like(source), matrix, (w, h), flags=cv2.INTER_LINEAR) > .999
    valid = support & interior
    overlap = float(valid.sum() / max(1, interior.sum()))
    mae = float(np.mean(np.abs(warped[valid] - target[valid]))) if valid.any() else None
    flags = []
    if not len(j): flags.append("empty_interior")
    if texture < config.min_texture_fraction: flags.append("low_texture")
    if ratio < config.min_eigenvalue_ratio: flags.append("rank_deficient")
    if overlap < config.min_overlap_fraction: flags.append("low_overlap")
    if mae is None or mae > config.max_mae: flags.append("high_residual")
    return dict(accepted=not flags, flags=flags, texture_fraction=texture,
                eigenvalue_ratio=ratio, overlap_fraction=overlap, mae=mae,
                thresholds=asdict(config))


class RegistrationError(ValueError):
    def __init__(self, pair_index, diagnostic):
        self.pair_index = pair_index
        self.diagnostic = diagnostic
        super().__init__(f"Pair {pair_index}: unreliable registration ({', '.join(diagnostic['flags'])})")
