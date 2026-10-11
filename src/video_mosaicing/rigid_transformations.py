import numpy as np
from typing import List, Optional
from .data_structures import RigidTransform, LKConfig
from .lucas_kanade import estimate_rigid_lk_pyramids

def invert_rigid_params(T: RigidTransform) -> RigidTransform:
    c, s = (np.cos(T.theta), np.sin(T.theta))
    dx = -(c * T.dx + s * T.dy)
    dy = -(-s * T.dx + c * T.dy)
    return RigidTransform(dx, dy, -T.theta)

def compose_rigid(A: RigidTransform, B: RigidTransform) -> RigidTransform:
    c, s = (np.cos(A.theta), np.sin(A.theta))
    dx = A.dx + c * B.dx - s * B.dy
    dy = A.dy + s * B.dx + c * B.dy
    return RigidTransform(dx, dy, A.theta + B.theta)

def estimate_pairwise(frames_gray: List[np.ndarray], cfg: LKConfig) -> List[RigidTransform]:
    """
    Estimate rigid motion between consecutive grayscale frames using your LK implementation.

    For K frames, returns K-1 transforms where transform i maps:
        frame_i  -> frame_{i+1}

    Args:
        :param frames_gray: List of grayscale float32 frames (H, W), typically in [0,1].
        :param cfg: LK configuration (pyramid levels, iterations, etc.).
    Returns:
        List of RigidTransform objects of length len(frames_gray)-1.
        
    """
    pairwise: List[RigidTransform] = []
    for i in range(len(frames_gray) - 1):
        motion = estimate_rigid_lk_pyramids(frames_gray[i], frames_gray[i + 1], cfg)
        pairwise.append(motion)
    return pairwise

def accumulate_to_0_ref(pairwise: List[RigidTransform]) -> List[RigidTransform]:
    """
    Accumulate pairwise transforms into transforms that map each frame to a common reference frame.

    With ref_index=0 (the default), we compute:
      cum[0] = I
      cum[i+1] = cum[i] @ inv(T_i->i+1)
    so that cum[i] maps:
      frame_i  -> frame_0

    Notes:
      - This convention ("map everything to frame 0") makes warping simple.
      - You could choose a different reference (e.g., middle frame), but the rest of the pipeline
        must be adjusted accordingly.

    Args:
        pairwise: List of RigidTransform, length K-1, each mapping frame i -> frame i+1.

    Returns:
        cum: List of 3x3 transforms, length K, where cum[i] maps frame i -> reference frame.
    """
    cum = [RigidTransform(0.0, 0.0, 0.0)]
    for Ti in pairwise:
        prev = cum[-1]
        invTi = invert_rigid_params(Ti)
        cum.append(compose_rigid(prev, invTi))
    return cum

def accumulate_to_ref(pairwise: List[RigidTransform], ref_index: int=0) -> List[RigidTransform]:
    """
    Return cum[i] mapping frame i -> frame ref_index.

    pairwise[k] maps frame k -> frame k+1.
    """
    K = len(pairwise) + 1
    if not 0 <= ref_index < K:
        raise ValueError('ref_index out of range')
    I = RigidTransform(0.0, 0.0, 0.0)
    cum: List[Optional[RigidTransform]] = [None] * K
    cum[ref_index] = I
    for i in range(ref_index - 1, -1, -1):
        cum[i] = compose_rigid(cum[i + 1], pairwise[i])
    for i in range(ref_index + 1, K):
        inv_prev = invert_rigid_params(pairwise[i - 1])
        cum[i] = compose_rigid(cum[i - 1], inv_prev)
    return [c for c in cum]
