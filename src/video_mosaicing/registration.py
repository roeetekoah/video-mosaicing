"""Robust, spatially checked motion with reserved correspondence evaluation."""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import GeometryError, checked_homography, project, symmetric_pair


class MotionError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


@dataclass(frozen=True)
class RegistrationConfig:
    features: int = 2400
    ratio: float = .75
    ransac_px: float = 2.0
    min_matches: int = 24
    min_inliers: int = 15
    min_cells: int = 4
    min_span_x: float = .30
    min_span_y: float = .20
    max_median_px: float = 1.5
    min_validation_fraction: float = .50
    seed: int = 47
    model: str = "auto"

    def __post_init__(self):
        if self.model not in ("auto","rigid","similarity","affine","homography"):
            raise ValueError("Unknown registration model")
        if self.features < 32 or not 0 < self.ratio < 1 or self.min_matches < 12:
            raise ValueError("Invalid feature settings")


@dataclass
class Features:
    xy: np.ndarray
    descriptors: np.ndarray | None


def detect(frame, cfg=RegistrationConfig()):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    sift = cv2.SIFT_create(nfeatures=cfg.features)
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    xy = np.array([p.pt for p in keypoints],float).reshape(-1,2)
    return Features(xy,descriptors)


def correspondences(a,b,cfg):
    if a.descriptors is None or b.descriptors is None or min(len(a.xy),len(b.xy)) < cfg.min_matches:
        raise MotionError("Insufficient textured features",dict(features=[len(a.xy),len(b.xy)]))
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(a.descriptors,b.descriptors,k=2)
    backward = matcher.knnMatch(b.descriptors,a.descriptors,k=2)
    reverse = {m.queryIdx:m.trainIdx for pair in backward if len(pair)==2 for m,n in [pair] if m.distance < cfg.ratio*n.distance}
    good = [m for pair in forward if len(pair)==2 for m,n in [pair]
            if m.distance < cfg.ratio*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    if len(good) < cfg.min_matches:
        raise MotionError("Too few mutually consistent correspondences",dict(matches=len(good)))
    return np.array([a.xy[m.queryIdx] for m in good]),np.array([b.xy[m.trainIdx] for m in good])


def fit_rigid(a,b):
    ac,bc = a.mean(0),b.mean(0)
    u,_,vt = np.linalg.svd((a-ac).T@(b-bc))
    r = vt.T@u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1; r = vt.T@u.T
    m = np.eye(3); m[:2,:2] = r; m[:2,2] = bc-r@ac
    return m


def support(points,hw):
    h,w = hw
    if not len(points): return dict(cells=0,span_x=0.,span_y=0.,hull_fraction=0.)
    cells = np.floor(points/np.array([w/4,h/3])).astype(int)
    cells = np.clip(cells,[0,0],[3,2])
    hull = cv2.convexHull(points.astype(np.float32))
    return dict(cells=int(len(np.unique(cells,axis=0))),
                span_x=float(np.ptp(points[:,0])/w),span_y=float(np.ptp(points[:,1])/h),
                hull_fraction=float(cv2.contourArea(hull)/(w*h)))


def fit_candidate(model,a,b,cfg):
    cv2.setRNGSeed(cfg.seed)
    if model in ("rigid","similarity"):
        fit,mask = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=cfg.ransac_px,maxIters=3000,confidence=.995,refineIters=10)
        if fit is None: return None
        if model == "rigid": return fit_rigid(a[mask.ravel()>0],b[mask.ravel()>0])
        return np.vstack((fit,[0,0,1]))
    if model == "affine":
        fit,_ = cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=cfg.ransac_px,maxIters=3000,confidence=.995,refineIters=10)
        return None if fit is None else np.vstack((fit,[0,0,1]))
    fit,_ = cv2.findHomography(a,b,cv2.RANSAC,cfg.ransac_px,maxIters=3000,confidence=.995)
    return fit


def estimate_features(a,b,hw,cfg=RegistrationConfig(),*,strip_alpha=None):
    aa,bb = correspondences(a,b,cfg)
    # Evaluation points never enter RANSAC or a post-selection refit.
    order = np.random.default_rng(cfg.seed).permutation(len(aa))
    ntrain = int(.70*len(aa))
    train,test = order[:ntrain],order[ntrain:]
    candidates, accepted = [],[]
    models = ("rigid","similarity","affine","homography") if cfg.model=="auto" else (cfg.model,)
    for model in models:
        m = fit_candidate(model,aa[train],bb[train],cfg)
        if m is None: continue
        try:
            m = checked_homography(m,hw)
            predicted = project(m,aa)
        except GeometryError: continue
        err = np.linalg.norm(predicted-bb,axis=1)
        inside = err[train] <= cfg.ransac_px
        spatial = support(aa[train][inside],hw)
        target_spatial = support(bb[train][inside],hw)
        validation = err[test]
        metric = dict(median_px=float(np.median(validation)),p90_px=float(np.percentile(validation,90)),
                      fraction_under_2px=float(np.mean(validation <= cfg.ransac_px)))
        flags = []
        if inside.sum() < cfg.min_inliers: flags.append("few_inliers")
        for s in (spatial,target_spatial):
            if s['cells'] < cfg.min_cells or s['span_x'] < cfg.min_span_x or s['span_y'] < cfg.min_span_y or s['hull_fraction'] < .025:
                flags.append("concentrated_support"); break
        if metric['median_px'] > cfg.max_median_px or metric['fraction_under_2px'] < cfg.min_validation_fraction:
            flags.append("validation_residual")
        # Reject implausible local magnification/shear, including near horizons.
        h,w = hw; probes = np.array([[w*.2,h*.2],[w*.8,h*.2],[w*.2,h*.8],[w*.8,h*.8],[w*.5,h*.5]])
        dx = project(m,probes+[1,0])-project(m,probes)
        dy = project(m,probes+[0,1])-project(m,probes)
        jac = np.stack((dx,dy),-1)
        singular = np.linalg.svd(jac,compute_uv=False)
        if singular.min() < .65 or singular.max() > 1.5 or np.min(np.linalg.det(jac)) <= 0:
            flags.append("implausible_warp")
        if strip_alpha is not None:
            center = np.array([(hw[1]-1)/2,(hw[0]-1)/2])
            along_scan = -(project(m,center)-center)[0]
            # Stationary/orthogonal jitter is not a strip. Let the selector
            # accumulate actual scan motion before imposing strip geometry.
            if along_scan >= .75:
                try: symmetric_pair(m,hw,strip_alpha)
                except GeometryError: flags.append("unsupported_strip_geometry")
        record = dict(model=model,train_inliers=int(inside.sum()),train_count=len(train),
                      validation_count=len(test),validation=metric,source_support=spatial,
                      target_support=target_spatial,flags=flags,matrix=m.tolist())
        candidates.append(record)
        if not flags: accepted.append((model,m,record))
    evidence = dict(matches=len(aa),candidates=candidates,split_seed=cfg.seed)
    if not accepted: raise MotionError("No spatially supported motion model passed validation",evidence)
    best = min(item[2]['validation']['median_px'] for item in accepted)
    # Prefer the simplest model within 0.12px of the best held-out median.
    eligible = [item for item in accepted if item[2]['validation']['median_px'] <= best+.12]
    model,m,chosen = eligible[0]
    c = np.array([(hw[1]-1)/2,(hw[0]-1)/2])
    delta = project(m,c)-c
    evidence.update(model=model,matrix=m.tolist(),center_displacement=delta.tolist(),validation=chosen['validation'])
    return m,evidence


def estimate_motion(a,b,cfg=RegistrationConfig()):
    return estimate_features(detect(a,cfg),detect(b,cfg),a.shape[:2],cfg)
