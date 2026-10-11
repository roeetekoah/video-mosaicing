"""Audited direct rigid refinement with exact bilinear image derivatives.

This comparison solver retains the centered rigid model and Gauss-Newton
structure of the original LK implementation. The production pipeline uses
validated feature motion; refinement is deliberately opt-in through this API.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .data_structures import RigidTransform
from .registration import MotionError


def bilinear(image,xy):
    """Return values, dI/dx, dI/dy, and genuine interpolation support."""
    h,w = image.shape
    xy = np.asarray(xy,float)
    valid = (xy[...,0]>=0)&(xy[...,0]<w-1)&(xy[...,1]>=0)&(xy[...,1]<h-1)
    x = np.clip(xy[...,0],0,w-1.000001);y = np.clip(xy[...,1],0,h-1.000001)
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    u,v = x-ix,y-iy
    a,b,c,d = image[iy,ix],image[iy,ix+1],image[iy+1,ix],image[iy+1,ix+1]
    value = (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d
    gx = (1-v)*(b-a)+v*(d-c)
    gy = (1-u)*(c-a)+u*(d-b)
    return value,gx,gy,valid


def residual_jacobian(source,target,points,parameters,*,photometric=False):
    """r = gain * target(W(source pixel)) + bias - source(pixel).

    Parameters are [dx,dy,radius*theta,(gain,bias)]. Exact rotation and
    bilinear derivatives make this Jacobian consistent with its objective.
    """
    h,w = source.shape
    radius = max(h,w)
    dx,dy,angle = parameters[:3]
    theta = angle/radius
    c = np.array([(w-1)/2,(h-1)/2])
    p = points-c
    co,si = np.cos(theta),np.sin(theta)
    r = np.array([[co,-si],[si,co]])
    warped = p@r.T+c+[dx,dy]
    value,gx,gy,valid = bilinear(target,warped)
    original,_,_,source_valid = bilinear(source,points)
    gain,bias = parameters[3:] if photometric else (1.,0.)
    rotation_derivative = p@np.array([[-si,-co],[co,-si]]).T/radius
    jac = np.column_stack((gain*gx,gain*gy,gain*(gx*rotation_derivative[:,0]+gy*rotation_derivative[:,1])))
    if photometric: jac = np.column_stack((jac,value,np.ones(len(points))))
    return gain*value+bias-original,jac,valid&source_valid


@dataclass(frozen=True)
class RefinementConfig:
    levels: int = 3
    iterations: int = 30
    border: int = 5
    photometric: bool = False

    def __post_init__(self):
        if self.levels<1 or self.iterations<1 or self.border<0:
            raise ValueError("levels/iterations must be positive; border nonnegative")


def refine_rigid(source,target,initial=None,cfg=RefinementConfig()):
    """Robust forward-additive rigid LK; initialize near the correct motion.

    The method rejects sparse/one-dimensional texture and insufficient true
    overlap. A line search compares the same supported samples for both
    parameter values; reflected borders never contribute to the objective.
    """
    if source.shape!=target.shape or source.ndim!=2: raise ValueError("Expected equal grayscale images")
    if min(source.shape)/(2**(cfg.levels-1))<12: raise ValueError("Too many pyramid levels for the image size")
    if not np.isfinite(source).all() or not np.isfinite(target).all(): raise ValueError("Nonfinite image")
    if min(source.min(),target.min())<0 or max(source.max(),target.max())>1: raise ValueError("Images must be in [0,1]")
    pyramids = [[cv2.GaussianBlur(im.astype(np.float64),(0,0),1.)] for im in (source,target)]
    for pyr in pyramids:
        for _ in range(1,cfg.levels): pyr.append(cv2.pyrDown(pyr[-1]))
    full = np.eye(3) if initial is None else np.asarray(initial,float).copy()
    gain,bias = 1.,0.
    records = []
    for level in reversed(range(cfg.levels)):
        a,b = pyramids[0][level],pyramids[1][level]
        h,w = a.shape
        border = max(2,min(cfg.border,min(h,w)//6))
        yy,xx = np.mgrid[border:h-border:2,border:w-border:2]
        points = np.column_stack((xx.ravel(),yy.ravel())).astype(float)
        scale = np.diag([2.**-level,2.**-level,1.])
        current = scale@full@np.linalg.inv(scale)
        center = np.array([(w-1)/2,(h-1)/2])
        translation = current[:2,:2]@center+current[:2,2]-center
        theta = np.arctan2(current[1,0],current[0,0])
        parameters = np.r_[translation,max(h,w)*theta]
        if cfg.photometric: parameters = np.r_[parameters,gain,bias]
        for iteration in range(cfg.iterations):
            residual,jac,valid = residual_jacobian(a,b,points,parameters,photometric=cfg.photometric)
            if valid.mean()<.5: raise MotionError("Direct refinement has insufficient overlap")
            j,r = jac[valid],residual[valid]
            textured = np.linalg.norm(j[:,:2],axis=1)>.002
            if textured.mean()<.01: raise MotionError("Direct refinement has insufficient texture")
            normal = j.T@j/len(j)
            eigen = np.linalg.eigvalsh(normal)
            if eigen[0]/max(eigen[-1],1e-12)<1e-6: raise MotionError("Direct refinement is rank deficient")
            huber = max(.01,1.5*1.4826*np.median(np.abs(r-np.median(r))))
            weights = np.minimum(1,huber/np.maximum(np.abs(r),1e-12))
            step = np.linalg.lstsq(j*np.sqrt(weights[:,None]),-r*np.sqrt(weights),rcond=None)[0]
            factor = min(1.,3/max(np.linalg.norm(step[:3]),1e-12))
            accepted = False
            def cost(v):
                av = np.abs(v)
                return np.where(av<=huber,.5*v*v,huber*(av-.5*huber)).mean()
            for shrink in (1.,.5,.25,.125,.0625):
                candidate = parameters+step*factor*shrink
                nr,_,nv = residual_jacobian(a,b,points,candidate,photometric=cfg.photometric)
                common = valid&nv
                if common.sum()<.98*valid.sum(): continue
                if cfg.photometric and not .5<candidate[3]<2: continue
                if cost(nr[common]) < cost(residual[common]):
                    parameters=candidate;accepted=True;break
            records.append(dict(level=level,iteration=iteration,accepted=accepted,valid_fraction=float(valid.mean()),median_residual=float(np.median(np.abs(r)))))
            if not accepted or np.linalg.norm(step*factor*shrink)<1e-3: break
        motion = RigidTransform(float(parameters[0]),float(parameters[1]),float(parameters[2]/max(h,w)))
        full = np.linalg.inv(scale)@motion.as_matrix(h,w)@scale
        if cfg.photometric: gain,bias = parameters[3:]
    return full,dict(objective="huber_forward_additive_rigid",gain=float(gain),bias=float(bias),iterations=records)
