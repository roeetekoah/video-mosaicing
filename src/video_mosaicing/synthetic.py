"""Calibrated pinhole fixtures with planar and occluding depth layers.

Scene rendering uses ray/plane intersection. Mosaicing never receives camera
poses; they are available separately as an independent geometric test oracle.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def rotation(yaw=0., pitch=0., roll=0.):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]) @ np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])


def texture(seed, hw=(1000,1800)):
    rng = np.random.default_rng(seed)
    h,w = hw
    noise = rng.integers(30,225,(h,w,3),dtype=np.uint8)
    image = cv2.GaussianBlur(noise,(0,0),.7)
    for x in range(20,w,97): cv2.line(image,(x,0),(x,h-1),(45,65,205),2)
    for y in range(30,h,83): cv2.line(image,(0,y),(w-1,y),(200,65,40),2)
    for _ in range(180):
        x,y = rng.integers((10,10),(w-40,h-40))
        color = tuple(int(v) for v in rng.integers(20,235,3))
        cv2.rectangle(image,(x,y),(x+int(rng.integers(8,35)),y+int(rng.integers(8,35))),color,-1)
    return image


@dataclass
class CalibratedScene:
    frames: list
    intrinsics: np.ndarray
    rotations: list  # camera to world
    centers: list
    plane_to_images: list
    foreground_masks: list

    @property
    def homographies(self):
        return [b @ np.linalg.inv(a) for a,b in zip(self.plane_to_images,self.plane_to_images[1:])]


def calibrated_sequence(*, seed=47, count=13, hw=(240,320), kind="translation", layered=False, exposure=False, moving=False):
    h,w = hw
    f = .94*w
    k = np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
    yy,xx = np.mgrid[:h,:w]
    rays = np.stack((xx,yy,np.ones_like(xx)),-1) @ np.linalg.inv(k).T
    back, front = texture(seed),texture(seed+1,(600,600))
    frames, rotations, centers, matrices, masks = [],[],[],[],[]
    for i,t in enumerate(np.linspace(0,1,count)):
        if kind == "translation": c,r = np.array([3*t,0,0]),rotation()
        elif kind == "tilted_pan": c,r = np.zeros(3),rotation(yaw=.8*(t-.5),pitch=.18,roll=.04)
        elif kind == "tilted_wall": c,r = np.array([3*t,.4*t,0]),rotation(yaw=.1,pitch=.12,roll=.10)
        elif kind == "variable_speed": c,r = np.array([3*t*t,0,0]),rotation(roll=.012*np.sin(6*t))
        elif kind == "vertical": c,r = np.array([0,2.2*t,0]),rotation()
        else: raise ValueError("Unknown camera path")
        directions = rays @ r.T
        world = c + directions*((6-c[2])/directions[...,2])[...,None]
        u = ((world[...,0]+9)*65).astype(np.float32)
        v = ((world[...,1]+6)*65).astype(np.float32)
        frame = cv2.remap(back,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        fg = np.zeros((h,w),bool)
        if layered:
            near = c+directions*((3-c[2])/directions[...,2])[...,None]
            shift = .8*np.sin(2*np.pi*t) if moving else 0
            fg = (near[...,0] > .7+shift)&(near[...,0] < 1.5+shift)&(np.abs(near[...,1]) < .75)
            patch = cv2.remap(front,((near[...,0]-.7-shift)*500).astype(np.float32),((near[...,1]+.75)*350).astype(np.float32),cv2.INTER_LINEAR)
            frame[fg] = patch[fg]
        if exposure:
            gain = 1+.25*t
            frame = np.clip(frame.astype(float)*gain+12*t,0,255).astype(np.uint8)
        p = k @ r.T @ np.array([[1,0,-c[0]],[0,1,-c[1]],[0,0,6-c[2]]])
        frames.append(frame); rotations.append(r); centers.append(c); matrices.append(p); masks.append(fg)
    return CalibratedScene(frames,k,rotations,centers,matrices,masks)
