"""Reusable rigid and depth-aware direct LK panorama sweeps."""
from dataclasses import asdict, dataclass, field
import time

import cv2
import numpy as np

from . import depth_lk
from .cached_seams import SeamCache
from .canvas import compute_canvas
from .data_structures import LKConfig
from .direction_normalization import normalize_scan_direction
from .lucas_kanade import estimate_rigid_lk_pyramids
from .preprocess import preprocess_frame
from .rigid_transformations import estimate_pairwise, accumulate_to_ref
from .seams import compute_frame_x_ranges, seams_for_view
from .strips import compose_strips_feathered
from .warping import warp_rgb_and_mask


@dataclass(frozen=True)
class DirectConfig:
    method: str = 'lk-original'
    registration_scale: float = .5
    blur_sigma: float = 1.5
    active_area: tuple | None = None
    max_canvas_pixels: int = 20_000_000
    lk: LKConfig = field(default_factory=lambda: LKConfig(num_levels=4,num_iters=30,border=8))

    def __post_init__(self):
        if self.method not in ('lk-depth','lk-original'): raise ValueError('Unknown direct LK method')
        if not .25 <= self.registration_scale <= 1: raise ValueError('registration_scale must be .25..1')
        if self.active_area is not None:
            l,t,r,b=self.active_area
            if not 0<=l<r<=1 or not 0<=t<b<=1: raise ValueError('Invalid active area')


def restore(image,direction):
    if direction=='TB': return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    if direction=='BT': return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


@dataclass
class DirectScan:
    frames: list
    direction: str
    indices: list
    config: DirectConfig
    evidence: dict
    poses: np.ndarray | None = None
    slopes: list | None = None
    original: dict | None = None

    def render(self,alpha):
        if not np.isfinite(alpha) or not 0<=alpha<=1: raise ValueError('alpha must lie in [0,1]')
        if self.config.method=='lk-depth':
            image,mask=depth_lk.render(self.frames,self.poses,self.slopes,float(alpha),len(self.frames)//2)
        else:
            s=self.original;seams=s['cache'].seams(alpha)
            if len(seams)>1 and np.max(abs(np.diff(seams)))>250: seams=seams_for_view(s['ranges'],alpha)
            image=compose_strips_feathered(s['images'],s['masks'],seams,s['hw'])
            mask=np.zeros(s['hw'],bool);left=0
            for i,m in enumerate(s['masks']):
                right=seams[i] if i<len(seams) else s['hw'][1]
                if right>left: mask[:,left:right]=m[:,left:right]
                left=right
        return restore(image,self.direction),restore(mask.astype(np.uint8),self.direction).astype(bool)


def prepare_direct(records,config=DirectConfig(),progress=None):
    started=time.perf_counter();indices=[];frames=[]
    for index,frame in records:
        if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3: raise ValueError('Expected uint8 BGR frames')
        if config.active_area:
            h,w=frame.shape[:2];l,t,r,b=config.active_area
            frame=frame[round(t*h):round(b*h),round(l*w):round(r*w)].copy()
        if min(frame.shape[:2])<32 or (frames and frame.shape!=frames[0].shape): raise ValueError('Equal frame dimensions >=32 required')
        frames.append(frame);indices.append(int(index))
    if len(frames)<2: raise ValueError('At least two source frames required')
    gray=[preprocess_frame(f,blur_sigma=config.blur_sigma) for f in frames]
    frames,gray,direction=normalize_scan_direction(frames,gray,config.lk,blur_sigma=config.blur_sigma)
    if direction=='RL': indices=indices[::-1]
    decoded=time.perf_counter()-started;tick=time.perf_counter()
    reference=len(frames)//2
    evidence=dict(method=config.method,source_indices=indices,direction=direction,reference_index=reference,
                  reference_source_index=indices[reference],source_hw=list(gray[0].shape),active_area=config.active_area,
                  timing=dict(decode_preprocess=decoded),registration_scale=config.registration_scale,config=asdict(config))
    scan=DirectScan(frames,direction,indices,config,evidence)
    if config.method=='lk-depth':
        initial=estimate_rigid_lk_pyramids(gray[0],gray[1],config.lk)
        if initial.dx>=-.05: raise ValueError(f'Initial scan displacement is not leftward: {initial.dx:.3f}')
        h,w=gray[0].shape;scale=config.registration_scale
        small=[cv2.resize(g,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) for g in gray]
        sx=small[0].shape[1]/w;sy=small[0].shape[0]/h
        poses,refs,diagnostics,selected=depth_lk.estimate(small,np.array([initial.dx*sx,initial.dy*sy,initial.theta]),progress,select_monotone=True)
        evidence['candidate_source_indices']=indices
        frames=[frames[i] for i in selected];indices=[indices[i] for i in selected]
        reference=len(frames)//2;scan.frames=frames;scan.indices=indices
        evidence.update(source_indices=indices,reference_index=reference,reference_source_index=indices[reference],
                        skipped_source_indices=[evidence['candidate_source_indices'][d['source_index']] for d in diagnostics if d.get('skipped')])
        poses[:,0]/=sx;poses[:,1]/=sy
        width=int(np.ceil(poses[0,0]-poses[-1,0]+w))
        if width<=0 or width*h>config.max_canvas_pixels: raise ValueError(f'Unsupported depth canvas: {h}x{width}')
        evidence['timing']['registration']=time.perf_counter()-tick;tick=time.perf_counter()
        slopes=[cv2.resize(depth_lk.slope_map(ref,small[0].shape),(w,h),interpolation=cv2.INTER_LINEAR) for ref in refs]
        scan.poses=poses;scan.slopes=slopes
        evidence.update(poses=poses.tolist(),registration_diagnostics=diagnostics,
                        reference_to_canvas=[[1,0,float(-(poses[reference,0]-poses[0,0]))],[0,1,0],[0,0,1]],
                        normalized_canvas_hw=[h,width])
    else:
        pair=estimate_pairwise(gray,config.lk)
        evidence['timing']['registration']=time.perf_counter()-tick;tick=time.perf_counter()
        cumulative=accumulate_to_ref(pair,reference);hw,shift=compute_canvas(cumulative,gray[0].shape,margin=50)
        if hw[0]*hw[1]>config.max_canvas_pixels: raise ValueError(f'Unsupported rigid canvas: {hw}')
        images=[];masks=[]
        for frame,t in zip(frames,cumulative):
            image,mask=warp_rgb_and_mask(frame,shift@t.as_matrix(*gray[0].shape),hw)
            images.append(image);masks.append(mask)
        ranges=compute_frame_x_ranges(masks,coverage_thresh=.5)
        scan.original=dict(images=images,masks=masks,ranges=ranges,hw=hw,
                           cache=SeamCache(images,masks,ranges,hw[1],int(np.ceil(.1*gray[0].shape[1]))))
        evidence.update(motions=[[t.dx,t.dy,t.theta] for t in pair],reference_to_canvas=shift.tolist(),normalized_canvas_hw=list(hw))
    evidence['timing']['preparation']=time.perf_counter()-tick
    evidence['timing']['preparation_total']=time.perf_counter()-started
    return scan
