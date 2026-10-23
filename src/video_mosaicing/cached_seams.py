"""Cache alpha-independent costs for the rigid LK seam objective."""
from dataclasses import dataclass
import cv2
import numpy as np
from .seams import _column_visual_cost


@dataclass
class Overlap:
    xs: np.ndarray
    visual: np.ndarray
    lo: int
    hi: int
    kind: str


class SeamCache:
    def __init__(self, warped_bgr, masks, x_ranges, out_w, pad_x, *,
                 lam=5e-4, mu=2e-3, stride=2, transition_budget_bytes=64*1024*1024):
        self.lam, self.mu = lam, mu
        self.overlaps = []
        self.safe_lo_hi = []
        self.transitions = {}
        self.transition_cache_bytes = 0
        if len(x_ranges) < 2:
            return
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                for im in warped_bgr]
        for i in range(len(x_ranges)-1):
            a0,a1=x_ranges[i]; b0,b1=x_ranges[i+1]
            ov0=max(a0,b0); ov1=min(a1,b1)
            if ov0 > ov1:
                lo=int(np.clip(min(a1,b0),0,out_w-1))
                hi=int(np.clip(max(a1,b0),0,out_w-1))
                if lo==hi:
                    xs=np.array([lo],dtype=np.int32)
                else:
                    k=max(3,int(np.ceil((hi-lo)/max(1,10*stride))))
                    xs=np.unique(np.round(np.linspace(lo,hi,num=k,dtype=np.float32)).astype(np.int32))
                o=Overlap(xs,np.full(xs.shape,1.,dtype=np.float64),lo,hi,'gap')
                safe=(int(xs.min()),int(xs.max()))
            else:
                lo=ov0+pad_x; hi=ov1-pad_x
                if lo>hi:
                    lo=int(np.clip(ov0,0,out_w-1)); hi=int(np.clip(ov1,0,out_w-1))
                    if lo>hi:
                        s=int(np.clip(round(.5*(a1+b0)),0,out_w-1))
                        xs=np.array([s],dtype=np.int32)
                        o=Overlap(xs,np.array([1.],dtype=np.float64),s,s,'constant')
                    else:
                        xs=np.arange(lo,hi+1,stride,dtype=np.int32)
                        if xs.size==0:
                            xs=np.array([int(round(.5*(lo+hi)))],dtype=np.int32)
                        v=_column_visual_cost(warped_bgr[i],warped_bgr[i+1],masks[i],masks[i+1],lo,hi)
                        o=Overlap(xs,v[xs-lo],lo,hi,'thin')
                    safe=(int(xs.min()),int(xs.max()))
                else:
                    lo=int(np.clip(lo,0,out_w-1));hi=int(np.clip(hi,0,out_w-1))
                    xs=np.arange(lo,hi+1,max(1,int(stride)),dtype=np.int32)
                    g1=gray[i][:,lo:hi+1];g2=gray[i+1][:,lo:hi+1]
                    m=masks[i][:,lo:hi+1]&masks[i+1][:,lo:hi+1]
                    diff=np.abs(g1-g2)*m.astype(np.float32)
                    denom=m.sum(axis=0).astype(np.float32)
                    with np.errstate(divide='ignore',invalid='ignore'):
                        full=np.where(denom>0,diff.sum(axis=0)/denom,np.inf).astype(np.float64)
                    v=full[::max(1,int(stride))][:xs.shape[0]]
                    o=Overlap(xs,v,lo,hi,'normal')
                    safe=(lo,hi)
            self.overlaps.append(o);self.safe_lo_hi.append(safe)
        del gray
        # Same arithmetic as original; cache only within an explicit memory cap.
        for i in range(1,len(self.overlaps)):
            xp=self.overlaps[i-1].xs.astype(np.float64)
            xc=self.overlaps[i].xs.astype(np.float64)
            needed=xp.size*xc.size*8
            if self.transition_cache_bytes+needed<=transition_budget_bytes:
                dx=xc[None,:]-xp[:,None]
                trans=self.mu*(dx**2)
                self.transitions[i]=np.where(dx>=0.,trans,np.inf)
                self.transition_cache_bytes+=needed

    def seams(self,alpha):
        if not self.overlaps:
            return []
        if alpha<=0.:
            return [lo for lo,hi in self.safe_lo_hi]
        if alpha>=1.:
            return [hi for lo,hi in self.safe_lo_hi]
        cand_xs=[o.xs for o in self.overlaps]
        cand_costs=[]
        for o in self.overlaps:
            xs=o.xs
            if o.kind=='normal':
                target=float(o.lo+alpha*(o.hi-o.lo))
                unary=o.visual+self.lam*(xs.astype(np.float64)-target)**2
            elif o.kind=='constant':
                unary=o.visual.copy()
            else:
                target=int(round(o.lo+alpha*(o.hi-o.lo)))
                penalty=self.lam*((xs-target)**2).astype(np.float64)
                unary=o.visual+penalty
                if o.kind=='thin' and not np.isfinite(unary).any():
                    unary=penalty+1.
            cand_costs.append(unary)
        K0=cand_xs[0].shape[0]
        dp_prev=cand_costs[0].copy()
        back_ptrs=[np.full(K0,-1,dtype=np.int32)]
        for i in range(1,len(cand_xs)):
            xs_prev=cand_xs[i-1].astype(np.float64)
            xs_cur=cand_xs[i].astype(np.float64)
            Kp,Kc=xs_prev.shape[0],xs_cur.shape[0]
            if i in self.transitions:
                trans=self.transitions[i]
            else:
                dx=xs_cur[None,:]-xs_prev[:,None]
                trans=self.mu*(dx**2)
                trans=np.where(dx>=0.,trans,np.inf)
            total=dp_prev[:,None]+trans
            if not np.isfinite(total).any():
                dx=xs_cur[None,:]-xs_prev[:,None]
                trans=self.mu*(dx**2)
                total=dp_prev[:,None]+trans
            argmin=np.argmin(total,axis=0).astype(np.int32)
            dp_cur=cand_costs[i]+total[argmin,np.arange(Kc)]
            back_ptrs.append(argmin);dp_prev=dp_cur
        result=[0]*len(cand_xs)
        k=int(np.argmin(dp_prev));result[-1]=int(cand_xs[-1][k])
        for i in range(len(cand_xs)-1,0,-1):
            k=int(back_ptrs[i][k]);result[i-1]=int(cand_xs[i-1][k])
        return result


def make_cached_api():
    """Drop-in closure for one original pipeline run; captures input by identity."""
    state={}
    def cached_api(warped_bgr,masks,x_ranges,alpha,out_w,pad_x,**kwargs):
        key=(id(warped_bgr),id(masks),id(x_ranges),out_w,pad_x,tuple(sorted(kwargs.items())))
        if state.get('key')!=key:
            state['cache']=SeamCache(warped_bgr,masks,x_ranges,out_w,pad_x,**kwargs)
            state['key']=key
        return state['cache'].seams(alpha)
    cached_api.state=state
    return cached_api
