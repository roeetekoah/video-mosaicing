"""Optional robust exposure alignment and narrow, validity-aware blending."""
import cv2
import numpy as np

from .geometry import project
from .registration import MotionError


def exposure_gains(frames,homographies):
    """Fit gain/bias over redundant overlaps, fixing the middle frame's gauge.

    The affine intensity model is grayscale, applied to all color channels.
    Saturated/dark observations are excluded and Huber weights limit moving
    content and parallax. Fits are recorded; excessive accumulated corrections
    are rejected instead of silently changing the clip's appearance.
    """
    h,w = frames[0].shape[:2]
    yy,xx = np.mgrid[8:h-8:4,8:w-8:4]
    coordinates = np.stack((xx,yy),-1).astype(float)
    # Estimate radiometry at low frequency; interpolation otherwise attenuates
    # target texture and an ordinary intensity regression mistakes it for gain.
    gray = [cv2.GaussianBlur(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)/255,(0,0),2.) for f in frames]
    unclipped = [cv2.erode(np.all((f>5)&(f<250),axis=-1).astype(np.uint8),np.ones((13,13),np.uint8)) for f in frames]
    records = []
    edges = []
    for i in range(len(frames)-1):
        m = np.eye(3)
        for j in range(i+1,min(len(frames),i+7)):
            m = homographies[j-1] @ m
            if j-i in (1,3,6): edges.append((i,j,m.copy()))
    for i,j,m in edges:
        xy = project(m,coordinates)
        target = cv2.remap(gray[j],xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR)
        source = gray[i][yy,xx]
        valid = (xy[...,0]>=3)&(xy[...,0]<w-4)&(xy[...,1]>=3)&(xy[...,1]<h-4)&(source>.08)&(source<.92)&(target>.08)&(target<.92)
        target_unclipped = cv2.remap(unclipped[j],xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_NEAREST)
        valid &= (unclipped[i][yy,xx]>0)&(target_unclipped>0)
        x,y = target[valid].astype(float),source[valid].astype(float)
        if len(x)<100 or np.std(x)<.005:
            if j-i>1: continue
            raise MotionError("Exposure model lacks intensity support",dict(pair=i,samples=len(x)))
        a = np.column_stack((x,np.ones(len(x))))
        beta = np.array([1.,0.])
        correlated = True
        for _ in range(8):
            residual = a@beta-y
            scale = max(.008,1.4826*np.median(np.abs(residual-np.median(residual))))
            weights = np.minimum(1,1.5*scale/np.maximum(np.abs(residual),1e-8))
            mx,my = np.average(x,weights=weights),np.average(y,weights=weights)
            vx = np.average((x-mx)**2,weights=weights)
            vy = np.average((y-my)**2,weights=weights)
            cov = np.average((x-mx)*(y-my),weights=weights)
            if cov <= 1e-5:
                correlated = False
                break
            slope = (vy-vx+np.sqrt((vy-vx)**2+4*cov*cov))/(2*cov)
            beta = np.array([slope,my-slope*mx])
        gain,bias = beta
        if not correlated:
            if j-i>1: continue
            raise MotionError("Exposure samples are not correlated",dict(indices=[i,j]))
        gain_standard_error = scale/np.sqrt(np.sum(weights*(x-mx)**2))
        if gain_standard_error > .03:
            if j-i>1: continue
            raise MotionError("Exposure gain is poorly constrained",dict(pair=i,gain_standard_error=float(gain_standard_error)))
        if not .65 < gain < 1.54 or abs(bias)>.20:
            if j-i>1: continue
            raise MotionError("Abrupt or unsupported exposure transition",dict(pair=i,gain=float(gain),bias=float(bias)))
        records.append(dict(indices=[i,j],gain=float(gain),bias_255=float(bias*255),samples=len(x),gain_standard_error=float(gain_standard_error),robust_residual_255=float(np.median(np.abs(a@beta-y))*255)))
    # Redundant overlap edges reduce the bias and drift of a one-way chain.
    # Solve relative log gains, then biases, with an explicit reference gauge.
    n = len(frames)
    ref = len(frames)//2
    system = np.zeros((len(records)+1,n)); gain_rhs = np.zeros(len(records)+1)
    bias_rhs = np.zeros(len(records)+1)
    for row,r in enumerate(records):
        i,j = r['indices']
        weight = 1/max(.5,r['robust_residual_255'])
        system[row,i],system[row,j] = -weight,weight
        gain_rhs[row] = np.log(r['gain'])*weight
    system[-1,ref] = 1000
    g = np.exp(np.linalg.lstsq(system,gain_rhs,rcond=None)[0])
    for row,r in enumerate(records):
        i,j = r['indices']
        bias_rhs[row] = r['bias_255']/255*g[i]*system[row,j]
    b = np.linalg.lstsq(system,bias_rhs,rcond=None)[0]
    if g.min()<.5 or g.max()>2 or np.abs(b).max()>.25:
        raise MotionError("Accumulated exposure correction is implausible",dict(gain_range=[float(g.min()),float(g.max())],max_bias_255=float(np.abs(b).max()*255)))
    transforms = np.column_stack((g,b*255))
    return transforms,dict(model="gray_gain_bias",reference_index=ref,transforms=transforms.tolist(),pairs=records)


def corrected(pixels,index,transforms):
    values = pixels.astype(np.float32)
    if transforms is not None:
        values = values*transforms[index,0]+transforms[index,1]
    return np.clip(values,0,255)


def blend_boundaries(result,frames,homographies,width,transforms=None):
    """Blend only across inter-frame boundaries, using neighboring observations.

    Source provenance continues to describe the dominant hard-strip sample.
    Blended pixels additionally depend on the adjacent frame and recorded H.
    Both samples must be observed; blending does not fill missing coverage.
    """
    output = result.image.copy()
    h,w = frames[0].shape[:2]
    blended = 0
    disagreement = []
    for i,seam in enumerate(result.evidence['seams']):
        columns = np.arange(max(0,int(np.floor(seam-width))),min(output.shape[1],int(np.ceil(seam+width))))
        if not len(columns): continue
        index = result.source_index[:,columns]
        xy = result.source_xy[:,columns].astype(float)
        valid = ((index==i)|(index==i+1)) & result.coverage[:,columns]
        if not valid.any(): continue
        earlier = np.zeros_like(xy);later = np.zeros_like(xy)
        is_a,is_b = valid&(index==i),valid&(index==i+1)
        earlier[is_a] = xy[is_a]; later[is_a] = project(homographies[i],xy[is_a])
        later[is_b] = xy[is_b]; earlier[is_b] = project(np.linalg.inv(homographies[i]),xy[is_b])
        for p in (earlier,later): valid &= (p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
        a = cv2.remap(frames[i],earlier[...,0].astype(np.float32),earlier[...,1].astype(np.float32),cv2.INTER_LINEAR)
        b = cv2.remap(frames[i+1],later[...,0].astype(np.float32),later[...,1].astype(np.float32),cv2.INTER_LINEAR)
        a,b = corrected(a,i,transforms),corrected(b,i+1,transforms)
        alpha = np.clip((columns-seam+width)/(2*width),0,1)[None,:,None]
        blended_values = np.round(a*(1-alpha)+b*alpha).astype(np.uint8)
        yy,xx = np.nonzero(valid)
        output[yy,columns[xx]] = blended_values[yy,xx]
        blended += int(valid.sum())
        if valid.any(): disagreement.append(float(np.abs(a[valid]-b[valid]).mean()))
    return output,dict(method="narrow_linear",half_width=width,blended_pixels=blended,
                       mean_overlap_disagreement_255=float(np.mean(disagreement)) if disagreement else None,
                       provenance="dominant source plus adjacent frame and pair homography in blended bands")
