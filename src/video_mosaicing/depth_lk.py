"""Direct multi-frame LK with local EPI slopes for lateral camera scans.

Adapted from Rav-Acha, Shor and Peleg (CVPRW 2004), sections 3-5.
This bounded implementation uses a lateral/vertical translation and in-plane
roll model, small 3x3 patches, and bounded robust Gauss-Newton iterations.
It is not a reproduction of the paper's complete planar/spherical formulation.
For a point p in reference j, predict its location in i by
    p_i = R(theta_i-theta_j)(p-c) + c + slope_j(p)*(t_i-t_j).
Slopes are relative inverse depths; the first translation fixes their scale.
"""
import cv2
import numpy as np


def sample(im,x,y):
    if x.shape[0]>=32767:
        return np.concatenate([sample(im,x[start:start+16000],y[start:start+16000])
                               for start in range(0,x.shape[0],16000)],axis=0)
    return cv2.remap(im,x.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def gradients(im):
    return (cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3,scale=1/8),
            cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3,scale=1/8))


def points(im,stride=6):
    h,w=im.shape
    yy,xx=np.mgrid[8:h-8:stride,8:w-8:stride]
    gx,gy=gradients(im)
    strength=cv2.boxFilter(gx*gx,(cv2.CV_32F),(5,5),normalize=True)
    keep=strength[yy,xx]>max(2e-5,float(np.quantile(strength[yy,xx],.35)))
    xx,yy=xx[keep].astype(float),yy[keep].astype(float)
    ox,oy=np.meshgrid([-2.,0.,2.],[-2.,0.,2.])
    x=xx[:,None]+ox.ravel();y=yy[:,None]+oy.ravel()
    return dict(x=x,y=y,xc=xx,yc=yy,values=sample(im,x,y),s=np.ones(len(xx)),
                confidence=np.ones(len(xx)),shape_error=np.full(len(xx),np.nan))


def warp_coords(ref,delta,shape):
    h,w=shape;cx=(w-1)/2;cy=(h-1)/2
    x,y=ref['x']-cx,ref['y']-cy
    co,si=np.cos(delta[2]),np.sin(delta[2]);s=ref['s'][:,None]
    u=co*x-si*y+cx+s*delta[0]
    v=si*x+co*y+cy+s*delta[1]
    return u,v,-si*x-co*y,co*x-si*y


def residual_terms(ref,target,grad,delta):
    h,w=target.shape
    x,y,rx,ry=warp_coords(ref,delta,target.shape)
    valid=(x>3)&(x<w-4)&(y>3)&(y<h-4)
    e=sample(target,x,y)-ref['values']
    gx,gy=sample(grad[0],x,y),sample(grad[1],x,y)
    return e,gx,gy,valid,rx,ry


def motion_step(refs,images,grads,poses,i,guess,iterations=10):
    q=guess.copy();history=[]
    shape=images[i].shape;radius=max(shape)
    for _ in range(iterations):
        mats=[];errs=[];weights=[]
        for j in range(max(0,i-3),i):
            ref=refs[j]
            e,gx,gy,valid,rx,ry=residual_terms(ref,images[i],grads[i],q-poses[j])
            confidence=ref['confidence'][:,None]
            wt=valid*confidence*np.minimum(1.,.035/np.maximum(abs(e),1e-6))
            s=ref['s'][:,None]
            J=np.stack([gx*s,gy*s,(gx*rx+gy*ry)/radius],axis=-1).reshape(-1,3)
            mats.append(J);errs.append(e.ravel());weights.append(wt.ravel())
        J=np.concatenate(mats);e=np.concatenate(errs);wt=np.concatenate(weights)
        good=wt>.05
        if good.sum()<100:break
        J,e,wt=J[good],e[good],wt[good]
        A=J.T@(wt[:,None]*J);b=J.T@(wt*e)
        dp=-np.linalg.solve(A+np.eye(3)*1e-5,b)
        dp[:2]=np.clip(dp[:2],-2.,2.);dp[2]=np.clip(dp[2],-1.5,1.5)
        dp[2]/=radius
        # Evaluate on fixed samples so discarding overlap cannot lower the objective.
        before=float(np.sum(wt*np.minimum(e*e,.07*abs(e)))/np.sum(wt))
        accepted=False
        for fraction in (1.,.5,.25):
            candidate=q+fraction*dp
            ee=[];vv=[]
            for j in range(max(0,i-3),i):
                er,_,_,val,_,_=residual_terms(refs[j],images[i],grads[i],candidate-poses[j])
                ee.append(er.ravel());vv.append(val.ravel())
            en=np.concatenate(ee)[good];vn=np.concatenate(vv)[good]
            after=float(np.sum(wt*np.where(vn,np.minimum(en*en,.07*abs(en)),.01))/np.sum(wt))
            if after<=before:
                q=candidate;accepted=True;break
        history.append(before)
        if not accepted or np.linalg.norm(dp[:2])+abs(dp[2])*radius<.003:break
    return q,dict(iterations=len(history),cost=history[-1] if history else None)


def shape_step(ref,images,grads,poses,j,neighbors,iterations=8):
    h,w=images[j].shape;s=ref['s'].copy()
    support=np.zeros(len(s));error=np.ones(len(s))
    for _ in range(iterations):
        ref['s']=s
        a=np.zeros(len(s));b=a.copy();count=a.copy();err=a.copy()
        for k in neighbors:
            if k==j:continue
            delta=poses[k]-poses[j]
            e,gx,gy,valid,_,_=residual_terms(ref,images[k],grads[k],delta)
            derivative=gx*delta[0]+gy*delta[1]
            wt=valid*np.minimum(1.,.035/np.maximum(abs(e),1e-6))
            a+=np.sum(wt*derivative**2,axis=1)
            b+=np.sum(wt*derivative*e,axis=1)
            count+=valid.sum(axis=1);err+=(abs(e)*valid).sum(axis=1)
        ds=-b/(a+1e-5)
        s=np.clip(s+np.clip(ds,-.15,.15),.2,3.)
        support=a;error=err/np.maximum(count,1)
    ref['s']=s
    ref['confidence']=np.where((support>1e-4)&(error<.065)&(s>.201)&(s<2.999),1.,.05)
    ref['shape_error']=error


def estimate(images,initial,progress=None,*,select_monotone=False):
    """Estimate at the supplied resolution; initialize with original rigid LK."""
    n=len(images);grads=[gradients(im) for im in images]
    refs=[points(im) for im in images];poses=np.zeros((n,3));poses[1]=initial
    if select_monotone:
        all_images,all_grads,all_refs=images,grads,refs
        images=list(images[:2]);grads=list(grads[:2]);refs=list(refs[:2]);selected=[0,1]
    shape_step(refs[0],images,grads,poses,0,[1])
    shape_step(refs[1],images,grads,poses,1,[0])
    records=[]
    for source in range(2,n):
        i=len(selected) if select_monotone else source
        if select_monotone:
            images.append(all_images[source]);grads.append(all_grads[source]);refs.append(all_refs[source])
        step=poses[i-1]-poses[i-2]
        guess=poses[i-1]+step
        poses[i],record=motion_step(refs,images,grads,poses,i,guess)
        displacement=poses[i,0]-poses[i-1,0]
        insufficient=not record['iterations']
        if insufficient or np.sign(displacement)!=np.sign(initial[0]) or abs(displacement)<.05:
            if not select_monotone: raise ValueError(f'Nonmonotone camera estimate at frame {i}')
            records.append(dict(source_index=source,skipped=True,reason='insufficient_photometric_support' if insufficient else 'small_or_reversing_displacement',measured_dx=float(displacement)))
            images.pop();grads.pop();refs.pop()
            if source-selected[-1]>=12:raise ValueError(f'No forward-supported observation in 12 candidates at {source}')
            continue
        if select_monotone:selected.append(source)
        shape_step(refs[i],images,grads,poses,i,list(range(max(0,i-5),i)))
        # Refresh the immediately preceding depth estimate with the new observation.
        shape_step(refs[i-1],images,grads,poses,i-1,list(range(max(0,i-4),i+1)),iterations=4)
        records.append(dict(frame=i,source_index=source,**record,valid_shapes=int((refs[i]['confidence']>.1).sum())))
        if progress and source%20==0:progress(source,n)
    if select_monotone:return poses[:len(selected)],refs,records,selected
    return poses,refs,records


def slope_map(ref,shape):
    """Interpolate reliable sampled slopes, retaining a confidence-weighted grid."""
    h,w=shape
    den=np.zeros((h,w),np.float32);num=den.copy()
    x,y=ref['xc'].astype(int),ref['yc'].astype(int)
    den[y,x]=ref['confidence'];num[y,x]=ref['confidence']*ref['s']
    num=cv2.GaussianBlur(num,(0,0),5);den=cv2.GaussianBlur(den,(0,0),5)
    return np.where(den>1e-6,num/np.maximum(den,1e-6),1).astype(np.float32)


def render(frames,poses,slopes,alpha,reference):
    """Resample a fixed slit through the estimated camera-position volume.

    Between adjacent camera positions, compensate local EPI displacement before
    linear interpolation. Endpoint images supply unswept side regions. No seam
    optimization, exposure correction, or post-render stabilization is applied.
    """
    h,w=frames[0].shape[:2];center=np.array([(w-1)/2,(h-1)/2])
    q=poses-poses[reference]
    if not np.all(np.diff(q[:,0])<0):raise ValueError('Renderer requires a leftward image scan')
    lo=float(-q[0,0]);span=float(q[0,0]-q[-1,0]);width=int(np.ceil(span+w))
    X=np.arange(width)+lo
    slit=alpha*(w-1)
    target=np.clip(slit-X,q[-1,0],q[0,0])
    source_x=X+target
    right=np.searchsorted(-q[:,0],-target,side='right').clip(1,len(q)-1)
    left=right-1
    weight=(target-q[left,0])/(q[right,0]-q[left,0])
    out=np.zeros((h,width,3),np.float32);valid_total=np.zeros((h,width),np.float32)
    def fetch_xy(i,xx,yy,tt):
        s=sample(slopes[i],xx.clip(0,w-1),yy)
        co,si=np.cos(q[i,2]),np.sin(q[i,2])
        for _ in range(2):
            dx=xx-center[0]+s*(q[i,0]-tt);dy=yy-center[1]
            mx=co*dx-si*dy+center[0]
            my=si*dx+co*dy+center[1]+s*q[i,1]
            s=sample(slopes[i],mx.clip(0,w-1),my.clip(0,h-1))
        valid=(mx>=0)&(mx<=w-1)&(my>=0)&(my<=h-1)
        return sample(frames[i],mx,my),valid

    def fetch(i,cols):
        xx=np.broadcast_to(source_x[cols][None,:],(h,len(cols))).astype(np.float32)
        yy=np.broadcast_to(np.arange(h)[:,None],xx.shape).astype(np.float32)
        return fetch_xy(i,xx,yy,target[cols][None,:])

    for i in np.union1d(left,right):
        cols=np.flatnonzero((left==i)|(right==i))
        weights=(left[cols]==i)*(1-weight[cols])+(right[cols]==i)*weight[cols]
        rgb,valid=fetch(i,cols)
        ww=valid*weights[None,:]
        out[:,cols]+=rgb.astype(np.float32)*ww[:,:,None]
        valid_total[:,cols]+=ww
    # At extreme slits, roll can move both immediate samples outside the sensor.
    # Use the nearest supported observation of the same model ray, never inpaint.
    rows,cols=np.nonzero(valid_total<1e-6)
    for distance in range(1,7):
        for indices in ((left-distance).clip(0,len(q)-1),(right+distance).clip(0,len(q)-1)):
            if not len(rows):break
            labels=indices[cols]
            found=np.zeros(len(rows),bool)
            for i in np.unique(labels):
                selected=labels==i;rr,cc=rows[selected],cols[selected]
                rgb,valid=fetch_xy(i,source_x[cc,None].astype(np.float32),rr[:,None].astype(np.float32),target[cc,None])
                use=valid[:,0]
                out[rr,cc]+=rgb[:,0].astype(np.float32)*use[:,None]
                valid_total[rr,cc]+=use
                found[selected]=use
            rows,cols=rows[~found],cols[~found]
    out/=np.maximum(valid_total[:,:,None],1e-6)
    return np.clip(out,0,255).astype(np.uint8),valid_total>1e-6
