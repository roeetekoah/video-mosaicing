"""Known-depth, source-selection and rendering checks for direct LK sweeps."""
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest

from video_mosaicing import depth_lk
from video_mosaicing.lk_pipeline import DirectConfig,prepare_direct
from video_mosaicing.cli import main
from video_mosaicing.io import iter_frames


def two_depths(count=18):
    rng=np.random.default_rng(47);h,w=120,180
    texture=cv2.GaussianBlur(rng.random((h,w+100)).astype(np.float32),(0,0),1)
    texture=(texture-texture.min())/(texture.max()-texture.min())
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    slope=np.where(yy<h/2,.65,1.5).astype(np.float32)
    t=np.arange(count,dtype=float);position=-1.1*(t+.3*np.sin(t*.4))
    images=[depth_lk.sample(texture,xx+25-slope*u,yy) for u in position]
    return images,position,slope,texture


def test_multiframe_lk_recovers_two_depths_and_camera_motion():
    images,truth,_,_=two_depths()
    q,refs,_=depth_lk.estimate(images,np.array([truth[1],0.,0.]))
    scale=np.dot(q[:,0],truth)/np.dot(truth,truth)
    assert np.sqrt(np.mean((q[:,0]-scale*truth)**2))<.1
    r=refs[-1];good=r['confidence']>.1
    top=np.median(r['s'][good&(r['yc']<52)]);bottom=np.median(r['s'][good&(r['yc']>68)])
    assert abs(bottom/top-1.5/.65)<.15
    assert abs(q[:,1]).max()<.1
    assert np.rad2deg(abs(q[:,2])).max()<.1


def test_ray_renderer_matches_analytic_two_plane_scene():
    images,position,slope,texture=two_depths()
    rgb=[np.repeat(np.rint(im*255).astype(np.uint8)[:,:,None],3,axis=2) for im in images]
    q=np.c_[position,np.zeros((len(images),2))];h,w=images[0].shape
    for alpha in (0.,.5,1.):
        im,mask=depth_lk.render(rgb,q,[slope]*len(images),alpha,0)
        X=np.arange(im.shape[1],dtype=np.float32)[None,:];target=np.clip(alpha*(w-1)-X,position[-1],0)
        source_x=X+target
        expected=depth_lk.sample(texture,np.broadcast_to(source_x,(h,im.shape[1]))-slope[:,:1]*target+25,np.broadcast_to(np.arange(h,dtype=np.float32)[:,None],(h,im.shape[1])))
        valid=mask.copy();valid[:8]=False;valid[-8:]=False;valid[55:65]=False
        assert abs(im[:,:,0].astype(float)-expected*255)[valid].mean()<4


def test_duplicate_frame_is_skipped_without_inventing_motion():
    images,truth,_,_=two_depths(10)
    images.insert(5,images[4].copy())
    q,refs,records,selected=depth_lk.estimate(images,np.array([truth[1],0.,0.]),select_monotone=True)
    assert 5 not in selected
    assert len(refs)==len(q)==len(selected)
    assert np.all(np.diff(q[:,0])<0)
    assert any(d.get('skipped') for d in records)


def test_cli_direct_modes_save_coverage_and_configuration(tmp_path):
    images,_,_,_=two_depths(9);folder=tmp_path/'input';folder.mkdir()
    for i,im in enumerate(images):cv2.imwrite(str(folder/f'{i:03d}.png'),np.repeat(np.rint(im*255).astype(np.uint8)[:,:,None],3,axis=2))
    for method in ('lk-depth','lk-original'):
        out=tmp_path/method
        assert main([str(folder),'--method',method,'--frame-stride','1','--sweep','.1','.5','.9','--no-provenance','--output',str(out)])==0
        record=json.loads((out/'run.json').read_text());assert record['method']==method
        assert len(record['views'])==3
        for i in range(3):
            image=cv2.imread(str(out/f'view-{i:03d}/panorama.png'))
            mask=cv2.imread(str(out/f'view-{i:03d}/coverage.png'),0)
            assert image.shape[:2]==mask.shape
            assert mask.any()


def test_depth_rejects_no_motion_instead_of_claiming_a_panorama():
    with pytest.raises(ValueError,match='Initial scan displacement'):
        prepare_direct([(0,np.zeros((80,100,3),np.uint8)),(1,np.zeros((80,100,3),np.uint8))],DirectConfig(method='lk-depth'))


def test_large_sparse_lookup_is_split_without_changing_samples():
    image=np.arange(80*100,dtype=np.float32).reshape(80,100)
    x=np.full((40000,1),23.25,np.float32);y=np.full_like(x,17.5)
    result=depth_lk.sample(image,x,y)
    assert result.shape==x.shape
    assert np.all(result==depth_lk.sample(image,x[:1],y[:1])[0,0])


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg is an optional video dependency')
def test_cli_encodes_actual_sweep_frames(tmp_path):
    images,_,_,_=two_depths(8);folder=tmp_path/'frames';folder.mkdir()
    for i,image in enumerate(images):cv2.imwrite(str(folder/f'{i:03d}.png'),np.repeat(np.rint(image*255).astype(np.uint8)[:,:,None],3,axis=2))
    out=tmp_path/'video'
    assert main([str(folder),'--method','lk-depth','--frame-stride','1','--video','--views','5','--no-provenance','--output',str(out)])==0
    data=json.loads((out/'run.json').read_text());assert data['decoded_output_frames']==5
    cap=cv2.VideoCapture(str(out/'sweep.mp4'));n=0
    while cap.read()[0]:n+=1
    cap.release();assert n==5


def test_resize_uses_original_rounded_dimensions(tmp_path):
    rng=np.random.default_rng(5);image=rng.integers(0,256,(114,64,3),dtype=np.uint8)
    cv2.imwrite(str(tmp_path/'frame.png'),image)
    _,actual=next(iter_frames(tmp_path,max_side=80))
    expected=cv2.resize(image,(round(64*80/114),80),interpolation=cv2.INTER_AREA)
    assert np.array_equal(actual,expected)
