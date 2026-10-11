import cv2
import numpy as np
import pytest

from video_mosaicing.geometry import GeometryError
from video_mosaicing.pipeline import MosaicConfig,build_mosaic
from video_mosaicing.registration import MotionError
from video_mosaicing.synthetic import calibrated_sequence


@pytest.mark.parametrize("direction",["LR","RL","TB","BT"])
def test_direction_and_original_coordinate_provenance(direction):
    scene = calibrated_sequence(seed=621,count=9)
    frames = scene.frames
    if direction=="RL": frames = [cv2.flip(f,1) for f in frames]
    if direction=="TB": frames = [cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE) for f in frames]
    if direction=="BT": frames = [cv2.rotate(f,cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
    r = build_mosaic([(i*3,f) for i,f in enumerate(frames)])
    assert r.evidence['direction']==direction
    assert r.coverage.mean() > .96
    ys,xs = np.nonzero(r.coverage)
    take = np.arange(0,len(xs),101)
    for sid in np.unique(r.source_index[ys[take],xs[take]]):
        select = r.source_index[ys[take],xs[take]]==sid
        y,x = ys[take][select],xs[take][select]
        xy = r.source_xy[y,x]
        pixels = cv2.remap(frames[sid//3],xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR)[:,0]
        np.testing.assert_allclose(r.image[y,x].astype(float),pixels.astype(float),atol=1)


def test_near_duplicates_are_skipped_but_scan_coverage_is_retained():
    scene = calibrated_sequence(seed=621,count=9)
    frames = [f for f in scene.frames for _ in range(3)]
    result = build_mosaic(enumerate(frames))
    assert len(result.evidence['selected_source_indices'])==9
    assert len(result.evidence['skipped'])==18
    assert result.evidence['decoded_frames']==27
    assert result.evidence['selected_source_indices'][-1] >= 24


def test_stationary_texture_is_reported():
    scene = calibrated_sequence(seed=621,count=2)
    with pytest.raises(MotionError,match="No observable scan"):
        build_mosaic(enumerate([scene.frames[0]]*10))


@pytest.mark.parametrize("alpha",[.1,.5,.9])
def test_explicit_matte_bounds_preserve_scene_and_original_coordinates(alpha):
    scene=calibrated_sequence(seed=621,count=9)
    frames=[cv2.copyMakeBorder(f,0,0,40,40,cv2.BORDER_CONSTANT) for f in scene.frames]
    result=build_mosaic(enumerate(frames),MosaicConfig(alpha=alpha,active_area=(.1,0,.9,1)))
    assert result.coverage.mean()>.96
    assert not np.all(result.image[result.coverage]==0,axis=-1).any()
    assert result.source_xy[result.coverage,0].min()>=40
    assert result.source_xy[result.coverage,0].max()<=359
    ys,xs=np.nonzero(result.coverage)
    for y,x in zip(ys[::211],xs[::211]):
        sid=result.source_index[y,x];xy=result.source_xy[y,x]
        expected=cv2.remap(frames[sid],np.array([[xy[0]]],np.float32),np.array([[xy[1]]],np.float32),cv2.INTER_LINEAR)[0,0]
        np.testing.assert_allclose(result.image[y,x].astype(float),expected.astype(float),atol=1)


def test_anchor_in_encoded_matte_is_reported_instead_of_a_black_panorama():
    scene=calibrated_sequence(seed=621,count=9)
    frames=[cv2.copyMakeBorder(f,0,0,40,40,cv2.BORDER_CONSTANT) for f in scene.frames]
    with pytest.raises(GeometryError,match="active-area"):
        build_mosaic(enumerate(frames),MosaicConfig(alpha=.05))
