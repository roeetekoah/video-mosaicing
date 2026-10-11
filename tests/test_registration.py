import numpy as np
import pytest

from video_mosaicing.geometry import project
from video_mosaicing.registration import MotionError, estimate_motion
from video_mosaicing.synthetic import calibrated_sequence, texture


@pytest.mark.parametrize("kind",["translation","tilted_pan","tilted_wall"])
@pytest.mark.parametrize("seed",[219,913])
def test_calibrated_registration_with_reserved_points(kind,seed):
    scene = calibrated_sequence(kind=kind,seed=seed,count=9)
    points = np.array([[50,45],[270,45],[270,195],[50,195],[160,120]],float)
    for i in (0,3,7):
        m,evidence = estimate_motion(scene.frames[i],scene.frames[i+1])
        err = np.linalg.norm(project(m,points)-project(scene.homographies[i],points),axis=1)
        assert err.max() < .5
        assert evidence['validation']['median_px'] < .5
        first = evidence['candidates'][0]
        assert first['train_count']+first['validation_count']==evidence['matches']


@pytest.mark.parametrize("kind",["constant","single_pixel","aperture","scene_cut","low_overlap"])
def test_degenerate_inputs_are_explicit_failures(kind):
    a = np.zeros((240,320,3),np.uint8)
    if kind=="single_pixel": a[80,110]=255
    if kind=="aperture": a[:,::8]=255
    b = np.roll(a,20,axis=1)
    if kind=="scene_cut": a,b = texture(13,(240,320)),texture(99,(240,320))
    if kind=="low_overlap":
        world = texture(15,(240,700)); a,b = world[:,:320],world[:,300:620]
    with pytest.raises(MotionError): estimate_motion(a,b)


def test_exposure_change_does_not_break_feature_registration():
    scene = calibrated_sequence(kind="translation",seed=829,count=13)
    b = np.clip(scene.frames[1].astype(float)*1.15+20,0,255).astype(np.uint8)
    m,_ = estimate_motion(scene.frames[0],b)
    p = np.array([[50,50],[270,190]],float)
    assert np.max(np.linalg.norm(project(m,p)-project(scene.homographies[0],p),axis=1)) < .5
