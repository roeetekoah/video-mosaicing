import numpy as np
import pytest

from video_mosaicing.geometry import GeometryError, project, render_rectified, symmetric_pair
from video_mosaicing.synthetic import calibrated_sequence


def test_translation_reduces_to_exact_strip_sampling():
    # Unique per-pixel coordinates make dropped or duplicated columns observable.
    h,w,step = 65,91,12
    yy,xx = np.mgrid[:h,:w+3*step]
    world = np.stack((xx,yy,(xx+yy)%256),-1).astype(np.uint8)
    frames = [world[:,i*step:i*step+w].copy() for i in range(4)]
    hom = np.array([[1,0,-step],[0,1,0],[0,0,1.]])
    result = render_rectified(frames,[hom]*3)
    assert result.image.shape == world.shape
    assert result.coverage.all()
    np.testing.assert_array_equal(result.image,world)
    assert result.evidence["inverse_stretch_range"] == pytest.approx([1,1],abs=1e-7)


@pytest.mark.parametrize("kind",["tilted_pan","tilted_wall","variable_speed"])
@pytest.mark.parametrize("alpha",[.1,.5,.9])
def test_oracle_boundaries_and_no_folds(kind,alpha):
    scene = calibrated_sequence(kind=kind,seed=913)
    result = render_rectified(scene.frames,scene.homographies,alpha=alpha)
    assert result.evidence["boundary_correspondence_max_px"] < .25
    assert result.coverage.mean() > .85
    assert result.evidence["inverse_stretch_range"][1] < 2
    for i,m in enumerate(scene.homographies):
        pair = symmetric_pair(m,scene.frames[0].shape[:2],alpha)
        v = np.linspace(0,1,151)
        a = pair.sample("right",np.ones_like(v),v)
        b = pair.sample("left",np.zeros_like(v),v)
        # Independent camera projection, not just renderer's reported score.
        world_xy = project(np.linalg.inv(scene.plane_to_images[i]),a)
        np.testing.assert_allclose(project(scene.plane_to_images[i+1],world_xy),b,atol=.25)
    assert np.isfinite(result.source_xy[result.coverage]).all()
    assert (result.source_index[result.coverage] >= 0).all()


@pytest.mark.parametrize("m",[np.zeros((3,3)),np.array([[1,0,10],[0,1,0],[0,0,1.]]),np.array([[1,0,-20],[0,-1,200],[0,0,1.]])])
def test_rejects_singular_reverse_and_folded_geometry(m):
    with pytest.raises(GeometryError): symmetric_pair(m,(240,320))


def test_vertical_offsets_preserve_translation_instead_of_detrending():
    p = symmetric_pair(np.array([[1,0,-18],[0,1,-4],[0,0,1.]]),(120,160))
    assert p.dy == pytest.approx(-4)
    assert p.width == pytest.approx(18)


@pytest.mark.parametrize("alpha",[0.,1.])
def test_border_anchor_reports_missing_strip_support(alpha):
    with pytest.raises(GeometryError,match="strip support"):
        symmetric_pair(np.array([[1,0,-18],[0,1,0],[0,0,1.]]),(120,160),alpha)
