import numpy as np

from video_mosaicing.geometry import render_rectified
from video_mosaicing.photometry import blend_boundaries,exposure_gains
from video_mosaicing.synthetic import calibrated_sequence


def test_known_exposure_is_recovered_without_changing_geometry():
    scene = calibrated_sequence(seed=632,count=13,exposure=True)
    transforms,evidence = exposure_gains(scene.frames,scene.homographies)
    t = np.linspace(0,1,len(scene.frames))
    expected_g = (1+.25*t[len(t)//2])/(1+.25*t)
    expected_b = 12*t[len(t)//2]-expected_g*12*t
    np.testing.assert_allclose(transforms[:,0],expected_g,atol=.02)
    np.testing.assert_allclose(transforms[:,1],expected_b,atol=2)
    raw = render_rectified(scene.frames,scene.homographies)
    corrected = render_rectified(scene.frames,scene.homographies,gains=transforms)
    np.testing.assert_array_equal(raw.coverage,corrected.coverage)
    np.testing.assert_array_equal(raw.source_index,corrected.source_index)
    clean = calibrated_sequence(seed=632,count=13)
    target = render_rectified([np.clip(f.astype(float)*1.125+6,0,255).astype(np.uint8) for f in clean.frames],clean.homographies)
    def error(result): return np.abs(result.image[result.coverage].astype(float)-target.image[result.coverage].astype(float)).mean()
    assert error(corrected) < .2*error(raw)


def test_narrow_blend_preserves_aligned_detail_and_never_fills_holes():
    scene = calibrated_sequence(seed=331,count=13)
    raw = render_rectified(scene.frames,scene.homographies)
    blended,evidence = blend_boundaries(raw,scene.frames,scene.homographies,2)
    assert not blended[~raw.coverage].any()
    assert np.abs(blended.astype(float)-raw.image.astype(float))[raw.coverage].mean() < 1
    assert evidence['blended_pixels'] > 0
