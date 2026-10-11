import cv2
import numpy as np
import pytest

from video_mosaicing.geometry import project
from video_mosaicing.pipeline import prepare_scan,render_scan_sweep
from video_mosaicing.sweep import prepare_atlas,render_atlas_view
from video_mosaicing.synthetic import calibrated_sequence,rotation


def test_translation_sweep_changes_observations_without_moving_the_world():
    h,w,step = 65,91,12
    yy,xx = np.mgrid[:h,:w+7*step]
    world = np.stack((xx,yy,(xx+yy)%256),-1).astype(np.uint8)
    frames = [world[:,i*step:i*step+w].copy() for i in range(8)]
    matrix = np.array([[1,0,-step],[0,1,0],[0,0,1.]])
    atlas = prepare_atlas(frames,[matrix]*7)
    views = [render_atlas_view(atlas,alpha=a) for a in (.1,.5,.9)]
    for result in views:
        np.testing.assert_array_equal(result.image,world)
        assert result.coverage.all()
    assert np.mean(views[0].source_index != views[2].source_index) > .5


@pytest.mark.parametrize('kind',['tilted_pan','tilted_wall','variable_speed'])
def test_perspective_sweep_keeps_independent_world_rays_fixed(kind):
    # Camera/plane matrices belong to the independent scene generator, not the
    # sweep atlas. Fixed canvas dimensions alone would not detect local wobble.
    scene = calibrated_sequence(kind=kind,seed=913)
    atlas = prepare_atlas(scene.frames,scene.homographies)
    base = atlas.reference
    source0 = np.empty_like(base.source_xy,dtype=float)
    for i,matrix in enumerate(scene.plane_to_images):
        valid = base.source_index == i
        source0[valid] = project(np.linalg.inv(matrix),base.source_xy[valid])
    changed = 0
    for alpha in np.linspace(.1,.9,9):
        view = render_atlas_view(atlas,alpha=float(alpha))
        assert view.image.shape == base.image.shape
        np.testing.assert_array_equal(view.coverage,base.coverage)
        assert view.coverage.mean() > .7
        changed += np.count_nonzero(view.coverage & (view.source_index != base.source_index))
        for i,matrix in enumerate(scene.plane_to_images):
            ys,xs = np.nonzero(view.source_index == i)
            ys,xs = ys[::59],xs[::59]
            if not len(ys): continue
            actual = project(np.linalg.inv(matrix),view.source_xy[ys,xs])
            np.testing.assert_allclose(actual,source0[ys,xs],atol=2e-6)
            pixels = cv2.remap(scene.frames[i],view.source_xy[ys,xs,0,None],
                               view.source_xy[ys,xs,1,None],cv2.INTER_LINEAR)[:,0]
            np.testing.assert_array_equal(view.image[ys,xs],pixels)
    assert changed > base.coverage.sum()


def test_wide_pan_does_not_wrap_behind_camera_or_across_visibility_gaps():
    h,w,count = 65,91,80
    k = np.array([[86,0,45],[0,86,32],[0,0,1.]])
    # One complete turn makes the first and last projective matrices nearly
    # identical. End caps must still use their own connected observations.
    poses = [k @ rotation(yaw=t).T for t in np.linspace(0,2*np.pi,count)]
    homographies = [b @ np.linalg.inv(a) for a,b in zip(poses,poses[1:])]
    frames = [np.full((h,w,3),i,np.uint8) for i in range(count)]
    atlas = prepare_atlas(frames,homographies)
    for alpha in (.1,.5,.9):
        result = render_atlas_view(atlas,alpha=alpha)
        assert result.source_index[h//2,0] < 3
        assert result.source_index[h//2,-2] > count-4
        np.testing.assert_array_equal(result.coverage,atlas.reference.coverage)


def test_layered_sweep_retains_real_view_changes_instead_of_freezing_a_still():
    scene = calibrated_sequence(count=17,layered=True,moving=True)
    atlas = prepare_atlas(scene.frames,scene.homographies)
    a,b = [render_atlas_view(atlas,alpha=alpha) for alpha in (.2,.8)]
    valid = a.coverage & b.coverage
    changed = np.max(np.abs(a.image.astype(float)-b.image.astype(float)),axis=2)>30
    assert (changed & valid).sum() > 500
    assert np.mean(a.source_index[valid] != b.source_index[valid]) > .5


@pytest.mark.parametrize('direction',['LR','RL','TB','BT'])
def test_sweep_restores_all_directions_and_source_pixel_provenance(direction):
    scene = calibrated_sequence(seed=621,count=9)
    frames = scene.frames
    if direction == 'RL': frames = [cv2.flip(f,1) for f in frames]
    if direction == 'TB': frames = [cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE) for f in frames]
    if direction == 'BT': frames = [cv2.rotate(f,cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
    scan = prepare_scan([(i*3,f) for i,f in enumerate(frames)])
    views = list(render_scan_sweep(scan,[.1,.5,.9]))
    assert len({v.image.shape for v in views}) == 1
    for view in views:
        assert view.evidence['direction'] == direction
        ys,xs = np.nonzero(view.coverage)
        ys,xs = ys[::337],xs[::337]
        for sid in np.unique(view.source_index[ys,xs]):
            take = view.source_index[ys,xs] == sid
            xy = view.source_xy[ys[take],xs[take]]
            pixels = cv2.remap(frames[sid//3],xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR)[:,0]
            np.testing.assert_allclose(view.image[ys[take],xs[take]],pixels,atol=1)
