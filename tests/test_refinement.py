import cv2
import numpy as np
import pytest

from video_mosaicing.data_structures import RigidTransform
from video_mosaicing.geometry import project
from video_mosaicing.refinement import RefinementConfig,refine_rigid,residual_jacobian
from video_mosaicing.registration import MotionError
from video_mosaicing.synthetic import texture


@pytest.mark.parametrize("photometric",[False,True])
def test_complete_jacobian_matches_finite_differences(photometric):
    rng = np.random.default_rng(317)
    a,b = rng.random((80,100)),rng.random((80,100))
    points = rng.uniform((10,10),(80,60),(150,2))
    params = np.array([1.37,-.72,.13,1.07,.03] if photometric else [1.37,-.72,.13])
    residual,jac,valid = residual_jacobian(a,b,points,params,photometric=photometric)
    for i in range(len(params)):
        delta = np.zeros_like(params);delta[i]=1e-6
        plus = residual_jacobian(a,b,points,params+delta,photometric=photometric)[0]
        minus = residual_jacobian(a,b,points,params-delta,photometric=photometric)[0]
        np.testing.assert_allclose(jac[valid,i],((plus-minus)/2e-6)[valid],atol=1e-7)


@pytest.mark.parametrize("angle",[0.,.035,-.025])
def test_direct_rigid_refinement_recovers_textured_motion(angle):
    a = cv2.cvtColor(texture(529,(160,192)),cv2.COLOR_BGR2GRAY).astype(float)/255
    truth = RigidTransform(5,-3,angle).as_matrix(*a.shape)
    b = cv2.warpPerspective(a,truth,(a.shape[1],a.shape[0]))
    m,evidence = refine_rigid(a,b)
    corners = np.array([[20,20],[170,20],[170,140],[20,140]])
    assert np.max(np.linalg.norm(project(m,corners)-project(truth,corners),axis=1))<.5


def test_direct_refinement_rejects_single_pixel():
    a = np.zeros((160,192));a[60,70]=1
    with pytest.raises(MotionError): refine_rigid(a,np.roll(a,20,axis=1))
