import json

import cv2
import numpy as np
import pytest

from video_mosaicing.cli import main
from video_mosaicing.synthetic import calibrated_sequence


def test_cli_writes_reproducible_sweep_and_provenance(tmp_path):
    scene=calibrated_sequence(seed=772,count=5)
    inputs=tmp_path/'input frames';inputs.mkdir()
    for i,f in enumerate(scene.frames): cv2.imwrite(str(inputs/f'frame-{i:04d}.png'),f)
    output=tmp_path/'results'
    assert main([str(inputs),'--output',str(output),'--sweep','.1','.5','.9'])==0
    run=json.loads((output/'run.json').read_text())
    assert len(run['views'])==3
    assert len(run['input_sha256'])==64
    for i in range(3):
        view=output/f'view-{i:02d}'
        im=cv2.imread(str(view/'panorama.png'))
        mask=cv2.imread(str(view/'coverage.png'),0)>0
        with np.load(view/'provenance.npz') as provenance:
            assert im.shape[:2]==mask.shape==provenance['source_index'].shape
            assert np.isfinite(provenance['source_xy'][mask]).all()
            assert (provenance['source_index'][mask]>=0).all()
    with pytest.raises(SystemExit): main([str(inputs),'--output',str(output)])


def test_cli_saves_an_explicit_failure_for_blank_input(tmp_path):
    inputs=tmp_path/'blank';inputs.mkdir()
    for i in range(2): cv2.imwrite(str(inputs/f'{i}.png'),np.zeros((100,140,3),np.uint8))
    output=tmp_path/'failed'
    with pytest.raises(SystemExit) as exc: main([str(inputs),'--output',str(output)])
    assert exc.value.code==2
    evidence=json.loads((output/'failure.json').read_text())
    assert 'insufficient' in evidence['error']
    assert not (output/'panorama.png').exists()
