"""Curated input selection, override semantics, and CLI evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from video_mosaicing import cli, direct_cli, presets
from video_mosaicing.io import input_digest
from video_mosaicing.presets import Preset, select_method


def test_reviewed_presets_match_recorded_source_configurations():
    record=Path(__file__).parents[1]/'docs/direct-lk-runs.json'
    runs=json.loads(record.read_text())['runs']
    enhanced={'boat','House','Garden','Kessaria'}
    assert len(presets.PRESETS)==len(runs)==9
    for run in runs:
        selected=select_method(run['input_sha256'])
        assert selected.max_side==800
        assert selected.preset_name==run['name']
        if run['name'] in enhanced:
            assert selected.method=='lk-depth'
            assert selected.frame_stride==run['depth_candidate_stride']
            assert selected.source_indices is None
            area=run['active_area']
        else:
            assert selected.method=='lk-original'
            assert selected.source_indices==tuple(sorted(run['original_source_indices']))
            assert selected.frame_stride==1
            area=run['original_active_area']
        assert selected.active_area==(tuple(area) if area else None)


def test_selection_follows_bytes_after_rename_and_ignores_unknown_names(tmp_path,monkeypatch):
    source=tmp_path/'sample.mp4';source.write_bytes(b'reviewed example bytes')
    digest=input_digest(source)
    monkeypatch.setitem(presets.PRESETS,digest,Preset('example','lk-depth'))
    before=select_method(input_digest(source))
    renamed=source.rename(tmp_path/'completely different name.mp4')
    assert select_method(input_digest(renamed))==before
    renamed.write_bytes(b'different video with the same filename')
    fallback=select_method(input_digest(renamed))
    assert fallback.method=='lk-original'
    assert fallback.preset_name is None
    assert fallback.max_side==640 and fallback.frame_stride==10


def test_explicit_method_preserves_manual_defaults_for_known_input():
    digest=next(h for h,p in presets.PRESETS.items() if p.name=='House')
    for method,stride in [('lk-original',10),('lk-depth',3),('rectified',1)]:
        selection=select_method(digest,requested_method=method)
        assert selection.method==method
        assert selection.max_side==640 and selection.frame_stride==stride
        assert selection.active_area is None and selection.source_indices is None
        assert selection.preset_name is None
        assert selection.requested_method==method


def test_explicit_options_override_only_the_corresponding_curated_defaults():
    digest=next(h for h,p in presets.PRESETS.items() if p.name=='Iguazu')
    selection=select_method(digest,frame_stride=4,max_side=512,active_area=(0,0,1,1))
    assert selection.method=='lk-original'
    assert selection.source_indices is None
    assert selection.frame_stride==4 and selection.max_side==512
    assert selection.active_area==(0,0,1,1)
    assert selection.applied_defaults==()
    # Overriding the scale alone retains the reviewed frame selection and crop.
    changed_scale=select_method(digest,max_side=512)
    assert changed_scale.source_indices is not None
    assert changed_scale.active_area==(.125,0,.875,1)


def test_default_cli_applies_curated_indices_once_and_records_selection(tmp_path,monkeypatch):
    folder=tmp_path/'input';folder.mkdir()
    for i in range(8):
        assert cv2.imwrite(str(folder/f'{i:02d}.png'),np.full((40,60,3),i,np.uint8))
    digest=input_digest(folder)
    monkeypatch.setitem(presets.PRESETS,digest,
                        Preset('reviewed fixture','lk-original',source_indices=(0,3,7),frame_stride=1))
    hash_calls=[]
    def hash_once(path):
        hash_calls.append(path)
        return input_digest(path)
    monkeypatch.setattr(cli,'input_digest',hash_once)
    def unexpected_hash(path):
        raise AssertionError('CLI digest must be reused')
    monkeypatch.setattr(direct_cli,'input_digest',unexpected_hash)
    def prepare(records,config):
        records=list(records)
        assert [index for index,_ in records]==[0,3,7]
        assert config.method=='lk-original'
        return SimpleNamespace(
            evidence=dict(method=config.method,source_indices=[0,3,7],timing=dict(preparation_total=0)),
            frames=[frame for _,frame in records],poses=None,
            render=lambda alpha:(records[0][1],np.ones((40,60),bool)))
    monkeypatch.setattr(direct_cli,'prepare_direct',prepare)
    output=tmp_path/'result'
    assert cli.main([str(folder),'--output',str(output),'--no-provenance'])==0
    assert len(hash_calls)==1
    manifest=json.loads((output/'run.json').read_text())
    assert manifest['requested_method']=='auto' and manifest['method']=='lk-original'
    assert manifest['input_sha256']==digest
    assert manifest['max_side']==800 and manifest['source_indices']==[0,3,7]
    selection=manifest['method_selection']
    assert selection['preset_name']=='reviewed fixture'
    assert selection['source_selection']=='curated_indices'
    assert 'SHA256' in selection['reason']
