"""Reproduce the curated baseline and the first rectified pipeline on a video.

Use only footage you are entitled to process. Inputs are never copied into the
result directory. Supplied coursework footage is for local evaluation only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from video_mosaicing.io import input_digest,iter_frames,save_result
from video_mosaicing.legacy_pipeline import MosaicConfig as LegacyConfig,mosaic as legacy_mosaic
from video_mosaicing.pipeline import MosaicConfig,crop_active,prepare_scan,render_scan
from video_mosaicing.registration import MotionError


def compare(path,output,area=None):
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);cv2.setRNGSeed(47)
    cap = cv2.VideoCapture(str(path)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    indices = set(np.unique(np.linspace(0,count-1,min(count,int(np.clip(.1*count,40,120)))).round().astype(int)))
    legacy_frames = [crop_active(f,area) for i,f in iter_frames(path) if i in indices]
    legacy_cfg = LegacyConfig()
    # Match the original wrapper's 30 iterations and preblur 1.5.
    from dataclasses import replace
    from video_mosaicing.data_structures import LKConfig
    legacy_cfg = replace(legacy_cfg,lk=LKConfig(num_iters=30,border=8),blur_sigma=1.5,margin=50)
    t = time.perf_counter()
    old,mask,evidence = legacy_mosaic(legacy_frames,legacy_cfg,allow_unreliable=True)
    baseline_time = time.perf_counter()-t
    legacy_dir = output/'legacy';legacy_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(legacy_dir/'panorama.png'),old)
    cv2.imwrite(str(legacy_dir/'coverage.png'),mask.astype(np.uint8)*255)
    evidence.update(source_indices=sorted(int(i) for i in indices),runtime_seconds=baseline_time,active_area=area)
    (legacy_dir/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(path.stem,'legacy',old.shape,'seconds',round(baseline_time,2),flush=True)
    t = time.perf_counter()
    try:
        scan = prepare_scan(iter_frames(path),MosaicConfig(active_area=area))
        (output/'motion.json').write_text(json.dumps(dict(direction=scan.direction,orientation_matrix=scan.orientation_matrix.tolist(),homographies=[m.tolist() for m in scan.homographies],evidence=scan.evidence),indent=2),encoding='utf-8')
        np.savez_compressed(output/'selected-frames.npz',frames=np.stack(scan.frames))
        new = render_scan(scan,MosaicConfig(active_area=area))
        new_time = time.perf_counter()-t
        new.evidence.update(runtime_seconds=new_time,input_name=path.name,input_sha256=input_digest(path),opencv_version=cv2.__version__,numpy_version=np.__version__)
        save_result(new,output/'rectified')
        print(path.stem,'rectified',new.image.shape,'selected',len(new.evidence['selected_source_indices']),'seconds',round(new_time,2),flush=True)
    except (MotionError,ValueError) as error:
        failure = dict(error=str(error),evidence=getattr(error,'evidence',{}),runtime_seconds=time.perf_counter()-t)
        (output/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        print(path.stem,'FAILED',str(error),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--active-area',type=float,nargs=4)
    args = parser.parse_args()
    for path in args.input: compare(path,args.output/path.stem,tuple(args.active_area) if args.active_area else None)
