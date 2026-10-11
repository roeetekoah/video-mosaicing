"""Compare motion estimators on the same reserved feature correspondences.

Validation residuals are correspondence proxies. These development clips are
not an untouched test set, and moving content can produce mixed motions.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_mosaicing.data_structures import LKConfig
from video_mosaicing.geometry import project
from video_mosaicing.io import iter_frames,input_digest
from video_mosaicing.lucas_kanade import estimate_rigid_lk_pyramids
from video_mosaicing.preprocess import preprocess_frame
from video_mosaicing.pipeline import crop_active
from video_mosaicing.registration import MotionError,RegistrationConfig,correspondences,detect,estimate_features


def benchmark(path,area=None):
    frames = {i:crop_active(f,area) for i,f in iter_frames(path)}
    n = len(frames)
    indices = np.unique(np.linspace(0,n-1,min(n,int(np.clip(.1*n,40,120)))).round().astype(int))
    cfg = RegistrationConfig()
    records = []
    for p in np.linspace(0,len(indices)-2,5).round().astype(int):
        ia,ib = int(indices[p]),int(indices[p+1])
        a,b = frames[ia],frames[ib]
        fa,fb = detect(a,cfg),detect(b,cfg)
        aa,bb = correspondences(fa,fb,cfg)
        order = np.random.default_rng(cfg.seed).permutation(len(aa))
        test = order[int(.7*len(aa)):]
        failure = None
        try: motion,evidence = estimate_features(fa,fb,a.shape[:2],cfg)
        except MotionError as error:
            motion,evidence,failure = None,error.evidence,str(error)
        lk = estimate_rigid_lk_pyramids(preprocess_frame(a,blur_sigma=1.5),preprocess_frame(b,blur_sigma=1.5),LKConfig(num_iters=30,border=8)).as_matrix(*a.shape[:2])
        item = dict(source_indices=[ia,ib],validation_count=len(test),selected_model=evidence.get('model'),failure=failure,candidates=evidence['candidates'],errors={})
        for model,m in [('legacy_lk',lk),('robust_selected',motion)]:
            if m is None:
                item['errors'][model] = None
                continue
            error = np.linalg.norm(project(m,aa[test])-bb[test],axis=1)
            item['errors'][model] = dict(median_px=float(np.median(error)),p90_px=float(np.percentile(error,90)),fraction_under_2px=float(np.mean(error<2)))
        records.append(item)
    comparable = [r for r in records if r['errors']['robust_selected'] is not None]
    summary = {model:float(np.median([r['errors'][model]['median_px'] for r in comparable])) if comparable else None for model in ('legacy_lk','robust_selected')}
    return dict(input_name=path.name,input_sha256=input_digest(path),seed=47,max_side=640,active_area=area,pairs=records,accepted_pairs=len(comparable),total_pairs=len(records),median_of_comparable_pair_medians_px=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--active-area',type=float,nargs=4)
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.setRNGSeed(47)
    results=[]
    for path in args.input:
        item=benchmark(path,tuple(args.active_area) if args.active_area else None);results.append(item)
        print(path.name,item['median_of_comparable_pair_medians_px'],'accepted',item['accepted_pairs'],'/',item['total_pairs'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(note=__doc__,clips=results),indent=2),encoding='utf-8')
