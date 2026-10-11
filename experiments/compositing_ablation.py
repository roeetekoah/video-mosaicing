"""Compare compositing while keeping the estimated frames and geometry fixed."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from video_mosaicing.io import input_digest,iter_frames,save_result
from video_mosaicing.pipeline import MosaicConfig,PreparedScan,crop_active,orient,render_scan
from video_mosaicing.registration import MotionError,RegistrationConfig


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('evidence',type=Path,help="evidence.json from a successful rectified run")
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    evidence=json.loads(args.evidence.read_text())
    if evidence.get('input_sha256')!=input_digest(args.input): parser.error("Input hash differs from motion evidence")
    cv2.setNumThreads(1);cv2.setRNGSeed(47)
    indices=evidence['selected_source_indices'];wanted=set(indices)
    area=evidence['config'].get('active_area')
    frames=[orient(crop_active(f,area),evidence['direction']) for i,f in iter_frames(args.input,evidence.get('max_side',640)) if i in wanted]
    scan=PreparedScan(frames,indices,[np.array(p['matrix']) for p in evidence['pairs']],evidence['direction'],
                      np.array(evidence['source_to_normalized']),
                      {k:evidence[k] for k in ('selected_source_indices','decoded_frames','skipped','pairs')})
    cfg=MosaicConfig(registration=RegistrationConfig(**evidence['config']['registration']),alpha=evidence['config']['alpha'],active_area=area)
    for name,settings in [('hard',cfg),('narrow',replace(cfg,blend_width=2)),('exposure',replace(cfg,exposure=True)),('exposure-narrow',replace(cfg,exposure=True,blend_width=2))]:
        output=args.output/name
        if output.exists(): parser.error("Ablation output already exists")
        try:
            result=render_scan(scan,settings)
            result.evidence.update(input_name=args.input.name,input_sha256=evidence['input_sha256'],motion_evidence=args.evidence.name)
            save_result(result,output,provenance=False)
            print(name,'completed',flush=True)
        except MotionError as error:
            output.mkdir(parents=True)
            (output/'failure.json').write_text(json.dumps(dict(error=str(error),evidence=error.evidence),indent=2),encoding='utf-8')
            print(name,'rejected:',error,flush=True)
