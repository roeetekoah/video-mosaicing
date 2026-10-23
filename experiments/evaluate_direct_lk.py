"""Reproduce one recorded direct-LK benchmark from the supplied input footage."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,hashlib
from pathlib import Path
import cv2,numpy as np
from video_mosaicing.io import iter_frames,input_digest
from video_mosaicing.lk_pipeline import prepare_direct,DirectConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--clip',required=True,help='clip key in the recorded protocol')
    parser.add_argument('--method',choices=['lk-original','lk-depth'],default='lk-depth')
    parser.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'docs/direct-lk-runs.json')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose an empty output directory')
    protocol=json.loads(args.protocol.read_text());record=next((r for r in protocol['runs'] if r['name']==args.clip),None)
    if record is None:parser.error('Unknown clip key')
    if input_digest(args.input)!=record['input_sha256']:parser.error('Input hash differs from the recorded footage')
    args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    if args.method=='lk-original':
        selected=set(record['original_source_indices'])
        inputs=((i,f) for i,f in iter_frames(args.input,800) if i in selected)
    else:inputs=iter_frames(args.input,800,stride=3)
    area=record['original_active_area'] if args.method=='lk-original' else record['active_area']
    scan=prepare_direct(inputs,DirectConfig(method=args.method,active_area=tuple(area) if area else None))
    elapsed=0.;hashes=[]
    for i in range(101):
        tick=time.perf_counter();image,mask=scan.render(i/100);elapsed+=time.perf_counter()-tick
        hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        if i in (0,25,50,75,100):
            cv2.imwrite(str(args.output/f'view-{i:03d}.png'),image)
            cv2.imwrite(str(args.output/f'coverage-{i:03d}.png'),mask.astype(np.uint8)*255)
    result=dict(scan.evidence,frame_hashes=hashes)
    result['timing'].update(render=elapsed,pipeline_total=scan.evidence['timing']['preparation_total']+elapsed)
    (args.output/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['timing']))


if __name__=='__main__':main()
