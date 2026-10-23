"""Streaming output bundles for direct LK panoramas and viewpoint movies."""
from pathlib import Path
import json,subprocess,time,shutil
import cv2,numpy as np
from .io import iter_frames,input_digest
from .lk_pipeline import DirectConfig,prepare_direct


def run_direct(args):
    if args.exposure or args.blend:
        raise ValueError('Direct LK modes use their own composition; exposure/blend flags apply to rectified mode')
    if args.video and not shutil.which('ffmpeg'):raise RuntimeError('FFmpeg is required for --video')
    cfg=DirectConfig(method=args.method,active_area=tuple(args.active_area) if args.active_area else None)
    records=iter_frames(args.input,args.max_side,stride=args.frame_stride)
    if getattr(args,'source_indices',None) is not None:
        selected=set(args.source_indices)
        records=(record for record in records if record[0] in selected)
    scan=prepare_direct(records,cfg)
    alphas=args.sweep or (np.linspace(0,1,args.views).tolist() if args.video else [args.alpha])
    if any(not 0<=a<=1 for a in alphas):raise ValueError('All anchors must lie in [0,1]')
    digest=getattr(args,'input_sha256',None) or input_digest(args.input)
    manifest=dict(scan.evidence,input_name=args.input.name,input_sha256=digest,max_side=args.max_side,
                  frame_stride=args.frame_stride,views=[],opencv_version=cv2.__version__,numpy_version=np.__version__,
                  **getattr(args,'selection_evidence',{}))
    writer=None;render_seconds=0.;encode_seconds=0.
    try:
        for i,alpha in enumerate(alphas):
            started=time.perf_counter();im,mask=scan.render(alpha);render_seconds+=time.perf_counter()-started
            manifest['views'].append(dict(alpha=alpha,canvas_hw=list(im.shape[:2]),coverage_fraction=float(mask.mean())))
            if args.video:
                started=time.perf_counter();h,w=im.shape[:2];hh=h+h%2;ww=w+w%2
                if writer is None:
                    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{ww}x{hh}','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'sweep.mp4')],stdin=subprocess.PIPE)
                padded=np.zeros((hh,ww,3),np.uint8);padded[:h,:w]=im;writer.stdin.write(padded.tobytes());encode_seconds+=time.perf_counter()-started
            if not args.video or i in {0,len(alphas)//4,len(alphas)//2,3*len(alphas)//4,len(alphas)-1}:
                folder=args.output/f'view-{i:03d}' if len(alphas)>1 else args.output;folder.mkdir(exist_ok=True)
                if not cv2.imwrite(str(folder/'panorama.png'),im) or not cv2.imwrite(str(folder/'coverage.png'),mask.astype(np.uint8)*255):raise OSError('Cannot save image output')
        if writer:
            started=time.perf_counter();writer.stdin.close()
            if writer.wait():raise RuntimeError('FFmpeg encoding failed')
            encode_seconds+=time.perf_counter()-started;writer=None
            cap=cv2.VideoCapture(str(args.output/'sweep.mp4'));count=0
            while cap.read()[0]:count+=1
            cap.release()
            if count!=len(alphas):raise RuntimeError(f'Encoded {count} of {len(alphas)} required views')
            manifest.update(decoded_output_frames=count,fps=args.fps)
        if not args.no_provenance and scan.poses is not None:
            np.savez_compressed(args.output/'geometry.npz',poses=scan.poses,slopes=np.asarray(scan.slopes))
        manifest['timing'].update(render=render_seconds,encode=encode_seconds,pipeline_total=manifest['timing']['preparation_total']+render_seconds)
        manifest['geometry_note']='Camera positions and local relative inverse depths; depth views interpolate source observations. Coverage marks observed samples. This mode does not export a single-owner pixel provenance map.'
        (args.output/'run.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    finally:
        if writer:
            writer.stdin.close();writer.wait()
    print(f'Saved {len(alphas)} {args.method} view(s) from {len(scan.frames)} sources in {args.output}')
    return 0
