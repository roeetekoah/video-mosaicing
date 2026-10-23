"""Command-line interface for single panoramas and controlled viewpoint sweeps."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .geometry import GeometryError
from .io import input_digest,iter_frames,save_result
from .pipeline import MosaicConfig,prepare_scan,render_scan,render_scan_sweep
from .presets import select_method
from .registration import MotionError,RegistrationConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rectified strip panoramas from a continuous camera scan")
    parser.add_argument('input',type=Path,help="video or lexically ordered image folder")
    parser.add_argument('--output',type=Path,default=Path('demo-output/custom'))
    parser.add_argument('--max-side',type=int,default=None,help='maximum source dimension; reviewed auto examples 800, otherwise 640')
    parser.add_argument('--method',choices=['auto','lk-depth','lk-original','rectified'],default='auto',help='auto uses reviewed configurations for exact example inputs and original LK for unknown inputs; explicit methods override selection')
    parser.add_argument('--frame-stride',type=int,default=None,help="decode every Nth frame; overrides curated source selection; other defaults: depth 3, original 10, rectified 1")
    parser.add_argument('--video',action='store_true',help='encode a complete sweep with FFmpeg')
    parser.add_argument('--views',type=int,default=101,help='number of video viewpoints')
    parser.add_argument('--fps',type=float,default=15)
    parser.add_argument('--spacing',type=float,default=6,help="minimum useful horizontal displacement in pixels")
    parser.add_argument('--alpha',type=float,default=.5,help="anchor fraction along scan axis; normally 0.1 to 0.9")
    parser.add_argument('--sweep',type=float,nargs='+',help="render several anchors using one motion-estimation pass")
    parser.add_argument('--model',choices=['auto','rigid','similarity','affine','homography'],default='auto')
    parser.add_argument('--active-area',type=float,nargs=4,metavar=('LEFT','TOP','RIGHT','BOTTOM'),help="explicit scene bounds as fractions of the input, excluding encoded matte")
    parser.add_argument('--exposure',action='store_true',help="robust overlap gain/bias correction")
    parser.add_argument('--blend',type=int,default=0,help="optional feather half-width, 0 to 16 pixels")
    parser.add_argument('--no-provenance',action='store_true',help="omit the larger source-coordinate file")
    args = parser.parse_args(argv)
    if not args.input.exists(): parser.error("Input does not exist")
    args.input_sha256=input_digest(args.input)
    try:
        selection=select_method(args.input_sha256,requested_method=args.method,max_side=args.max_side,
                                frame_stride=args.frame_stride,active_area=args.active_area)
    except ValueError as error:parser.error(str(error))
    args.method=selection.method;args.max_side=selection.max_side;args.frame_stride=selection.frame_stride
    args.active_area=selection.active_area;args.source_indices=selection.source_indices
    args.selection_evidence=selection.evidence()
    if args.views<2 or args.fps<=0:parser.error('At least two video views and positive fps required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve prior evidence")
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);cv2.setRNGSeed(47)
    if args.method!='rectified':
        from .direct_cli import run_direct
        try:return run_direct(args)
        except (ValueError,OSError,RuntimeError,cv2.error) as error:
            (args.output/'failure.json').write_text(json.dumps(dict(error=str(error),method=args.method,**args.selection_evidence)),encoding='utf-8')
            parser.exit(2,f'{error}\nDetails: {args.output / "failure.json"}\n')
    if args.video:parser.error('--video is available for direct LK methods; use export_sweep_video.py for rectified sweeps')
    started = time.perf_counter()
    try:
        cfg = MosaicConfig(registration=RegistrationConfig(model=args.model),alpha=args.alpha,
                           min_step_px=args.spacing,exposure=args.exposure,blend_width=args.blend,
                           active_area=tuple(args.active_area) if args.active_area else None)
        views = args.sweep or [args.alpha]
        if any(not 0<=v<=1 for v in views): raise ValueError("All anchors must lie in [0,1]")
        scan = prepare_scan(iter_frames(args.input,args.max_side,stride=args.frame_stride),cfg)
        estimate_seconds = time.perf_counter()-started
        digest = args.input_sha256
        manifest = dict(input_name=args.input.name,input_sha256=digest,max_side=args.max_side,
                        frame_stride=args.frame_stride,opencv_version=cv2.__version__,numpy_version=np.__version__,
                        registration_seconds=estimate_seconds,views=[],**args.selection_evidence)
        results = render_scan_sweep(scan,views,cfg) if args.sweep else iter([render_scan(scan,cfg)])
        for i,alpha in enumerate(views):
            render_start = time.perf_counter()
            result = next(results)
            result.evidence.update(input_name=args.input.name,input_sha256=digest,max_side=args.max_side,
                                   frame_stride=args.frame_stride,opencv_version=cv2.__version__,numpy_version=np.__version__,
                                   registration_seconds=estimate_seconds,render_seconds=time.perf_counter()-render_start)
            destination = args.output/f'view-{i:02d}' if args.sweep else args.output
            save_result(result,destination,provenance=not args.no_provenance)
            manifest['views'].append(dict(alpha=alpha,directory=destination.name,canvas_hw=list(result.image.shape[:2]),coverage_fraction=result.evidence['coverage_fraction']))
        (args.output/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    except (MotionError,GeometryError,ValueError) as error:
        failure = dict(error=str(error),evidence=getattr(error,'evidence',{}))
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        parser.exit(2,f'{error}\nDetails: {args.output / "failure.json"}\n')
    print(f"Saved {len(views)} view(s) from {len(scan.frames)} selected frames in {args.output}")
    return 0


if __name__=='__main__': main()
