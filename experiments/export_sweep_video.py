"""Export real pipeline panoramas as a full-canvas H.264 viewpoint-sweep video.

Requires FFmpeg on PATH. Each view is rendered separately by render_scan;
encoding adds black padding to a common canvas, without morphing between views.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from video_mosaicing.io import input_digest, iter_frames, save_result
from video_mosaicing.pipeline import MosaicConfig, prepare_scan, render_scan


def encode_views(images, output, fps=8):
    """Encode BGR images at their native pixel scale, with padding only."""
    height = max(im.shape[0] for im in images)
    width = max(im.shape[1] for im in images)
    height += height % 2
    width += width % 2
    sequence = list(range(len(images))) + list(range(len(images)-2, 0, -1))
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for index in sequence:
            image = images[index]
            canvas = np.zeros((height, width, 3), np.uint8)
            canvas[:image.shape[0], :image.shape[1]] = image
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg could not encode the sweep')
    return dict(canvas_hw=[height, width], fps=fps, sequence=sequence,
                presentation='Native pixel scale; black padding only; no interpolated or fabricated views')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--active-area', type=float, nargs=4)
    parser.add_argument('--steps', type=int, default=21)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--max-side', type=int, default=640)
    args = parser.parse_args()
    if args.steps < 2 or args.fps < 1:
        parser.error('At least two views and a positive frame rate are required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(47)
    cfg = MosaicConfig(active_area=tuple(args.active_area) if args.active_area else None)
    scan = prepare_scan(iter_frames(args.input, args.max_side), cfg)
    digest = input_digest(args.input)
    images = []
    alphas = np.linspace(.1, .9, args.steps)
    for index, alpha in enumerate(alphas):
        result = render_scan(scan, replace(cfg, alpha=float(alpha)))
        result.evidence.update(input_name=args.input.name, input_sha256=digest, max_side=args.max_side)
        save_result(result, args.output/f'view-{index:02d}', provenance=False)
        images.append(result.image)
    evidence = encode_views(images, args.output/'sweep.mp4', args.fps)
    evidence.update(input_name=args.input.name, input_sha256=digest, anchors=alphas.tolist(),
                    selected_source_indices=scan.indices)
    (args.output/'video.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(f'Encoded {len(images)} actual pipeline views in {args.output}')


if __name__ == '__main__':
    main()
