"""Rebuild the public moving preview from procedural observations and real output.

Requires ffmpeg on PATH in addition to the project dependencies. No external
footage, ground-truth transforms, or fabricated panorama pixels are used.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import cv2
from PIL import Image, ImageDraw, ImageFont

from video_mosaicing.pipeline import build_mosaic
from video_mosaicing.scene import camera_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('assets'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(47)
    _, frames, _ = camera_sequence()
    result = build_mosaic(enumerate(frames))
    title = ImageFont.load_default(size=28)
    body = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=15)
    rgb = lambda a: Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    pano = rgb(result.image)
    pano.thumbnail((944, 252), Image.Resampling.LANCZOS)
    with tempfile.TemporaryDirectory(prefix='mosaic-showcase-') as temporary:
        temp = Path(temporary)
        # Each visible panorama pixel is revealed when its actual owner arrives.
        # The registered final canvas stays fixed; this is a provenance animation,
        # not a claim that final geometry is estimated online.
        timeline = [0]*5 + list(range(len(frames))) + [len(frames)-1]*14
        for n, index in enumerate(timeline):
            board = Image.new('RGB', (1000, 740), '#102631')
            draw = ImageDraw.Draw(board)
            draw.text((28, 20), 'VIDEO MOSAICING', font=title, fill='#ead5ac')
            draw.text((28, 61), 'Input camera scan', font=body, fill='#b9ccce')
            source = rgb(frames[index]).resize((432, 288), Image.Resampling.LANCZOS)
            board.paste(source, (28, 91))
            draw.text((500, 112), 'Observe. Register. Rectify.', font=title, fill='#ead5ac')
            for y, text in [(166, 'Robust motion estimation'),
                            (204, 'Locally rectified manifold strips'),
                            (242, 'Every pixel linked to its source')]:
                draw.text((500, y), text, font=body, fill='#b9ccce')
            draw.text((500, 306), f'Source frame {index+1:02d} / {len(frames)}', font=body, fill='#ead5ac')
            draw.text((28, 402), 'Panorama / revealed by source-frame provenance', font=body, fill='#b9ccce')
            visible = result.image.copy()
            visible[(~result.coverage) | (result.source_index > index)] = (49, 38, 16)
            revealed = rgb(visible).resize(pano.size, Image.Resampling.LANCZOS)
            board.paste(revealed, (28, 438))
            draw.text((28, 712), 'Procedural scene / actual pipeline output / Peleg et al., PAMI 2000', font=small, fill='#91aeb4')
            board.save(temp/f'{n:03d}.png')
        common = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '8', '-i', str(temp/'%03d.png')]
        subprocess.run(common + ['-c:v', 'libx264', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output/'demo.mp4')], check=True)
        subprocess.run(common + ['-filter_complex', '[0:v]scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3', '-loop', '0', str(args.output/'demo.gif')], check=True)
        rgb(result.image).save(args.output/'harbor.png')
        Image.open(temp/'034.png').save(args.output/'demo-poster.png')
    print('Generated demo.mp4, demo.gif, demo-poster.png and harbor.png')


if __name__ == '__main__':
    main()
