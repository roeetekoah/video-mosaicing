"""Sequential decoding and reproducible result bundles."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def iter_frames(path,max_side=640,*,stride=1):
    path = Path(path)
    if max_side < 32 or stride < 1: raise ValueError("max_side >= 32 and stride >= 1 required")
    def resize(frame):
        if frame is None: raise ValueError("An image could not be decoded")
        scale = min(1.,max_side/max(frame.shape[:2]))
        return cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else frame
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
        for index in range(0,len(files),stride): yield index,resize(cv2.imread(str(files[index])))
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {path.name}")
    index = 0
    try:
        while True:
            ok,frame = cap.read()
            if not ok: break
            if index%stride==0: yield index,resize(frame)
            index += 1
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if declared > 0 and index < declared-1:
            raise ValueError(f"Decode stopped early at frame {index} of {declared}")
    finally: cap.release()


def input_digest(path):
    path = Path(path)
    digest = hashlib.sha256()
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')) if path.is_dir() else [path]
    for p in paths:
        if path.is_dir(): digest.update(p.name.encode('utf-8'))
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def save_result(result,output,*,provenance=True):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output/'panorama.png'),result.image): raise OSError("Cannot save panorama")
    if not cv2.imwrite(str(output/'coverage.png'),result.coverage.astype(np.uint8)*255): raise OSError("Cannot save coverage")
    if provenance:
        np.savez_compressed(output/'provenance.npz',source_index=result.source_index,source_xy=result.source_xy)
    (output/'evidence.json').write_text(json.dumps(result.evidence,indent=2,allow_nan=False),encoding='utf-8')
