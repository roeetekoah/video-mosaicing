"""Motion-selected video -> locally rectified manifold panorama."""
from dataclasses import asdict, dataclass, field
from itertools import chain

import cv2
import numpy as np

from .geometry import project, render_rectified
from .registration import MotionError, RegistrationConfig, detect, estimate_features


@dataclass(frozen=True)
class MosaicConfig:
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    alpha: float = .5
    min_step_px: float = 6.
    max_step_fraction: float = .15
    max_selected: int = 1500
    exposure: bool = False
    blend_width: int = 0
    active_area: tuple | None = None  # fractional left, top, right, bottom

    def __post_init__(self):
        if not 0 <= self.alpha <= 1: raise ValueError("alpha must be in [0,1]")
        if self.min_step_px < .5 or not 0 < self.max_step_fraction <= .5 or self.max_selected < 2:
            raise ValueError("Invalid motion-selection limits")
        if self.blend_width < 0 or self.blend_width > 16:
            raise ValueError("blend_width must be between 0 and 16 pixels")
        if self.active_area is not None:
            l,t,r,b = self.active_area
            if not (0<=l<r<=1 and 0<=t<b<=1): raise ValueError("Invalid fractional active area")


def active_bounds(shape,area=None):
    h,w = shape[:2]
    l,t,r,b = area or (0,0,1,1)
    return tuple(int(round(x)) for x in (l*w,t*h,r*w,b*h))


def crop_active(frame,area=None):
    l,t,r,b = active_bounds(frame.shape,area)
    return frame[t:b,l:r].copy() if area is not None else frame


def thumbnail_difference(a,b):
    x = cv2.resize(a,(64,48),interpolation=cv2.INTER_AREA).astype(float)
    y = cv2.resize(b,(64,48),interpolation=cv2.INTER_AREA).astype(float)
    return float(np.mean(np.abs(x-y)))


def orientation(matrix,hw):
    h,w = hw
    c = np.array([(w-1)/2,(h-1)/2])
    dx,dy = project(matrix,c)-c
    if abs(dx) >= abs(dy):
        if dx < 0: return "LR",np.eye(3)
        return "RL",np.array([[-1,0,w-1],[0,1,0],[0,0,1.]])
    if dy < 0: return "TB",np.array([[0,1,0],[-1,0,w-1],[0,0,1.]])
    return "BT",np.array([[0,-1,h-1],[1,0,0],[0,0,1.]])


def orient(frame,direction):
    if direction == "LR": return frame
    if direction == "RL": return cv2.flip(frame,1)
    return cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE if direction=="TB" else cv2.ROTATE_90_CLOCKWISE)


def restore(array,direction):
    if direction == "LR": return array
    if direction == "RL": return cv2.flip(array,1)
    return cv2.rotate(array,cv2.ROTATE_90_CLOCKWISE if direction=="TB" else cv2.ROTATE_90_COUNTERCLOCKWISE)


@dataclass
class PreparedScan:
    frames: list
    indices: list
    homographies: list
    direction: str
    orientation_matrix: np.ndarray
    evidence: dict


def prepare_scan(records,config=MosaicConfig()):
    """Consume (source_index, BGR frame) records without retaining the full video.

    Near-identical frames are discarded cheaply. Every useful candidate is
    registered against the last selected frame, so small motions accumulate
    until the requested spacing is reached. An unsupported gap stops the run;
    it is never silently bridged by a guessed transform.
    """
    iterator = iter(records)
    try: first_index,first = next(iterator)
    except StopIteration: raise ValueError("No decodable frames") from None
    original_shape = first.shape
    bounds = active_bounds(original_shape,config.active_area)
    first = crop_active(first,config.active_area)
    shape = first.shape
    if len(shape)!=3 or shape[2]!=3 or min(shape[:2])<32 or first.dtype!=np.uint8:
        raise ValueError("Expected uint8 BGR frames at least 32 pixels per side")
    def check(record):
        index,frame = record
        if frame.shape!=original_shape or frame.dtype!=np.uint8: raise ValueError("Frame shape or dtype changed within shot")
        return index,crop_active(frame,config.active_area)
    initial_features = detect(first,config.registration)
    if len(initial_features.xy) < config.registration.min_matches:
        raise MotionError("First frame has insufficient textured features",dict(source_index=int(first_index),features=len(initial_features.xy)))
    skipped, decoded = [],1
    direction = None
    for raw in iterator:
        index,frame = check(raw); decoded += 1
        if thumbnail_difference(first,frame)<.5:
            skipped.append(dict(source_index=int(index),reason="near_duplicate")); continue
        try:
            initial_h,initial_evidence = estimate_features(initial_features,detect(frame,config.registration),shape[:2],config.registration)
        except MotionError as error:
            error.evidence.update(source_indices=[int(first_index),int(index)])
            raise
        if np.linalg.norm(initial_evidence['center_displacement']) < .75:
            skipped.append(dict(source_index=int(index),reason="subpixel_motion")); continue
        direction,q = orientation(initial_h,shape[:2])
        second = (index,raw[1])
        break
    if direction is None: raise MotionError("No observable scan motion",dict(decoded_frames=decoded,skipped=skipped))
    selected = [orient(first,direction)]
    indices = [int(first_index)]
    previous_features = detect(selected[0],config.registration)
    homographies, pairs = [],[]
    last_record = None
    last_estimate = None
    pending = chain([second],iterator)
    for raw in pending:
        index,raw_frame = check(raw)
        if index != second[0]: decoded += 1
        frame = orient(raw_frame,direction)
        last_record = (index,frame)
        if thumbnail_difference(selected[-1],frame)<.5:
            skipped.append(dict(source_index=int(index),reason="near_duplicate")); last_estimate=None; continue
        features = detect(frame,config.registration)
        try: m,evidence = estimate_features(previous_features,features,frame.shape[:2],config.registration,
                                           strip_alpha=(1-config.alpha if direction in ("RL","BT") else config.alpha))
        except MotionError as error:
            error.evidence.update(source_indices=[indices[-1],int(index)])
            raise
        dx,dy = evidence['center_displacement']
        if dx > config.min_step_px:
            raise MotionError("Camera reversed direction; split the shot",dict(source_indices=[indices[-1],int(index)],center_displacement=[dx,dy]))
        if -dx > config.max_step_fraction*frame.shape[1]:
            raise MotionError("Motion gap is too wide; provide denser source frames",dict(source_indices=[indices[-1],int(index)],center_displacement=[dx,dy]))
        last_estimate = (m,evidence,features)
        if -dx < config.min_step_px:
            skipped.append(dict(source_index=int(index),reason="small_motion",displacement_px=float(-dx))); continue
        selected.append(frame); indices.append(int(index)); homographies.append(m)
        evidence.update(source_indices=[indices[-2],indices[-1]])
        pairs.append(evidence); previous_features=features; last_estimate=None
        if len(selected)>config.max_selected:
            raise MotionError("Selected-frame limit reached; split the shot",dict(max_selected=config.max_selected))
    if last_estimate is not None and last_record[0] != indices[-1]:
        m,evidence,_ = last_estimate
        if -evidence['center_displacement'][0] >= .75:
            selected.append(last_record[1]); indices.append(int(last_record[0])); homographies.append(m)
            evidence.update(source_indices=[indices[-2],indices[-1]])
            pairs.append(evidence)
            skipped = [s for s in skipped if s['source_index'] != indices[-1]]
    if len(selected)<2: raise MotionError("Scan displacement is too small",dict(decoded_frames=decoded))
    cropping = np.array([[1,0,-bounds[0]],[0,1,-bounds[1]],[0,0,1.]])
    return PreparedScan(selected,indices,homographies,direction,q@cropping,
                        dict(selected_source_indices=indices,decoded_frames=decoded,skipped=skipped,pairs=pairs,
                             input_frame_hw=list(original_shape[:2]),active_bounds=list(bounds),
                             registration_config=asdict(config.registration),selection_config=dict(min_step_px=config.min_step_px,max_step_fraction=config.max_step_fraction)))


def render_scan(scan,config=MosaicConfig()):
    selected,indices,homographies = scan.frames,scan.indices,scan.homographies
    direction,q = scan.direction,scan.orientation_matrix
    alpha = 1-config.alpha if direction in ("RL","BT") else config.alpha
    col = int(round(alpha*(selected[0].shape[1]-1)))
    in_matte = []
    for f in selected[::max(1,len(selected)//10)]:
        values = f.astype(float)
        black = (values.mean(axis=(0,2))<1)&(values.std(axis=(0,2))<.5)
        nonblack = np.flatnonzero(~black)
        edge_margin = max(3,.02*f.shape[1])
        in_matte.append(bool(len(nonblack) and ((col<nonblack[0] and nonblack[0]>=edge_margin) or
                                                (col>nonblack[-1] and f.shape[1]-1-nonblack[-1]>=edge_margin))))
    if np.mean(in_matte)>.8:
        from .geometry import GeometryError
        raise GeometryError("Anchor falls in a constant black border; declare the scene's --active-area")
    gains = None; photometry = None
    if config.exposure:
        from .photometry import exposure_gains
        gains,photometry = exposure_gains(selected,homographies)
    result = render_rectified(selected,homographies,alpha=alpha,gains=gains)
    if config.blend_width:
        from .photometry import blend_boundaries
        result.image,blend_evidence = blend_boundaries(result,selected,homographies,config.blend_width,gains)
        result.evidence['blending'] = blend_evidence
    valid = result.coverage
    result.source_xy[valid] = project(np.linalg.inv(q),result.source_xy[valid])
    result.source_index[valid] = np.array(indices,dtype=np.int32)[result.source_index[valid]]
    result.evidence.update(scan.evidence)
    result.evidence.update(config=asdict(config),direction=direction,source_to_normalized=q.tolist(),photometry=photometry)
    result.image = restore(result.image,direction)
    result.coverage = restore(result.coverage.astype(np.uint8),direction).astype(bool)
    result.source_index = restore(result.source_index,direction)
    result.source_xy = restore(result.source_xy,direction)
    result.evidence['normalized_canvas_hw'] = result.evidence['canvas_hw']
    result.evidence['canvas_hw'] = list(result.image.shape[:2])
    result.evidence['geometry_coordinates'] = "direction-normalized active area; provenance coordinates refer to resized input frames before active-area extraction"
    return result


def build_mosaic(records,config=MosaicConfig()):
    return render_scan(prepare_scan(records,config),config)


def mosaic(frames,config=MosaicConfig()):
    """Convenience API returning image, coverage, and JSON-safe evidence."""
    result = build_mosaic(enumerate(frames),config)
    return result.image,result.coverage,result.evidence
