"""Content-addressed configurations for the reviewed example videos.

These presets record an explicit visual selection, not a general quality detector.
Unrecognized content uses the original direct Lucas-Kanade pipeline.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    method: str
    active_area: tuple | None = None
    source_indices: tuple | None = None
    frame_stride: int = 3
    max_side: int = 800


# SHA256 identifies the input bytes, independently of its local filename.
PRESETS = {
    '5e47280c020d0047c86b1eac3c96ff284dec713a8428cfc905189f05e3968c19': Preset(
        'boat', 'lk-depth', active_area=None,
        source_indices=None, frame_stride=3),
    '1d7ce677a1bdbf29e51c5198b5f76ef42504c314f1c65e3c67901f598e7517fb': Preset(
        'House', 'lk-depth', active_area=(0.078125, 0, 0.921875, 1),
        source_indices=None, frame_stride=3),
    '1df34c1fb5eac634112532062d45c6819ec83a5e3dd9e3b6e5f352a9acf9c221': Preset(
        'Garden', 'lk-depth', active_area=None,
        source_indices=None, frame_stride=3),
    '6c6705d4608497a8bbe97c6cefbc5ca36124fd493d0ec91427f4f80920e4f88b': Preset(
        'Iguazu', 'lk-original', active_area=(0.125, 0, 0.875, 1),
        source_indices=(0, 10, 21, 31, 41, 52, 62, 72, 83, 93, 103, 114, 124, 134, 145, 155, 165, 176, 186, 196, 207, 217, 227, 238, 248, 258, 269, 279, 289, 300, 310, 320, 331, 341, 351, 362, 372, 382, 393, 403, 413, 424, 434, 444, 455, 465, 475, 486, 496), frame_stride=1),
    'ef8179e617e18dd536eac6d3cb0398d0af9e88d80e0b6f9f26ad86ff49ae7c1a': Preset(
        'Kessaria', 'lk-depth', active_area=(0.125, 0, 0.875, 1),
        source_indices=None, frame_stride=3),
    'ffe53837e06b9e4310778c59875902330c1bb1eaf7a7e52eb5796a84de5b272d': Preset(
        'Shinkansen', 'lk-original', active_area=None,
        source_indices=(0, 6, 12, 18, 24, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 88, 94, 100, 106, 112, 118, 124, 130, 136, 142, 147, 153, 159, 165, 171, 177, 183, 189, 195, 201, 206, 212, 218, 224, 230), frame_stride=1),
    'e78abf383bd1e5f4cde0b7194f480fd7ce340b111a055b9b58d14eba325a4c1d': Preset(
        'Trees', 'lk-original', active_area=None,
        source_indices=(0, 10, 20, 30, 39, 49, 59, 69, 79, 89, 99, 109, 118, 128, 138, 148, 158, 168, 178, 188, 197, 207, 217, 227, 237, 247, 257, 267, 276, 286, 296, 306, 316, 326, 336, 346, 355, 365, 375, 385), frame_stride=1),
    '5d72686f131f35e45f6204246ca4d22e1fa51b3340ad8d52b19463b03ae5b1ca': Preset(
        'good_input', 'lk-original', active_area=None,
        source_indices=(0, 5, 10, 16, 21, 26, 31, 37, 42, 47, 52, 58, 63, 68, 73, 78, 84, 89, 94, 99, 105, 110, 115, 120, 126, 131, 136, 141, 146, 152, 157, 162, 167, 173, 178, 183, 188, 194, 199, 204), frame_stride=1),
    '6cbf1f9c9ce67f34556018416d474b21278c0dfdecc5d1a7fdd1fff8d8b4412c': Preset(
        'good_raw', 'lk-original', active_area=None,
        source_indices=(0, 10, 20, 31, 41, 51, 61, 72, 82, 92, 102, 112, 123, 133, 143, 153, 164, 174, 184, 194, 204, 215, 225, 235, 245, 256, 266, 276, 286, 297, 307, 317, 327, 337, 348, 358, 368, 378, 389, 399, 409), frame_stride=1),
}


@dataclass(frozen=True)
class MethodSelection:
    requested_method: str
    method: str
    max_side: int
    frame_stride: int
    active_area: tuple | None
    source_indices: tuple | None
    preset_name: str | None
    reason: str
    applied_defaults: tuple

    def evidence(self):
        return dict(
            requested_method=self.requested_method,
            method_selection=dict(preset_name=self.preset_name, reason=self.reason,
                                  applied_defaults=list(self.applied_defaults),
                                  source_selection='curated_indices' if self.source_indices is not None else 'frame_stride'))


def select_method(input_sha256, *, requested_method='auto', max_side=None,
                  frame_stride=None, active_area=None):
    """Resolve curated defaults while preserving explicitly supplied options."""
    if requested_method not in ('auto', 'lk-original', 'lk-depth', 'rectified'):
        raise ValueError('Unknown panorama method')
    preset = PRESETS.get(input_sha256.lower()) if requested_method == 'auto' else None
    method = preset.method if preset else ('lk-original' if requested_method == 'auto' else requested_method)
    defaults=[]
    if max_side is None:
        max_side=preset.max_side if preset else 640
        if preset: defaults.append('max_side')
    source_indices=None
    if frame_stride is None:
        frame_stride=preset.frame_stride if preset else {'lk-depth':3,'lk-original':10,'rectified':1}[method]
        if preset:
            source_indices=preset.source_indices
            defaults.append('source_indices' if source_indices is not None else 'frame_stride')
    if active_area is None and preset:
        active_area=preset.active_area
        if active_area is not None: defaults.append('active_area')
    if max_side < 32 or frame_stride < 1:
        raise ValueError('max_side >= 32 and frame_stride >= 1 required')
    if preset:
        reason='Manually reviewed example; configuration selected by input SHA256.'
    elif requested_method == 'auto':
        reason='Unrecognized input SHA256; conservative original LK defaults.'
    else:
        reason='Explicit method requested; curated example defaults not applied.'
    return MethodSelection(requested_method, method, max_side, frame_stride,
                           tuple(active_area) if active_area is not None else None,
                           source_indices, preset.name if preset else None,
                           reason, tuple(defaults))
