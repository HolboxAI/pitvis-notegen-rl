"""Keyframes for the facts + keyframes -> MedGemma architecture.

For every step segment the perception system predicted, frames are picked from inside the
segment and shown to MedGemma next to that segment's line in the prompt, so the note writer
can check each detection against what the endoscope saw:

  - candidates are the seconds where the MS-TCN is most confident in the segment's step;
  - out-of-patient seconds (non-surgical step probability >= out_of_patient_max) are skipped;
  - dark frames (mean intensity < min_brightness) are skipped, then frames much blurrier than
    the sharpest candidate (variance of the Laplacian < min_relative_sharpness x the best) are
    skipped; the most confident remaining candidate is chosen. Both are measured on a centred
    square inside the endoscope's circular field of view, so the black border around it does
    not count;
  - segments lasting at least long_segment_s get frames_per_long_segment frames, one from each
    equal part of the segment; others get frames_per_segment;
  - at most max_images_per_prompt frames per prompt: extra frames of long segments go first,
    then the frames of the most confident segments.

Keyframes are chosen from predictions only (never labels), so training prompts (out-of-fold
predictions) and test prompts (final step model) are built the same way.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

QUALITY_CROP = 0.6           # centred square, side = 0.6 x the shorter frame side (inside the endoscope circle)
QUALITY_SIZE = 256           # the square is resized to this before brightness / sharpness


@dataclass
class Keyframe:
    segment: int             # index into facts.segments
    t: int                   # second of the video
    path: str                # image file
    primary: bool = True     # first frame of its segment (extra frames of long segments: False)


@lru_cache(maxsize=65536)
def frame_quality(path: str) -> tuple[float, float]:
    """(mean intensity 0-255, variance of the Laplacian) of the frame's centre square; (0, 0) if unreadable."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, 0.0
    h, w = img.shape
    side = max(1, int(QUALITY_CROP * min(h, w)))
    y, x = (h - side) // 2, (w - side) // 2
    img = cv2.resize(img[y:y + side, x:x + side], (QUALITY_SIZE, QUALITY_SIZE), interpolation=cv2.INTER_AREA)
    return float(img.mean()), float(cv2.Laplacian(img, cv2.CV_64F).var())


class FrameDir:
    """Pre-extracted 1 fps frames: <dir>/<second>.jpg."""

    def __init__(self, directory):
        self.dir = Path(directory)

    def __call__(self, t: int) -> str | None:
        p = self.dir / f"{int(t)}.jpg"
        return str(p) if p.exists() else None


class VideoFrames:
    """Frames decoded on demand from a video file (new cases), cached as <cache>/<second>.jpg."""

    def __init__(self, video, cache_dir):
        self.video, self.cache = str(video), Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._cap = None

    def __call__(self, t: int) -> str | None:
        p = self.cache / f"{int(t)}.jpg"
        if p.exists():
            return str(p)
        import cv2
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video)
        self._cap.set(cv2.CAP_PROP_POS_MSEC, int(t) * 1000.0)
        ok, frame = self._cap.read()
        if not ok:
            return None
        cv2.imwrite(str(p), frame)
        return str(p)


def _pick(idx: np.ndarray, conf: np.ndarray, times: np.ndarray, frame_path, kcfg) -> tuple[int, str] | None:
    """Most confident usable frame among the seconds `idx` (conf aligned with idx)."""
    order = idx[np.argsort(-conf, kind="stable")][: int(kcfg.get("candidates", 8))]
    cands = []
    for i in order:
        p = frame_path(int(times[i]))
        if p is None:
            continue
        bright, sharp = frame_quality(p)
        if bright >= float(kcfg.get("min_brightness", 25)):
            cands.append((int(times[i]), p, sharp))
    if not cands:
        return None
    best = max(s for _, _, s in cands)
    rel = float(kcfg.get("min_relative_sharpness", 0.5))
    for t, p, s in cands:               # still in confidence order
        if s >= rel * best:
            return t, p
    return None


def select_keyframes(facts, times, step_probs, tax, frame_path, kcfg) -> list:
    """Keyframes for the predicted segments of `facts` (built from the same times / step_probs)."""
    times = np.asarray(times)
    step_idx = {n: i for i, n in enumerate(tax.step_names)}
    non_surgical = [i for i, n in enumerate(tax.step_names) if n in tax.non_surgical]
    p_out = step_probs[:, non_surgical].sum(1) if non_surgical else np.zeros(len(times))
    out_max = float(kcfg.get("out_of_patient_max", 0.5))
    long_s = float(kcfg.get("long_segment_s", 300))
    kfs = []
    for si, seg in enumerate(facts.segments):
        cls = step_idx.get(seg.step)
        if cls is None:
            continue
        sec = np.flatnonzero((times >= seg.start_s) & (times < seg.end_s) & (p_out < out_max))
        if len(sec) == 0:
            continue
        n = int(kcfg.get("frames_per_long_segment", 2) if seg.duration >= long_s
                else kcfg.get("frames_per_segment", 1))
        for k, part in enumerate(np.array_split(sec, max(n, 1))):
            if len(part) == 0:
                continue
            got = _pick(part, step_probs[part, cls], times, frame_path, kcfg)
            if got:
                kfs.append(Keyframe(si, got[0], got[1], primary=not any(x.segment == si for x in kfs)))
    return cap_keyframes(kfs, facts, int(kcfg.get("max_images_per_prompt", 16)))


def cap_keyframes(kfs: list, facts, max_images: int) -> list:
    """Keep at most max_images: one frame per segment before any extra frame, and frames of
    less confident segments before those of more confident ones."""
    if max_images <= 0:
        return []
    if len(kfs) <= max_images:
        return kfs
    conf = {i: s.conf for i, s in enumerate(facts.segments)}
    ranked = sorted(kfs, key=lambda k: (not k.primary, conf.get(k.segment, 1.0), k.t))
    return sorted(ranked[:max_images], key=lambda k: (k.segment, k.t))


def store_keyframes(kfs: list, out_dir) -> list:
    """Copy the chosen frames to out_dir/<second>.jpg (shared by every prompt of a video) and
    return keyframes pointing at the copies (absolute paths)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = []
    for k in kfs:
        dst = out_dir / f"{k.t}.jpg"
        if not dst.exists():
            shutil.copyfile(k.path, dst)
        stored.append(Keyframe(k.segment, k.t, str(dst.resolve()), k.primary))
    return stored


def image_paths(kfs: list) -> list:
    """Image list in the order the prompt shows the keyframes (segment order, then time)."""
    return [k.path for k in sorted(kfs, key=lambda k: (k.segment, k.t))]
