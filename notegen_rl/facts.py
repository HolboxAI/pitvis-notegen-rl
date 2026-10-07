"""Per-second predictions / labels -> time-ordered step segments with instruments.

The same `VideoFacts` shape comes out of both paths, so the prompt builder never
knows (or needs to know) whether it is looking at perception output or labels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-8


@dataclass
class Segment:
    step: str
    start_s: float
    end_s: float
    conf: float = 1.0
    instruments: dict = field(default_factory=dict)  # {name: confidence}

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class VideoFacts:
    video_id: str
    source: str                      # "predicted" | "label"
    segments: list = field(default_factory=list)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "VideoFacts":
        d = json.loads(Path(path).read_text())
        d["segments"] = [Segment(**s) for s in d["segments"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def viterbi(log_emis: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    T, S = log_emis.shape
    score = log_init + log_emis[0]
    back = np.zeros((T, S), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + log_trans          # [from, to]
        back[t] = cand.argmax(0)
        score = cand.max(0) + log_emis[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def mode_filter(labels: np.ndarray, window: int, n_classes: int) -> np.ndarray:
    half = max(window // 2, 0)
    out = labels.copy()
    for t in range(len(labels)):
        lo, hi = max(0, t - half), min(len(labels), t + half + 1)
        out[t] = np.bincount(labels[lo:hi], minlength=n_classes).argmax()
    return out


def runs(path: np.ndarray) -> list:
    """[(cls, start_idx, end_idx_exclusive)] for consecutive equal values."""
    if len(path) == 0:
        return []
    cut = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(path)]])
    return [(int(path[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def merge_short(rs: list, min_len: int) -> list:
    """Absorb runs shorter than min_len into their longer neighbour, repeatedly."""
    rs = [list(r) for r in rs]
    while len(rs) > 1:
        lengths = [e - s for _, s, e in rs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            break
        if i == 0:
            j = 1
        elif i == len(rs) - 1:
            j = i - 1
        else:
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        keep_cls = rs[j][0]
        rs[lo:hi + 1] = [[keep_cls, rs[lo][1], rs[hi][2]]]
        # neighbours that now share a class collapse too
        k = 0
        while k < len(rs) - 1:
            if rs[k][0] == rs[k + 1][0]:
                rs[k:k + 2] = [[rs[k][0], rs[k][1], rs[k + 1][2]]]
            else:
                k += 1
    return [tuple(r) for r in rs]


def smooth_path(step_probs: np.ndarray, fcfg: dict, log_trans=None, log_init=None) -> np.ndarray:
    raw = step_probs.argmax(1)
    mode = fcfg["smoothing"]
    if mode == "viterbi" and log_trans is not None:
        if log_init is None:  # excerpt that does not start at the beginning of the procedure
            log_init = np.full(step_probs.shape[1], -np.log(step_probs.shape[1]))
        return viterbi(np.log(step_probs + EPS), log_trans, log_init)
    if mode in ("median", "viterbi"):
        return mode_filter(raw, int(fcfg["median_window_s"]), step_probs.shape[1])
    return raw


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
def facts_from_predictions(video_id, times, step_probs, instr_probs, tax, fcfg,
                           log_trans=None, log_init=None) -> tuple[VideoFacts, np.ndarray]:
    """Returns (facts, smoothed per-second step path). Non-surgical steps are dropped."""
    path = smooth_path(step_probs, fcfg, log_trans, log_init)
    segs = []
    thr, min_frac = fcfg["instrument_threshold"], fcfg["instrument_min_fraction"]
    for cls, s, e in merge_short(runs(path), int(fcfg["min_segment_s"])):
        name = tax.step_names[cls]
        if name in tax.non_surgical:
            continue
        instruments = {}
        if instr_probs.shape[1]:
            p = instr_probs[s:e]
            hit = p >= thr
            for k in np.flatnonzero(hit.mean(0) >= min_frac):
                instruments[tax.instr_names[k]] = round(float(p[hit[:, k], k].mean()), 3)
        segs.append(Segment(name, float(times[s]), float(times[e - 1] + 1),
                            round(float(step_probs[s:e, cls].mean()), 3), instruments))
    return VideoFacts(str(video_id), "predicted", segs), path


def facts_from_labels(video_id, labels, tax) -> VideoFacts:
    segs = []
    for cls, s, e in runs(labels.steps):
        name = tax.step_names[cls]
        if name in tax.non_surgical:
            continue
        present = np.flatnonzero(labels.instruments[s:e].any(0))
        segs.append(Segment(name, float(labels.times[s]), float(labels.times[e - 1] + 1), 1.0,
                            {tax.instr_names[k]: 1.0 for k in present}))
    return VideoFacts(str(video_id), "label", segs)


def gold_sets(labels, tax, min_step_s: int, min_instr_s: int) -> tuple[list, list]:
    """Steps / instruments present for at least the given number of labeled seconds."""
    counts = np.bincount(labels.steps, minlength=len(tax.step_names))
    steps = [n for n, c in zip(tax.step_names, counts) if c >= min_step_s and n not in tax.non_surgical]
    surgical = np.array([tax.step_names[c] not in tax.non_surgical for c in labels.steps], dtype=bool)
    icounts = labels.instruments[surgical].sum(0) if len(labels.times) else np.zeros(len(tax.instr_names))
    instr = [n for n, c in zip(tax.instr_names, icounts) if c >= min_instr_s]
    return steps, instr
