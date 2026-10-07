"""PitVis-2023 discovery, taxonomy and per-second labels.

Class names come from PitVis's own map_steps.csv / map_instrument(s).csv -- nothing here
hardcodes a class list. Codes seen in the labels but missing from the maps are kept
under a generic name rather than dropped, so an incomplete map never crashes a stage.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import wpath

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_NUM_RE = re.compile(r"(\d+)")


def norm(s) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def display_name(s) -> str:
    n = norm(s)
    return n[:1].upper() + n[1:]


def _vid(p: Path):
    m = _NUM_RE.search(p.stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Locating the dataset
# ---------------------------------------------------------------------------
def resolve_pitvis_root(cfg, need_videos: bool = True, need_frames: bool = False) -> Path:
    """Local Path to the dataset. An s3:// root is synced (incrementally) into
    work_dir/<data.local_cache>; `need_videos=False` syncs only the annotations,
    `need_frames=True` also syncs the pre-extracted frames (data.frames_subdir)."""
    root = cfg["data"].get("pitvis_root")
    if not root:
        raise SystemExit("config.yaml: set data.pitvis_root to your PitVis location (local dir or s3://...).")
    root = str(root)
    if root.startswith("s3://"):
        local = wpath(cfg, cfg["data"]["local_cache"], mkdir=True)
        subs = [cfg["data"]["annotations_subdir"]] + ([cfg["data"]["videos_subdir"]] if need_videos else [])
        if need_frames and cfg["data"].get("frames_subdir"):
            subs.append(cfg["data"]["frames_subdir"])
        for sub in subs:
            cmd = ["aws", "s3", "sync", f"{root.rstrip('/')}/{sub}", str(local / sub), "--only-show-errors"]
            print("sync:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        return local
    p = Path(root).expanduser()
    if not p.exists():
        raise SystemExit(f"data.pitvis_root does not exist: {p}")
    return p


@dataclass
class PitVisLayout:
    root: Path
    videos: dict          # {video_id: Path}
    annotations: dict     # {video_id: Path}
    map_steps: Path | None
    map_instruments: Path | None
    frames: dict = field(default_factory=dict)   # {video_id: dir of pre-extracted JPEGs}

    @property
    def ids(self) -> list:
        return sorted((set(self.videos) | set(self.frames)) & set(self.annotations))


def discover(root: Path, cfg) -> PitVisLayout:
    vdir = root / cfg["data"]["videos_subdir"]
    adir = root / cfg["data"]["annotations_subdir"]
    videos = {_vid(p): p for p in sorted(vdir.glob("*"))
              if p.suffix.lower() in VIDEO_EXTS and _vid(p) is not None} if vdir.exists() else {}
    annots = {_vid(p): p for p in sorted(adir.glob("*.csv"))
              if "annotation" in p.name.lower() and _vid(p) is not None} if adir.exists() else {}

    def first(*names):
        for n in names:
            if (adir / n).exists():
                return adir / n
        return None

    frames = {}
    fsub = cfg["data"].get("frames_subdir")
    if fsub and (root / fsub).exists():
        frames = {_vid(p): p for p in sorted((root / fsub).iterdir()) if p.is_dir() and _vid(p) is not None}
    return PitVisLayout(root, videos, annots, first("map_steps.csv"),
                        first("map_instruments.csv", "map_instrument.csv"), frames)


def split_ids(cfg, ids) -> tuple[list, list]:
    test = sorted(set(cfg["data"]["test_video_ids"]) & set(ids))
    train = sorted(set(ids) - set(test))
    return train, test


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
def _read_map_all(path) -> dict:
    """{code: [every name listed for it]} -- PitVis maps list some codes more than once
    (e.g. instrument 0 = no_visible_instrument AND occluded_image_inside_patient)."""
    out = {}
    if path is None or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().lstrip("-").isdigit():
                out.setdefault(int(row[0]), []).append(row[1].strip())
    return out


def _read_map(path) -> dict:
    return {c: names[0] for c, names in _read_map_all(path).items()}


_ABSENT_RE = re.compile(r"^(no|none)\b|occluded|out of patient")


@dataclass
class Taxonomy:
    step_codes: list
    step_names: list               # display names, index = step class index
    instr_codes: list
    instr_names: list              # display names, index = instrument class index
    non_surgical: list = field(default_factory=list)   # display names of non-surgical steps

    @property
    def surgical_steps(self) -> list:
        return [n for n in self.step_names if n not in self.non_surgical]

    def step_index(self) -> dict:
        return {c: i for i, c in enumerate(self.step_codes)}

    def instr_index(self) -> dict:
        return {c: i for i, c in enumerate(self.instr_codes)}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "Taxonomy":
        return cls(**json.loads(Path(path).read_text()))


def build_taxonomy(layout: PitVisLayout, tables: dict, cfg) -> Taxonomy:
    step_map, instr_map = _read_map(layout.map_steps), _read_map(layout.map_instruments)
    instr_all_names = _read_map_all(layout.map_instruments)
    seen_steps = {int(c) for t in tables.values() for c in np.unique(t[:, 1])}
    seen_instr = {int(c) for t in tables.values() for c in np.unique(t[:, 2:4])}

    non_surgical_norm = {norm(s) for s in cfg["data"]["non_surgical_steps"]}
    step_all_names = _read_map_all(layout.map_steps)
    step_codes = sorted(set(step_map) | seen_steps)

    def step_display(c):
        # a code listed under several names (PitVis -1 = operation_ended / operation_not_started /
        # out_of_patient) is shown by the non-surgical name if it has one, so the filter matches
        names = step_all_names.get(c, [])
        for n in names:
            if norm(n) in non_surgical_norm:
                return display_name(n)
        return display_name(names[0] if names else f"step {c}")

    step_names = [step_display(c) for c in step_codes]

    absent = set(cfg["data"]["instrument_absent_codes"])
    instr_codes = sorted(
        c for c in set(instr_map) | seen_instr
        if c not in absent and not any(_ABSENT_RE.search(norm(n)) for n in instr_all_names.get(c, []))
    )
    instr_names = [display_name(instr_map.get(c, f"instrument {c}")) for c in instr_codes]

    non_surgical = [n for n in step_names if norm(n) in non_surgical_norm]
    return Taxonomy(step_codes, step_names, instr_codes, instr_names, non_surgical)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def read_label_table(path) -> np.ndarray:
    """[N, 4] int array: int_time, int_step, int_instrument1, int_instrument2 (sorted, unique times)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(float(r["int_time"])), int(float(r["int_step"])),
                         int(float(r["int_instrument1"])), int(float(r["int_instrument2"]))))
    arr = np.array(sorted(rows), dtype=np.int64).reshape(-1, 4)
    _, keep = np.unique(arr[:, 0], return_index=True)
    return arr[keep]


@dataclass
class VideoLabels:
    times: np.ndarray        # [T] int seconds
    steps: np.ndarray        # [T] step class index
    instruments: np.ndarray  # [T, K] uint8 multi-hot

    def subset(self, mask) -> "VideoLabels":
        return VideoLabels(self.times[mask], self.steps[mask], self.instruments[mask])


def labels_from_table(table: np.ndarray, tax: Taxonomy) -> VideoLabels:
    s_idx, i_idx = tax.step_index(), tax.instr_index()
    steps = np.array([s_idx[int(c)] for c in table[:, 1]], dtype=np.int64)
    instr = np.zeros((len(table), len(tax.instr_codes)), dtype=np.uint8)
    for col in (2, 3):
        for row, code in enumerate(table[:, col]):
            k = i_idx.get(int(code))
            if k is not None:
                instr[row, k] = 1
    return VideoLabels(table[:, 0].astype(np.int64), steps, instr)


def load_all(cfg, need_videos: bool = False, need_frames: bool = False):
    """(layout, taxonomy, {vid: VideoLabels}). Uses the saved taxonomy when present so
    every stage sees identical class indices."""
    root = resolve_pitvis_root(cfg, need_videos=need_videos, need_frames=need_frames)
    layout = discover(root, cfg)
    tables = {vid: read_label_table(layout.annotations[vid]) for vid in layout.ids}
    tax_path = wpath(cfg, "taxonomy.json")
    tax = Taxonomy.load(tax_path) if tax_path.exists() else build_taxonomy(layout, tables, cfg)
    labels = {vid: labels_from_table(t, tax) for vid, t in tables.items()}
    return layout, tax, labels
