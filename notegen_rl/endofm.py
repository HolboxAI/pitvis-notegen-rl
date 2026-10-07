"""Endo-FM backbone loading and streaming per-second feature extraction.

Videos are decoded sequentially, once, straight from the .mp4 (no JPEG intermediate),
one video per DataLoader worker; the GPU only ever sees preprocessed 224x224 frames.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import torch

from .config import resolve

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_SIZE = 224


def load_backbone(cfg, device, pretrained: bool = True):
    repo = resolve(cfg["endofm"]["repo_dir"])
    if not (repo / "models" / "timesformer.py").exists():
        raise SystemExit(f"Endo-FM repo not found at {repo} -- run setup_env.sh first.")
    sys.path.insert(0, str(repo))
    from models.timesformer import get_vit_base_patch16_224

    num_frames = cfg["endofm"]["num_frames"]

    class SimpleCfg:  # stand-in for the repo's fvcore CfgNode; only the fields the builder reads
        class DATA:
            TRAIN_CROP_SIZE = IMG_SIZE
            NUM_FRAMES = num_frames

        class MODEL:
            NUM_CLASSES = 0

        class TIMESFORMER:
            ATTENTION_TYPE = "divided_space_time"
            PRETRAINED_MODEL = ""

    model = get_vit_base_patch16_224(cfg=SimpleCfg(), no_head=True)
    if pretrained:
        ckpt_path = resolve(cfg["endofm"]["checkpoint"])
        # weights_only=False: the release bundles a Namespace + optimizer state (trusted source)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt = ckpt.get("teacher", ckpt)
        state = {k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")}
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Endo-FM checkpoint is missing {len(missing)} backbone keys, e.g. {missing[:3]}")
    return model.eval().to(device)


def preprocess_bgr(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 frame -> normalized [3, 224, 224] float tensor (resize short side 256, center crop)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = 256.0 / min(h, w)
    rgb = cv2.resize(rgb, (max(IMG_SIZE, round(w * s)), max(IMG_SIZE, round(h * s))),
                     interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    y0, x0 = (h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2
    rgb = rgb[y0:y0 + IMG_SIZE, x0:x0 + IMG_SIZE]
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    return (t - _MEAN) / _STD


def video_duration_s(path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(n / fps) if n > 0 else 0.0


def iter_video_clips(vid, path, times, num_frames: int, mode: str, stride: int):
    """Yields (vid, t, clip) for each requested second t, decoding the video once.
    repeat mode: clip is [3, H, W] (replicated over time on the GPU).
    window mode: clip is [3, num_frames, H, W], frames ending at t, `stride` frames apart."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    targets = sorted({int(round(float(t) * fps)): int(t) for t in times}.items())
    if not targets:
        return
    if mode == "repeat":
        needed = {f for f, _ in targets}
    else:
        needed = {max(0, f - k * stride) for f, _ in targets for k in range(num_frames)}
    last = max(needed)

    cache, ti, idx = {}, 0, 0
    while idx <= last:
        if idx in needed:
            ok, frame = cap.read()
            if not ok:
                break
            cache[idx] = preprocess_bgr(frame)
        elif not cap.grab():
            break
        while ti < len(targets) and targets[ti][0] == idx:
            f, t = targets[ti]
            if mode == "repeat":
                clip = cache.pop(f)
            else:
                clip = torch.stack([cache[max(0, f - (num_frames - 1 - k) * stride)]
                                    for k in range(num_frames)], dim=1)
            yield vid, t, clip
            ti += 1
            if mode != "repeat":
                lo = targets[ti][0] - (num_frames - 1) * stride if ti < len(targets) else idx + 1
                for k in [k for k in cache if k < lo]:
                    del cache[k]
        idx += 1
    cap.release()


class ClipStream(torch.utils.data.IterableDataset):
    """jobs: list of (vid, video_path, times). Each DataLoader worker decodes its own videos."""

    def __init__(self, jobs, num_frames, mode, stride):
        self.jobs, self.num_frames, self.mode, self.stride = jobs, num_frames, mode, stride

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        jobs = self.jobs[info.id::info.num_workers] if info else self.jobs
        for vid, path, times in jobs:
            yield from iter_video_clips(vid, path, times, self.num_frames, self.mode, self.stride)


class FrameFiles(torch.utils.data.Dataset):
    """Pre-extracted 1 fps JPEGs: items = [(vid, t, path)]. Unreadable files yield None."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, t, path = self.items[i]
        img = cv2.imread(str(path))
        return vid, t, (preprocess_bgr(img) if img is not None else None)


def _collate(batch):
    batch = [b for b in batch if b[2] is not None]
    if not batch:
        return [], [], None
    vids, times, clips = zip(*batch)
    return list(vids), list(times), torch.stack(clips)


@torch.no_grad()
def extract(model, jobs, cfg, device, on_video_done):
    """Decodes videos. jobs: [(vid, video_path, times)]. Calls on_video_done(vid, times[T],
    feats[T,768] float16) as soon as every requested second of a video has been embedded."""
    ecfg = cfg["endofm"]
    expected = {vid: len({int(t) for t in times}) for vid, _, times in jobs}
    stream = ClipStream(jobs, ecfg["num_frames"], ecfg["clip_mode"], ecfg["window_stride_frames"])
    workers = min(ecfg["decode_workers"], len(jobs))
    dl = torch.utils.data.DataLoader(stream, batch_size=ecfg["batch_size"], num_workers=workers,
                                     collate_fn=_collate, pin_memory=device.type == "cuda")
    _embed(model, dl, expected, cfg, device, on_video_done)


@torch.no_grad()
def extract_frames(model, items, cfg, device, on_video_done):
    """Reads pre-extracted JPEGs instead of decoding video (repeat clip mode only).
    items: [(vid, t, jpg_path)], ordered by video so videos complete (and save) one by one."""
    ecfg = cfg["endofm"]
    expected = {}
    for vid, _, _ in items:
        expected[vid] = expected.get(vid, 0) + 1
    dl = torch.utils.data.DataLoader(FrameFiles(items), batch_size=ecfg["batch_size"], shuffle=False,
                                     num_workers=ecfg["frame_workers"], collate_fn=_collate,
                                     pin_memory=device.type == "cuda", persistent_workers=False)
    _embed(model, dl, expected, cfg, device, on_video_done)


def _embed(model, dl, expected, cfg, device, on_video_done):
    ecfg = cfg["endofm"]
    T = ecfg["num_frames"]
    buf = {vid: ([], []) for vid in expected}
    done = {vid: 0 for vid in expected}
    use_amp = bool(ecfg["amp"]) and device.type == "cuda"
    n_batches = 0
    for vids, times, clips in dl:
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"  ... {n_batches} batches embedded", flush=True)
        if clips is None:
            continue
        x = clips.to(device, non_blocking=True)
        if x.dim() == 4:  # repeat mode
            x = x.unsqueeze(2).expand(-1, -1, T, -1, -1).contiguous()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            f = model(x)
        f = f.float().cpu().numpy().astype(np.float16)
        for v, t, row in zip(vids, times, f):
            buf[v][0].append(t)
            buf[v][1].append(row)
            done[v] += 1
            if done[v] == expected[v]:
                _flush(buf, v, on_video_done)
    for v in list(buf):  # streams that ended early, or unreadable frames
        if buf[v][0]:
            print(f"WARNING: video {v}: only {len(buf[v][0])}/{expected[v]} seconds embedded")
            _flush(buf, v, on_video_done)


def _flush(buf, v, cb):
    times, feats = buf.pop(v)
    order = np.argsort(times)
    cb(v, np.asarray(times, dtype=np.int64)[order], np.stack(feats)[order])
