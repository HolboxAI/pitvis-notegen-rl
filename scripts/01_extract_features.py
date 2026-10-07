"""Stage 1: Endo-FM features for every labeled second of every video.

Resumable (skips videos whose feature file exists). Split work across GPUs with
--shard i --num-shards n (one process per GPU, CUDA_VISIBLE_DEVICES=i)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from notegen_rl.config import load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import load_all  # noqa: E402
from notegen_rl.endofm import extract, extract_frames, load_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--videos", type=int, nargs="*", help="restrict to these video ids")
    ap.add_argument("--allow-cpu", action="store_true", help="extract on CPU if no GPU (very slow)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    frames_sub = cfg["data"].get("frames_subdir")
    use_frames = bool(frames_sub) and cfg["endofm"]["clip_mode"] == "repeat"
    layout, tax, labels = load_all(cfg, need_videos=not use_frames, need_frames=use_frames)
    out_dir = wpath(cfg, "features", mkdir=True)
    ids = [v for v in layout.ids if not args.videos or v in args.videos]
    ids = ids[args.shard::args.num_shards]
    todo = [v for v in ids if not (out_dir / f"video{v:02d}.npz").exists()]
    print(f"{len(ids)} videos in shard, {len(todo)} still to extract "
          f"(clip_mode={cfg['endofm']['clip_mode']}, source={'frames' if use_frames else 'video'})", flush=True)
    if not todo:
        return

    if use_frames:
        frames_root = layout.root / frames_sub
        items, missing = [], 0
        for v in todo:
            d = frames_root / f"video{v:02d}"
            have = {int(p.stem) for p in d.glob("*.jpg") if p.stem.isdigit()} if d.exists() else set()
            want = [int(t) for t in labels[v].times]
            missing += sum(t not in have for t in want)
            items += [(v, t, d / f"{t}.jpg") for t in want if t in have]
        print(f"{len(items)} frames found, {missing} labeled seconds without a frame (skipped)", flush=True)
        if not items:
            raise SystemExit(f"no frames under {frames_root} -- check data.frames_subdir")
    else:
        jobs = [(v, layout.videos[v], labels[v].times) for v in todo]

    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no usable CUDA GPU (driver/torch mismatch?) -- refusing to extract on CPU; "
                         "pass --allow-cpu to force")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    model = load_backbone(cfg, device)

    def save(vid, times, feats):
        np.savez(out_dir / f"video{vid:02d}.npz", times=times, feats=feats,
                 clip_mode=cfg["endofm"]["clip_mode"])
        print(f"  video{vid:02d}: {len(times)} seconds -> {feats.shape}", flush=True)

    if use_frames:
        extract_frames(model, items, cfg, device, save)
    else:
        extract(model, jobs, cfg, device, save)
    upload_results(cfg, "features")


if __name__ == "__main__":
    main()
