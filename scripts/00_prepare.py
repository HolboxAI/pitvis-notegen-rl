"""Stage 0: locate/sync PitVis, validate the layout, freeze the taxonomy and the split."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from notegen_rl.config import load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import (build_taxonomy, discover, read_label_table,  # noqa: E402
                             resolve_pitvis_root, split_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--no-videos", action="store_true", help="only sync annotations (s3 roots)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    use_frames = bool(cfg["data"].get("frames_subdir")) and cfg["endofm"]["clip_mode"] == "repeat"
    root = resolve_pitvis_root(cfg, need_videos=not args.no_videos and not use_frames, need_frames=use_frames)
    layout = discover(root, cfg)
    sources = set(layout.frames) if use_frames else set(layout.videos)
    print(f"PitVis root: {root}  (feature source: {'pre-extracted frames' if use_frames else 'video'})")
    print(f"videos={len(layout.videos)} frame dirs={len(layout.frames)} annotation csvs={len(layout.annotations)} "
          f"matched={len(sources & set(layout.annotations))}")
    for name, missing in (("annotations", sources - set(layout.annotations)),
                          ("frames" if use_frames else "videos", set(layout.annotations) - sources)):
        if missing:
            print(f"WARNING: no {name} for video ids {sorted(missing)}")
    if not layout.ids:
        raise SystemExit("no matched videos+annotations -- check data.videos_subdir / annotations_subdir")
    print("map_steps:", layout.map_steps, "| map_instruments:", layout.map_instruments)

    tables = {vid: read_label_table(layout.annotations[vid]) for vid in layout.ids}
    tax = build_taxonomy(layout, tables, cfg)
    tax.save(wpath(cfg, "taxonomy.json"))
    print(f"\n{len(tax.step_names)} step classes: {tax.step_names}")
    print(f"non-surgical (excluded from notes): {tax.non_surgical}")
    print(f"{len(tax.instr_names)} instrument classes: {tax.instr_names}")
    if not tax.non_surgical and cfg["data"]["non_surgical_steps"]:
        raise SystemExit("no step matched data.non_surgical_steps -- non-surgical time would leak into "
                         "the notes; check the step names above and the config")

    train, test = split_ids(cfg, layout.ids)
    seconds = {vid: int(len(t)) for vid, t in tables.items()}
    split = {"train": train, "test": test, "labeled_seconds": seconds}
    wpath(cfg, "split.json").write_text(json.dumps(split, indent=2))
    print(f"\ntrain videos ({len(train)}): {train}\ntest videos ({len(test)}): {test}")
    print(f"labeled seconds: train={sum(seconds[v] for v in train)} test={sum(seconds[v] for v in test)}")
    all_steps = np.concatenate([t[:, 1] for t in tables.values()])
    print("step code counts:", dict(zip(*[x.tolist() for x in np.unique(all_steps, return_counts=True)])))
    upload_results(cfg, "taxonomy.json")
    upload_results(cfg, "split.json")


if __name__ == "__main__":
    main()
