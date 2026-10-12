"""Stage 4: GRPO / SFT / eval datasets (ms-swift jsonl).

One prompt per full procedure plus overlapping sub-windows. Prompts are built from
PREDICTED facts (out-of-fold for training videos); the reward columns hold the LABELS.
All structured columns are JSON strings so the HF datasets/Arrow loader never has to
infer types for mixed lists.

With keyframes.enabled, every predicted segment also gets keyframes (notegen_rl/keyframes.py):
`messages` carries one <image> tag per keyframe, `images` the matching files (copied to
work_dir/keyframes/videoNN/), and `messages_text` the same prompt without keyframes. The
datasets then go to work_dir/datasets_keyframes/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from notegen_rl.config import dataset_path, datasets_subdir, load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import load_all  # noqa: E402
from notegen_rl.keyframes import FrameDir, image_paths, select_keyframes, store_keyframes  # noqa: E402
from notegen_rl.facts import facts_from_labels, facts_from_predictions, gold_sets  # noqa: E402
from notegen_rl.prompts import build_messages, render_template_note  # noqa: E402


def windows(t_min, t_end, dcfg):
    out = [(t_min, t_end, False)] if dcfg["include_full_video"] else []
    w, s = int(dcfg["window_minutes"] * 60), int(dcfg["window_stride_minutes"] * 60)
    if w > 0 and t_end - t_min > w:
        starts = list(range(int(t_min), int(t_end - w) + 1, s))
        if starts[-1] + w < t_end:
            starts.append(int(t_end - w))
        out += [(a, a + w, True) for a in starts]
    return out


def build_rows(cfg, tax, labels, v, reliability, frames=None):
    """frames: FrameDir of this video's 1 fps frames when keyframes are enabled, else None."""
    dcfg, fcfg, kcfg = cfg["dataset"], cfg["facts"], cfg.get("keyframes") or {}
    d = np.load(wpath(cfg, "predictions", f"video{v:02d}.npz"))
    times = d["times"]
    rows = []
    for t0, t1, is_window in windows(float(times.min()), float(times.max() + 1), dcfg):
        m = (times >= t0) & (times < t1)
        if m.sum() < 10:
            continue
        facts, _ = facts_from_predictions(v, times[m], d["step_probs"][m], d["instr_probs"][m], tax, fcfg,
                                          d["log_trans"], None if is_window else d["log_init"])
        lab = labels[v].subset((labels[v].times >= t0) & (labels[v].times < t1))
        gold_steps, gold_instr = gold_sets(lab, tax, dcfg["min_gold_step_s"], dcfg["min_gold_instrument_s"])
        any_steps, any_instr = gold_sets(lab, tax, 1, 1)
        gold_facts = facts_from_labels(v, lab, tax)
        if not gold_steps and not facts.segments:
            continue
        window = (t0, t1) if is_window else None
        prompt_args = (facts, reliability, tax.surgical_steps, tax.instr_names, dcfg["procedure_name"],
                       window, dcfg.get("max_step_lines", 25))
        row = {"messages": build_messages(*prompt_args)}
        if frames is not None:
            kfs = select_keyframes(facts, times[m], d["step_probs"][m], tax, frames, kcfg)
            kfs = store_keyframes(kfs, wpath(cfg, "keyframes", f"video{v:02d}"))
            row = {"messages": build_messages(*prompt_args, keyframes=kfs),
                   "images": image_paths(kfs),
                   "messages_text": json.dumps(row["messages"]),
                   "keyframe_times": json.dumps([k.t for k in sorted(kfs, key=lambda k: (k.segment, k.t))])}
        rows.append({
            **row,
            "video_id": int(v),
            "window": json.dumps([t0, t1]) if window else "",
            "gold_steps": json.dumps(gold_steps),
            "gold_steps_any": json.dumps(any_steps),
            "gold_instruments": json.dumps(gold_instr),
            "gold_instruments_any": json.dumps(any_instr),
            "gold_segments": json.dumps([[s.step, s.start_s, s.end_s] for s in gold_facts.segments]),
            "step_vocab": json.dumps(tax.surgical_steps),
            "instrument_vocab": json.dumps(tax.instr_names),
            "reliability": json.dumps(reliability),
            "reliability_floor": float(dcfg["reliability_floor"]),
            "length_soft_tokens": float(dcfg.get("length_soft_tokens", 600)),
            "length_hard_tokens": float(dcfg.get("length_hard_tokens", 900)),
            "chars_per_token": float(dcfg.get("chars_per_token", 3.8)),
            "truncation_credit": float(dcfg.get("truncation_credit", 0.5)),
            "template_note": render_template_note(facts, dcfg["procedure_name"]),
        })
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    use_kf = bool((cfg.get("keyframes") or {}).get("enabled"))
    if use_kf and not cfg["data"].get("frames_subdir"):
        raise SystemExit("keyframes.enabled needs data.frames_subdir (pre-extracted 1 fps frames)")
    layout, tax, labels = load_all(cfg, need_frames=use_kf)
    split = json.loads(wpath(cfg, "split.json").read_text())
    reliability = json.loads(wpath(cfg, "facts", "reliability.json").read_text())
    out = dataset_path(cfg, mkdir=True)

    def frames(v):
        return FrameDir(layout.root / cfg["data"]["frames_subdir"] / f"video{v:02d}") if use_kf else None

    # validation videos: training videos kept out of GRPO, used only to choose the checkpoint
    val_ids = [v for v in cfg["data"].get("val_video_ids") or [] if v in split["train"]]
    grpo_ids = [v for v in split["train"] if v not in val_ids]
    train = [r for v in grpo_ids for r in build_rows(cfg, tax, labels, v, reliability, frames(v))]
    val = [r for v in val_ids for r in build_rows(cfg, tax, labels, v, reliability, frames(v))]
    test = [r for v in split["test"] for r in build_rows(cfg, tax, labels, v, reliability, frames(v))]

    write_jsonl(out / "grpo_train.jsonl",
                [{k: x for k, x in r.items() if k not in ("template_note", "messages_text")} for r in train])
    write_jsonl(out / "grpo_val.jsonl", val)
    write_jsonl(out / "grpo_eval.jsonl", test)
    print(f"GRPO training videos {grpo_ids} | validation videos {val_ids} | test videos {split['test']}")
    # SFT target = the deterministic template note from the same predicted facts: teaches the
    # output FORMAT only. GRPO is what teaches deviating from the perception output.
    write_jsonl(out / "sft_train.jsonl",
                [{"messages": r["messages"] + [{"role": "assistant", "content": r["template_note"]}],
                  **({"images": r["images"]} if "images" in r else {})} for r in train])
    full = sum(1 for r in train if not r["window"])
    print(f"train: {full} full-procedure + {len(train) - full} window prompts from {len(grpo_ids)} videos; "
          f"validation: {len(val)} prompts")
    if use_kf:
        n_img = [len(r["images"]) for r in train + val + test]
        print(f"keyframes per prompt: mean {np.mean(n_img):.1f}, max {max(n_img)}; "
              f"{sum(n_img)} images in all prompts")
        upload_results(cfg, "keyframes")
    upload_results(cfg, datasets_subdir(cfg))


if __name__ == "__main__":
    main()
