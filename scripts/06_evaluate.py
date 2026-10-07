"""Stage 6: score notes on the held-out videos with the same reward functions.

Systems compared (each gets eval/<name>/completions.jsonl + metrics.json):
  template  deterministic note from the perception facts (no LLM) -- the bar to beat
  base      llm.base (+ cold_start_adapter if set), no note-specific training
  grpo      newest GRPO checkpoint under work_dir/grpo
  extra:    --adapter name=/path/to/checkpoint (repeatable)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import rewards as R  # noqa: E402
from notegen_rl.config import load_config, resolve, upload_results, wpath  # noqa: E402
from notegen_rl.llm import Generator, find_latest_checkpoint  # noqa: E402
from notegen_rl.render import to_markdown  # noqa: E402

METRICS = ("weighted_total", "format", "parsed_ok", "grounding", "calibration", "temporal", "safety",
           "step_precision", "step_recall", "step_f1", "instr_precision", "instr_recall", "instr_f1")


def summarize(scores: list, rows: list) -> dict:
    out = {}
    for subset, keep in (("all", lambda r: True), ("full_procedure", lambda r: not r["window"]),
                         ("windows", lambda r: bool(r["window"]))):
        sel = [s for s, r in zip(scores, rows) if keep(r)]
        if not sel:
            continue
        m = {k: sum(s.get(k, 0.0) for s in sel) / len(sel) for k in METRICS}
        m["ece"] = R.expected_calibration_error([p for s in sel for p in s["conf_pairs"]])
        m["n"] = len(sel)
        out[subset] = m
    return out


def evaluate(name, completions, rows, w_by_func, out_root, hedge):
    scores = [R.score_all(c, r) for c, r in zip(completions, rows)]
    for s in scores:
        s["weighted_total"] = sum(w * s[comp] for comp, w in w_by_func.items())
    metrics = summarize(scores, rows)
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "completions.jsonl", "w") as f:
        for r, c, s in zip(rows, completions, scores):
            f.write(json.dumps({"video_id": r["video_id"], "window": r["window"], "completion": c,
                                "scores": {k: v for k, v in s.items() if k != "conf_pairs"}}) + "\n")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    notes = d / "notes"
    notes.mkdir(exist_ok=True)
    for r, c, s in zip(rows, completions, scores):
        tag = f"video{r['video_id']:02d}" + ("_" + "-".join(str(int(x)) for x in json.loads(r["window"])) if r["window"] else "")
        md = to_markdown(c, hedge, f"{tag} ({name})")
        md += "\n---\nScores: " + ", ".join(f"{k}={s[k]:.2f}" for k in ("grounding", "step_f1", "instr_f1", "calibration", "temporal", "safety", "format") if k in s) + "\n"
        (notes / f"{tag}.md").write_text(md)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--systems", nargs="*", default=["template", "base", "grpo"])
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--limit", type=int, help="only the first N eval rows")
    ap.add_argument("--dataset", default="grpo_eval.jsonl", help="file under work_dir/datasets (or a path)")
    ap.add_argument("--full-only", action="store_true", help="only full-procedure rows (no windows)")
    ap.add_argument("--out-subdir", default="eval", help="results go to work_dir/<out-subdir>")
    ap.add_argument("--grpo-dir", default="grpo", help="work_dir subdir searched for the newest GRPO checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = Path(args.dataset) if Path(args.dataset).is_absolute() else wpath(cfg, "datasets", args.dataset)
    rows = [json.loads(line) for line in open(ds)]
    rows = [r for r in rows if not r["window"]] if args.full_only else rows
    rows = rows[:args.limit] if args.limit else rows
    print(f"{len(rows)} eval rows from {ds.name}")
    comp_of = {f: f.replace("note_", "") for f in cfg["grpo"]["reward_funcs"]}
    w_by_func = {comp_of[f]: w for f, w in zip(cfg["grpo"]["reward_funcs"], cfg["grpo"]["reward_weights"])}
    out_root = wpath(cfg, args.out_subdir, mkdir=True)
    cold = str(resolve(cfg["llm"]["cold_start_adapter"])) if cfg["llm"].get("cold_start_adapter") else None

    systems = {}
    for s in args.systems:
        if s == "template":
            systems[s] = None
        elif s == "base":
            systems[s] = cold or ""
        elif s == "grpo":
            ck = find_latest_checkpoint(wpath(cfg, args.grpo_dir))
            if ck:
                systems[s] = ck
            else:
                print("no GRPO checkpoint yet -- skipping 'grpo'")
    for spec in args.adapter:
        name, path = spec.split("=", 1)
        systems[name] = path

    results = {}
    convs = [r["messages"] for r in rows]
    for name, adapter in systems.items():
        print(f"\n=== {name} {('(adapter ' + adapter + ')') if adapter else ''}")
        if adapter is None:
            completions = [r["template_note"] for r in rows]
        else:
            gen = Generator(cfg["llm"]["base"], adapter or None, cfg["llm"])
            completions = gen.generate(convs)
            del gen
            try:
                import gc

                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except ImportError:
                pass
        results[name] = evaluate(name, completions, rows, w_by_func, out_root, cfg["eval"]["hedge_threshold"])

    cols = ("weighted_total", "parsed_ok", "grounding", "step_f1", "instr_f1",
            "calibration", "ece", "temporal", "safety")
    lines = ["| system | subset | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
    # summary covers every system evaluated into this folder so far, not just this invocation
    for mf in sorted(out_root.glob("*/metrics.json")):
        results.setdefault(mf.parent.name, json.loads(mf.read_text()))
    for name, m in results.items():
        for subset, v in m.items():
            vals = ["-" if v.get(c) is None else f"{v[c]:.3f}" for c in cols]
            lines.append(f"| {name} | {subset} | {v['n']} | " + " | ".join(vals) + " |")
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n")
    print("\n" + table)
    upload_results(cfg, args.out_subdir)


if __name__ == "__main__":
    main()
