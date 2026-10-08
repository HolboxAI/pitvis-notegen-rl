"""Choose the GRPO checkpoint to report: evaluate checkpoints on the validation videos
(data.val_video_ids, never used for GRPO training) and keep the best weighted reward.

  python scripts/10_select_checkpoint.py --grpo-dir grpo_v2

Checkpoints at every `grpo.select_every` steps (plus the last one) are evaluated, each in its own
process so GPU memory is released between models. Checkpoints after a watchdog stop are skipped.
Writes <grpo-dir>/best_checkpoint.json and validation results under <work_dir>/select_<grpo-dir>/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl.config import PROJECT_DIR, load_config, upload_results, wpath  # noqa: E402
from notegen_rl.llm import list_checkpoints  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--grpo-dir", default="grpo")
    ap.add_argument("--dataset", default="grpo_val.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)
    grpo_dir = wpath(cfg, args.grpo_dir)
    every = int(cfg["grpo"].get("select_every", 50))

    cks = {int(c.split("-")[-1]): c for c in list_checkpoints(grpo_dir)}
    stop_file = grpo_dir / ".watchdog_stop"
    if stop_file.exists():
        trip = json.loads(stop_file.read_text())
        healthy = trip.get("last_healthy_checkpoint")
        limit = int(healthy.split("-")[-1]) if healthy else 0
        cks = {s: c for s, c in cks.items() if s <= limit}
        print(f"watchdog stopped training ({trip['reason']}); considering checkpoints up to step {limit}")
    if not cks:
        raise SystemExit(f"no checkpoints to evaluate under {grpo_dir}")
    chosen = sorted(s for s in cks if s % every == 0) or []
    if max(cks) not in chosen:
        chosen.append(max(cks))

    sub = f"select_{args.grpo_dir}"
    scores = {}
    for step in chosen:
        name = f"step{step:04d}"
        metrics = wpath(cfg, sub, name, "metrics.json")
        if not metrics.exists():
            subprocess.run([sys.executable, "scripts/06_evaluate.py", "--dataset", args.dataset, "--systems",
                            "--adapter", f"{name}={cks[step]}", "--out-subdir", sub]
                           + (["--config", args.config] if args.config else []), check=True, cwd=PROJECT_DIR)
        scores[step] = json.loads(metrics.read_text())["all"]["weighted_total"]
        print(f"step {step}: validation weighted total {scores[step]:.3f}", flush=True)

    best = max(scores, key=scores.get)
    result = {"step": best, "path": cks[best], "validation_weighted_total": scores[best],
              "all": {str(k): v for k, v in sorted(scores.items())}}
    (grpo_dir / "best_checkpoint.json").write_text(json.dumps(result, indent=2))
    print("best checkpoint:", json.dumps(result), flush=True)
    upload_results(cfg, sub)
    upload_results(cfg, args.grpo_dir)


if __name__ == "__main__":
    main()
