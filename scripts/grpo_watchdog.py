"""GRPO training watchdog: follows ms-swift's logging.jsonl and stops training when it destabilises.

  python scripts/grpo_watchdog.py --grpo-dir grpo_v2 --pid <training pid (process-group leader)>

Every `interval` seconds it averages the last `window` logged steps and stops the training process
group if any limit in config.yaml (grpo.watchdog) is exceeded:
  mean KL, mean gradient norm, share of clipped completions, mean completion length relative to the
  first `window` steps, or mean reward relative to the best window so far.
On a stop it writes <grpo-dir>/.watchdog_stop (JSON: step, reason, last_healthy_checkpoint), where
last_healthy_checkpoint is the newest full checkpoint saved before the window that tripped.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl.config import load_config, wpath  # noqa: E402
from notegen_rl.llm import list_checkpoints  # noqa: E402


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_steps(grpo_dir: Path) -> list:
    """Per-step log records from every logging.jsonl under grpo_dir (resumed runs append new files)."""
    rows = {}
    for log in sorted(grpo_dir.rglob("logging.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in log.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gs = d.get("global_step/max_steps")
            if d.get("reward") is None or not gs:
                continue
            step = int(str(gs).split("/")[0])
            rows[step] = {"step": step, "reward": _f(d.get("reward")), "kl": _f(d.get("kl")),
                          "grad_norm": _f(d.get("grad_norm")),
                          "clipped": _f(d.get("completions/clipped_ratio")),
                          "length": _f(d.get("completions/mean_length"))}
    return [rows[k] for k in sorted(rows)]


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def check(rows: list, w: dict) -> str | None:
    n = int(w["window"])
    if len(rows) < max(n * 2, 1) or rows[-1]["step"] < int(w["min_step"]):
        return None
    base_len = mean(rows[:n], "length")
    recent = rows[-n:]
    best = max(mean(rows[i:i + n], "reward") or 0.0 for i in range(0, len(rows) - n + 1))
    tests = [
        ("mean KL", mean(recent, "kl"), w["max_kl"], ">"),
        ("mean gradient norm", mean(recent, "grad_norm"), w["max_grad_norm"], ">"),
        ("clipped completions", mean(recent, "clipped"), w["max_clipped_ratio"], ">"),
        ("completion length vs start", (mean(recent, "length") or 0) / base_len if base_len else None,
         w["max_length_ratio"], ">"),
        ("reward vs best window", (mean(recent, "reward") or 0) / best if best > 0 else None,
         w["min_reward_ratio"], "<"),
    ]
    for name, value, limit, op in tests:
        if value is not None and ((op == ">" and value > limit) or (op == "<" and value < limit)):
            return f"{name} {value:.3f} {op} {limit} over steps {recent[0]['step']}-{recent[-1]['step']}"
    return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid: int):
    for sig, wait in ((signal.SIGTERM, 90), (signal.SIGKILL, 0)):
        try:
            os.killpg(pid, sig)
        except OSError:
            return
        for _ in range(wait):
            if not alive(pid):
                return
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--grpo-dir", default="grpo")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config(args.config)
    w = cfg["grpo"]["watchdog"]
    grpo_dir = wpath(cfg, args.grpo_dir)

    while alive(args.pid):
        rows = read_steps(grpo_dir)
        reason = check(rows, w)
        if reason:
            first_bad = rows[-int(w["window"])]["step"]
            healthy = [c for c in list_checkpoints(grpo_dir, resumable=True) if int(c.split("-")[-1]) < first_bad]
            info = {"step": rows[-1]["step"], "reason": reason,
                    "last_healthy_checkpoint": healthy[-1] if healthy else None}
            (grpo_dir / ".watchdog_stop").write_text(json.dumps(info, indent=2))
            print("WATCHDOG STOP:", json.dumps(info), flush=True)
            stop(args.pid)
            return
        if rows:
            r = rows[-1]
            print(f"watchdog ok @ step {r['step']}: reward {r['reward']} kl {r['kl']} "
                  f"grad {r['grad_norm']} len {r['length']}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
