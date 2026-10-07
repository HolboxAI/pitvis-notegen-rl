"""Stage 3: predictions -> facts (smoothed step segments + instruments), gold facts from
labels, and per-class reliability (F1 on out-of-fold training predictions only -- test
videos never inform it)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from notegen_rl.config import load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import load_all  # noqa: E402
from notegen_rl.facts import facts_from_labels, facts_from_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fcfg = cfg["facts"]

    _, tax, labels = load_all(cfg)
    split = json.loads(wpath(cfg, "split.json").read_text())
    pdir, gdir = wpath(cfg, "facts", "pred", mkdir=True), wpath(cfg, "facts", "gold", mkdir=True)
    S, K = len(tax.step_names), len(tax.instr_names)

    acc = {"oof": ([], [], [], []), "test": ([], [], [], [])}  # y_step, p_step, y_instr, p_instr
    for v in split["train"] + split["test"]:
        d = np.load(wpath(cfg, "predictions", f"video{v:02d}.npz"))
        times = d["times"]
        facts, path = facts_from_predictions(v, times, d["step_probs"], d["instr_probs"], tax, fcfg,
                                             d["log_trans"], d["log_init"])
        facts.save(pdir / f"video{v:02d}.json")
        lab = labels[v].subset(np.isin(labels[v].times, times))
        facts_from_labels(v, labels[v], tax).save(gdir / f"video{v:02d}.json")
        bucket = acc[str(d["kind"])]
        bucket[0].append(lab.steps)
        bucket[1].append(path)
        bucket[2].append(lab.instruments)
        bucket[3].append((d["instr_probs"] >= fcfg["instrument_threshold"]).astype(np.uint8))
        print(f"video{v:02d} [{d['kind']}]: {len(facts.segments)} predicted segments")

    metrics = {}
    reliability = {}
    for kind, (ys, ps, yi, pi) in acc.items():
        if not ys:
            continue
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        per_step = f1_score(ys, ps, labels=list(range(S)), average=None, zero_division=0)
        present = np.unique(ys)
        m = {"step_accuracy": float(accuracy_score(ys, ps)),
             "step_macro_f1": float(f1_score(ys, ps, labels=present, average="macro", zero_division=0)),
             "per_step_f1": {tax.step_names[c]: float(per_step[c]) for c in present}}
        if K:
            yi, pi = np.concatenate(yi), np.concatenate(pi)
            per_instr = f1_score(yi, pi, average=None, zero_division=0)
            m["instrument_macro_f1"] = float(per_instr[yi.sum(0) > 0].mean()) if (yi.sum(0) > 0).any() else 0.0
            m["per_instrument_f1"] = {tax.instr_names[k]: float(per_instr[k]) for k in range(K) if yi[:, k].any()}
        metrics[kind] = m
        print(f"{kind}: smoothed step acc={m['step_accuracy']:.4f} macro-F1={m['step_macro_f1']:.4f}"
              + (f" | instrument macro-F1={m['instrument_macro_f1']:.4f}" if K else ""))
        if kind == "oof":
            reliability = {**m["per_step_f1"], **m.get("per_instrument_f1", {})}

    wpath(cfg, "facts", "perception_metrics.json").write_text(json.dumps(metrics, indent=2))
    wpath(cfg, "facts", "reliability.json").write_text(json.dumps(reliability, indent=2))
    print("\nreliability (OOF per-class F1, used in prompts and reward weights):")
    for n, r in sorted(reliability.items(), key=lambda x: -x[1]):
        print(f"  {r:.2f}  {n}")
    upload_results(cfg, "facts")


if __name__ == "__main__":
    main()
