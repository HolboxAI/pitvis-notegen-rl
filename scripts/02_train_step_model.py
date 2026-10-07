"""Stage 2: temporal step/instrument model.

Training videos get OUT-OF-FOLD predictions (k-fold by video): each is predicted by a model
that never saw it, so GRPO training prompts carry the same kind of errors the system makes
on a new video. Test videos are predicted by the final model trained on all training videos.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from notegen_rl.config import load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import load_all  # noqa: E402
from notegen_rl.step_model import Sequence, estimate_transitions, fit, predict  # noqa: E402


def load_sequences(cfg, labels, ids):
    seqs = {}
    for v in ids:
        f = wpath(cfg, "features", f"video{v:02d}.npz")
        if not f.exists():
            raise SystemExit(f"missing features for video {v} -- run 01_extract_features.py")
        d = np.load(f)
        lab = labels[v]
        common, fi, li = np.intersect1d(d["times"], lab.times, return_indices=True)
        seqs[v] = Sequence(v, d["feats"][fi].astype(np.float32), lab.steps[li], lab.instruments[li])
    return seqs


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0)
    print(f"  {name}: per-second accuracy={acc:.4f} macro-F1={f1:.4f} (raw argmax, before smoothing)")
    return {"accuracy": float(acc), "macro_f1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["step_model"]

    _, tax, labels = load_all(cfg)
    split = json.loads(wpath(cfg, "split.json").read_text())
    train_ids, test_ids = split["train"], split["test"]
    seqs = load_sequences(cfg, labels, train_ids + test_ids)
    S, K = len(tax.step_names), len(tax.instr_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir = wpath(cfg, "predictions", mkdir=True)
    model_dir = wpath(cfg, "step_model", mkdir=True)

    def save_pred(v, bundle, log_trans, log_init, kind):
        sp, ip = predict(bundle, seqs[v].feats, device)
        times = np.intersect1d(np.load(wpath(cfg, "features", f"video{v:02d}.npz"))["times"], labels[v].times)
        np.savez(pred_dir / f"video{v:02d}.npz", times=times, step_probs=sp, instr_probs=ip,
                 log_trans=log_trans, log_init=log_init, kind=kind)
        return sp

    # --- out-of-fold predictions for training videos ---
    order = list(train_ids)
    random.Random(mcfg["seed"]).shuffle(order)
    k = max(2, min(mcfg["cv_folds"], len(order)))
    folds = [order[i::k] for i in range(k)]
    oof_true, oof_pred = [], []
    for fi, held in enumerate(folds):
        tr = [v for v in train_ids if v not in held]
        print(f"fold {fi + 1}/{k}: train on {len(tr)} videos, predict {held}")
        bundle = fit([seqs[v] for v in tr], S, K, mcfg, device, tag=f"fold{fi + 1}")
        lt, li = estimate_transitions([seqs[v].steps for v in tr], S)
        for v in held:
            sp = save_pred(v, bundle, lt, li, "oof")
            oof_true.append(seqs[v].steps)
            oof_pred.append(sp.argmax(1))

    # --- final model on all training videos ---
    print(f"final model: train on {len(train_ids)} videos")
    bundle = fit([seqs[v] for v in train_ids], S, K, mcfg, device, tag="final")
    lt, li = estimate_transitions([seqs[v].steps for v in train_ids], S)
    bundle.update(log_trans=lt, log_init=li, taxonomy=tax.__dict__)
    torch.save(bundle, model_dir / "final.pt")
    test_true, test_pred = [], []
    for v in test_ids:
        sp = save_pred(v, bundle, lt, li, "test")
        test_true.append(seqs[v].steps)
        test_pred.append(sp.argmax(1))

    metrics = {"oof_train": report("OOF train", np.concatenate(oof_true), np.concatenate(oof_pred))}
    if test_ids:
        metrics["test"] = report("test", np.concatenate(test_true), np.concatenate(test_pred))
    (model_dir / "metrics_raw.json").write_text(json.dumps(metrics, indent=2))
    print("benchmark reference (frozen linear probe, same split): step accuracy 0.535, macro-F1 0.366")
    upload_results(cfg, "predictions")
    upload_results(cfg, "step_model")


if __name__ == "__main__":
    main()
