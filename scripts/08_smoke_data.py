"""Smoke-test datasets: a 10-procedure evaluation set plus GRPO/SFT training data that
excludes every evaluated procedure.

  smoke_eval.jsonl        full-procedure prompts for the 5 held-out videos + N training videos
                          (training videos use their out-of-fold perception, so they are fair to
                          evaluate as long as the pilot policy never trains on them)
  smoke_grpo_train.jsonl  GRPO prompts from the remaining training videos only
  smoke_sft_train.jsonl   matching format warm-start data
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl.config import load_config, upload_results, wpath  # noqa: E402
from notegen_rl.data import load_all  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_datasets", Path(__file__).with_name("04_build_datasets.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--train-videos", type=int, nargs="*", default=[3, 7, 11, 15, 19],
                    help="training videos to evaluate (held out from the pilot policy)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    _, tax, labels = load_all(cfg)
    split = json.loads(wpath(cfg, "split.json").read_text())
    reliability = json.loads(wpath(cfg, "facts", "reliability.json").read_text())
    eval_train = [v for v in args.train_videos if v in split["train"]]
    eval_ids = split["test"] + eval_train
    pilot_ids = [v for v in split["train"] if v not in eval_train]

    eval_rows = [r for v in eval_ids for r in B.build_rows(cfg, tax, labels, v, reliability) if not r["window"]]
    pilot = [r for v in pilot_ids for r in B.build_rows(cfg, tax, labels, v, reliability)]

    out = wpath(cfg, "datasets", mkdir=True)
    B.write_jsonl(out / "smoke_eval.jsonl", eval_rows)
    B.write_jsonl(out / "smoke_grpo_train.jsonl", [{k: x for k, x in r.items() if k != "template_note"} for r in pilot])
    B.write_jsonl(out / "smoke_sft_train.jsonl",
                  [{"messages": r["messages"] + [{"role": "assistant", "content": r["template_note"]}]} for r in pilot])
    print(f"smoke eval: {len(eval_rows)} full procedures (held-out {split['test']} + training {eval_train})")
    print(f"pilot training: {len(pilot)} prompts from {len(pilot_ids)} videos {pilot_ids}")
    upload_results(cfg, "datasets")


if __name__ == "__main__":
    main()
