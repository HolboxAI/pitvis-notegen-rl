#!/usr/bin/env bash
# Stabilised GRPO run with crash-safe resume, watchdog, validation-based checkpoint selection and
# held-out evaluation. Safe to re-run at any point: every boot continues where the last one stopped.
#
# Resume behaviour
#   - full checkpoints (adapter + optimizer + scheduler + step) every grpo.save_steps, all kept;
#   - the output dir is synced to S3 every 5 minutes and at exit, and restored from S3 at start,
#     so a new instance or a lost disk resumes from the newest checkpoint in S3;
#   - training restarts with --resume from the highest-step full checkpoint;
#   - to resume from a specific checkpoint instead, put its path (relative to the project dir,
#     e.g. runs/pitvis-notegen/grpo_v2/v0-.../checkpoint-150) in $RUNS_S3/$OUT/resume_from.txt;
#   - after a watchdog stop, training is not restarted automatically: delete $OUT/.watchdog_stop
#     (locally and in S3) and set resume_from.txt to its last_healthy_checkpoint to continue.
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
OUT=${OUT:-grpo_v2}
EVAL=${EVAL:-eval_v2}
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
W=runs/pitvis-notegen
mkdir -p "$W/$OUT"

push_ckpts() { aws s3 sync "$W/$OUT" "$RUNS_S3/$OUT" --only-show-errors || true; }
# restore: a fresh disk gets every checkpoint and log back from S3 before anything else
aws s3 sync "$RUNS_S3/$OUT" "$W/$OUT" --only-show-errors || true
( while sleep 300; do push_ckpts; done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null; push_ckpts; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT

source .venv/bin/activate
[[ -f /usr/include/python3.12/Python.h ]] || { sudo apt-get update -qq && sudo apt-get install -y python3-dev; }
python -m pip install -q "qwen-vl-utils>=0.0.14"
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pytest -q tests

echo "=== [v2 0] datasets (validation split, length-controlled prompts)"
if [[ ! -f $W/datasets/.v2 ]]; then
  python scripts/04_build_datasets.py && touch $W/datasets/.v2
fi

if [[ -f $W/$OUT/.done ]]; then
  echo "=== [v2 1] training already finished -- skipping"
elif [[ -f $W/$OUT/.watchdog_stop ]]; then
  echo "=== [v2 1] training was stopped by the watchdog -- not restarting:"; cat "$W/$OUT/.watchdog_stop"
else
  RESUME_FROM=$(aws s3 cp "$RUNS_S3/$OUT/resume_from.txt" - 2>/dev/null | tr -d '[:space:]' || true)
  echo "=== [v2 1] GRPO training -> $OUT (resume: ${RESUME_FROM:-newest full checkpoint, if any})"
  setsid python scripts/05_train.py --stage grpo --output-subdir "$OUT" --resume \
    ${RESUME_FROM:+--resume-from "$RESUME_FROM"} &
  TRAIN=$!
  python scripts/grpo_watchdog.py --grpo-dir "$OUT" --pid "$TRAIN" &
  WATCH=$!
  set +e; wait "$TRAIN"; RC=$?; set -e
  kill "$WATCH" 2>/dev/null || true
  push_ckpts
  if [[ -f $W/$OUT/.watchdog_stop ]]; then
    echo "training stopped by the watchdog:"; cat "$W/$OUT/.watchdog_stop"
  elif [[ $RC -ne 0 ]]; then
    echo "training exited with rc=$RC -- restart the instance to resume from the newest checkpoint"
    exit "$RC"
  else
    touch "$W/$OUT/.done"
  fi
fi

echo "=== [v2 2] checkpoint selection on validation videos"
python scripts/10_select_checkpoint.py --grpo-dir "$OUT"
BEST=$(python -c "import json;print(json.load(open('$W/$OUT/best_checkpoint.json'))['path'])")
echo "selected: $BEST"

echo "=== [v2 3] held-out evaluation (template, base, selected GRPO checkpoint)"
python scripts/06_evaluate.py --systems template base --out-subdir "$EVAL"
python scripts/06_evaluate.py --systems --adapter "grpo=$BEST" --out-subdir "$EVAL"
cat "$W/$EVAL/summary.md"
