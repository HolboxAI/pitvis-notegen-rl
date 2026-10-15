#!/usr/bin/env bash
# Facts + keyframes -> MedGemma: keyframe datasets, GRPO with crash-safe resume and watchdog,
# validation-based checkpoint selection and held-out evaluation. Safe to re-run at any point:
# every boot continues where the last one stopped (same resume behaviour as job_grpo_v2.sh).
#
#   aws s3 sync . s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-keyframes \
#     --exclude "runs/*" --exclude ".git/*"
#   aws s3 cp infra/job_grpo_keyframes.sh s3://stanford-segment-clips/debjyoti/operative-notes/control/job.sh
#
# Uses the perception outputs of stages 0-3 (taxonomy, split, out-of-fold and test predictions,
# reliability), restored from S3 when the disk does not have them.
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-keyframes}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
OUT=${OUT:-grpo_keyframes}
EVAL=${EVAL:-eval_keyframes}
# MedGemma 1.5 4B: the copy in S3 (operative-notes/models/) is used when present, otherwise the model
# is downloaded from Hugging Face (gated: the token is read from S3, never stored in the repository)
MODEL_S3=s3://stanford-segment-clips/debjyoti/operative-notes/models/medgemma-1.5-4b-it
MODEL_DIR=/opt/notegen/models/medgemma-1.5-4b-it
HF_TOKEN=$(aws s3 cp s3://stanford-segment-clips/debjyoti/operative-notes/control/hf_token.txt - 2>/dev/null || true)
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
if aws s3 ls "$MODEL_S3/config.json" >/dev/null 2>&1; then
  aws s3 sync "$MODEL_S3" "$MODEL_DIR" --only-show-errors
  sed -i "s#^  base: google/medgemma-1.5-4b-it#  base: $MODEL_DIR#" config.yaml   # this instance's copy only
  echo "MedGemma 1.5 4B from $MODEL_S3 -> $MODEL_DIR"
else
  [[ -n "${HF_TOKEN:-}" ]] || { echo "no model at $MODEL_S3 and no Hugging Face token at control/hf_token.txt"; exit 1; }
fi
mkdir -p logs
W=runs/pitvis-notegen
mkdir -p "$W/$OUT"

echo "=== [kf 0] perception outputs (stages 0-3)"
for f in taxonomy.json split.json; do
  [[ -f $W/$f ]] || aws s3 cp "$RUNS_S3/$f" "$W/$f" --only-show-errors
done
for d in predictions facts keyframes; do
  aws s3 sync "$RUNS_S3/$d" "$W/$d" --only-show-errors || true
done
[[ -f $W/facts/reliability.json ]] || { echo "missing facts/reliability.json -- run stages 0-3 first"; exit 1; }

push_ckpts() { aws s3 sync "$W/$OUT" "$RUNS_S3/$OUT" --only-show-errors || true; }
# restore: a fresh disk gets every checkpoint and log back from S3 before anything else
aws s3 sync "$RUNS_S3/$OUT" "$W/$OUT" --only-show-errors || true
( while sleep 300; do push_ckpts; done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null; push_ckpts; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT

source .venv/bin/activate
[[ -f /usr/include/python3.12/Python.h ]] || { sudo apt-get update -qq && sudo apt-get install -y python3-dev; }
python -m pip install -q "qwen-vl-utils>=0.0.14" opencv-python-headless pillow
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pytest -q tests

echo "=== [kf 1] keyframe datasets (work_dir/datasets_keyframes, frames in work_dir/keyframes)"
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
echo "keyframe datasets -> $W/$DS"
aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors || true
if [[ ! -f $W/$DS/.built ]]; then
  python scripts/04_build_datasets.py && touch "$W/$DS/.built"
fi

if [[ -f $W/$OUT/.done ]]; then
  echo "=== [kf 2] training already finished -- skipping"
elif [[ -f $W/$OUT/.watchdog_stop ]]; then
  echo "=== [kf 2] training was stopped by the watchdog -- not restarting:"; cat "$W/$OUT/.watchdog_stop"
else
  RESUME_FROM=$(aws s3 cp "$RUNS_S3/$OUT/resume_from.txt" - 2>/dev/null | tr -d '[:space:]' || true)
  echo "=== [kf 2] GRPO training -> $OUT (resume: ${RESUME_FROM:-newest full checkpoint, if any})"
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

echo "=== [kf 3] checkpoint selection on validation videos"
python scripts/10_select_checkpoint.py --grpo-dir "$OUT"
BEST=$(python -c "import json;print(json.load(open('$W/$OUT/best_checkpoint.json'))['path'])")
echo "selected: $BEST"

echo "=== [kf 4] held-out evaluation: template, base and GRPO with keyframes; base and GRPO without"
python scripts/06_evaluate.py --systems template base --out-subdir "$EVAL"
python scripts/06_evaluate.py --systems --adapter "grpo=$BEST" --out-subdir "$EVAL"
python scripts/06_evaluate.py --systems base --adapter "grpo=$BEST" --text-only --out-subdir "$EVAL"
cat "$W/$EVAL/summary.md"
