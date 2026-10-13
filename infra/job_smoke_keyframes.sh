#!/usr/bin/env bash
# Smoke test for facts + keyframes -> MedGemma 1.5 4B (job_grpo_keyframes.sh): the full-run settings
# with 40 steps and 4 candidate notes per prompt, the same as the text-only smoke test.
#
#   0. perception outputs (stages 0-3) restored from S3; keyframe datasets built (datasets_keyframes/)
#   1. template, untrained MedGemma with keyframes and without (base_text) on the 44 validation prompts
#   2. GRPO pilot with keyframes: 40 steps, 4 generations, watchdog on, checkpoints at steps 20 and 40
#   3. pilot with keyframes and without (grpo_text) on the same prompts (last checkpoint, or the last
#      healthy one after a watchdog stop)
#   4. training diagnostics
#
#   aws s3 cp infra/job_smoke_keyframes.sh s3://stanford-segment-clips/debjyoti/operative-notes/control/job.sh
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-keyframes}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
OUT=${OUT:-grpo_keyframes_smoke_r5}
EVAL=${EVAL:-eval_keyframes_smoke_r5}
STEPS=${STEPS:-40}
GENS=${GENS:-4}
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
grep -q '^  enabled: true' config.yaml || { echo "keyframes.enabled must be true in this code copy"; exit 1; }
mkdir -p logs
W=runs/pitvis-notegen
mkdir -p "$W/$OUT"

push() { aws s3 sync "$W/$OUT" "$RUNS_S3/$OUT" --only-show-errors || true
         aws s3 sync "$W/$EVAL" "$RUNS_S3/$EVAL" --only-show-errors || true; }
trap 'push; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT

echo "=== [kf-smoke 0] perception outputs and keyframe datasets"
for f in taxonomy.json split.json; do
  [[ -f $W/$f ]] || aws s3 cp "$RUNS_S3/$f" "$W/$f" --only-show-errors
done
for d in predictions facts keyframes; do
  aws s3 sync "$RUNS_S3/$d" "$W/$d" --only-show-errors || true
done
source .venv/bin/activate
python -m pip install -q opencv-python-headless pillow
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pytest -q tests
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
echo "keyframe datasets -> $W/$DS"
aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors || true
if [[ ! -f $W/$DS/.built ]]; then
  python scripts/04_build_datasets.py && touch "$W/$DS/.built"
fi

echo "=== [kf-smoke 1] template and untrained MedGemma 1.5 4B, with and without keyframes"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems template base --out-subdir "$EVAL"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems base --text-only --out-subdir "$EVAL"

echo "=== [kf-smoke 2] GRPO pilot with keyframes: $STEPS steps, $GENS generations per prompt"
if [[ -f $W/$OUT/.done ]]; then
  echo "pilot already trained -- skipping"
else
  train_pilot() {  # $1 = extra --set overrides; output streamed to the job log and logs/kf_smoke_train.log
    setsid bash -c "set -o pipefail; python scripts/05_train.py --stage grpo --dataset grpo_train.jsonl \
      --output-subdir $OUT --resume --set grpo.max_steps=$STEPS grpo.num_generations=$GENS \
      grpo.warmup_steps=5 grpo.save_steps=20 $1 2>&1 | tee logs/kf_smoke_train.log" &
    TRAIN=$!
    python scripts/grpo_watchdog.py --grpo-dir "$OUT" --pid "$TRAIN" &
    WATCH=$!
    set +e; wait "$TRAIN"; RC=$?; set -e
    kill "$WATCH" 2>/dev/null || true
  }
  train_pilot ""
  grep -h "\[notegen\]" logs/kf_smoke_train.log | head -6 || true
  if [[ $RC -ne 0 ]] && grep -q "prompt placeholders" logs/kf_smoke_train.log; then
    echo "vLLM rollouts failed on image markers -- retrying the pilot with ms-swift generation (grpo.use_vllm=false)"
    rm -rf "$W/$OUT"; mkdir -p "$W/$OUT"
    train_pilot "grpo.use_vllm=false"
  fi
  if [[ -f $W/$OUT/.watchdog_stop ]]; then
    echo "watchdog stopped the pilot:"; cat "$W/$OUT/.watchdog_stop"; echo
  elif [[ $RC -ne 0 ]]; then
    echo "pilot training exited with rc=$RC"; exit "$RC"
  else
    touch "$W/$OUT/.done"
  fi
fi

echo "=== [kf-smoke 3] pilot with and without keyframes"
if [[ -f $W/$OUT/.watchdog_stop ]]; then
  CK=$(python -c "import json;print(json.load(open('$W/$OUT/.watchdog_stop')).get('last_healthy_checkpoint') or '')")
else
  CK=$(python -c "from notegen_rl.llm import find_latest_checkpoint as f;print(f('$W/$OUT') or '')")
fi
[[ -n "$CK" ]] || { echo "no pilot checkpoint to evaluate"; exit 1; }
echo "pilot checkpoint: $CK"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems --adapter "grpo=$CK" --out-subdir "$EVAL"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems --adapter "grpo=$CK" --text-only --out-subdir "$EVAL"

echo "=== [kf-smoke 4] training diagnostics"
python - "$W/$OUT" <<'EOF'
import json, sys
from pathlib import Path
logs = sorted(Path(sys.argv[1]).rglob("logging.jsonl"))
rows = [r for r in map(json.loads, open(logs[-1])) if r.get("reward") is not None] if logs else []
if not rows:
    sys.exit("no training log found")
keys = [("reward", "reward"), ("reward_std", "reward std"), ("frac_reward_zero_std", "zero-std groups"),
        ("rewards/NoteFormat/mean", "format"), ("rewards/NoteGrounding/mean", "grounding"),
        ("rewards/NoteCalibration/mean", "calibration"), ("rewards/NoteConcise/mean", "concise"),
        ("completions/mean_length", "length (tokens)"), ("completions/clipped_ratio", "clipped"),
        ("kl", "KL"), ("grad_norm", "grad norm"), ("step_time", "s / step")]
blocks = [rows[i:i + 10] for i in range(0, len(rows), 10)]
print("steps      " + " | ".join(f"{k:>10.10}" for _, k in keys))
for i, b in enumerate(blocks):
    vals = [sum(r.get(key, 0) or 0 for r in b) / len(b) for key, _ in keys]
    print(f"{10 * i + 1:>3}-{10 * i + len(b):<5}  " + " | ".join(f"{v:10.4f}" for v in vals))
EOF
echo
cat "$W/$EVAL/summary.md"
