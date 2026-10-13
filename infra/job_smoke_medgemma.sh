#!/usr/bin/env bash
# Smoke test for the text-only MedGemma 1.5 4B GRPO experiment (job_grpo_v2.sh on this branch):
# the full-run settings with 40 steps and 4 candidate notes per prompt instead of 400 and 8.
#
#   1. template and untrained MedGemma on the 44 validation prompts (videos 5, 12, 18)
#   2. GRPO pilot: 40 steps, 4 generations, watchdog on, checkpoints at steps 20 and 40
#   3. pilot on the same validation prompts (last checkpoint, or the last healthy one after a watchdog stop)
#   4. training diagnostics from the pilot's log
#
# Datasets are built from the stage 0-3 perception outputs into the folder named by dataset.subdir
# (or the default), so the prompts follow this branch's prompt rules. Held-out videos 21-25 are not
# touched.
#
#   aws s3 cp infra/job_smoke_medgemma.sh s3://stanford-segment-clips/debjyoti/operative-notes/control/job.sh
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-medgemma}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
OUT=${OUT:-grpo_medgemma_smoke_r8}
EVAL=${EVAL:-eval_medgemma_smoke_r8}
STEPS=${STEPS:-40}
GENS=${GENS:-4}
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
for f in taxonomy.json split.json; do
  [[ -f $W/$f ]] || aws s3 cp "$RUNS_S3/$f" "$W/$f" --only-show-errors
done
for d in predictions facts; do
  aws s3 sync "$RUNS_S3/$d" "$W/$d" --only-show-errors || true
done

push() { aws s3 sync "$W/$OUT" "$RUNS_S3/$OUT" --only-show-errors || true
         aws s3 sync "$W/$EVAL" "$RUNS_S3/$EVAL" --only-show-errors || true; }
trap 'push; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT

source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pytest -q tests
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
echo "=== [mg-smoke 0] datasets -> $W/$DS"
if [[ ! -f $W/$DS/.built ]]; then
  python scripts/04_build_datasets.py && touch "$W/$DS/.built"
fi

echo "=== [mg-smoke 1] template and untrained MedGemma 1.5 4B on the validation prompts"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems template base --out-subdir "$EVAL"

echo "=== [mg-smoke 2] GRPO pilot: $STEPS steps, $GENS generations per prompt"
if [[ -f $W/$OUT/.done ]]; then
  echo "pilot already trained -- skipping"
else
  setsid python scripts/05_train.py --stage grpo --dataset grpo_train.jsonl --output-subdir "$OUT" --resume \
    --set grpo.max_steps="$STEPS" grpo.num_generations="$GENS" grpo.warmup_steps=5 grpo.save_steps=20 &
  TRAIN=$!
  python scripts/grpo_watchdog.py --grpo-dir "$OUT" --pid "$TRAIN" &
  WATCH=$!
  set +e; wait "$TRAIN"; RC=$?; set -e
  kill "$WATCH" 2>/dev/null || true
  if [[ -f $W/$OUT/.watchdog_stop ]]; then
    echo "watchdog stopped the pilot:"; cat "$W/$OUT/.watchdog_stop"; echo
  elif [[ $RC -ne 0 ]]; then
    echo "pilot training exited with rc=$RC"; exit "$RC"
  else
    touch "$W/$OUT/.done"
  fi
fi

echo "=== [mg-smoke 3] pilot on the validation prompts"
if [[ -f $W/$OUT/.watchdog_stop ]]; then
  CK=$(python -c "import json;print(json.load(open('$W/$OUT/.watchdog_stop')).get('last_healthy_checkpoint') or '')")
else
  CK=$(python -c "from notegen_rl.llm import find_latest_checkpoint as f;print(f('$W/$OUT') or '')")
fi
[[ -n "$CK" ]] || { echo "no pilot checkpoint to evaluate"; exit 1; }
echo "pilot checkpoint: $CK"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems --adapter "grpo=$CK" --out-subdir "$EVAL"

echo "=== [mg-smoke 4] training diagnostics"
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
echo
echo "Qwen3.5-9B GRPO on the same 44 validation prompts (select_grpo_v2):"
aws s3 cp "$RUNS_S3/select_grpo_v2/summary.md" - 2>/dev/null | grep -E '^\| (system|step0050|step0400) .*\| (subset|all) ' || true
