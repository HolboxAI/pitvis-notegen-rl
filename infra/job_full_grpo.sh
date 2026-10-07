#!/usr/bin/env bash
# Full GRPO run on all 20 training videos (302 prompts), then evaluation on the 5 held-out videos
# (85 prompts: 5 full procedures + 80 windows) for template, base and GRPO.
# Restart-safe: training resumes from the newest checkpoint after a Spot interruption, and is
# skipped once finished (grpo/.done), so a restart during evaluation goes straight to evaluation.
# Settings that differ from config.yaml are sized for one 48 GB GPU (see the pilot in smoke/).
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
W=runs/pitvis-notegen
# progress: push training logs to S3 every 10 min while the job runs
( while sleep 600; do
    aws s3 sync $W/grpo "$RUNS_S3/grpo" --exclude "*" --include "*/logging.jsonl" --include "*/args.json" --only-show-errors || true
  done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
[[ -f /usr/include/python3.12/Python.h ]] || { sudo apt-get update -qq && sudo apt-get install -y python3-dev; }
python -m pip install -q "qwen-vl-utils>=0.0.14"
export VLLM_USE_FLASHINFER_SAMPLER=0
test -f $W/datasets/grpo_train.jsonl || { echo "datasets missing -- run job_rebuild.sh first"; exit 1; }

if [[ ! -f $W/grpo/.done ]]; then
  echo "=== [full 1] GRPO training (1000 steps)"
  python scripts/05_train.py --stage grpo --resume \
    --set grpo.num_generations=8 grpo.per_device_train_batch_size=1 grpo.gradient_accumulation_steps=8 \
          grpo.learning_rate=1.0e-5 grpo.max_completion_length=1280 grpo.vllm_gpu_memory_utilization=0.5 \
          grpo.save_steps=100
  touch $W/grpo/.done
else
  echo "=== [full 1] GRPO training already finished -- skipping"
fi

echo "=== [full 2] evaluation on held-out videos (template, base, grpo)"
python scripts/06_evaluate.py --systems template base --out-subdir eval
python scripts/06_evaluate.py --systems grpo --out-subdir eval   # separate process: frees vLLM memory between models
cat $W/eval/summary.md
