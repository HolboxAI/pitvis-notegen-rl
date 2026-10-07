#!/usr/bin/env bash
# Smoke test, part 2 (smoke datasets + template results already exist on this instance):
#   1. score the base model with a 1536-token generation limit
#   2. GRPO pilot (40 steps) on the 15 training videos not in the smoke evaluation set
#   3. score the pilot model on the same 10 procedures
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
trap 'aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
[[ -f /usr/include/python3.12/Python.h ]] || { sudo apt-get update -qq && sudo apt-get install -y python3-dev; }
python -m pip install -q "qwen-vl-utils>=0.0.14"
export VLLM_USE_FLASHINFER_SAMPLER=0
W=runs/pitvis-notegen
test -f $W/datasets/smoke_grpo_train.jsonl || python scripts/08_smoke_data.py

echo "=== [smoke 1b] base model, 1536-token limit"
python scripts/06_evaluate.py --dataset smoke_eval.jsonl --out-subdir smoke --systems base

echo "=== [smoke 3] GRPO pilot"
python scripts/05_train.py --stage grpo --dataset smoke_grpo_train.jsonl --output-subdir grpo_smoke \
  --set grpo.max_steps=40 grpo.num_generations=4 grpo.per_device_train_batch_size=1 \
        grpo.gradient_accumulation_steps=8 grpo.max_completion_length=1280 grpo.learning_rate=1.0e-5 \
        grpo.warmup_steps=5 grpo.save_steps=20 grpo.vllm_gpu_memory_utilization=0.5

echo "=== [smoke 4] GRPO pilot model"
python scripts/06_evaluate.py --dataset smoke_eval.jsonl --out-subdir smoke --systems grpo --grpo-dir grpo_smoke
cat $W/smoke/summary.md
