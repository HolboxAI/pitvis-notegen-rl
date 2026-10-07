#!/usr/bin/env bash
# Smoke test on 10 full procedures (5 held-out + 5 training videos excluded from the pilot):
#   1. template baseline + base model notes, scored
#   2. if <50% of base notes parse: short SFT format warm-start, scored
#   3. GRPO pilot (40 steps) on the other 15 training videos
#   4. pilot model notes, scored
# Results (scores, summary table, one Markdown note per procedure and system) are uploaded
# after every step to runs/pitvis-notegen/smoke/, so a later failure never loses earlier results.
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
export VLLM_USE_FLASHINFER_SAMPLER=0
W=runs/pitvis-notegen
python -m pytest -q tests

echo "=== [smoke 0] datasets"
python scripts/08_smoke_data.py

echo "=== [smoke 1] template + base model"
python scripts/06_evaluate.py --dataset smoke_eval.jsonl --out-subdir smoke --systems template base

PARSED=$(python -c "import json;print(json.load(open('$W/smoke/base/metrics.json'))['all']['parsed_ok'])")
echo "base model parse rate: $PARSED"
START=""
if python -c "import sys; sys.exit(0 if $PARSED < 0.5 else 1)"; then
  echo "=== [smoke 2] SFT format warm-start (base parse rate < 0.5)"
  python scripts/05_train.py --stage sft --dataset smoke_sft_train.jsonl --output-subdir sft_smoke --set sft.enabled=true
  START=$(python -c "from notegen_rl.llm import find_latest_checkpoint as f; print(f('$W/sft_smoke') or '')")
  python scripts/06_evaluate.py --dataset smoke_eval.jsonl --out-subdir smoke --systems --adapter "sft=$START"
else
  echo "=== [smoke 2] skipped: base model already follows the format"
fi

echo "=== [smoke 3] GRPO pilot"
python scripts/05_train.py --stage grpo --dataset smoke_grpo_train.jsonl --output-subdir grpo_smoke \
  ${START:+--start-adapter "$START"} \
  --set grpo.max_steps=40 grpo.num_generations=4 grpo.per_device_train_batch_size=2 \
        grpo.gradient_accumulation_steps=4 grpo.max_completion_length=768 grpo.learning_rate=1.0e-5 \
        grpo.warmup_steps=5 grpo.save_steps=20 grpo.vllm_gpu_memory_utilization=0.5

echo "=== [smoke 4] GRPO pilot model"
python scripts/06_evaluate.py --dataset smoke_eval.jsonl --out-subdir smoke --systems grpo --grpo-dir grpo_smoke
cat $W/smoke/summary.md
