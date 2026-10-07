#!/usr/bin/env bash
# Base-model format check only (stages 0-4 already done on this instance's disk):
# template vs base model on 20 held-out prompts -> decides whether SFT warm-start is needed.
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
test -f runs/pitvis-notegen/datasets/grpo_eval.jsonl || { echo "datasets missing -- run job_rebuild.sh"; exit 1; }
python scripts/06_evaluate.py --systems template base --limit 20 2>&1 | tee logs/06_format_check.log
