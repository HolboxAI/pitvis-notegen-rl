#!/usr/bin/env bash
# Job 1: environment + stages 0-4 + a 20-prompt format check of the base model.
# Stops before GRPO so the format-compliance result can decide whether SFT warm-start is needed.
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
trap 'aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python3 -c "import ensurepip" 2>/dev/null || { sudo apt-get update -qq && sudo apt-get install -y python3-venv; }
if [[ ! -f .venv/.setup_done ]]; then
  VENV=.venv bash setup_env.sh
  touch .venv/.setup_done
fi
source .venv/bin/activate
aws s3 cp env_lock.txt "$RUNS_S3/env_lock.txt" --only-show-errors || true

NUM_GPUS=$(nvidia-smi -L | wc -l) STAGES="0 1 2 3 4" bash run_pipeline.sh
python scripts/06_evaluate.py --systems template base --limit 20 2>&1 | tee logs/06_format_check.log
aws s3 sync logs "$RUNS_S3/logs" --only-show-errors
