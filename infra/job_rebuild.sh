#!/usr/bin/env bash
# Sync the latest code and config, rebuild the taxonomy, and rerun every stage that depends on it.
# Stage 1 is included only to resume unfinished videos -- finished feature files are reused
# (features do not depend on the taxonomy). Ends with the base-model format check.
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
trap 'aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
# Triton (vLLM) JIT-compiles a C helper and needs Python.h -- missing from the AMI by default
[[ -f /usr/include/python3.12/Python.h ]] || { sudo apt-get update -qq && sudo apt-get install -y python3-dev; }
python -m pytest -q tests

W=runs/pitvis-notegen
rm -f $W/taxonomy.json $W/split.json
rm -rf $W/predictions $W/step_model $W/facts $W/datasets $W/eval
NUM_GPUS=$(nvidia-smi -L | wc -l) STAGES="0 1 2 3 4" bash run_pipeline.sh
python scripts/06_evaluate.py --systems template base --limit 20 2>&1 | tee logs/06_format_check.log
