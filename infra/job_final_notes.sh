#!/usr/bin/env bash
# Stage 9 with the real writer model: regenerate the final human-readable notes with
# MedGemma-27B-text-it (FP8) from structured notes. Queue only AFTER the full GRPO job has finished
# (control/job.sh is re-run on every boot, so replacing it mid-training would break resume).
# Requires a Hugging Face token with access to google/medgemma-27b-text-it (gated model):
#   aws s3 cp hf_token.txt s3://stanford-segment-clips/debjyoti/operative-notes/control/hf_token.txt
set -euo pipefail
PROJ_S3=s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
mkdir -p logs
trap 'aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
HF_TOKEN=$(aws s3 cp s3://stanford-segment-clips/debjyoti/operative-notes/control/hf_token.txt - 2>/dev/null || true)
[[ -n "$HF_TOKEN" ]] || { echo "no HF token at control/hf_token.txt -- MedGemma is gated"; exit 1; }
export HF_TOKEN
python -m pytest -q tests/test_final_notes.py
W=runs/pitvis-notegen

# smoke-test notes (the 10 procedures in the current document)
python scripts/09_final_notes.py --completions $W/smoke/grpo/completions.jsonl --dataset smoke_eval.jsonl \
  --out-subdir final_notes/smoke_grpo_medgemma
# held-out notes from the full GRPO run, once its evaluation exists
if [[ -f $W/eval/grpo/completions.jsonl ]]; then
  python scripts/09_final_notes.py --completions $W/eval/grpo/completions.jsonl --dataset grpo_eval.jsonl \
    --out-subdir final_notes/heldout_grpo_full
fi
