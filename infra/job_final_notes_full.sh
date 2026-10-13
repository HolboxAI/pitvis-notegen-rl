#!/usr/bin/env bash
# Narrative final notes (stage 9) for the MedGemma smoke test: the full-procedure structured notes of
# the template, untrained MedGemma and the GRPO pilot on the validation videos (eval_medgemma, held-out videos 21-25),
# written by WRITER (default: the MedGemma 1.5 4B copy in S3) and checked by the verifier.
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-medgemma}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
EVAL=${EVAL:-eval_medgemma}
MODEL_S3=s3://stanford-segment-clips/debjyoti/operative-notes/models/medgemma-1.5-4b-it
MODEL_DIR=/opt/notegen/models/medgemma-1.5-4b-it
WRITER=${WRITER:-$MODEL_DIR}
HF_TOKEN=$(aws s3 cp s3://stanford-segment-clips/debjyoti/operative-notes/control/hf_token.txt - 2>/dev/null || true)
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
aws s3 sync "$MODEL_S3" "$MODEL_DIR" --only-show-errors
mkdir -p logs
W=runs/pitvis-notegen
trap 'aws s3 sync $W/final_notes "$RUNS_S3/final_notes" --only-show-errors || true; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
aws s3 sync "$RUNS_S3/$EVAL" "$W/$EVAL" --exclude "*/notes/*" --only-show-errors
DS=$(cd /opt/notegen/pitvis-notegen-rl && .venv/bin/python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))"); aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pytest -q tests/test_final_notes.py

for s in template base grpo; do
  echo "=== [final notes] $s"
  python scripts/09_final_notes.py --completions "$W/$EVAL/$s/completions.jsonl" --dataset grpo_eval.jsonl \
    --out-subdir "final_notes/medgemma_full_${s}" --writer-model "$WRITER" --writer-quantization none
done
for s in template base grpo; do
  echo "--- $s"
  python - "$W/final_notes/medgemma_full_${s}/final_notes.jsonl" <<'EOF'
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    print(f"video {r['video_id']}: writer {r['writer']}, attempts {r['attempts']}, rejections {len(r['rejections'])}")
EOF
done
