#!/usr/bin/env bash
# Narrative final notes (stage 9) for the surgeon-review cases: the pilot's structured notes in
# runs/pitvis-notegen/review_pilot (job_review_notes.sh), written by MedGemma 1.5 4B and verified.
# Notes that are not complete (cut off) are skipped by stage 9.
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-medgemma}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
SRC=${SRC:-review_pilot}
VIDEOS=${VIDEOS:-"3 7 11 15 19 21 22 23 24 25"}
MODEL_S3=s3://stanford-segment-clips/debjyoti/operative-notes/models/medgemma-1.5-4b-it
MODEL_DIR=/opt/notegen/models/medgemma-1.5-4b-it
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
aws s3 sync "$MODEL_S3" "$MODEL_DIR" --only-show-errors
mkdir -p logs
W=runs/pitvis-notegen
trap 'aws s3 sync $W/final_notes/$SRC "$RUNS_S3/final_notes/$SRC" --only-show-errors || true; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors
aws s3 sync "$RUNS_S3/$SRC" "$W/$SRC" --exclude "*/notes/*" --only-show-errors
if [[ ! -f $W/$DS/review_full.jsonl ]]; then
  python - "$W/$DS" "$VIDEOS" <<'EOF'
import json, sys
from pathlib import Path
d, want = Path(sys.argv[1]), {int(v) for v in sys.argv[2].split()}
rows = {}
for f in ("grpo_train.jsonl", "grpo_val.jsonl", "grpo_eval.jsonl"):
    for r in map(json.loads, open(d / f)):
        if not r["window"] and r["video_id"] in want:
            rows[r["video_id"]] = r
with open(d / "review_full.jsonl", "w") as f:
    for v in sorted(rows):
        f.write(json.dumps(rows[v]) + "\n")
EOF
fi
python -m pytest -q tests/test_final_notes.py

echo "=== [final notes] $SRC"
python scripts/09_final_notes.py --completions "$W/$SRC/pilot/completions.jsonl" --dataset review_full.jsonl \
  --out-subdir "final_notes/$SRC" --writer-model "$MODEL_DIR" --writer-quantization none
python - "$W/final_notes/$SRC/final_notes.jsonl" <<'EOF'
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    print(f"video {r['video_id']}: writer {r['writer']}, attempts {r['attempts']}, rejections {len(r['rejections'])}")
EOF
