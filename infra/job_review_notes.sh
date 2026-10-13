#!/usr/bin/env bash
# Structured notes from the MedGemma pilot (grpo_medgemma_smoke_r5, last checkpoint) for the full procedures
# of the videos in VIDEOS, for surgeon review. Output: runs/pitvis-notegen/review_pilot/pilot/{completions.jsonl,notes/}
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-medgemma}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
PILOT=${PILOT:-grpo_medgemma_smoke_r5}
OUT=${OUT:-review_pilot}
VIDEOS=${VIDEOS:-"3 7 11 15 19 21 22 23 24 25"}
MODEL_S3=s3://stanford-segment-clips/debjyoti/operative-notes/models/medgemma-1.5-4b-it
MODEL_DIR=/opt/notegen/models/medgemma-1.5-4b-it
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
aws s3 sync "$MODEL_S3" "$MODEL_DIR" --only-show-errors
sed -i "s#^  base: google/medgemma-1.5-4b-it#  base: $MODEL_DIR#" config.yaml
mkdir -p logs
W=runs/pitvis-notegen
trap 'aws s3 sync "$W/$OUT" "$RUNS_S3/$OUT" --only-show-errors || true; aws s3 sync logs "$RUNS_S3/logs" --only-show-errors || true' EXIT
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors
aws s3 sync "$RUNS_S3/$PILOT" "$W/$PILOT" --only-show-errors
CK=$(python -c "from notegen_rl.llm import find_latest_checkpoint as f;print(f('$W/$PILOT') or '')")
[[ -n "$CK" ]] || { echo "no checkpoint under $W/$PILOT"; exit 1; }
echo "pilot checkpoint: $CK"

echo "=== [review] full-procedure prompts for videos: $VIDEOS"
python - "$W/$DS" "$VIDEOS" <<'EOF'
import json, sys
from pathlib import Path
d, want = Path(sys.argv[1]), {int(v) for v in sys.argv[2].split()}
rows = {}
for f in ("grpo_train.jsonl", "grpo_val.jsonl", "grpo_eval.jsonl"):
    for r in map(json.loads, open(d / f)):
        if not r["window"] and r["video_id"] in want:
            rows[r["video_id"]] = r
missing = sorted(want - set(rows))
if missing:
    sys.exit(f"no full-procedure prompt for videos {missing}")
with open(d / "review_full.jsonl", "w") as f:
    for v in sorted(rows):
        f.write(json.dumps(rows[v]) + "\n")
print(f"wrote {len(rows)} prompts -> {d / 'review_full.jsonl'}")
EOF

echo "=== [review] pilot notes"
python scripts/06_evaluate.py --dataset review_full.jsonl --systems --adapter "pilot=$CK" --out-subdir "$OUT"
cat "$W/$OUT/summary.md"
