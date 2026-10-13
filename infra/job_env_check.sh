#!/usr/bin/env bash
# Environment check for a new GPU instance made from the pipeline image (e.g. g7e.4xlarge, RTX PRO 6000
# Blackwell): GPU and driver, torch / vLLM / ms-swift on this GPU, the test suite, MedGemma with
# keyframes through vLLM, and two GRPO steps with keyframes (training path incl. sdpa attention).
#
#   aws s3 cp infra/job_env_check.sh s3://stanford-segment-clips/debjyoti/operative-notes/control-g7e/job.sh
set -euo pipefail
PROJ_S3=${PROJ_S3:-s3://stanford-segment-clips/debjyoti/operative-notes/pitvis-notegen-rl-keyframes}
RUNS_S3=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
OUT=${OUT:-grpo_env_check}
EVAL=${EVAL:-eval_env_check}
MODEL_S3=s3://stanford-segment-clips/debjyoti/operative-notes/models/medgemma-1.5-4b-it
MODEL_DIR=/opt/notegen/models/medgemma-1.5-4b-it
cd /opt/notegen
aws s3 sync "$PROJ_S3/" pitvis-notegen-rl/ --exclude "runs/*" --exclude "control/*" --only-show-errors
cd pitvis-notegen-rl
aws s3 sync "$MODEL_S3" "$MODEL_DIR" --only-show-errors
sed -i "s#^  base: google/medgemma-1.5-4b-it#  base: $MODEL_DIR#" config.yaml   # this instance's copy only
mkdir -p logs
W=runs/pitvis-notegen
trap 'aws s3 sync "$W/$EVAL" "$RUNS_S3/$EVAL" --only-show-errors || true' EXIT

echo "=== [env 0] GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== [env 1] torch / vLLM / ms-swift on this GPU"
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0),
      "| compiled for", torch.cuda.get_arch_list())
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("bf16 matmul ok:", float((a @ a).float().abs().mean()) > 0)
q = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
print("sdpa ok:", torch.nn.functional.scaled_dot_product_attention(q, q, q).shape)
import vllm, swift, transformers
print("vllm", vllm.__version__, "| ms-swift", swift.__version__, "| transformers", transformers.__version__)
EOF

echo "=== [env 2] tests"
python -m pytest -q tests

echo "=== [env 3] keyframe datasets"
DS=$(python -c "from notegen_rl.config import load_config, datasets_subdir; print(datasets_subdir(load_config()))")
echo "datasets -> $W/$DS"
aws s3 sync "$RUNS_S3/$DS" "$W/$DS" --only-show-errors || true
for d in predictions facts keyframes; do aws s3 sync "$RUNS_S3/$d" "$W/$d" --only-show-errors || true; done
if [[ ! -f $W/$DS/.built ]]; then
  python scripts/04_build_datasets.py && touch "$W/$DS/.built"
fi

echo "=== [env 4] untrained MedGemma with keyframes through vLLM (8 validation prompts)"
python scripts/06_evaluate.py --dataset grpo_val.jsonl --systems template base --limit 8 --out-subdir "$EVAL"

echo "=== [env 5] two GRPO steps with keyframes"
rm -rf "$W/$OUT"
python scripts/05_train.py --stage grpo --dataset grpo_train.jsonl --output-subdir "$OUT" \
  --set grpo.max_steps=2 grpo.num_generations=4 grpo.warmup_steps=1 grpo.save_steps=100 2>&1 | tee logs/env_check_train.log
python - "$W/$OUT" <<'EOF'
import json, sys
from pathlib import Path
logs = sorted(Path(sys.argv[1]).rglob("logging.jsonl"))
rows = [r for r in map(json.loads, open(logs[-1])) if r.get("reward") is not None] if logs else []
if not rows:
    sys.exit("no GRPO step logged")
for r in rows:
    print({k: r.get(k) for k in ("global_step/max_steps", "reward", "kl", "grad_norm", "completions/mean_length",
                                  "step_time", "memory(GiB)")})
EOF
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "=== environment check passed"
