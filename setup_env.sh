#!/usr/bin/env bash
# One-time environment setup on the GPU server. Safe to re-run.
#   bash setup_env.sh            # installs into the active Python env
#   VENV=.venv bash setup_env.sh # creates/uses a virtualenv first
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${VENV:-}" ]]; then
  [[ -d "$VENV" ]] || python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || \
  echo "WARNING: nvidia-smi not found -- feature extraction and GRPO need a CUDA GPU"
command -v aws >/dev/null || echo "NOTE: aws CLI not found -- needed only for s3:// pitvis_root / results_uri"

# Triton (used by vLLM) JIT-compiles a C helper at first use and needs the Python headers.
if [[ ! -f "/usr/include/python3.$(python3 -c 'import sys;print(sys.version_info.minor)')/Python.h" ]]; then
  sudo apt-get update -qq && sudo apt-get install -y python3-dev
fi

python3 -m pip install -U pip
python3 -m pip install vllm
python3 -m pip install -r requirements.txt
python3 -m pip freeze > env_lock.txt

ENDOFM_DIR=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['endofm']['repo_dir'])")
ENDOFM_URL=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['endofm']['repo_url'])")
CKPT=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['endofm']['checkpoint'])")
GDRIVE_ID=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['endofm']['gdrive_id'])")
CKPT_S3=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['endofm'].get('checkpoint_s3') or '')")

if [[ ! -f "$ENDOFM_DIR/models/timesformer.py" ]]; then
  git clone --depth 1 "$ENDOFM_URL" "$ENDOFM_DIR"
fi
if [[ ! -f "$CKPT" ]]; then
  mkdir -p "$(dirname "$CKPT")"
  if [[ -n "$CKPT_S3" ]] && aws s3 cp "$CKPT_S3" "$CKPT" --only-show-errors; then
    echo "Endo-FM checkpoint copied from $CKPT_S3"
  else
    python3 -m gdown "$GDRIVE_ID" -O "$CKPT"
  fi
fi
ls -lh "$CKPT"

python3 -m pytest -q tests
echo "setup OK. Next: set data.pitvis_root (and llm.*) in config.yaml, then: bash run_pipeline.sh"
