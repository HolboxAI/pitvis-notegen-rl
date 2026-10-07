#!/usr/bin/env bash
# Runs the whole pipeline. Every stage is resumable / idempotent, so re-running after a
# failure picks up where it stopped (feature extraction skips finished videos).
#
#   bash run_pipeline.sh                      # all stages, config.yaml
#   CONFIG=my.yaml bash run_pipeline.sh
#   STAGES="3 4 5 6" bash run_pipeline.sh     # a subset (e.g. after changing facts settings)
#   NUM_GPUS=4 bash run_pipeline.sh           # shard feature extraction across 4 GPUs
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${CONFIG:-config.yaml}
STAGES=${STAGES:-"0 1 2 3 4 5 6"}
NUM_GPUS=${NUM_GPUS:-1}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

sft_enabled=$(python3 -c "import yaml;print(str(yaml.safe_load(open('$CONFIG'))['sft']['enabled']).lower())")

run() {  # run <log name> <command...>
  local name=$1; shift
  echo "=== [$name] $*"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

for s in $STAGES; do
  case $s in
    0) run 00_prepare python3 scripts/00_prepare.py --config "$CONFIG" ;;
    1)
      if (( NUM_GPUS > 1 )); then
        pids=()
        for ((i = 0; i < NUM_GPUS; i++)); do
          CUDA_VISIBLE_DEVICES=$i python3 scripts/01_extract_features.py --config "$CONFIG" \
            --shard "$i" --num-shards "$NUM_GPUS" > "$LOG_DIR/01_extract_features.$i.log" 2>&1 &
          pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        python3 scripts/01_extract_features.py --config "$CONFIG"   # verifies nothing is left
      else
        run 01_extract_features python3 scripts/01_extract_features.py --config "$CONFIG"
      fi ;;
    2) run 02_train_step_model python3 scripts/02_train_step_model.py --config "$CONFIG" ;;
    3) run 03_build_facts python3 scripts/03_build_facts.py --config "$CONFIG" ;;
    4) run 04_build_datasets python3 scripts/04_build_datasets.py --config "$CONFIG" ;;
    5)
      if [[ "$sft_enabled" == "true" ]]; then
        run 05_train_sft python3 scripts/05_train.py --config "$CONFIG" --stage sft
      fi
      run 05_train_grpo python3 scripts/05_train.py --config "$CONFIG" --stage grpo ;;
    6) run 06_evaluate python3 scripts/06_evaluate.py --config "$CONFIG" ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
