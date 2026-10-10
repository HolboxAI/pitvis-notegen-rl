#!/usr/bin/env bash
# One-shot status of the EC2 job: status file, instance state, recent log lines, run outputs.
#   bash infra/status.sh            # last 15 log lines
#   N=60 bash infra/status.sh       # more lines
export AWS_PROFILE=${AWS_PROFILE:-stanford-gpu} AWS_REGION=${AWS_REGION:-us-east-1}
C=s3://stanford-segment-clips/debjyoti/operative-notes/control
RUNS=s3://stanford-segment-clips/debjyoti/operative-notes/runs/pitvis-notegen
N=${N:-15}

echo "now:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "status:   $(aws s3 cp $C/status.txt - 2>/dev/null || echo 'none')"
aws ec2 describe-instances --filters Name=tag:Name,Values=pitvis-notegen-rl \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,InstanceLifecycle]' --output text \
  | sed 's/^/instance: /'
L=$(aws s3 ls $C/logs/ | sort | tail -1 | awk '{print $4}')
if [[ -n $L ]]; then
  echo "log:      $L (uploaded every 2 min while running)"
  log=$(aws s3 cp "$C/logs/$L" - | grep -v -E 'Downloading|━|^\s*$|FutureWarning|warnings.warn')
  echo "stages:   $(echo "$log" | grep -o '=== \[[0-9a-z_]*\]' | tr -d '=[] ' | tr '\n' ' ')"
  echo "--- last $N lines"
  echo "$log" | tail -"$N"
fi
echo "--- run outputs ($RUNS)"
aws s3 ls "$RUNS/" | awk '{print "  " $NF}'
# GRPO progress for GRPO_DIR (default grpo_v2; logs and checkpoints are pushed to S3 every 5 min)
G=${GRPO_DIR:-grpo_v2}
P=$(aws s3 ls "$RUNS/$G/" --recursive 2>/dev/null | grep 'logging.jsonl' | sort | tail -1 | awk '{print $4}')
if [[ -n $P ]]; then
  echo "--- GRPO ($G):"
  aws s3 cp "s3://stanford-segment-clips/$P" - 2>/dev/null | jq -r -s \
    'map(select(.reward != null and .["global_step/max_steps"] != null)) | last | "  step \(.["global_step/max_steps"]) | reward \(.reward) | grounding \(.["rewards/NoteGrounding/mean"]) | concise \(.["rewards/NoteConcise/mean"] // "-") | length \(.["completions/mean_length"]) | kl \(.kl) | grad \(.grad_norm) | elapsed \(.elapsed_time) | remaining \(.remaining_time)"' 2>/dev/null
  echo "  checkpoints in S3: $(aws s3 ls "$RUNS/$G/" --recursive | grep -o 'checkpoint-[0-9]*/optimizer.pt' | sed 's#/optimizer.pt##' | sort -t- -k2n | tr '\n' ' ')"
  aws s3 cp "$RUNS/$G/.watchdog_stop" - 2>/dev/null | sed 's/^/  WATCHDOG STOP: /'
  aws s3 cp "$RUNS/$G/best_checkpoint.json" - 2>/dev/null | jq -r '"  selected checkpoint: step \(.step) (validation weighted total \(.validation_weighted_total))"' 2>/dev/null
fi
E=${EVAL_DIR:-eval_v2}
aws s3 cp "$RUNS/$E/summary.md" - 2>/dev/null | sed "s/^/  [$E] /"
