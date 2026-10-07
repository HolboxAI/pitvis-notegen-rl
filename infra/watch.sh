#!/usr/bin/env bash
# Block until the EC2 job reaches a milestone, then print it and exit:
#   job finished/failed | Python traceback | first video features saved | stage 1 stalled 25 min.
#   bash infra/watch.sh                       # follow the newest log
#   bash infra/watch.sh <old-log-name>        # wait for a NEWER boot than <old-log-name> first
export AWS_PROFILE=${AWS_PROFILE:-stanford-gpu}
C=s3://stanford-segment-clips/debjyoti/operative-notes/control
OLD=${1:-}
end=$(( $(date +%s) + 3 * 3600 ))
s1_start=""
while (( $(date +%s) < end )); do
  L=$(aws s3 ls $C/logs/ | sort | tail -1 | awk '{print $4}')
  if [[ -z $L || $L == "$OLD" ]]; then sleep 30; continue; fi
  status=$(aws s3 cp $C/status.txt - 2>/dev/null)
  [[ -z $status ]] && { sleep 30; continue; }   # transient S3/network error
  log=$(aws s3 cp "$C/logs/$L" - 2>/dev/null | grep -v -E 'Downloading|━|^\s*$|FutureWarning|warnings.warn')
  stages=$(echo "$log" | grep -o '=== \[[0-9a-z_]*\]' | tr '\n' ' ')
  if [[ $status != RUNNING* ]]; then echo "STATUS: $status"; echo "stages: $stages"; echo "$log" | tail -40; exit 0; fi
  if echo "$log" | grep -q 'Traceback'; then echo "TRACEBACK"; echo "stages: $stages"; echo "$log" | grep -A40 'Traceback' | tail -45; exit 0; fi
  if echo "$log" | grep -qE '^  video[0-9]+: [0-9]+ seconds'; then
    echo "STAGE 1 PROGRESSING ($L)"; echo "stages: $stages"
    echo "$log" | grep -E 'passed|instrument classes|source=|frames found|torch |GPU:|batches embedded|video[0-9]+:' | tail -14; exit 0; fi
  if [[ -z $s1_start ]] && echo "$log" | grep -q '=== \[01_extract_features\]'; then s1_start=$(date +%s); fi
  if [[ -n $s1_start ]] && (( $(date +%s) - s1_start > 1500 )); then
    echo "STAGE 1 SLOW: no video saved 25 min after stage 1 started"; echo "$log" | tail -25; exit 0; fi
  sleep 60
done
echo "WATCH TIMEOUT"; echo "$log" | tail -25
