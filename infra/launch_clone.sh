#!/usr/bin/env bash
# Launch a second pipeline instance from an image of the first one (models, venv and data included),
# with its own S3 control folder so the two instances never pick up each other's jobs.
#
#   bash infra/launch_clone.sh <ami-id> <instance-type> <control-folder> <name> [subnet ...]
#   e.g. bash infra/launch_clone.sh ami-0123 g7e.4xlarge control-g7e pitvis-notegen-rl-g7e subnet-a subnet-b
#
# The image carries the per-boot job runner from infra/user_data.sh, which reads <bucket>/control/job.sh.
# A cloud-config bootcmd (runs on every boot, before the per-boot scripts) points it at <control-folder>.
# Persistent Spot request, stop on interruption and on shutdown, like the first instance. Subnets are
# tried in order until one has capacity. Security group and instance profile are taken from the first
# instance (tag Name=$SOURCE_NAME) unless SG / PROFILE_ARN are set.
set -euo pipefail
AMI=$1; TYPE=$2; CTRL_NAME=$3; NAME=$4; shift 4
SUBNETS=("$@")
BASE=s3://stanford-segment-clips/debjyoti/operative-notes
AWS=(aws --profile "${AWS_PROFILE_NAME:-stanford-gpu}" --region us-east-1)
SOURCE_NAME=${SOURCE_NAME:-pitvis-notegen-rl}
if [[ -z "${SG:-}" || -z "${PROFILE_ARN:-}" ]]; then
  read -r SRC_SG SRC_PROFILE < <("${AWS[@]}" ec2 describe-instances --filters "Name=tag:Name,Values=$SOURCE_NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].[SecurityGroups[0].GroupId,IamInstanceProfile.Arn]' --output text)
  SG=${SG:-$SRC_SG}; PROFILE_ARN=${PROFILE_ARN:-$SRC_PROFILE}
fi
[[ -n "$SG" && "$SG" != None && -n "$PROFILE_ARN" && "$PROFILE_ARN" != None ]] || { echo "no security group / instance profile"; exit 1; }
UD=$(mktemp)
cat > "$UD" <<EOF
#cloud-config
bootcmd:
  - [sed, -i, "s#^CTRL=.*#CTRL=$BASE/$CTRL_NAME#", /var/lib/cloud/scripts/per-boot/10-notegen-job.sh]
EOF
for SUBNET in "${SUBNETS[@]}"; do
  echo "trying $TYPE in $SUBNET"
  if ID=$("${AWS[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
      --security-group-ids "$SG" --iam-instance-profile Arn="$PROFILE_ARN" \
      --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
      --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=owner,Value=debjyoti}]" \
      --query 'Instances[0].InstanceId' --output text); then
    echo "launched $ID ($TYPE, $SUBNET), control folder $BASE/$CTRL_NAME"
    rm -f "$UD"; exit 0
  fi
done
rm -f "$UD"; echo "no capacity for $TYPE in: ${SUBNETS[*]}"; exit 1
