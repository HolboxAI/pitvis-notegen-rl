#!/bin/bash
# EC2 user-data. Installs a per-boot runner: on EVERY boot the instance downloads
# <CTRL>/job.sh, runs it as `ubuntu`, streams its log to <CTRL>/logs/, writes
# <CTRL>/status.txt, and then shuts down (launch with instance-initiated-shutdown-behavior=stop
# so the EBS volume -- venv, data, features, checkpoints -- survives between jobs).
# No SSH or SSM needed: upload a new job.sh and start the instance to run the next stage.
set -u
cat > /var/lib/cloud/scripts/per-boot/10-notegen-job.sh <<'EOF'
#!/bin/bash
CTRL=s3://stanford-segment-clips/debjyoti/operative-notes/control
BOOT=$(date -u +%Y%m%dT%H%M%SZ)
LOG=/var/log/notegen-job-$BOOT.log
exec > >(tee -a "$LOG") 2>&1
shutdown -h +2160 "notegen watchdog: 36h limit"
mkdir -p /opt/notegen && chown ubuntu:ubuntu /opt/notegen
if ! aws s3 cp "$CTRL/job.sh" /opt/notegen/job.sh --only-show-errors; then
  echo "no job.sh at $CTRL -- shutting down"; shutdown -h now; exit 0
fi
chmod +x /opt/notegen/job.sh
echo "RUNNING since $BOOT on $(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-id)" \
  | aws s3 cp - "$CTRL/status.txt"
( while sleep 120; do aws s3 cp "$LOG" "$CTRL/logs/$BOOT.log" --only-show-errors; done ) &
UPLOADER=$!
sudo -iu ubuntu bash /opt/notegen/job.sh
RC=$?
kill $UPLOADER 2>/dev/null
[[ $RC -eq 0 ]] && S=SUCCEEDED || S="FAILED rc=$RC"
echo "$S boot=$BOOT finished=$(date -u +%Y%m%dT%H%M%SZ)" | aws s3 cp - "$CTRL/status.txt"
aws s3 cp "$LOG" "$CTRL/logs/$BOOT.log" --only-show-errors
shutdown -h now
EOF
chmod +x /var/lib/cloud/scripts/per-boot/10-notegen-job.sh
# per-boot scripts run before user-data on the first boot, so start it explicitly once
nohup /var/lib/cloud/scripts/per-boot/10-notegen-job.sh >/dev/null 2>&1 &
