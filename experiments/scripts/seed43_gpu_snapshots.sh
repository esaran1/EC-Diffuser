#!/usr/bin/env bash
# Periodic GPU provenance snapshots during seed-43 training.
# PROVENANCE ONLY: this never kills or restarts anything.
TRAIN_PID=${1:?training pid required}
OUT=/home/jren313/EC-Diffuser-1/linux_logs/seed43_gpu_snapshots.log
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  ts=$(date -Iseconds)
  total=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  # One line per GPU process, tagged with whether it is our trainer.
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
    while IFS=, read -r pid mem; do
      pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
      [ -z "$pid" ] && continue
      cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-90)
      tag="other"; [ "$pid" = "$TRAIN_PID" ] && tag="SEED43"
      echo "$ts total=$total pid=$pid mem=${mem}MiB [$tag] $cl"
    done >> "$OUT"
  sleep 600   # every 10 minutes
done
echo "$(date -Iseconds) trainer $TRAIN_PID exited; snapshots stopped" >> "$OUT"
