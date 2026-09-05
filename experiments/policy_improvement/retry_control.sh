#!/bin/bash
# Retry the remaining control sets whenever the GPU is genuinely clean.
# Each attempt re-checks per-set; sets that start or become contended are
# aborted/marked invalid by the harness itself.
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:$LD_LIBRARY_PATH"
export WANDB_MODE=disabled WANDB_SILENT=true
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
LOG=experiments/policy_improvement/control_retry.log

remaining() {
  $PY - <<'P'
import json,os
f="experiments/policy_improvement/phase2/phase2_20k_diagnostics.json"
c=json.load(open(f)).get("control",{}) if os.path.exists(f) else {}
need=[k for k in ("E0","E1s","E2s","E3s","E4s")
      if c.get(k,{}).get("status")!="VALID"]
print(" ".join(need))
P
}
foreign_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=1000 '$2+0>=m {s+=$2} END {print s+0}'
}

for attempt in $(seq 1 60); do
  NEED=$(remaining)
  if [ -z "$NEED" ]; then
    echo "[$(date -Iseconds)] all five sets VALID" >> "$LOG"; break
  fi
  F=$(foreign_mib)
  if [ "${F:-0}" -eq 0 ]; then
    sleep 45
    if [ "$(foreign_mib)" -eq 0 ]; then
      echo "[$(date -Iseconds)] attempt $attempt: GPU clean, running: $NEED" >> "$LOG"
      $PY experiments/policy_improvement/phase2_diagnostics.py \
          --skip-offline --merge --only-sets $NEED >> "$LOG" 2>&1
      continue
    fi
  fi
  sleep 120
done
echo "[$(date -Iseconds)] retry loop finished. remaining: $(remaining)" >> "$LOG"
