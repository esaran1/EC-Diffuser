#!/bin/bash
# Wait for a genuinely clean GPU, then resume the registered chain.
# NEVER kills or preempts a foreign job -- it only waits.
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
LOG=experiments/policy_improvement/tal_followup/uwm/resume_chain.log

log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }

log "=== resumption armed; waiting for a clean GPU (foreign jobs are NEVER touched) ==="
# poll until no foreign compute app holds the GPU
while true; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$N" -eq 0 ]; then
    log "GPU shows 0 compute apps; handing off to the lease (which re-verifies stable-clean)"
    break
  fi
  sleep 120
done

# 1. finish the control extension (the lease enforces its own stable-clean check)
UWM_EXT=1 $PY experiments/policy_improvement/tal_followup/uwm/run_pilot.py >> $LOG 2>&1
rc=$?; log "control extension rc=$rc"
if [ $rc -ne 0 ]; then log "STOPPING: extension did not complete"; exit $rc; fi

# 2. diagnostics on the extended arms
UWM_EXT=1 $PY experiments/policy_improvement/tal_followup/uwm/run_diagnostics.py >> $LOG 2>&1
log "ext diagnostics rc=$?"

log "=== resumption chain finished ==="
