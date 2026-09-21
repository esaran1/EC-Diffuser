#!/bin/bash
# Waits for the GPU driver state to be repaired (nvidia-smi answerable again),
# then resumes the registered chain. Touches nothing while blind.
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
D=$TF/uwm
LOG=$D/chain_v3.log
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }

log "=== CHAIN V3 ARMED: waiting for the nvidia driver/library mismatch to clear ==="
while true; do
  if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    log "nvidia-smi answerable again; contention monitoring restored"
    break
  fi
  sleep 300
done

# double-check through the guard itself before any GPU work
$PY - <<'PYEOF' >> $LOG 2>&1
import sys; sys.path.insert(0,"diffuser")
from diffuser.utils import gpu_guard
assert gpu_guard.monitoring_ok(), "monitoring still unavailable"
print("[chain_v3] gpu_guard.monitoring_ok() == True")
PYEOF
[ $? -ne 0 ] && { log "guard still reports blind; aborting"; exit 1; }

log "--- step 1: closed-loop R=3, uwm5k vs ctl5k (per-E-set chunks, resumes) ---"
$PY $D/run_closed_loop_chunked.py --R 3 \
  --jobs "uwm5k:uwm_uwm/$RUN:final,ctl5k:uwm_ctl/$RUN:final" >> $LOG 2>&1
log "closed-loop rc=$?"

log "--- step 2: direct-action training ---"
for attempt in $(seq 1 10); do
  [ -f "data/panda_push/qaction/$RUN/state_final.pt" ] && { log "qaction complete"; break; }
  if [ -d "data/panda_push/qaction/$RUN" ]; then
    ts=$(date +%H%M%S); mkdir -p data/panda_push/_contended_attempts
    mv data/panda_push/qaction data/panda_push/_contended_attempts/qaction_$ts 2>/dev/null \
      && log "quarantined partial qaction -> qaction_$ts"
  fi
  log "qaction attempt $attempt"
  $PY $TF/direct_action/run_qaction.py >> $LOG 2>&1
  [ -f "data/panda_push/qaction/$RUN/state_final.pt" ] && { log "qaction COMPLETE"; break; }
  for _ in $(seq 1 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || break
    sleep 60
  done
done

if [ -f "data/panda_push/qaction/$RUN/state_final.pt" ]; then
  log "--- step 3: direct-action metrics + closed-loop R=3 ---"
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/qaction/$RUN \
      --epoch final --out $TF/direct_action/qaction_metrics.json >> $LOG 2>&1
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/uwm_ctl/$RUN \
      --epoch final --out $TF/direct_action/ctl5k_metrics.json >> $LOG 2>&1
  $PY $D/run_closed_loop_chunked.py --R 3 --jobs "qaction:qaction/$RUN:final" >> $LOG 2>&1
  log "qaction closed-loop rc=$?"
else
  log "qaction never completed -- HARD GATE"
fi
log "=== CHAIN V3 FINISHED ==="
