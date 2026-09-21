#!/bin/bash
# Full registered chain, armed to run unattended once the GPU is clean.
# NEVER kills or preempts a foreign job -- it only waits.
#
#   1. control extension 5k->10k (RESTART from the 5k parent, not a resume)
#   2. extended-arm diagnostics (field + replay, both arms)
#   3. closed-loop R=3: UWM 10k vs matched control 10k, SAME diagonal path
#   4. direct-action training + its metrics
#   5. closed-loop R=3: direct-action
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
LOG=$TF/uwm/full_chain.log
D=$TF/uwm
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42

log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }

log "=== FULL CHAIN ARMED; waiting for a clean GPU (foreign jobs never touched) ==="
while true; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$N" -eq 0 ] && { log "GPU clear; proceeding (lease re-verifies stable-clean)"; break; }
  sleep 120
done

# ---- 1. control extension (restart), with retry across foreign contention ----
# The foreign workload (alloc.*) is intermittent and has stopped this arm twice.
# Retry when the GPU clears rather than abandoning after one contention. Each
# attempt RESTARTS from the 5k parent -- never a mid-stream resume.
log "--- step 1: control extension 5k->10k (restart from 5k parent) ---"
rc=1
for attempt in 1 2 3 4 5 6 7 8; do
  # quarantine any partial artifacts so each attempt starts clean
  PART=data/panda_push/uwm_ctl_ext/$RUN
  if [ -f "$PART/state_contention_"*.pt ] 2>/dev/null || [ -d "$PART" ]; then
    if [ ! -f "$PART/state_final.pt" ] && [ -d "$PART" ]; then
      ts=$(date +%H%M%S)
      mkdir -p data/panda_push/_contended_attempts
      mv data/panda_push/uwm_ctl_ext data/panda_push/_contended_attempts/ctl_ext_$ts 2>/dev/null         && log "quarantined partial attempt -> _contended_attempts/ctl_ext_$ts"
    fi
  fi
  log "control extension attempt $attempt"
  UWM_EXT=1 $PY $D/run_pilot.py >> $LOG 2>&1
  rc=$?
  log "control extension attempt $attempt rc=$rc"
  if [ -f "data/panda_push/uwm_ctl_ext/$RUN/state_final.pt" ]; then
    log "control extension COMPLETE on attempt $attempt"; rc=0; break
  fi
  log "attempt $attempt did not finish (likely foreign contention); waiting for a clean GPU"
  # wait for the GPU to clear before retrying; NEVER touch the foreign job
  for _ in $(seq 1 90); do
    N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$N" -eq 0 ] && break
    sleep 60
  done
done
[ $rc -ne 0 ] && { log "HARD STOP at step 1 after retries"; exit $rc; }

# ---- 2. extended-arm diagnostics ----
log "--- step 2: extended-arm diagnostics ---"
UWM_EXT=1 $PY $D/run_diagnostics.py >> $LOG 2>&1
log "ext diagnostics rc=$?"
$PY $D/compare_5k_10k.py >> $LOG 2>&1

# ---- 3. decisive closed-loop R=3 on the 10k matched pair, SAME diagonal path ----
log "--- step 3: closed-loop R=3, UWM 10k vs control 10k (diagonal path both) ---"
$PY $D/run_closed_loop.py --R 3 --jobs \
  "uwm10k:uwm_uwm_ext/$RUN:final,ctl10k:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
rc=$?; log "closed-loop rc=$rc"
[ $rc -ne 0 ] && { log "HARD STOP at step 3"; exit $rc; }

# ---- 4. direct-action training ----
log "--- step 4: direct-action training ---"
$PY $TF/direct_action/run_qaction.py >> $LOG 2>&1
rc=$?; log "qaction training rc=$rc"
[ $rc -ne 0 ] && { log "HARD STOP at step 4"; exit $rc; }

# ---- 5. direct-action metrics + closed-loop ----
log "--- step 5: direct-action metrics and closed-loop R=3 ---"
$PY $TF/direct_action/qaction_metrics.py \
  --savepath data/panda_push/qaction/$RUN --epoch final \
  --out $TF/direct_action/qaction_metrics.json >> $LOG 2>&1
$PY $TF/direct_action/qaction_metrics.py \
  --savepath data/panda_push/uwm_ctl/$RUN --epoch final \
  --out $TF/direct_action/ctl5k_metrics.json >> $LOG 2>&1
$PY $D/run_closed_loop.py --R 3 --jobs "qaction:qaction/$RUN:final" >> $LOG 2>&1
log "qaction closed-loop rc=$?"

log "=== FULL CHAIN FINISHED ==="
