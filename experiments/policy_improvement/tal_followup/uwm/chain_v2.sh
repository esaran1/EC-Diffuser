#!/bin/bash
# Contention-resilient plan. Training could not be chunked (dataloader state),
# but closed-loop evaluation CAN be: each E set is an independent evaluation of
# a FROZEN checkpoint with no trajectory state to preserve.
#
#   1. closed-loop R=3 on the COMPLETED 5k matched pair (per-E-set chunks)
#   2. direct-action training (one 19-min window needed; retried)
#   3. direct-action metrics + closed-loop R=3 (per-E-set chunks)
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
D=$TF/uwm
LOG=$D/chain_v2.log
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }

log "=== CHAIN V2: closed-loop first, on the completed 5k matched pair ==="

# ---- 1. THE DECISIVE COMPARISON: UWM 5k vs matched control 5k, diagonal path ----
log "--- step 1: closed-loop R=3, uwm5k vs ctl5k (diagonal, equal calls) ---"
$PY $D/run_closed_loop_chunked.py --R 3 \
  --jobs "uwm5k:uwm_uwm/$RUN:final,ctl5k:uwm_ctl/$RUN:final" >> $LOG 2>&1
log "closed-loop rc=$?"

# ---- 2. direct-action training (needs one clear ~19 min window; retry) ----
log "--- step 2: direct-action training ---"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if [ -f "data/panda_push/qaction/$RUN/state_final.pt" ]; then
    log "qaction already complete"; break
  fi
  if [ -d "data/panda_push/qaction/$RUN" ]; then
    ts=$(date +%H%M%S); mkdir -p data/panda_push/_contended_attempts
    mv data/panda_push/qaction data/panda_push/_contended_attempts/qaction_$ts 2>/dev/null \
      && log "quarantined partial qaction -> qaction_$ts"
  fi
  log "qaction training attempt $attempt"
  $PY $TF/direct_action/run_qaction.py >> $LOG 2>&1
  log "qaction attempt $attempt rc=$?"
  [ -f "data/panda_push/qaction/$RUN/state_final.pt" ] && { log "qaction COMPLETE"; break; }
  for _ in $(seq 1 60); do
    N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$N" -eq 0 ] && break
    sleep 60
  done
done

# ---- 3. direct-action metrics + closed-loop ----
if [ -f "data/panda_push/qaction/$RUN/state_final.pt" ]; then
  log "--- step 3: direct-action metrics + closed-loop R=3 ---"
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/qaction/$RUN \
      --epoch final --out $TF/direct_action/qaction_metrics.json >> $LOG 2>&1
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/uwm_ctl/$RUN \
      --epoch final --out $TF/direct_action/ctl5k_metrics.json >> $LOG 2>&1
  $PY $D/run_closed_loop_chunked.py --R 3 --jobs "qaction:qaction/$RUN:final" >> $LOG 2>&1
  log "qaction closed-loop rc=$?"
else
  log "qaction training never completed -- HARD GATE"
fi
log "=== CHAIN V2 FINISHED ==="
