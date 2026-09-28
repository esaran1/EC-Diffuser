#!/bin/bash
# REGISTERED ORDER (per instruction):
#   1. verify monitoring genuinely restored (fail closed if not)
#   2. FINISH THE MATCHED 10K CONTROL -- restart from the valid 5k parent
#   3. extended diagnostics on UWM 10k + control 10k
#   4. PRIMARY closed-loop R=3: UWM 10k vs CONTROL 10k, diagonal, 4 calls
#   5. direct-action experiment
#
# The 3 already-VALID chunks tagged tf_uwm5k_R1 are a SEPARATE 5k pilot
# artifact and are NOT merged into the 10k comparison. Different tags
# (tf_uwm10k_*, tf_ctl10k_*) guarantee this.
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
D=$TF/uwm
LOG=$D/chain_v4.log
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }

# ---------- 1. monitoring gate (fail closed, permanently) ----------
log "=== CHAIN V4: verifying monitoring before ANY GPU work ==="
while true; do
  if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    $PY - <<'PYEOF' >/dev/null 2>&1
import sys; sys.path.insert(0,"diffuser")
sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from diffuser.utils import gpu_guard
from gpu_lease import foreign_mib
assert gpu_guard.monitoring_ok(), "monitoring_ok() False"
assert foreign_mib() != -1, "foreign_mib() == -1 (blind)"
PYEOF
    [ $? -eq 0 ] && { log "monitoring VERIFIED: nvidia-smi rc=0, monitoring_ok, foreign_mib real"; break; }
  fi
  log "monitoring unavailable/ambiguous -- launch forbidden; waiting"
  sleep 300
done

# ---------- 2. matched 10k CONTROL (restart from the 5k parent) ----------
log "--- step 2: control extension 5k->10k (restart from valid 5k parent) ---"
for attempt in $(seq 1 12); do
  if [ -f "data/panda_push/uwm_ctl_ext/$RUN/state_final.pt" ]; then
    log "control 10k COMPLETE"; break
  fi
  if [ -d "data/panda_push/uwm_ctl_ext/$RUN" ]; then
    ts=$(date +%m%d_%H%M%S); mkdir -p data/panda_push/_contended_attempts
    mv data/panda_push/uwm_ctl_ext data/panda_push/_contended_attempts/ctl_ext_$ts 2>/dev/null \
      && log "quarantined partial attempt -> ctl_ext_$ts"
  fi
  log "control 10k attempt $attempt"
  UWM_EXT=1 $PY $D/run_pilot.py >> $LOG 2>&1
  log "attempt $attempt rc=$?"
  [ -f "data/panda_push/uwm_ctl_ext/$RUN/state_final.pt" ] && { log "control 10k COMPLETE on attempt $attempt"; break; }
  log "attempt $attempt incomplete; waiting for a clean GPU"
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$n" = "0" ] && break
    sleep 60
  done
done
if [ ! -f "data/panda_push/uwm_ctl_ext/$RUN/state_final.pt" ]; then
  log "HARD GATE: control 10k never completed. 10k comparison NOT runnable."; exit 2
fi

# ---------- 3. extended diagnostics on the matched 10k pair ----------
log "--- step 3: extended diagnostics (UWM 10k + control 10k) ---"
UWM_EXT=1 $PY $D/run_diagnostics.py >> $LOG 2>&1
log "ext diagnostics rc=$?"
$PY $D/compare_5k_10k.py >> $LOG 2>&1

# ---------- 4. PRIMARY closed-loop: UWM 10k vs CONTROL 10k ----------
log "--- step 4: PRIMARY closed-loop R=3, uwm10k vs ctl10k (diagonal, 4 calls) ---"
$PY $D/run_closed_loop_chunked.py --R 3 \
  --jobs "uwm10k:uwm_uwm_ext/$RUN:final,ctl10k:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
log "primary closed-loop rc=$?"

# ---------- 5. direct-action ----------
log "--- step 5: direct-action training ---"
for attempt in $(seq 1 12); do
  [ -f "data/panda_push/qaction/$RUN/state_final.pt" ] && { log "qaction COMPLETE"; break; }
  if [ -d "data/panda_push/qaction/$RUN" ]; then
    ts=$(date +%m%d_%H%M%S); mkdir -p data/panda_push/_contended_attempts
    mv data/panda_push/qaction data/panda_push/_contended_attempts/qaction_$ts 2>/dev/null \
      && log "quarantined partial qaction -> qaction_$ts"
  fi
  log "qaction attempt $attempt"
  $PY $TF/direct_action/run_qaction.py >> $LOG 2>&1
  [ -f "data/panda_push/qaction/$RUN/state_final.pt" ] && { log "qaction COMPLETE on attempt $attempt"; break; }
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$n" = "0" ] && break
    sleep 60
  done
done
if [ -f "data/panda_push/qaction/$RUN/state_final.pt" ]; then
  log "--- step 5b: direct-action metrics + closed-loop R=3 ---"
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/qaction/$RUN \
      --epoch final --out $TF/direct_action/qaction_metrics.json >> $LOG 2>&1
  $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/uwm_ctl_ext/$RUN \
      --epoch final --out $TF/direct_action/ctl10k_metrics.json >> $LOG 2>&1
  $PY $D/run_closed_loop_chunked.py --R 3 --jobs "qaction:qaction/$RUN:final" >> $LOG 2>&1
  log "qaction closed-loop rc=$?"
else
  log "qaction never completed -- HARD GATE"
fi
log "=== CHAIN V4 FINISHED ==="
