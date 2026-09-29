#!/bin/bash
# Queued after chain v5. Phase 5 (baseline-only harder-task selection),
# Phase 6 (candidates on the harder task), Phases 8-9 (same-dim latent action).
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
LOG=$TF/chain_v6.log
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42
ENVS="cd $PWD && export PYTHONPATH=$PWD:$PWD/diffuser LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:\${LD_LIBRARY_PATH-} CUDA_VISIBLE_DEVICES=0 &&"
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }
gate(){ while ! { nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1 && $PY -c "
import sys; sys.path.insert(0,'diffuser'); sys.path.insert(0,'experiments/policy_improvement/loss_finetune/mechanism')
from diffuser.utils import gpu_guard; from gpu_lease import foreign_mib
assert gpu_guard.monitoring_ok() and foreign_mib() != -1" >/dev/null 2>&1; }; do log "monitoring unavailable -- waiting"; sleep 300; done; }
guarded(){ # stage cmd -> retries lease-level failures
  local st=$1; shift
  for i in $(seq 1 30); do
    $PY - "$st" "$*" <<'PYEOF' >> $LOG 2>&1
import sys, time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
L=open("experiments/policy_improvement/tal_followup/chain_v6_lease.log","a")
def log(m): L.write("[%s] %s\n"%(time.strftime("%Y-%m-%dT%H:%M:%S"),m)); L.flush(); print(m,flush=True)
sys.exit(run_guarded(sys.argv[1], sys.argv[2], log=log))
PYEOF
    rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || return $rc; sleep 120
  done; return 9; }

log "=== CHAIN V6 queued; waiting for chain v5 to finish ==="
while systemctl --user is-active --quiet ecd-chain-v5b; do sleep 120; done
log "chain v5 finished: $(tail -1 $TF/chain_v5.log)"
gate

# ---------- Phase 5: baseline-only harder-task selection ----------
log "--- P5: E0 scenario sets for 4/5/6 cubes ---"
for N in 4 5 6; do
  guarded "mc:gen:c$N" "$ENVS $PY $TF/multicube/gen_sets.py $N E0 >> linux_logs/mc_gen.txt 2>&1" || log "gen c$N rc=$?"
done
log "--- P5: CANONICAL baseline on 4/5/6 cubes (E0, 96 eps each) ---"
for N in 4 5 6; do
  $PY $TF/uwm/run_closed_loop_chunked.py --R 1 --num_entity $N --sets E0 --max_attempts 10 \
      --jobs "c${N}sel_canonical:flow/$RUN:400000" >> $LOG 2>&1
done
$PY - >> $LOG 2>&1 <<'PYEOF'
import json
R={}
for N in (4,5,6):
    d=json.load(open("experiments/policy_improvement/phase2/tf_c%dsel_canonical_R1_diagnostics.json"%N))
    c=d["control"]["E0"]; R[N]={"success":c.get("success"),"status":c.get("status"),"trace":c.get("trace")}
# rule fixed BEFORE any candidate runs: smallest N with 0.20 <= success <= 0.70
ok=[N for N in (4,5,6) if R[N]["status"]=="VALID" and 0.20<=R[N]["success"]<=0.70]
sel={"baseline":"canonical 861dc34434474455","per_N":R,"rule":"smallest N in [0.20,0.70]","chosen":ok[0] if ok else None}
json.dump(sel,open("experiments/policy_improvement/tal_followup/multicube/SELECTION.json","w"),indent=1)
print("[P5] selection:", {N:(R[N]["success"],R[N]["status"]) for N in R}, "-> chosen", sel["chosen"])
PYEOF
NSEL=$($PY -c "import json;print(json.load(open('$TF/multicube/SELECTION.json'))['chosen'])")
log "P5 chosen cube count: $NSEL"

# ---------- Phase 6: matched control + both candidates on the harder task ----------
if [ "$NSEL" != "None" ]; then
  guarded "mc:gen:c$NSEL:all" "$ENVS $PY $TF/multicube/gen_sets.py $NSEL >> linux_logs/mc_gen.txt 2>&1"
  jobs="c${NSEL}_ctl10k:uwm_ctl_ext/$RUN:final,c${NSEL}_uwm10k:uwm_uwm_ext/$RUN:final"
  [ -f data/panda_push/qaction_ext/$RUN/state_final.pt ] && jobs="$jobs,c${NSEL}_qaction10k:qaction_ext/$RUN:final"
  log "--- P6: R=3 on $NSEL cubes: $jobs ---"
  $PY $TF/uwm/run_closed_loop_chunked.py --R 3 --num_entity $NSEL --max_attempts 120 --jobs "$jobs" >> $LOG 2>&1
  log "P6 rc=$?"
else
  log "P5: no cube count in the usable window -- harder-task comparison skipped"
fi

# ---------- Phases 8-9: same-dim latent action (VAE z=3 passed its gate) ----------
log "--- P9: latent-action 5k -> 10k ---"
$PY $TF/resume/train_resumable.py --config config.pandapush_latact_seed42 --prefix latact \
    --tag latact5k --reserve 0.75 >> $LOG 2>&1 || { log "HARD GATE latact5k"; exit 4; }
$PY $TF/resume/train_resumable.py --config config.pandapush_latact_seed42_ext --prefix latact_ext \
    --tag latact10k --reserve 0.7 >> $LOG 2>&1 || { log "HARD GATE latact10k"; exit 5; }
guarded "latact:metrics" "$ENVS $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/latact_ext/$RUN --out $TF/action_latent/latact10k_metrics.json > linux_logs/latact_metrics.txt 2>&1"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 50 --jobs "latact10k:latact_ext/$RUN:final" >> $LOG 2>&1
log "=== CHAIN V6 FINISHED ==="
