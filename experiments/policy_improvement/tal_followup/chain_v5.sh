#!/bin/bash
# Order (per instruction): monitoring gate -> matched 10k control (exact resume)
# -> divergence + matched-semantics check -> ext diagnostics -> PRIMARY 10k
# closed loop -> direct action 5k -> 10k -> metrics -> direct-action closed loop.
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
TF=experiments/policy_improvement/tal_followup
LOG=$TF/chain_v5.log
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a $LOG; }
gate(){
  while true; do
    if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1 && $PY - <<'PYEOF' >/dev/null 2>&1
import sys; sys.path.insert(0,"diffuser"); sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from diffuser.utils import gpu_guard; from gpu_lease import foreign_mib
assert gpu_guard.monitoring_ok() and foreign_mib() != -1
PYEOF
    then return 0; fi
    log "monitoring unavailable/ambiguous -- launch forbidden; waiting"; sleep 300
  done
}
gate; log "=== CHAIN V5: monitoring verified ==="

log "--- 1: control 10k (exact-resume runner) ---"
$PY $TF/resume/train_resumable.py --config config.pandapush_uwm_ctl_seed42_ext \
    --prefix uwm_ctl_ext --tag ctl10k --reserve 2.0 >> $LOG 2>&1 || { log "HARD GATE: control 10k failed"; exit 2; }

log "--- 2: 10k pair integrity ---"
$PY - >> $LOG 2>&1 <<'PYEOF'
import hashlib, json, torch, importlib, sys
sys.path.insert(0,"diffuser"); sys.path.insert(0,"diffuser/config")
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"
def sha(p):
    h=hashlib.sha256(); f=open(p,"rb")
    for c in iter(lambda:f.read(1<<22),b""): h.update(c)
    return h.hexdigest()[:16]
P={"uwm10k":"data/panda_push/uwm_uwm_ext/%s/state_final.pt"%R,"ctl10k":"data/panda_push/uwm_ctl_ext/%s/state_final.pt"%R,
   "uwm5k":"data/panda_push/uwm_uwm/%s/state_final.pt"%R,"ctl5k":"data/panda_push/uwm_ctl/%s/state_final.pt"%R}
S={k:sha(v) for k,v in P.items()}; print("[integrity] sha", S)
assert len(set(S.values()))==4, "byte-identical checkpoints -- INVALID pair"
for k in ("uwm10k","ctl10k"):
    a=json.load(open(P[k].replace("state_final.pt","args.json")))
    print("[integrity] %s p_diag=%s ds=%s ws=%s"%(k,a.get("p_diag"),a.get("dataloader_seed"),a.get("warm_start_checkpoint").split("/")[2]))
A=importlib.import_module("config.pandapush_uwm_uwm_seed42_ext").base["diffusion"]
B=importlib.import_module("config.pandapush_uwm_ctl_seed42_ext").base["diffusion"]
d=sorted(k for k in set(A)|set(B) if A.get(k)!=B.get(k)); print("[integrity] config diff", d)
assert set(d)=={"p_diag","prefix","warm_start_checkpoint"}
print("[integrity] PASS")
PYEOF
[ $? -ne 0 ] && { log "HARD GATE: 10k pair integrity failed"; exit 3; }

log "--- 3: extended diagnostics ---"
gate; UWM_EXT=1 $PY $TF/uwm/run_diagnostics.py >> $LOG 2>&1; log "ext diag rc=$?"
$PY $TF/uwm/compare_5k_10k.py >> $LOG 2>&1

log "--- 4: PRIMARY closed loop R=3: uwm10k vs ctl10k, diagonal, 4 calls ---"
gate; $PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 80 \
  --jobs "uwm10k:uwm_uwm_ext/$RUN:final,ctl10k:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
log "primary closed loop rc=$?"

log "--- 5: direct action 5k then 10k ---"
$PY $TF/resume/train_resumable.py --config config.pandapush_qaction_seed42 \
    --prefix qaction --tag qaction5k --reserve 1.0 >> $LOG 2>&1 || { log "HARD GATE: qaction 5k"; exit 4; }
$PY $TF/resume/train_resumable.py --config config.pandapush_qaction_seed42_ext \
    --prefix qaction_ext --tag qaction10k --reserve 0.8 >> $LOG 2>&1 || { log "HARD GATE: qaction 10k"; exit 5; }

log "--- 6: direct-action offline metrics (qaction10k, ctl10k) ---"
$PY - >> $LOG 2>&1 <<'PYEOF'
import os, sys, time, subprocess
sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; TF="experiments/policy_improvement/tal_followup"
PY="/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
L=open(TF+"/direct_action/metrics.log","a")
def log(m): L.write("[%s] %s\n"%(time.strftime("%Y-%m-%dT%H:%M:%S"),m)); L.flush(); print(m, flush=True)
cmd=("cd /home/jren313/EC-Diffuser-1 && export PYTHONPATH=$PWD:$PWD/diffuser CUDA_VISIBLE_DEVICES=0 && "
     "{py} {tf}/direct_action/qaction_metrics.py --savepath data/panda_push/qaction_ext/{r} --out {tf}/direct_action/qaction10k_metrics.json && "
     "{py} {tf}/direct_action/qaction_metrics.py --savepath data/panda_push/uwm_ctl_ext/{r} --out {tf}/direct_action/ctl10k_metrics.json").format(py=PY,tf=TF,r=R)
log("rc=%s"%run_guarded("qaction:metrics", cmd+" > linux_logs/qaction_metrics.txt 2>&1", log=log))
PYEOF

log "--- 7: direct-action closed loop R=3 (ctl10k already evaluated in step 4) ---"
gate; $PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 50 \
  --jobs "qaction10k:qaction_ext/$RUN:final" >> $LOG 2>&1
log "direct-action closed loop rc=$?"
log "=== CHAIN V5 FINISHED ==="
