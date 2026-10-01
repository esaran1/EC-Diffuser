#!/bin/bash
# After the primary run: (3) best intermediate state-leading path closed loop,
# (4) direct action, (5) action-to-action source. Then lower priority (v6 items).
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_v7.log
ENVS="cd $PWD && export PYTHONPATH=$PWD:$PWD/diffuser LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:\${LD_LIBRARY_PATH-} CUDA_VISIBLE_DEVICES=0 &&"
log(){ echo "[$(date +%FT%T)] $*" | tee -a $LOG; }
guarded(){ local st=$1; shift; for i in $(seq 1 30); do $PY - "$st" "$*" >> $LOG 2>&1 <<'PYEOF'
import sys, time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
L=open("experiments/policy_improvement/tal_followup/chain_v7_lease.log","a")
def log(m): L.write("[%s] %s\n"%(time.strftime("%Y-%m-%dT%H:%M:%S"),m)); L.flush(); print(m,flush=True)
sys.exit(run_guarded(sys.argv[1], sys.argv[2], log=log))
PYEOF
rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || return $rc; sleep 120; done; return 9; }
log "=== CHAIN V7 queued behind the primary closed loop ==="
while systemctl --user is-active --quiet ecd-primary; do sleep 60; done
log "--- 3: sweet-spot closed loop: UWM10k, path lead0.25 (same checkpoint, 4 calls) ---"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 60 \
   --jobs "uwm10k_lead025:uwm_uwm_ext/$RUN:final:lead0.25" >> $LOG 2>&1; log "sweet-spot CL rc=$?"
log "--- 4: direct action 5k -> 10k ---"
$PY $TF/resume/train_resumable.py --config config.pandapush_qaction_seed42 --prefix qaction --tag qaction5k --reserve 2.2 >> $LOG 2>&1 || { log "HARD GATE qaction5k"; exit 4; }
$PY $TF/resume/train_resumable.py --config config.pandapush_qaction_seed42_ext --prefix qaction_ext --tag qaction10k --reserve 1.9 >> $LOG 2>&1 || { log "HARD GATE qaction10k"; exit 5; }
guarded "qaction:metrics" "$ENVS $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/qaction_ext/$RUN --out $TF/direct_action/qaction10k_metrics.json > linux_logs/qaction_metrics.txt 2>&1 && $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/uwm_ctl_ext/$RUN --out $TF/direct_action/ctl10k_metrics.json >> linux_logs/qaction_metrics.txt 2>&1"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 60 --jobs "qaction10k:qaction_ext/$RUN:final" >> $LOG 2>&1; log "direct-action CL rc=$?"
log "--- 5: action-to-action source (prev action) 5k -> 10k ---"
$PY $TF/resume/train_resumable.py --config config.pandapush_prevact_seed42 --prefix prevact --tag prevact5k --reserve 1.0 >> $LOG 2>&1 || { log "HARD GATE prevact5k"; exit 6; }
$PY $TF/resume/train_resumable.py --config config.pandapush_prevact_seed42_ext --prefix prevact_ext --tag prevact10k --reserve 0.7 >> $LOG 2>&1 || { log "HARD GATE prevact10k"; exit 7; }
guarded "prevact:metrics" "$ENVS $PY $TF/direct_action/qaction_metrics.py --savepath data/panda_push/prevact_ext/$RUN --out $TF/direct_action/prevact10k_metrics.json > linux_logs/prevact_metrics.txt 2>&1"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 60 --jobs "prevact10k:prevact_ext/$RUN:final" >> $LOG 2>&1; log "prevact CL rc=$?"
log "=== CHAIN V7 FINISHED ==="
