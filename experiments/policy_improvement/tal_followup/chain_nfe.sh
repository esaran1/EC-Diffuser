#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_nfe.log
log(){ echo "[$(date +%FT%T)] $*" | tee -a $LOG; }
guarded(){ local st=$1; shift; for i in $(seq 1 60); do $PY - "$st" "$*" >> $LOG 2>&1 <<'PYEOF'
import sys, time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
def log(m): open("experiments/policy_improvement/tal_followup/chain_nfe.log","a").write("[%s] %s\n"%(time.strftime("%Y-%m-%dT%H:%M:%S"),m))
sys.exit(run_guarded(sys.argv[1], sys.argv[2], log=log))
PYEOF
rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || return $rc; sleep 120; done; return 9; }
log "--- 1: validation, control 10k, batch 96, NFE 4, R=3 ---"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --num_envs 96 --nfe 4 --max_attempts 60 --jobs "ctl10k_b96:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
$PY $TF/nfe/decide.py validate >> $LOG 2>&1 || { log "VALIDATION REJECTED batch 96 -- stopping"; log "=== CHAIN NFE FINISHED ==="; exit 2; }
log "--- 2: batch-96 timing ---"
[ -f $TF/nfe/timing_b96.json ] || guarded "nfe:timing96" "cd $PWD && export PYTHONPATH=$PWD:$PWD/diffuser CUDA_VISIBLE_DEVICES=0 && $PY $TF/nfe/time_b96.py >> $LOG 2>&1"
$PY $TF/nfe/decide.py project >> $LOG 2>&1 || { log "PROJECTION EXCEEDS 15 GPU-h -- stopping before launch"; log "=== CHAIN NFE FINISHED ==="; exit 3; }
log "--- 3: NFE 100, R=3, control vs direct action, batch 96 ---"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --num_envs 96 --nfe 100 --max_attempts 80 \
   --jobs "ctl10k_b96n100:uwm_ctl_ext/$RUN:final,qaction10k_b96n100:qaction_ext/$RUN:final" >> $LOG 2>&1
log "--- 4: direct action NFE 4 at batch 96 (same-protocol gap) ---"
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --num_envs 96 --nfe 4 --max_attempts 40 --jobs "qaction10k_b96:qaction_ext/$RUN:final" >> $LOG 2>&1
log "=== CHAIN NFE FINISHED ==="
