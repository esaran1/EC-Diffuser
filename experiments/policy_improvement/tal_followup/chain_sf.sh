#!/bin/bash
# Pre-declared: direct action NFE1 (= readout at t=0, chosen on REPLAY data only)
# vs existing control NFE4, batch 96, R1-R2. Plus control NFE1 (step sensitivity),
# then the 3 missing control NFE100 R1 sets. Speed benchmark first (tiny).
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_sf.log
log(){ echo "[$(date +%FT%T)] $*" | tee -a $LOG; }
for i in $(seq 1 60); do $PY - >> $LOG 2>&1 <<'PYEOF'
import sys,time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
def log(m): open("experiments/policy_improvement/tal_followup/chain_sf.log","a").write("[%s] %s\n"%(time.strftime("%FT%T"),m))
sys.exit(run_guarded("speedup",("cd /home/jren313/EC-Diffuser-1 && export PYTHONPATH=$PWD:$PWD/diffuser CUDA_VISIBLE_DEVICES=0 && "
 "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python experiments/policy_improvement/tal_followup/profile/speedup.py > experiments/policy_improvement/tal_followup/profile/speedup.txt 2>&1"),log=log))
PYEOF
rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || break; sleep 120; done
CL(){ $PY $TF/uwm/run_closed_loop_chunked.py --num_envs 96 --max_attempts 80 "$@" >> $LOG 2>&1; }
log "--- direct action NFE1 + control NFE1, batch96, R1-R2 ---"
CL --R 2 --nfe 1 --jobs "qaction10k_b96n1:qaction_ext/$RUN:final,ctl10k_b96n1:uwm_ctl_ext/$RUN:final"
log "--- control NFE100 R1 (remaining sets) ---"
CL --R 1 --nfe 100 --jobs "ctl10k_b96n100:uwm_ctl_ext/$RUN:final"
log "=== CHAIN SF FINISHED ==="
