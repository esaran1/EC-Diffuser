#!/bin/bash
# 4-cube zero-shot (same 3-cube ctl10k checkpoint, no retraining), SDPA+FP16,
# batch 96, episode length 100. NFE1/NFE4 R1-R2 (paired), then NFE100 R1 only
# if spend + 1.0 <= 17. No second NFE100 realization.
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_c4.log
log(){ echo "[$(date +%FT%T)] $*" >> $LOG; }
log "--- generate c4 scenario sets (seeds as 3-cube) ---"
for i in $(seq 1 60); do $PY - >> $LOG 2>&1 <<'PYEOF'
import sys,time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
def log(m): open("experiments/policy_improvement/tal_followup/chain_c4.log","a").write("[%s] %s\n"%(time.strftime("%FT%T"),m))
sys.exit(run_guarded("gen_c4",("cd /home/jren313/EC-Diffuser-1 && export PYTHONPATH=$PWD:$PWD/diffuser LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib CUDA_VISIBLE_DEVICES=0 && "
 "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python experiments/policy_improvement/tal_followup/multicube/gen_sets.py 4 "
 ">> experiments/policy_improvement/tal_followup/multicube/gen_c4.txt 2>&1"),log=log))
PYEOF
rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || break; sleep 120; done
n=$(ls experiments/isaacgym_episode_sets/c4_replicate*_n96.pkl 2>/dev/null | wc -l)
[ "$n" -eq 5 ] || { log "ABORT: only $n/5 c4 sets"; exit 1; }
CL(){ $PY $TF/uwm/run_closed_loop_chunked.py --num_envs 96 --max_attempts 40 --num_entity 4 --fast sdpa_fp16 "$@" >> $LOG 2>&1; }
log "--- c4 NFE1 + NFE4 R1-R2 ---"
CL --R 2 --nfe 4 --jobs "c4ctl10k_n4:uwm_ctl_ext/$RUN:final"
CL --R 2 --nfe 1 --jobs "c4ctl10k_n1:uwm_ctl_ext/$RUN:final"
S=$(cat $TF/uwm/SPEND_FLOOR.txt)
if python3 -c "import sys; sys.exit(0 if $S + 1.0 <= 17 else 1)"; then
  log "--- c4 NFE100 R1 (spend $S) ---"
  CL --R 1 --nfe 100 --jobs "c4ctl10k_n100:uwm_ctl_ext/$RUN:final"
else log "SKIP NFE100: spend $S + 1.0 > 17"; fi
log "=== CHAIN C4 FINISHED ==="
