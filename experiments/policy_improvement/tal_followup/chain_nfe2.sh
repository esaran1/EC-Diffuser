#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_nfe2.log
log(){ echo "[$(date +%FT%T)] $*" | tee -a $LOG; }
CL(){ $PY $TF/uwm/run_closed_loop_chunked.py --num_envs 96 --max_attempts 80 "$@" >> $LOG 2>&1; }
log "--- 0: direct action NFE4 batch96 R1-R2 ---"
CL --R 2 --nfe 4 --jobs "qaction10k_b96:qaction_ext/$RUN:final"
log "--- 1: NFE100 R1 control + direct action ---"
CL --R 1 --nfe 100 --jobs "ctl10k_b96n100:uwm_ctl_ext/$RUN:final,qaction10k_b96n100:qaction_ext/$RUN:final"
$PY $TF/nfe/gap.py 1 r1check >> $LOG 2>&1; rc=$?
if [ $rc -eq 10 ]; then log "R1: gap exploded and consistent -- stopping (screening answer)"; log "=== CHAIN NFE2 FINISHED ==="; exit 0; fi
log "--- 2: NFE100 R2 ---"
CL --R 2 --nfe 100 --jobs "ctl10k_b96n100:uwm_ctl_ext/$RUN:final,qaction10k_b96n100:qaction_ext/$RUN:final"
$PY $TF/nfe/gap.py 2 >> $LOG 2>&1
log "=== CHAIN NFE2 FINISHED ==="
