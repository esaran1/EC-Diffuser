#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_primary.log
echo "[$(date +%FT%T)] primary closed loop (resume)" >> $LOG
$PY $TF/uwm/run_closed_loop_chunked.py --R 3 --max_attempts 120 \
  --jobs "uwm10k:uwm_uwm_ext/$RUN:final,ctl10k:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
echo "[$(date +%FT%T)] PRIMARY DONE rc=$?" >> $LOG
