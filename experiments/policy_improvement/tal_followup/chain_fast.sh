#!/bin/bash
# Pre-declared: control NFE4 batch96, R1-R2, paired vs tf_ctl10k_b96 (77.92%).
# Step 1 SDPA alone (exact, 1.19x). Step 2 SDPA+FP16 (2.6x). Keep a variant if
# paired delta CI lower bound > -3pp (TOST-style) and no set INVALID.
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_fast.log
while pgrep -f "^bash $TF/chain_sf.sh" >/dev/null; do sleep 60; done
for V in sdpa sdpa_fp16; do
  echo "[$(date +%FT%T)] control NFE4 batch96 fast=$V R1-R2" >> $LOG
  $PY $TF/uwm/run_closed_loop_chunked.py --num_envs 96 --max_attempts 40 --R 2 --nfe 4 --fast $V \
     --jobs "ctl10k_b96${V/_/}:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
done
echo "[$(date +%FT%T)] === CHAIN FAST FINISHED ===" >> $LOG
