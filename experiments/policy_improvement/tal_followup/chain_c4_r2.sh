#!/bin/bash
# FINAL experiment: 4-cube NFE100 realization 2 (R2), same checkpoint/sets/protocol.
# Per-set budget gate: spend + 0.25 * sets_remaining <= 17, else stop.
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; TF=experiments/policy_improvement/tal_followup
RUN=3C_dlp_adalnpint_randcolor_H5_T4_seed42; LOG=$TF/chain_c4.log
log(){ echo "[$(date +%FT%T)] $*" >> $LOG; }
log "--- c4 NFE100 R2 (final) ---"
SETS=(E0 E1s E2s E3s E4s)
for i in "${!SETS[@]}"; do
  S=$(cat $TF/uwm/SPEND_FLOOR.txt); left=$(( ${#SETS[@]} - i ))
  if ! python3 -c "import sys; sys.exit(0 if $S + 0.25*$left <= 17 else 1)"; then
    log "STOP budget: spend $S + 0.25*$left > 17"; break; fi
  $PY $TF/uwm/run_closed_loop_chunked.py --num_envs 96 --max_attempts 40 --num_entity 4 --fast sdpa_fp16 \
     --R 2 --nfe 100 --sets ${SETS[$i]} --jobs "c4ctl10k_n100:uwm_ctl_ext/$RUN:final" >> $LOG 2>&1
done
log "=== CHAIN C4 R2 FINISHED ==="
