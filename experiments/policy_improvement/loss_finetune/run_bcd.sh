#!/bin/bash
# Sequential B/C/D runner. ONE launch path. Advances only on a validated
# RUN_COMPLETE.json sentinel + verified final internal step, never on process
# name matching. Each arm starts independently from canonical.
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:$LD_LIBRARY_PATH"
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
RUNROOT=data/panda_push
SUB=3C_dlp_adalnpint_randcolor_H5_T4_seed42
LOG=experiments/policy_improvement/loss_finetune/bcd_runner.log

foreign_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=1000 '$2+0>=m {s+=$2} END {print s+0}'
}
wait_clean() {
  for i in $(seq 1 240); do
    [ "$(foreign_mib)" -eq 0 ] && { sleep 30; [ "$(foreign_mib)" -eq 0 ] && return 0; }
    sleep 60
  done
  echo "[$(date -Iseconds)] GPU never became clean" >> "$LOG"; return 1
}

for ARM in B C D; do
  D_ARM="$RUNROOT/lossft_arm${ARM}/$SUB"
  # sentinel-based skip: arm already complete?
  if [ -f "$D_ARM/RUN_COMPLETE.json" ]; then
    echo "[$(date -Iseconds)] arm $ARM already complete (sentinel present)" >> "$LOG"
  else
    wait_clean || exit 1
    echo "[$(date -Iseconds)] launching arm $ARM training" >> "$LOG"
    WANDB_MODE=offline $PY diffuser/scripts/train.py \
      --config config.pandapush_lossft_arm${ARM} --num_entity 3 --rand_color --seed 42 \
      > linux_logs/lossft_arm${ARM}.txt 2>&1
    rc=$?
    echo "[$(date -Iseconds)] arm $ARM training exited rc=$rc" >> "$LOG"
    if [ ! -f "$D_ARM/RUN_COMPLETE.json" ]; then
      echo "[$(date -Iseconds)] arm $ARM NO SENTINEL - stopping" >> "$LOG"; exit 1
    fi
    ok=$($PY -c "import json;print(json.load(open('$D_ARM/RUN_COMPLETE.json'))['step_matches'])")
    if [ "$ok" != "True" ]; then
      echo "[$(date -Iseconds)] arm $ARM step mismatch - stopping" >> "$LOG"; exit 1
    fi
  fi
  # --- SECTION 16: early scientific divergence check -------------------------
  # From the ACTUAL production runs, not a preflight: once two arms exist, their
  # first periodic checkpoint (step 500) must no longer be byte-identical. If
  # different runtime objectives still yield identical weights, the objective is
  # not reaching optimization and there is no point burning the remaining arms.
  # This is an engineering validity check, not an adaptive scientific gate.
  $PY experiments/policy_improvement/loss_finetune/rerun/early_divergence_check.py \
    --runroot "$RUNROOT" --sub "$SUB" --arm "$ARM" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date -Iseconds)] arm $ARM EARLY DIVERGENCE CHECK FAILED - STOPPING ALL" >> "$LOG"
    exit 1
  fi

  # diagnostics + five-seed control
  if [ ! -f "experiments/policy_improvement/phase2/lossft_${ARM}_diagnostics.json" ]; then
    wait_clean || exit 1
    echo "[$(date -Iseconds)] arm $ARM diagnostics+control" >> "$LOG"
    WANDB_MODE=disabled WANDB_SILENT=true $PY \
      experiments/policy_improvement/phase2_diagnostics.py \
      --loadpath lossft_arm${ARM}/$SUB --epoch final --tag lossft_${ARM} \
      > linux_logs/lossft_${ARM}_diag.txt 2>&1
    echo "[$(date -Iseconds)] arm $ARM diagnostics exited rc=$?" >> "$LOG"
  fi
done
echo "[$(date -Iseconds)] B/C/D COMPLETE" >> "$LOG"
