#!/bin/bash
# Stage C: one additional clean control evaluation (R=2) for all eight A/B policies.
#
# Same frozen evaluator, same checkpoints, same NFE, same success criterion as the
# accepted originals -- only the simulator realization differs. Offline diagnostics
# are skipped (--skip-offline): they are deterministic and already recorded, so a
# repeat would add nothing while costing GPU.
#
# Predeclared balanced order (applies to the NEW repeat only; historical run order
# is not rewritten):
#   seed42: A then B
#   seed43: B then A
#   seed44: A then B
#   seed45: B then A
# A and B never run concurrently.
set -u
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:${LD_LIBRARY_PATH-}"
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
M=experiments/policy_improvement/loss_finetune/mechanism
LOG=$M/repeat_eval_runner.log
P=experiments/policy_improvement/phase2

foreign_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=1000 '$2+0>=m {s+=$2} END {print s+0}'
}
STABLE_PROBES=10
STABLE_INTERVAL=18
wait_clean() {
  local streak=0
  for i in $(seq 1 1200); do
    if [ "$(foreign_mib)" -eq 0 ]; then
      streak=$((streak+1))
      if [ "$streak" -ge "$STABLE_PROBES" ]; then
        echo "[$(date -Iseconds)] GPU stable-clean for $((STABLE_PROBES*STABLE_INTERVAL))s" >> "$LOG"
        return 0
      fi
      sleep "$STABLE_INTERVAL"
    else
      [ "$streak" -gt 0 ] && echo "[$(date -Iseconds)] clean streak broken at $streak (foreign $(foreign_mib) MiB)" >> "$LOG"
      streak=0
      sleep 60
    fi
  done
  echo "[$(date -Iseconds)] GPU never became stable-clean" >> "$LOG"; return 1
}

# seed : arm order  (predeclared, balanced)
PLAN=("42:A B" "43:B A" "44:A B" "45:B A")

echo "[$(date -Iseconds)] STAGE C repeat evaluation start, commit $(git rev-parse --short HEAD)" >> "$LOG"

for SPEC in "${PLAN[@]}"; do
  SEED="${SPEC%%:*}"; ORDER="${SPEC#*:}"
  SUB="3C_dlp_adalnpint_randcolor_H5_T4_seed${SEED}"
  for ARM in $ORDER; do
    TAG="lossft_rep2_${ARM}${SEED}"
    if [ -f "$P/${TAG}_diagnostics.json" ]; then
      echo "[$(date -Iseconds)] $TAG already complete" >> "$LOG"; continue
    fi
    wait_clean || exit 1
    echo "[$(date -Iseconds)] repeat eval $TAG (arm $ARM seed $SEED)" >> "$LOG"
    WANDB_MODE=disabled WANDB_SILENT=true $PY \
      experiments/policy_improvement/phase2_diagnostics.py \
      --loadpath "lossft_arm${ARM}/$SUB" --epoch final --tag "$TAG" --skip-offline \
      > "linux_logs/${TAG}.txt" 2>&1
    rc=$?
    echo "[$(date -Iseconds)] $TAG exited rc=$rc" >> "$LOG"
    if [ "$rc" -ne 0 ]; then
      echo "[$(date -Iseconds)] $TAG FAILED - STOPPING" >> "$LOG"; exit 1
    fi
  done
done

echo "[$(date -Iseconds)] STAGE C REPEAT EVALUATION COMPLETE" >> "$LOG"
