#!/bin/bash
# Paired multi-seed replication of A vs B (terminal-action mask).
#
# The replication unit is a PAIRED A/B fine-tuning run. Within a replicate, A and
# B share everything -- canonical source checkpoint, initial model and EMA, fresh
# Adam semantics, fine-tuning RNG seed, dataloader seed, minibatch order, LR,
# warmup, 2000-step budget, EMA schedule, architecture, evaluator. The ONLY
# intended difference is terminal-action supervision (and the registered 0.207%
# active-coordinate renormalization it inherits).
#
# These are FINE-TUNING STOCHASTIC SEEDS, not independent pretrained policies:
# every replicate warm-starts from the SAME competent canonical checkpoint
# flow/3C_..._seed42/state_400000.pt. The pretrained flow/..._seed43 and
# flow/..._seed44 models exist in this repo and are deliberately NOT used.
#
# Predeclared, chosen before any results were seen:
#   replicate 1: finetune seed 43, dataloader seed 20260906, order B then A
#   replicate 2: finetune seed 44, dataloader seed 20260907, order A then B
#   replicate 3: finetune seed 45, dataloader seed 20260908, order B then A
# The balanced order avoids a systematic A-always-first temporal confound. The
# original replicate (seed 42, dataloader 20260905) keeps its historical order
# and is not re-run.
#
# ONE launch path. Advances only on a validated RUN_COMPLETE.json sentinel plus a
# verified final internal step. All pinned fail-loud safeguards retained.
set -u
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:${LD_LIBRARY_PATH-}"
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
RUNROOT=data/panda_push
LOG=experiments/policy_improvement/loss_finetune/replication/replication_runner.log
RERUN=experiments/policy_improvement/loss_finetune/rerun

# replicate : finetune_seed : dataloader_seed : arm order
REPLICATES=(
  "1:43:20260906:B A"
  "2:44:20260907:A B"
  "3:45:20260908:B A"
)

foreign_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=1000 '$2+0>=m {s+=$2} END {print s+0}'
}
# STABLE-clean: external jobs relaunch under new PIDs, so one clean reading can be
# a momentary gap rather than a free GPU. Require 10 consecutive clean probes over
# 180s; any foreign process resets the streak.
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

echo "[$(date -Iseconds)] REPLICATION start, commit $(git rev-parse --short HEAD)" >> "$LOG"

for SPEC in "${REPLICATES[@]}"; do
  R="${SPEC%%:*}"; REST="${SPEC#*:}"
  FSEED="${REST%%:*}"; REST="${REST#*:}"
  DLSEED="${REST%%:*}"; ORDER="${REST#*:}"
  SUB="3C_dlp_adalnpint_randcolor_H5_T4_seed${FSEED}"

  echo "[$(date -Iseconds)] === replicate $R: finetune_seed=$FSEED dataloader_seed=$DLSEED order='$ORDER' ===" >> "$LOG"

  # --- zero-GPU step-0 pairing precheck for this pair ------------------------
  $PY experiments/policy_improvement/loss_finetune/replication/verify_pair.py \
    --seed "$FSEED" --dataloader_seed "$DLSEED" --stage preflight >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date -Iseconds)] replicate $R PREFLIGHT PAIRING FAILED - STOPPING ALL" >> "$LOG"; exit 1
  fi

  for ARM in $ORDER; do
    D_ARM="$RUNROOT/lossft_arm${ARM}/$SUB"
    if [ -f "$D_ARM/RUN_COMPLETE.json" ]; then
      echo "[$(date -Iseconds)] r$R arm $ARM already complete (sentinel present)" >> "$LOG"
    else
      wait_clean || exit 1
      echo "[$(date -Iseconds)] r$R launching arm $ARM (seed $FSEED)" >> "$LOG"
      WANDB_MODE=offline $PY diffuser/scripts/train.py \
        --config config.pandapush_lossft_arm${ARM}_seed${FSEED} \
        --num_entity 3 --rand_color --seed "$FSEED" \
        > "linux_logs/lossft_r${R}_arm${ARM}_seed${FSEED}.txt" 2>&1
      rc=$?
      echo "[$(date -Iseconds)] r$R arm $ARM training exited rc=$rc" >> "$LOG"
      if [ ! -f "$D_ARM/RUN_COMPLETE.json" ]; then
        echo "[$(date -Iseconds)] r$R arm $ARM NO SENTINEL - STOPPING ALL" >> "$LOG"; exit 1
      fi
      ok=$($PY -c "import json;print(json.load(open('$D_ARM/RUN_COMPLETE.json'))['step_matches'])")
      if [ "$ok" != "True" ]; then
        echo "[$(date -Iseconds)] r$R arm $ARM step mismatch - STOPPING ALL" >> "$LOG"; exit 1
      fi
    fi

    # mandatory runtime objective manifests, step 0 and after 10 real steps
    for M in RUNTIME_OBJECTIVE_MANIFEST.json RUNTIME_OBJECTIVE_MANIFEST_after10.json; do
      if [ ! -f "$D_ARM/$M" ]; then
        echo "[$(date -Iseconds)] r$R arm $ARM MISSING $M - STOPPING ALL" >> "$LOG"; exit 1
      fi
      v=$($PY -c "import json;print(json.load(open('$D_ARM/$M'))['verdict'])")
      echo "[$(date -Iseconds)] r$R arm $ARM $M verdict=$v" >> "$LOG"
      if [ "$v" != "PASS" ]; then
        echo "[$(date -Iseconds)] r$R arm $ARM OBJECTIVE VERDICT $v - STOPPING ALL" >> "$LOG"; exit 1
      fi
    done

    # diagnostics + five-set control
    TAG="lossft_r${R}_${ARM}"
    if [ ! -f "experiments/policy_improvement/phase2/${TAG}_diagnostics.json" ]; then
      wait_clean || exit 1
      echo "[$(date -Iseconds)] r$R arm $ARM diagnostics+control" >> "$LOG"
      WANDB_MODE=disabled WANDB_SILENT=true $PY \
        experiments/policy_improvement/phase2_diagnostics.py \
        --loadpath lossft_arm${ARM}/$SUB --epoch final --tag "$TAG" \
        > "linux_logs/${TAG}_diag.txt" 2>&1
      echo "[$(date -Iseconds)] r$R arm $ARM diagnostics exited rc=$?" >> "$LOG"
    fi
  done

  # --- post-pair verification: A and B must have diverged, and only as intended
  $PY experiments/policy_improvement/loss_finetune/replication/verify_pair.py \
    --seed "$FSEED" --dataloader_seed "$DLSEED" --stage postrun >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date -Iseconds)] replicate $R POSTRUN PAIRING FAILED - STOPPING ALL" >> "$LOG"; exit 1
  fi
  echo "[$(date -Iseconds)] === replicate $R COMPLETE ===" >> "$LOG"
done

echo "[$(date -Iseconds)] REPLICATION COMPLETE" >> "$LOG"
