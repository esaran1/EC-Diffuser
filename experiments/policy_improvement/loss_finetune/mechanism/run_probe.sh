#!/bin/bash
# SUPERSEDED -- DO NOT RUN DIRECTLY.
#
# This script launched GPU work on its own stable-clean check. With more than one
# such runner armed, two of OUR jobs could observe the same clean window and
# launch simultaneously, contaminating each other. All GPU work in this program
# now goes through the single orchestrator (orchestrate.py), which serializes
# every launch behind one program-wide lease (gpu_lease.py).
#
# Kept for provenance only. Refuses to run.
echo "SUPERSEDED: use orchestrate.py (single serialized GPU path). Refusing." >&2
exit 64
# Stage B probe runner: B then B', each launched only on a STABLE-clean GPU.
#
# The first attempt ran both probes while a foreign RLinf job was resident and
# had to be invalidated (CONTENTION_EVENT.md). This runner applies the same
# stable-clean condition the pinned scientific runners use: 10 consecutive clean
# probes over 180s, any foreign process resetting the streak. It never kills an
# unrelated job -- it waits.
set -u
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:${LD_LIBRARY_PATH-}"
export WANDB_MODE=offline
export WANDB_SILENT=true
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
M=experiments/policy_improvement/loss_finetune/mechanism
LOG=$M/probe_runner.log

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

echo "[$(date -Iseconds)] STAGE B probe start, commit $(git rev-parse --short HEAD)" >> "$LOG"

for CFG in config.pandapush_lossft_armB_seed42 config.pandapush_lossft_armBprime_seed42; do
  NAME="${CFG#config.}"
  if [ -f "$M/probe_runs/$NAME/PROBE_COMPLETE.json" ]; then
    echo "[$(date -Iseconds)] $NAME already complete" >> "$LOG"; continue
  fi
  wait_clean || exit 1
  echo "[$(date -Iseconds)] launching probe $NAME" >> "$LOG"
  $PY $M/bprime_probe.py --config "$CFG" > "linux_logs/probe_${NAME}.txt" 2>&1
  rc=$?
  echo "[$(date -Iseconds)] probe $NAME exited rc=$rc" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] probe $NAME FAILED/CONTENDED - STOPPING" >> "$LOG"; exit 1
  fi
done

echo "[$(date -Iseconds)] STAGE B PROBES COMPLETE" >> "$LOG"
