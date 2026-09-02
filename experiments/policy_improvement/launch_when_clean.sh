#!/bin/bash
# Waits for an uncontended GPU, then launches the frozen 20k MeanFlow run.
# The in-process gpu_guard remains authoritative: if it disagrees at launch,
# training refuses to start and this script records that and exits.
cd /home/jren313/EC-Diffuser-1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/diffuser"
export LD_LIBRARY_PATH="/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:$LD_LIBRARY_PATH"
export WANDB_MODE=offline
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
MINMIB=1000
LOG=linux_logs/imf_viability_20k_final.txt
GATE=experiments/policy_improvement/launch_gate.log

foreign_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=$MINMIB '$2+0 >= m {s+=$2} END {print s+0}'
}

echo "[$(date -Iseconds)] waiting for uncontended GPU (threshold ${MINMIB} MiB)" >> "$GATE"
while true; do
  F=$(foreign_mib)
  if [ "${F:-0}" -eq 0 ]; then
    # require the GPU to stay clean for two consecutive checks 60s apart
    sleep 60
    F2=$(foreign_mib)
    if [ "${F2:-0}" -eq 0 ]; then break; fi
  fi
  sleep 60
done

{
  echo "=== LAUNCH GATE PASSED $(date -Iseconds) ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo "--- compute processes ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "--- commit ---"
  git rev-parse HEAD
} >> "$GATE" 2>&1

echo "[$(date -Iseconds)] launching 20k run" >> "$GATE"
$PY diffuser/scripts/train.py \
  --config config.pandapush_imf_viability --num_entity 3 --rand_color --seed 42 \
  > "$LOG" 2>&1
echo "[$(date -Iseconds)] training process exited rc=$?" >> "$GATE"
