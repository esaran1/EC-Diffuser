#!/usr/bin/env bash
# Gated launcher for the canonical Flow seed-44 replication.
#
# SAFETY DESIGN: this script performs every GPU check itself and then REPLACES
# itself with the training process via `exec`. It never spawns a dormant child
# capable of launching training, so killing this PID is sufficient and final.
# Before exec there is no training process; after exec this PID *is* training.

set -u

REPO=/home/jren313/EC-Diffuser-1
WORKTREE=/home/jren313/ecdiff-seed44-7506ce48
EXPECT_SHA=7506ce48cc5e0ccbaf8ae41be7f3b8acf4944ba7
LOCK=$REPO/linux_logs/seed44.lock
MANIFEST=$REPO/experiments/audit/seed44_prelaunch_manifest.json
NEED_CLEAR=8
INTERVAL=60
PROC_MEM_LIMIT=1000   # no single unrelated process may hold >= this (MiB)
TOTAL_MEM_LIMIT=2000  # total GPU memory must stay below this (MiB)
UTIL_LIMIT=5          # total GPU utilization must stay at or below this (%)

log() { echo "$(date -Iseconds) $*"; }

# Returns 0 when no MATERIAL competing GPU workload is present.
# A small resident CUDA context (e.g. a CPU-bound job holding ~274 MiB at 0%
# utilization) is permitted; a process holding >= PROC_MEM_LIMIT is not.
gpu_state_ok() {
  local biggest=0 total util pid mem
  total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  while IFS=, read -r pid mem; do
    pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' MiB')
    [ -z "$pid" ] && continue
    [ "$mem" -gt "$biggest" ] && biggest=$mem
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  GPU_TOTAL=$total; GPU_UTIL=$util; GPU_BIGGEST=$biggest
  [ "$biggest" -lt "$PROC_MEM_LIMIT" ] \
    && [ "$total" -lt "$TOTAL_MEM_LIMIT" ] \
    && [ "$util" -le "$UTIL_LIMIT" ]
}

# --- refuse to launch if seed-44 training already exists -------------------
# Match only real python trainers, not shell strings that merely mention the
# command. -f matches the full argv, so restrict to processes whose executable
# is python AND whose argv contains train.py and the seed flag.
existing=$(pgrep -f "^[^ ]*python[0-9.]* .*diffuser/scripts/train\.py" 2>/dev/null || true)
for pid in $existing; do
  [ "$pid" = "$$" ] && continue
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--seed 44"; then
    log "REFUSING: a seed-44 training process already exists: PID $pid"
    tr '\0' ' ' < "/proc/$pid/cmdline"; echo
    exit 3
  fi
done

# --- unique lock ----------------------------------------------------------
if [ -e "$LOCK" ]; then
  other=$(cat "$LOCK" 2>/dev/null || echo "?")
  if kill -0 "$other" 2>/dev/null; then
    log "REFUSING: lock held by live PID $other ($LOCK)"; exit 4
  fi
  log "stale lock from dead PID $other -- reclaiming"
fi
echo $$ > "$LOCK"
log "GATE_PID=$$ lock=$LOCK"

# Clean the lock on any pre-launch cancellation. `exec` replaces the process,
# so this trap does NOT fire once training starts (by design: the lock then
# marks a live training run).
cleanup() {
  log "gate cancelled -- killing pending sleep and removing lock"
  [ -n "${SLEEP_PID:-}" ] && kill "$SLEEP_PID" 2>/dev/null
  rm -f "$LOCK"
  exit 130
}
trap cleanup INT TERM HUP

# --- GPU idle gate --------------------------------------------------------
clear=0
while [ "$clear" -lt "$NEED_CLEAR" ]; do
  if gpu_state_ok; then clear=$((clear+1)); else clear=0; fi   # any violation resets
  log "biggest_proc=${GPU_BIGGEST}MiB total=${GPU_TOTAL}MiB util=${GPU_UTIL}% sustained_clear=$clear/$NEED_CLEAR"
  [ "$clear" -ge "$NEED_CLEAR" ] && break
  # Backgrounded sleep + wait: `wait` is interruptible by traps, whereas a
  # foreground `sleep` would defer TERM until it returned.
  sleep "$INTERVAL" & SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null
  SLEEP_PID=""
done

# --- one final state check immediately before launch ----------------------
if gpu_state_ok; then
  log "FINAL_CHECK PASS biggest_proc=${GPU_BIGGEST}MiB total=${GPU_TOTAL}MiB util=${GPU_UTIL}%"
else
  log "ABORT: material competing GPU workload at final check "\
      "(biggest_proc=${GPU_BIGGEST}MiB total=${GPU_TOTAL}MiB util=${GPU_UTIL}%)"
  rm -f "$LOCK"; exit 5
fi
# Snapshot every GPU process for the manifest.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$REPO/linux_logs/seed44_gpu_procs_at_launch.txt" 2>/dev/null || true

cd "$WORKTREE" || { log "ABORT: worktree missing"; rm -f "$LOCK"; exit 6; }
ACTUAL_SHA=$(git rev-parse HEAD)
if [ "$ACTUAL_SHA" != "$EXPECT_SHA" ]; then
  log "ABORT: worktree at $ACTUAL_SHA, expected $EXPECT_SHA"; rm -f "$LOCK"; exit 7
fi

# --- pre-launch manifest --------------------------------------------------
log "writing pre-launch manifest"
/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python \
    "$REPO/experiments/scripts/seed44_manifest.py" \
    --out "$MANIFEST" --gate-pid $$ \
    --gpu-total "$GPU_TOTAL" --gpu-util "$GPU_UTIL" --gpu-biggest "$GPU_BIGGEST" \
    || { log "ABORT: manifest failed"; rm -f "$LOCK"; exit 8; }

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    > "$REPO/linux_logs/seed44_gpu_at_launch.txt"
log "GPU_GENUINELY_IDLE -- exec'ing training, this PID becomes the trainer"
log "COMMIT=$ACTUAL_SHA"
log "START=$(date -Iseconds)"

source /home/jren313/miniconda3/etc/profile.d/conda.sh
conda activate ecdiffuser-linux
export PYTHONPATH="$PWD:$PWD/diffuser"
export WANDB_MODE=offline

# exec: replace this shell with training. No detached child is ever created.
exec python diffuser/scripts/train.py \
    --config config.pandapush_flow_single_gpu \
    --num_entity 3 \
    --rand_color \
    --seed 44
