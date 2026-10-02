#!/bin/bash
# cron: @reboot and */15. Relaunches the research chain ONLY if no chain/runner
# is already running. Never kills or preempts anything. GPU safety is unchanged:
# every GPU stage still goes through the global lease and the fail-closed guard,
# which wait while any foreign job is on the GPU or monitoring is unavailable.
cd /home/jren313/EC-Diffuser-1; TF=experiments/policy_improvement/tal_followup
export PATH=/usr/local/bin:/usr/bin:/bin
exec 9>$TF/.watchdog.lock; flock -n 9 || exit 0          # one watchdog at a time
[ -f $TF/CHAIN_DISABLED ] && exit 0                       # manual off-switch
RUNNING='^(/usr)?(/bin/)?bash [^ ]*(chain_master|chain_primary|chain_v7|chain_v5b|chain_v6)\.sh|^[^ ]*python[0-9.]* [^ ]*(run_closed_loop_chunked|train_resumable|phase2_diagnostics)\.py|^[^ ]*python[0-9.]* diffuser/scripts/train\.py'
if pgrep -u "$(id -u)" -f "$RUNNING" >/dev/null 2>&1; then
  exit 0                                                  # already running
fi
if grep -q "CHAIN V7 FINISHED" $TF/chain_v7.log 2>/dev/null; then exit 0; fi
echo "[$(date +%FT%T)] watchdog: nothing running -> launching chain_master" >> $TF/watchdog.log
setsid nohup bash $TF/chain_master.sh >> $TF/chain_master.out 2>&1 < /dev/null &
