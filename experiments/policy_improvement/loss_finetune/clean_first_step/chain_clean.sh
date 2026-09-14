#!/bin/bash
# Start the CLEAN program once the OLD program reaches ANY terminal state.
#
# Directive section 48: the clean experiment is scientifically separable, has its
# own 2.90 GPU-h budget, and is HIGHER PRIORITY. An old-allocation Stage-C
# failure -- including a hard budget stop -- must NOT strand it. So this waits
# for the close-out process to exit for any reason, then launches CLEAN.
#
# Exactly one GPU-launching process exists at any moment: we wait for
# finish_stage_c.py to be gone before exec'ing the clean orchestrator, and the
# global GPU lease backstops it regardless.
cd /home/jren313/EC-Diffuser-1
M=experiments/policy_improvement/loss_finetune/mechanism
C=experiments/policy_improvement/loss_finetune/clean_first_step
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python

# Wait for the old close-out to finish, whatever its outcome.
while pgrep -f "finish_stage_c\.py" >/dev/null; do sleep 30; done

if grep -q "STAGE C COMPLETE" $M/orchestrator.log 2>/dev/null; then
  STATUS="Stage C COMPLETE"
elif grep -qE "\[finish_stage_c\] STOP" $M/orchestrator.log 2>/dev/null; then
  STATUS="Stage C PARTIAL DUE HARD COMPUTE CAP"
else
  STATUS="Stage C ended without a completion marker (treated as PARTIAL)"
fi
echo "[$(date -Iseconds)] old allocation closed: ${STATUS} -> starting CLEAN program" >> $C/clean_orchestrator.log
exec $PY $C/orchestrate_clean.py
