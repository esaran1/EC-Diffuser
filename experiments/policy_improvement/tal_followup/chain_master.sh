#!/bin/bash
# Sequential, idempotent: completed chunks / checkpoints are skipped on rerun.
cd /home/jren313/EC-Diffuser-1; TF=experiments/policy_improvement/tal_followup
echo "[$(date +%FT%T)] master start" >> $TF/watchdog.log
bash $TF/chain_primary.sh
bash $TF/chain_v7.sh
echo "[$(date +%FT%T)] master finished" >> $TF/watchdog.log
