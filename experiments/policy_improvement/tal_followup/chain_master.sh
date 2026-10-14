#!/bin/bash
cd /home/jren313/EC-Diffuser-1; TF=experiments/policy_improvement/tal_followup
echo "[$(date +%FT%T)] master start (nfe2)" >> $TF/watchdog.log
bash $TF/chain_nfe2.sh
echo "[$(date +%FT%T)] master finished" >> $TF/watchdog.log
