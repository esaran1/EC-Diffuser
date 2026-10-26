#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; S=experiments/submission; L=$S/logs/chain0b.log
$PY $S/run.py $S/plans/p0b_batch_invariance.json >> $L 2>&1
for N in 1 2 5; do $PY $S/leased.py gen:c$N "$PY experiments/policy_improvement/tal_followup/multicube/gen_sets.py $N >> $S/logs/gen_sets.txt 2>&1"; done
echo "[$(date +%FT%T)] sets: $(ls experiments/isaacgym_episode_sets/ | grep -c '^c[125]_')" >> $L
echo "[$(date +%FT%T)] === CHAIN0B FINISHED ===" >> $L
