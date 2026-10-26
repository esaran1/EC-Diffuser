#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser; PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; S=experiments/submission
$PY $S/leased.py test:policy_invariance "$PY $S/tests/test_policy_batch_invariance.py > $S/logs/test_policy_invariance.txt 2>&1"
echo "[$(date +%FT%T)] policy invariance rc=$?" >> $S/logs/chain0.log
$PY $S/run.py $S/plans/p0_batch_invariance.json >> $S/logs/chain0.log 2>&1
echo "[$(date +%FT%T)] === CHAIN0 FINISHED ===" >> $S/logs/chain0.log
