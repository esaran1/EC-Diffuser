#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; S=experiments/submission; L=$S/logs/chain1.log
$PY $S/run.py $S/plans/p1_nfe_curve_s42.json >> $L 2>&1
$PY $S/leased.py latency "$PY $S/latency.py > $S/logs/latency.txt 2>&1"
echo "[$(date +%FT%T)] latency rc=$?" >> $L
echo "[$(date +%FT%T)] === CHAIN1 FINISHED ===" >> $L
