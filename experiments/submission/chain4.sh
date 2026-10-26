#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; S=experiments/submission; L=$S/logs/chain2.log
while pgrep -f "run.py experiments/submission/plans/p4_solver.json" >/dev/null; do sleep 60; done
echo "[$(date +%FT%T)] p4_solver finished (detached)" >> $L
for p in p4_solver p7_nfe16 p3_difficulty p5_maturity p6_fp16; do
  echo "[$(date +%FT%T)] --- $p ---" >> $L
  $PY $S/run.py $S/plans/$p.json >> $L 2>&1; rc=$?
  echo "[$(date +%FT%T)] $p rc=$rc" >> $L
  [ $rc -eq 2 ] && { echo "[$(date +%FT%T)] budget stop" >> $L; break; }
done
echo "[$(date +%FT%T)] === CHAIN2 FINISHED ===" >> $L
