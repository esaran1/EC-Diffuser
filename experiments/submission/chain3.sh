#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib PYTHONPATH=/home/jren313/EC-Diffuser-1:/home/jren313/EC-Diffuser-1/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python; S=experiments/submission; L=$S/logs/chain3.log
while pgrep -f "^bash chain4.sh" >/dev/null; do sleep 60; done
$PY $S/leased.py latency_dlp "$PY $S/latency_dlp.py > $S/logs/latency_dlp.txt 2>&1"; echo "[$(date +%FT%T)] latency_dlp rc=$?" >> $L
