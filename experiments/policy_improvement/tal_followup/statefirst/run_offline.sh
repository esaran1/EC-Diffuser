#!/bin/bash
cd /home/jren313/EC-Diffuser-1; export PYTHONPATH=$PWD:$PWD/diffuser
PY=/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python
for i in $(seq 1 60); do
$PY - <<'PYEOF'
import sys,time; sys.path.insert(0,"experiments/policy_improvement/loss_finetune/mechanism")
from gpu_lease import run_guarded
def log(m): open("experiments/policy_improvement/tal_followup/statefirst/lease.log","a").write("[%s] %s\n"%(time.strftime("%FT%T"),m))
sys.exit(run_guarded("statefirst:offline",("cd /home/jren313/EC-Diffuser-1 && export PYTHONPATH=$PWD:$PWD/diffuser CUDA_VISIBLE_DEVICES=0 && "
  "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python experiments/policy_improvement/tal_followup/statefirst/offline.py > experiments/policy_improvement/tal_followup/statefirst/offline.txt 2>&1; /home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python experiments/policy_improvement/tal_followup/profile/speedup.py > experiments/policy_improvement/tal_followup/profile/speedup.txt 2>&1"),log=log))
PYEOF
rc=$?; [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ] || break; sleep 120; done
echo "[$(date +%FT%T)] offline done rc=$rc" >> experiments/policy_improvement/tal_followup/statefirst/lease.log
