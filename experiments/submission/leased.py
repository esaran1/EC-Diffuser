"""Run one GPU command under the program lease, logged for phase spend. Retries
while the GPU is busy. Usage: leased.py STAGE "shell command" """
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../policy_improvement/loss_finetune/mechanism"))
sys.path.insert(0, HERE)
from gpu_lease import run_guarded  # noqa: E402
from run import ENV, log  # noqa: E402
stage, cmd = sys.argv[1], sys.argv[2]
for _ in range(120):
    rc = run_guarded(stage, ENV + cmd, log=log)
    if rc not in (3, 4, 5):
        sys.exit(rc)
    time.sleep(120)
sys.exit(rc)
