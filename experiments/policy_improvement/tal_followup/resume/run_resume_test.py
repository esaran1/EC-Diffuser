"""GPU exact-resume test: uninterrupted 40 steps (A) vs forced interrupt at
step 25 + exact resume (B). One lease for all three runs. Compares final
model / EMA / optimizer tensors bit-for-bit."""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "loss_finetune", "mechanism"))
from gpu_lease import run_guarded
ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "resume_test.log")
RUN = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
def log(m):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), m); print(line, flush=True)
    open(LOG, "a").write(line + "\n")
ENV = ("cd {r} && export PYTHONPATH={r}:{r}/diffuser && export LD_LIBRARY_PATH="
       "/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:${{LD_LIBRARY_PATH-}} && "
       "export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline && ").format(r=ROOT)
T = "{py} diffuser/scripts/train.py --config config.pandapush_resumetest_{a} --num_entity 3 --rand_color --seed 42"
cmd = ENV + " && ".join([
    "rm -rf data/panda_push/resumetest_A data/panda_push/resumetest_B",
    T.format(py=PY, a="A") + " > linux_logs/resumetest_A.txt 2>&1",
    "EC_TEST_FORCE_CONTENTION_STEP=25 " + T.format(py=PY, a="B") + " > linux_logs/resumetest_B1.txt 2>&1",
    "EC_RESUME_FROM=contention_25 " + T.format(py=PY, a="B") + " > linux_logs/resumetest_B2.txt 2>&1",
])
log("=== exact-resume GPU test ===")
rc = run_guarded("resume_test", cmd, log=log)
log("rc={}".format(rc))
sys.exit(rc)
