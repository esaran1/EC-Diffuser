"""Stage C direct-action training runner.

One arm, warm-started from canonical, clean-first-step, matched to the UWM
pilot in every training respect. Fails closed on the program cap.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UWM = os.path.join(os.path.dirname(HERE), "uwm")
MECH = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "loss_finetune", "mechanism")
sys.path.insert(0, MECH)
sys.path.insert(0, UWM)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "qaction_runner.log")
RUNDIR = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
SAVE = os.path.join(ROOT, "data/panda_push/qaction", RUNDIR)
CANON = os.path.join(ROOT, "data/panda_push/flow", RUNDIR, "state_400000.pt")

CAP_GPU_H = 10.00
TRAIN_COST = 0.33          # measured from the UWM pilot arms
RESERVED = 0.70            # its own R=3 closed-loop evaluation

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def program_spent():
    import re
    from datetime import datetime as dt
    total = 0.0
    for p in (os.path.join(UWM, "pilot_runner.log"),
              os.path.join(UWM, "closed_loop.log"),
              os.path.join(UWM, "diagnostics.log"), LOG):
        if not os.path.exists(p):
            continue
        acq = {}
        for line in open(p):
            m = re.search(r"\[([0-9T:\-]+)\].*lease acquired \(stage ([^,]+),", line)
            if m:
                acq[m.group(2)] = dt.fromisoformat(m.group(1))
            m = re.search(r"\[([0-9T:\-]+)\].*lease released \(stage ([^)]+)\)", line)
            if m and m.group(2) in acq:
                total += (dt.fromisoformat(m.group(1))
                          - acq.pop(m.group(2))).total_seconds() / 3600
    return total


def main():
    log("=== STAGE C: direct-action / query-token training ===")
    log("foreign GPU MiB: {}".format(foreign_mib()))
    spent = program_spent()
    log("program spent so far: {:.4f} GPU-h".format(spent))
    if spent + TRAIN_COST + RESERVED > CAP_GPU_H:
        log("STOP: cap. {:.4f} + {:.2f} train + {:.2f} reserved > {:.2f}".format(
            spent, TRAIN_COST, RESERVED, CAP_GPU_H))
        return 2

    final = os.path.join(SAVE, "state_final.pt")
    if os.path.exists(final) and os.path.exists(os.path.join(SAVE, "RUN_COMPLETE")):
        log("already complete")
        return 0

    rc = run_guarded(
        "qaction:train",
        ENV + ("WANDB_MODE=offline {py} diffuser/scripts/train.py "
               "--config config.pandapush_qaction_seed42 "
               "--num_entity 3 --rand_color --seed 42 "
               "> linux_logs/qaction.txt 2>&1").format(py=PY),
        log=log)
    log("training rc={}".format(rc))
    if rc != 0 or not os.path.exists(final):
        log("FAILED")
        return 1
    with open(os.path.join(SAVE, "RUN_COMPLETE"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")

    sh = sha256_of(final)
    canon_sha = sha256_of(CANON)
    log("qaction final sha = {}".format(sh[:24]))
    if sh == canon_sha:
        log("FATAL: byte-identical to canonical -- no training occurred")
        return 3
    with open(os.path.join(HERE, "QACTION_MANIFEST.json"), "w") as fh:
        json.dump({"final_sha256": sh, "canonical_sha256": canon_sha,
                   "savepath": os.path.relpath(SAVE, ROOT),
                   "gpu_h": program_spent(),
                   "code_commit": subprocess.run(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()},
                  fh, indent=2, sort_keys=True)
    log("=== STAGE C COMPLETE: program total {:.4f} GPU-h ===".format(program_spent()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
