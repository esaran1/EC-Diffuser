"""Run all Stage-E offline GPU diagnostics for both arms under ONE lease.

No training. No optimizer steps. Evaluates frozen checkpoints only.
Batches every diagnostic into a single GPU lease acquisition to avoid paying
the stable-clean wait repeatedly.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "loss_finetune", "mechanism")
sys.path.insert(0, MECH)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "diagnostics.log")
RUNDIR = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
CANON = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
         "state_400000.pt")

import os as _os
_EXT = _os.environ.get("UWM_EXT") == "1"
if _EXT:
    ARMS = {"uwm_ext": "data/panda_push/uwm_uwm_ext/" + RUNDIR,
            "ctl_ext": "data/panda_push/uwm_ctl_ext/" + RUNDIR}
else:
    ARMS = {"uwm": "data/panda_push/uwm_uwm/" + RUNDIR,
            "ctl": "data/panda_push/uwm_ctl/" + RUNDIR}

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)
D = "experiments/policy_improvement/tal_followup/uwm"


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main():
    log("=== STAGE-E OFFLINE DIAGNOSTICS (no training) ===")
    log("foreign GPU MiB: {}".format(foreign_mib()))
    for arm, sp in ARMS.items():
        if not os.path.exists(os.path.join(ROOT, sp, "state_final.pt")):
            log("MISSING final checkpoint for arm {} -- aborting".format(arm))
            return 1

    cmds = []
    for arm, sp in ARMS.items():
        cmds.append("{py} {d}/field_diagnostics.py --savepath {sp} "
                    "--epoch final --out {d}/field_{arm}.json --n_batch 32"
                    .format(py=PY, d=D, sp=sp, arm=arm))
        cmds.append("{py} {d}/replay_eval.py --savepath {sp} --epoch final "
                    "--out {d}/replay_{arm}.json --n_batch 64 --steps 4 "
                    "--canonical {canon}"
                    .format(py=PY, d=D, sp=sp, arm=arm, canon=CANON))
    if not _EXT:
        # adaptation-over-training: field at intermediate UWM checkpoints
        for ep in ("500", "1000", "2000"):
            cmds.append("{py} {d}/field_diagnostics.py --savepath {sp} "
                        "--epoch {ep} --out {d}/field_uwm_{ep}.json --n_batch 32"
                        .format(py=PY, d=D, sp=ARMS["uwm"], ep=ep))
        # registers diagnostic on the diagonal control
        cmds.append("{py} {d}/../registers/register_diagnostic.py --savepath {sp} "
                    "--epoch final --out {d}/../registers/register_ctl.json "
                    "--n_batch 32".format(py=PY, d=D, sp=ARMS["ctl"]))

    script = " && ".join(cmds)
    rc = run_guarded("diag:stageE_ext" if _EXT else "diag:stageE",
                     ENV + "(" + script + ") > linux_logs/tf_diagnostics.txt 2>&1",
                     log=log)
    log("diagnostics rc={}".format(rc))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
