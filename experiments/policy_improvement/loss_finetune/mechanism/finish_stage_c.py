"""Close the OLD allocation: finish Stage C only. Touches nothing else.

The general orchestrator re-derives every stage from scratch on restart, and
because the B' checkpoint directory had been cleaned up after its diagnostics
were already banked, a restart re-trained B' seed42 (twice, 0.2664 GPU-h wasted).
The B' scientific results were never at risk -- they live in the phase2
diagnostics files -- but the compute was spent for nothing.

This runner does one thing: complete the two outstanding Stage-C repeat
evaluations under the same program-wide GPU lease and the same validity rule.
It cannot start training, cannot touch B', and cannot start the clean program.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "orchestrator.log")
P2 = os.path.join(ROOT, "experiments/policy_improvement/phase2")
SUB_T = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
SETS = ("E0", "E1s", "E2s", "E3s", "E4s")

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)

REMAINING = [(45, "B"), (45, "A")]      # the only two still outstanding


def log(msg):
    line = "[{}] [finish_stage_c] {}".format(
        time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def eval_is_valid(tag):
    path = os.path.join(P2, "{}_diagnostics.json".format(tag))
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
    except Exception:
        return False
    if d.get("control_mean") is None or d.get("n_valid_sets") != len(SETS):
        return False
    return all(d.get("control", {}).get(s, {}).get("status") == "VALID"
               for s in SETS)


def quarantine(tag):
    path = os.path.join(P2, "{}_diagnostics.json".format(tag))
    if os.path.exists(path) and not eval_is_valid(tag):
        dest = os.path.join(P2, "{}_diagnostics_INVALID_contended.json".format(tag))
        i = 1
        while os.path.exists(dest):
            dest = os.path.join(
                P2, "{}_diagnostics_INVALID_contended.{}.json".format(tag, i))
            i += 1
        os.replace(path, dest)
        log("quarantined {} -> {}".format(os.path.basename(path),
                                          os.path.basename(dest)))


def main():
    log("START (old allocation close-out). foreign GPU MiB: {}".format(foreign_mib()))
    for seed, arm in REMAINING:
        tag = "lossft_rep2_{}{}".format(arm, seed)
        for attempt in range(1, 4):
            if eval_is_valid(tag):
                log("{} complete and VALID".format(tag))
                break
            quarantine(tag)
            rc = run_guarded(
                "stageC-close:{}".format(tag),
                ENV + ("WANDB_MODE=disabled WANDB_SILENT=true {py} "
                       "experiments/policy_improvement/phase2_diagnostics.py "
                       "--loadpath lossft_arm{arm}/{sub} --epoch final --tag {tag} "
                       "--skip-offline > linux_logs/{tag}.txt 2>&1").format(
                           py=PY, arm=arm, sub=SUB_T.format(seed), tag=tag),
                log=log)
            log("{} attempt {} rc={} valid={}".format(
                tag, attempt, rc, eval_is_valid(tag)))
            if rc != 0:
                return 1
        if not eval_is_valid(tag):
            log("{} NOT VALID after 3 attempts".format(tag))
            return 1
    log("STAGE C COMPLETE -- old allocation closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
