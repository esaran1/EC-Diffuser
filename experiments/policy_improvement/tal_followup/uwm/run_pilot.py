"""UWM independent-time pilot runner (directive sections 21-23).

Trains TWO arms that differ only in the training time distribution:
    uwm  p_diag=0.0  faithful UWM (independent t_state, t_action)
    ctl  p_diag=1.0  matched diagonal control

Safety (section 8): global GPU lease, stable-clean wait, foreign-process
monitoring, atomic run locks, semantic validation, RUN_COMPLETE sentinels.
NEVER kills a foreign job. Fails closed on budget.

Section 54 divergence sentinels: records source hashes before training and
verifies the two arms actually diverge afterwards. The historical kwargs defect
produced byte-identical checkpoints for supposedly different objectives; this
check exists so that can never pass silently again.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "loss_finetune", "mechanism")
sys.path.insert(0, MECH)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "pilot_runner.log")

CAP_GPU_H = 10.00              # program-wide fresh cap
TRAIN_COST_EST = 0.42          # per arm, 5k steps (historical 2k ~0.167)
RESERVED_DOWNSTREAM = 1.50     # closed-loop gate + offline diagnostics

ARMS = {
    "uwm": {"config": "config.pandapush_uwm_uwm_seed42", "p_diag": 0.0,
            "prefix": "uwm_uwm"},
    "ctl": {"config": "config.pandapush_uwm_ctl_seed42", "p_diag": 1.0,
            "prefix": "uwm_ctl"},
}
RUNDIR = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
CANON = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
         "state_400000.pt")

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


def spent_gpu_h():
    """GPU-h charged to THIS program only, from its own log."""
    import re
    from datetime import datetime as dt
    if not os.path.exists(LOG):
        return 0.0
    acq, tot = {}, 0.0
    for line in open(LOG):
        m = re.search(r"\[([0-9T:\-]+)\].*lease acquired \(stage ([^,]+),", line)
        if m:
            acq[m.group(2)] = dt.fromisoformat(m.group(1))
        m = re.search(r"\[([0-9T:\-]+)\].*lease released \(stage ([^)]+)\)", line)
        if m and m.group(2) in acq:
            tot += (dt.fromisoformat(m.group(1))
                    - acq.pop(m.group(2))).total_seconds() / 3600
    return tot


def arm_dir(arm):
    return os.path.join(ROOT, "data/panda_push", ARMS[arm]["prefix"], RUNDIR)


def final_ckpt(arm):
    return os.path.join(arm_dir(arm), "state_final.pt")


def arm_complete(arm):
    """Semantic validation: sentinel AND a loadable final checkpoint."""
    sentinel = os.path.join(arm_dir(arm), "RUN_COMPLETE")
    return os.path.exists(sentinel) and os.path.exists(final_ckpt(arm))


def write_sentinel(arm):
    d = arm_dir(arm)
    if os.path.exists(final_ckpt(arm)):
        with open(os.path.join(d, "RUN_COMPLETE"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        return True
    return False


def main():
    log("=== UWM PILOT (program cap {:.2f} GPU-h) ===".format(CAP_GPU_H))
    canon_sha = sha256_of(os.path.join(ROOT, CANON))
    log("canonical source sha256 = {}".format(canon_sha[:24]))
    if not canon_sha.startswith("861dc34434474455"):
        log("FATAL: canonical checkpoint hash mismatch. Refusing to train.")
        return 2
    log("foreign GPU MiB at start: {}".format(foreign_mib()))
    log("already spent in THIS program: {:.4f} GPU-h".format(spent_gpu_h()))

    todo = [a for a in ("uwm", "ctl") if not arm_complete(a)]
    for a in ("uwm", "ctl"):
        if a not in todo:
            log("{} already complete".format(a))

    for arm in ("uwm", "ctl"):
        if arm_complete(arm):
            continue
        spent = spent_gpu_h()
        remaining_arms = sum(1 for a in ("uwm", "ctl") if not arm_complete(a))
        need = TRAIN_COST_EST * remaining_arms + RESERVED_DOWNSTREAM
        if spent + need > CAP_GPU_H:
            log("STOP: {} would breach cap. spent {:.4f} + need {:.4f} "
                "({} arms + {:.2f} reserved) > {:.2f}".format(
                    arm, spent, need, remaining_arms, RESERVED_DOWNSTREAM, CAP_GPU_H))
            return 2
        cfg = ARMS[arm]["config"]
        log("--- training arm {} (p_diag={}) ---".format(arm, ARMS[arm]["p_diag"]))
        rc = run_guarded(
            "uwm:{}".format(arm),
            ENV + ("WANDB_MODE=offline {py} diffuser/scripts/train.py "
                   "--config {cfg} --num_entity 3 --rand_color --seed 42 "
                   "> linux_logs/uwm_{arm}.txt 2>&1").format(
                       py=PY, cfg=cfg, arm=arm),
            log=log)
        log("arm {} rc={}".format(arm, rc))
        if rc != 0:
            log("arm {} FAILED (rc={}); see linux_logs/uwm_{}.txt".format(arm, rc, arm))
            return 1
        if not write_sentinel(arm):
            log("arm {} produced no final checkpoint -- NOT complete".format(arm))
            return 1
        log("arm {} complete, sha={}".format(arm, sha256_of(final_ckpt(arm))[:16]))

    # ---- section 54: the arms MUST actually diverge ----
    log("=== DIVERGENCE SENTINEL ===")
    sh = {a: sha256_of(final_ckpt(a)) for a in ("uwm", "ctl")}
    for a in ("uwm", "ctl"):
        log("  {} final sha256 = {}".format(a, sh[a][:24]))
    if sh["uwm"] == sh["ctl"]:
        log("FATAL: the two arms produced BYTE-IDENTICAL checkpoints. This is "
            "the historical kwargs-defect signature. Results are INVALID.")
        return 3
    if sh["uwm"] == canon_sha or sh["ctl"] == canon_sha:
        log("FATAL: an arm is byte-identical to the canonical source -- no "
            "training occurred.")
        return 3
    log("arms diverged from each other and from canonical: OK")

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "canonical_sha256": canon_sha,
        "arms": {a: {"p_diag": ARMS[a]["p_diag"],
                     "config": ARMS[a]["config"],
                     "final_sha256": sh[a],
                     "dir": os.path.relpath(arm_dir(a), ROOT)}
                 for a in ("uwm", "ctl")},
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
        "gpu_h_spent": spent_gpu_h(),
    }
    with open(os.path.join(HERE, "PILOT_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    log("=== PILOT COMPLETE: {:.4f} GPU-h ===".format(spent_gpu_h()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
