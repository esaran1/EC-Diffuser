"""Closed-loop R=3 evaluation for the tal_followup program (sections 10, 29).

Evaluates frozen checkpoints only. NO training, NO optimizer steps. Uses the
frozen five fixed scenario sets and the production evaluator unchanged.

Fails closed on the program's 10.0 GPU-h cap.
"""
import argparse
import json
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
LOG = os.path.join(HERE, "closed_loop.log")
P2 = os.path.join(ROOT, "experiments/policy_improvement/phase2")
SETS = ("E0", "E1s", "E2s", "E3s", "E4s")

CAP_GPU_H = 15.00
EVAL_COST = 0.2212          # measured, one five-set evaluation

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def program_spent():
    """GPU-h across ALL logs of this program (training + evaluation)."""
    import re
    from datetime import datetime as dt
    total = 0.0
    for name in ("pilot_runner.log", "closed_loop.log", "qaction_runner.log"):
        p = os.path.join(HERE, name)
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


def eval_valid(tag):
    path = os.path.join(P2, "{}_diagnostics.json".format(tag))
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
    except Exception:
        return False
    if d.get("control_mean") is None or d.get("n_valid_sets") != len(SETS):
        return False
    c = d.get("control", {})
    return (all(c.get(s, {}).get("status") == "VALID" for s in SETS)
            and not any(c.get(s, {}).get("midset_max_foreign_mib") or 0
                        for s in SETS))


def quarantine(tag):
    p = os.path.join(P2, "{}_diagnostics.json".format(tag))
    if os.path.exists(p) and not eval_valid(tag):
        d = os.path.join(P2, "{}_diagnostics_INVALID_contended.json".format(tag))
        i = 1
        while os.path.exists(d):
            d = os.path.join(P2, "{}_diagnostics_INVALID_contended.{}.json".format(tag, i))
            i += 1
        os.replace(p, d)
        log("quarantined {}".format(os.path.basename(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True,
                    help="comma list of name:loadpath:epoch[:path]")
    ap.add_argument("--R", type=int, default=3)
    a = ap.parse_args()

    jobs = []
    for spec in a.jobs.split(","):
        parts = spec.split(":")
        jobs.append({"name": parts[0], "loadpath": parts[1],
                     "epoch": parts[2] if len(parts) > 2 else "final",
                     "path": parts[3] if len(parts) > 3 else None})

    log("=== CLOSED-LOOP R={} ({} models) ===".format(a.R, len(jobs)))
    log("program spent so far: {:.4f} GPU-h".format(program_spent()))
    log("foreign GPU MiB: {}".format(foreign_mib()))

    plan = [(j, r) for j in jobs for r in range(1, a.R + 1)]
    for j, r in plan:
        tag = "tf_{}_R{}".format(j["name"], r)
        if eval_valid(tag):
            log("{} already VALID".format(tag))
            continue
        quarantine(tag)
        remaining = sum(1 for jj, rr in plan
                        if not eval_valid("tf_{}_R{}".format(jj["name"], rr)))
        spent = program_spent()
        if spent + EVAL_COST * remaining > CAP_GPU_H:
            log("STOP: cap. spent {:.4f} + {} remaining evals x {:.4f} > {:.2f}"
                .format(spent, remaining, EVAL_COST, CAP_GPU_H))
            return 2
        extra = ""
        if j["path"]:
            extra = " --inference_path {}".format(j["path"])
        rc = run_guarded(
            "cl:{}".format(tag),
            ENV + ("WANDB_MODE=disabled WANDB_SILENT=true {py} "
                   "experiments/policy_improvement/phase2_diagnostics.py "
                   "--loadpath {lp} --epoch {ep} --tag {tag} --skip-offline{extra} "
                   "> linux_logs/{tag}.txt 2>&1").format(
                       py=PY, lp=j["loadpath"], ep=j["epoch"], tag=tag, extra=extra),
            log=log)
        log("{} rc={} valid={}".format(tag, rc, eval_valid(tag)))
        if rc != 0 or not eval_valid(tag):
            log("{} FAILED".format(tag))
            return 1

    log("=== CLOSED-LOOP COMPLETE: program total {:.4f} GPU-h ==="
        .format(program_spent()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
