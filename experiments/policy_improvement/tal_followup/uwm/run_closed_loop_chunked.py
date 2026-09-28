"""Contention-resilient closed-loop R=3 runner.

Evaluates ONE E set per lease acquisition (~2.7 min) instead of five (~13 min),
merging results into the same diagnostics json. This fits inside the short
clear windows the foreign pipeline currently leaves.

This introduces NO scientific compromise: each E set is an independent
evaluation of a FROZEN checkpoint. There is no optimizer, no data-order, and
no trajectory state to preserve across chunks -- unlike training, which is why
training could not be chunked.

Per-set VALID/CONTENDED status is preserved by the evaluator, and control_mean
is computed from valid sets only.
"""
import argparse
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
LOG = os.path.join(HERE, "closed_loop.log")
P2 = os.path.join(ROOT, "experiments/policy_improvement/phase2")
SETS = ("E0", "E1s", "E2s", "E3s", "E4s")
CAP = 10.00
SET_COST = 0.2212 / 5.0

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def program_spent():
    """Cumulative program spend across ALL runner logs. See program_spend.py."""
    try:
        from program_spend import spent
        return spent()
    except Exception:
        f = os.path.join(HERE, "SPEND_FLOOR.txt")
        if os.path.exists(f):
            try:
                return float(open(f).read().strip())
            except ValueError:
                pass
        return 0.0


def set_status(tag, s):
    p = os.path.join(P2, "{}_diagnostics.json".format(tag))
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    return (d.get("control", {}) or {}).get(s, {}).get("status")


def all_valid(tag):
    return all(set_status(tag, s) == "VALID" for s in SETS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="name:loadpath[:epoch]")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--max_attempts", type=int, default=40)
    a = ap.parse_args()

    jobs = []
    for spec in a.jobs.split(","):
        parts = spec.split(":")
        jobs.append({"name": parts[0], "loadpath": parts[1],
                     "epoch": parts[2] if len(parts) > 2 else "final"})

    log("=== CHUNKED CLOSED-LOOP R={} ({} models, one E set per lease) ==="
        .format(a.R, len(jobs)))
    log("program spent: {:.4f} GPU-h".format(program_spent()))

    # interleave models within each (realization, set) so no model sits
    # entirely earlier or later in wall-clock time
    work = []
    for r in range(1, a.R + 1):
        for s in SETS:
            for j in jobs:
                work.append((j, r, s))

    attempts = 0
    while attempts < a.max_attempts:
        remaining = [(j, r, s) for j, r, s in work
                     if set_status("tf_{}_R{}".format(j["name"], r), s) != "VALID"]
        if not remaining:
            log("ALL SETS VALID for every model/realization")
            break
        spent = program_spent()
        if spent + SET_COST * len(remaining) > CAP:
            log("STOP: cap. spent {:.4f} + {} sets remaining > {:.2f}".format(
                spent, len(remaining), CAP))
            return 2
        j, r, s = remaining[0]
        tag = "tf_{}_R{}".format(j["name"], r)
        attempts += 1
        log("[{}/{}] {} set {}  ({} sets left)".format(
            attempts, a.max_attempts, tag, s, len(remaining)))
        rc = run_guarded(
            "cl:{}:{}".format(tag, s),
            ENV + ("WANDB_MODE=disabled WANDB_SILENT=true {py} "
                   "experiments/policy_improvement/phase2_diagnostics.py "
                   "--loadpath {lp} --epoch {ep} --tag {tag} --skip-offline "
                   "--only-sets {s} --merge --trace "
                   "> linux_logs/{tag}_{s}.txt 2>&1").format(
                       py=PY, lp=j["loadpath"], ep=j["epoch"], tag=tag, s=s),
            log=log)
        st = set_status(tag, s)
        log("  rc={} status={}".format(rc, st))
        if st != "VALID":
            log("  set {} not VALID (contention?); will retry after a clean GPU"
                .format(s))
            for _ in range(60):
                if foreign_mib() == 0:
                    break
                time.sleep(60)

    for j in jobs:
        for r in range(1, a.R + 1):
            tag = "tf_{}_R{}".format(j["name"], r)
            p = os.path.join(P2, "{}_diagnostics.json".format(tag))
            if os.path.exists(p):
                d = json.load(open(p))
                log("{}: n_valid_sets={} control_mean={}".format(
                    tag, d.get("n_valid_sets"), d.get("control_mean")))
    log("=== CHUNKED CLOSED-LOOP DONE: {:.4f} GPU-h ===".format(program_spent()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
