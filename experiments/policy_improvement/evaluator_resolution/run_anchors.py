"""Evaluator-resolution anchor study: 6 new realizations x 2 fixed checkpoints.

NO TRAINING. This script can only invoke the frozen control evaluator on
already-existing checkpoints; it has no path to an optimizer step. It fails
closed on budget, contention, or provenance.

Anchors were chosen BEFORE any new repeat outcome was observed:
  canonical  -- the stable pre-fine-tuning reference policy
  cleanA42   -- the predeclared clean baseline from the corrected protocol
Neither was selected for having an unusual treatment effect.

Predeclared interleaved order (never reordered on outcomes), so neither anchor
sits entirely earlier or later in wall-clock time:
  canonical R1, cleanA42 R1, cleanA42 R2, canonical R2,
  canonical R3, cleanA42 R3, cleanA42 R4, canonical R4,
  canonical R5, cleanA42 R5, cleanA42 R6, canonical R6

Own ledger, hard cap 3.00 GPU-h, no borrowing from any other program.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(os.path.dirname(HERE), "loss_finetune", "mechanism")
sys.path.insert(0, MECH)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "anchor_runner.log")
P2 = os.path.join(ROOT, "experiments/policy_improvement/phase2")
SETS = ("E0", "E1s", "E2s", "E3s", "E4s")

CAP_GPU_H = 3.00
EVAL_COST_EST = 0.2225          # measured mean of a five-set control evaluation

ANCHORS = {
    "canonical": {
        "loadpath": "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
        "epoch": "400000",
        "checkpoint": ("data/panda_push/flow/"
                       "3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"),
    },
    "cleanA42": {
        "loadpath": "lossft_armAclean/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
        "epoch": "final",
        "checkpoint": ("data/panda_push/lossft_armAclean/"
                       "3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_final.pt"),
    },
}

PLAN = [("canonical", 1), ("cleanA42", 1), ("cleanA42", 2), ("canonical", 2),
        ("canonical", 3), ("cleanA42", 3), ("cleanA42", 4), ("canonical", 4),
        ("canonical", 5), ("cleanA42", 5), ("cleanA42", 6), ("canonical", 6)]

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


def tag_for(anchor, r):
    return "evalres_{}_R{}".format(anchor, r)


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
    ctrl = d.get("control", {})
    return (all(ctrl.get(s, {}).get("status") == "VALID" for s in SETS)
            and not any(ctrl.get(s, {}).get("midset_max_foreign_mib") or 0
                        for s in SETS))


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


def write_manifest():
    man = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "evaluator_script": "experiments/policy_improvement/phase2_diagnostics.py",
           "code_commit": subprocess.run(
               ["git", "rev-parse", "HEAD"], cwd=ROOT,
               capture_output=True, text=True).stdout.strip(),
           "nfe": 4, "solver": "fixed-step forward Euler (ConditionalFlowMatching)",
           "weights_evaluated": "EMA",
           "success_criterion": "max_obj_dist <= 0.04",
           "episodes_per_set": 96,
           "scenario_seeds": {"E0": 20260820, "E1s": 20261820, "E2s": 20262820,
                              "E3s": 20263820, "E4s": 20264820},
           "policy_noise_seed_constant": 20260902,
           "anchors": {}}
    for name, a in ANCHORS.items():
        p = os.path.join(ROOT, a["checkpoint"])
        if not os.path.exists(p):
            raise SystemExit("anchor checkpoint missing: {}".format(p))
        import torch
        ck = torch.load(p, map_location="cpu")
        man["anchors"][name] = {
            "path": a["checkpoint"],
            "sha256": sha256_of(p),
            "checkpoint_step": int(ck["step"]),
            "loadpath": a["loadpath"],
            "epoch_arg": a["epoch"],
            "selection_rationale": (
                "stable pre-fine-tuning reference" if name == "canonical"
                else "predeclared clean baseline from the corrected protocol"),
        }
        del ck
    with open(os.path.join(HERE, "ANCHOR_MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    log("anchor manifest written: " + ", ".join(
        "{}={}".format(k, v["sha256"][:16]) for k, v in man["anchors"].items()))
    return man


def remaining_jobs(done):
    return sum(1 for (a, r) in PLAN if tag_for(a, r) not in done)


def main():
    log("=== EVALUATOR-RESOLUTION ANCHOR STUDY (cap {:.2f} GPU-h) ===".format(
        CAP_GPU_H))
    write_manifest()
    log("already spent in THIS program: {:.4f} GPU-h".format(spent_gpu_h()))
    log("foreign GPU MiB at start: {}".format(foreign_mib()))

    done = set()
    for anchor, r in PLAN:
        tag = tag_for(anchor, r)
        if eval_is_valid(tag):
            log("{} already complete and VALID".format(tag))
            done.add(tag)

    for anchor, r in PLAN:
        tag = tag_for(anchor, r)
        for attempt in range(1, 4):
            if eval_is_valid(tag):
                done.add(tag)
                break
            quarantine(tag)
            # FAIL CLOSED: this job plus ALL remaining registered jobs must fit.
            spent = spent_gpu_h()
            need = EVAL_COST_EST * remaining_jobs(done)
            if spent + need > CAP_GPU_H:
                log("STOP: {} would breach the cap. spent {:.4f} + remaining "
                    "registered {:.4f} ({} jobs) > {:.2f}. Preserving completed "
                    "repeats; not shortening any evaluation.".format(
                        tag, spent, need, remaining_jobs(done), CAP_GPU_H))
                return 2
            a = ANCHORS[anchor]
            rc = run_guarded(
                "evalres:{}".format(tag),
                ENV + ("WANDB_MODE=disabled WANDB_SILENT=true {py} "
                       "experiments/policy_improvement/phase2_diagnostics.py "
                       "--loadpath {lp} --epoch {ep} --tag {tag} --skip-offline "
                       "> linux_logs/{tag}.txt 2>&1").format(
                           py=PY, lp=a["loadpath"], ep=a["epoch"], tag=tag),
                log=log)
            log("{} attempt {} rc={} valid={}".format(
                tag, attempt, rc, eval_is_valid(tag)))
            if rc != 0:
                return 1
        if not eval_is_valid(tag):
            log("{} NOT VALID after 3 attempts".format(tag))
            return 1
        done.add(tag)

    log("=== ANCHOR STUDY COMPLETE: {:.4f} GPU-h of {:.2f} ===".format(
        spent_gpu_h(), CAP_GPU_H))
    return 0


if __name__ == "__main__":
    sys.exit(main())
