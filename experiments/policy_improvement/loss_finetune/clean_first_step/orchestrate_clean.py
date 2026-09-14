"""THE single GPU orchestrator for the CLEAN-FIRST-STEP replication program.

Separate program, separate ledger, separate cap (2.90 GPU-h). Shares the
program-wide GPU lease with everything else so only one GPU job ever runs.

Scientific question: the legacy fine-tuning path lets a random-init sanity
gradient enter the first optimizer update, and that residual is
OBJECTIVE-DEPENDENT. So part of the legacy A-vs-B divergence could come from the
mask changing that residual rather than from the intended objective change. This
program removes the residual and repeats the four-seed paired A/B experiment.

  LEGACY: sanity backward on random-init net -> residual survives warm-start ->
          step 1 = g_sanity + g_batch1 + g_batch2
  CLEAN:  same sanity check -> explicit grad clear -> warm-start -> verify zero ->
          step 1 = g_batch1 + g_batch2

Everything else is identical, including the exact historical dataloader-seed
mapping (42->20260905, 43->20260906, 44->20260907, 45->20260908), verified
against the legacy args.json files.

PREDECLARED balanced training order (fixed before launch, never reordered on
intermediate results):
  seed42: A then B
  seed43: B then A
  seed44: A then B
  seed45: B then A

This runner cannot touch legacy or B' artifacts.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(os.path.dirname(HERE), "mechanism")
sys.path.insert(0, MECH)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "clean_orchestrator.log")
P2 = os.path.join(ROOT, "experiments/policy_improvement/phase2")
SUB_T = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
SETS = ("E0", "E1s", "E2s", "E3s", "E4s")
CAP_GPU_H = 2.90

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)

# (seed, arm) in predeclared balanced order
PLAN = [(42, "A"), (42, "B"),
        (43, "B"), (43, "A"),
        (44, "A"), (44, "B"),
        (45, "B"), (45, "A")]


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def spent_gpu_h():
    """GPU-h charged to THIS program, from its own log. Never shared."""
    import re
    from datetime import datetime as dt
    acq, tot = {}, 0.0
    if not os.path.exists(LOG):
        return 0.0
    for line in open(LOG):
        m = re.match(r"\[([0-9T:\-]+)\] \[gpu_lease\] lease acquired \(stage ([^,]+),", line)
        if m:
            acq[m.group(2)] = dt.fromisoformat(m.group(1))
        m = re.match(r"\[([0-9T:\-]+)\] \[gpu_lease\] lease released \(stage ([^)]+)\)", line)
        if m and m.group(2) in acq:
            tot += (dt.fromisoformat(m.group(1)) - acq.pop(m.group(2))).total_seconds() / 3600
    return tot


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


def budget_ok(need, what):
    s = spent_gpu_h()
    if s + need > CAP_GPU_H:
        log("STOP: {} needs ~{:.3f} GPU-h; spent {:.3f} of {:.2f} cap".format(
            what, need, s, CAP_GPU_H))
        return False
    return True


def train_arm(seed, arm):
    """Train one clean arm. Skips if its RUN_COMPLETE sentinel already exists."""
    sub = SUB_T.format(seed)
    d_arm = os.path.join(ROOT, "data/panda_push/lossft_arm{}clean".format(arm), sub)
    sentinel = os.path.join(d_arm, "RUN_COMPLETE.json")
    if os.path.exists(sentinel):
        try:
            if json.load(open(sentinel)).get("step_matches"):
                log("clean {}{} training already complete".format(arm, seed))
                return True
        except Exception:
            pass
    if not budget_ok(0.14, "clean {}{} training".format(arm, seed)):
        return False
    rc = run_guarded(
        "clean:train:{}{}".format(arm, seed),
        ENV + ("WANDB_MODE=offline {py} diffuser/scripts/train.py "
               "--config config.pandapush_lossft_arm{arm}clean_seed{seed} "
               "--num_entity 3 --rand_color --seed {seed} "
               "> linux_logs/clean_{arm}{seed}.txt 2>&1").format(
                   py=PY, arm=arm, seed=seed),
        log=log)
    log("clean {}{} training rc={}".format(arm, seed, rc))
    if rc != 0:
        return False

    # MANDATORY: the first-step manifest must exist and prove zero residual.
    man = os.path.join(d_arm, "FIRST_STEP_GRADIENT_MANIFEST.json")
    if not os.path.exists(man):
        log("clean {}{}: MISSING FIRST_STEP_GRADIENT_MANIFEST".format(arm, seed))
        return False
    m = json.load(open(man))
    if m["n_params_with_grad_after_clear"] != 0 or \
            m["total_grad_norm_after_clear"] != 0.0:
        log("clean {}{}: residual sanity gradient SURVIVED -- abort".format(arm, seed))
        return False
    log("clean {}{}: cleared {} grads (norm {:.6e}); residual now 0".format(
        arm, seed, m["n_params_with_grad_before_clear"],
        m["total_grad_norm_before_clear"]))
    return True


def eval_arm(seed, arm):
    sub = SUB_T.format(seed)
    tag = "lossft_clean_{}{}".format(arm, seed)
    for attempt in range(1, 4):
        if eval_is_valid(tag):
            log("clean {} evaluation complete and VALID".format(tag))
            return True
        quarantine(tag)
        if not budget_ok(0.24, "clean {} evaluation".format(tag)):
            return False
        rc = run_guarded(
            "clean:eval:{}{}".format(arm, seed),
            ENV + ("WANDB_MODE=disabled WANDB_SILENT=true {py} "
                   "experiments/policy_improvement/phase2_diagnostics.py "
                   "--loadpath lossft_arm{arm}clean/{sub} --epoch final --tag {tag} "
                   "> linux_logs/{tag}.txt 2>&1").format(
                       py=PY, arm=arm, sub=sub, tag=tag),
            log=log)
        log("clean {} attempt {} rc={} valid={}".format(
            tag, attempt, rc, eval_is_valid(tag)))
        if rc != 0:
            return False
    return eval_is_valid(tag)


def main():
    log("=== CLEAN-FIRST-STEP PROGRAM START (cap {:.2f} GPU-h) commit {} ===".format(
        CAP_GPU_H,
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()))
    log("already spent in THIS program: {:.4f} GPU-h".format(spent_gpu_h()))
    log("foreign GPU MiB at start: {}".format(foreign_mib()))

    for seed, arm in PLAN:
        if not train_arm(seed, arm):
            log("STOP at clean {}{} training".format(arm, seed))
            return 1
        if not eval_arm(seed, arm):
            log("STOP at clean {}{} evaluation".format(arm, seed))
            return 1

    log("=== CLEAN PROGRAM COMPLETE: {:.4f} GPU-h of {:.2f} ===".format(
        spent_gpu_h(), CAP_GPU_H))
    return 0


if __name__ == "__main__":
    sys.exit(main())
