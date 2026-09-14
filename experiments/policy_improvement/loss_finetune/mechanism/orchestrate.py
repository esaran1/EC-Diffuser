"""THE single GPU orchestrator for the terminal-mask mechanism program.

There is exactly one of these. No other script may launch GPU work. Every GPU
job goes through `gpu_lease.run_guarded`, which waits for a stable-clean GPU,
takes the program-wide lease, re-checks cleanliness while holding it (closing
the TOCTOU race), and runs exactly one process.

Frozen execution priority:

  1  Stage B    deterministic B vs B' 500-step denominator probe
  2             apply the PREDECLARED denominator gate
  3A NEGLIGIBLE -> skip the four-seed B' branch entirely
  3B NON-NEGL.  -> run the four-seed B' causal branch FIRST
  4  Stage C    second simulator realization for all eight A/B policies
  5  Stage D    remaining mechanism diagnostics, registered priority order
  6             figures / red-team / report are produced by the caller

Stage C never starts while Stage B is running: this is one sequential process.
"""
import json
import os
import subprocess
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

ENV = ("cd {root} && export PYTHONPATH={root}:{root}/diffuser && "
       "export LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:"
       "${{LD_LIBRARY_PATH-}} && export CUDA_VISIBLE_DEVICES=0 && ").format(root=ROOT)


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def gpu(stage, command, max_wait=None):
    """Run ONE GPU job under the global lease. Returns its exit code."""
    return run_guarded(stage, ENV + command, max_wait_s=max_wait, log=log)


def cpu(command):
    return subprocess.call(ENV + command, shell=True)


# ---------------------------------------------------------------- stage B ---
def stage_b():
    probe = os.path.join(HERE, "bprime_probe.py")
    runs = os.path.join(HERE, "probe_runs")
    for cfg in ("config.pandapush_lossft_armB_seed42",
                "config.pandapush_lossft_armBprime_seed42"):
        name = cfg.replace("config.", "")
        if os.path.exists(os.path.join(runs, name, "PROBE_COMPLETE.json")):
            log("stage B: {} already complete".format(name))
            continue
        rc = gpu("stageB:{}".format(name),
                 "WANDB_MODE=offline WANDB_SILENT=true {py} {probe} --config {cfg} "
                 "> linux_logs/probe_{name}.txt 2>&1".format(
                     py=PY, probe=probe, cfg=cfg, name=name))
        log("stage B: {} exited rc={}".format(name, rc))
        if rc != 0:
            return False
    return True


def denominator_gate():
    rc = cpu("CUDA_VISIBLE_DEVICES='' {py} {s} > {log} 2>&1".format(
        py=PY, s=os.path.join(HERE, "analyze_bprime_probe.py"),
        log=os.path.join(HERE, "bprime_gate.txt")))
    path = os.path.join(HERE, "BPRIME_PROBE_DIVERGENCE.json")
    if rc != 0 or not os.path.exists(path):
        log("denominator gate: analysis FAILED rc={}".format(rc))
        return None
    d = json.load(open(path))
    log("denominator gate: {} | {}".format(d["verdict"], json.dumps(d["gate"])))
    return d["verdict"]


# --------------------------------------------------- conditional B' branch ---
def stage_bprime_full():
    """Only runs if the denominator effect is NON-negligible."""
    order = [(42, "B'"), (43, "B'"), (44, "B'"), (45, "B'")]
    for seed, _ in order:
        sub = SUB_T.format(seed)
        d_arm = os.path.join(ROOT, "data/panda_push/lossft_armBprime", sub)
        tag = "lossft_bprime_{}".format(seed)
        if not os.path.exists(os.path.join(d_arm, "RUN_COMPLETE.json")):
            rc = gpu("stageB':train:seed{}".format(seed),
                     "WANDB_MODE=offline {py} diffuser/scripts/train.py "
                     "--config config.pandapush_lossft_armBprime_seed{seed} "
                     "--num_entity 3 --rand_color --seed {seed} "
                     "> linux_logs/bprime_seed{seed}.txt 2>&1".format(py=PY, seed=seed))
            log("stage B' seed{} training rc={}".format(seed, rc))
            if rc != 0:
                return False
        if not os.path.exists(os.path.join(P2, tag + "_diagnostics.json")):
            rc = gpu("stageB':eval:seed{}".format(seed),
                     "WANDB_MODE=disabled WANDB_SILENT=true {py} "
                     "experiments/policy_improvement/phase2_diagnostics.py "
                     "--loadpath lossft_armBprime/{sub} --epoch final --tag {tag} "
                     "> linux_logs/{tag}.txt 2>&1".format(py=PY, sub=sub, tag=tag))
            log("stage B' seed{} diagnostics rc={}".format(seed, rc))
            if rc != 0:
                return False
    return True


# ---------------------------------------------------------------- stage C ---
def stage_c():
    """One extra clean realization for all eight EXISTING A/B checkpoints."""
    plan = [(42, "A"), (42, "B"), (43, "B"), (43, "A"),
            (44, "A"), (44, "B"), (45, "B"), (45, "A")]   # predeclared balanced
    for seed, arm in plan:
        tag = "lossft_rep2_{}{}".format(arm, seed)
        if os.path.exists(os.path.join(P2, tag + "_diagnostics.json")):
            log("stage C: {} already complete".format(tag))
            continue
        rc = gpu("stageC:{}".format(tag),
                 "WANDB_MODE=disabled WANDB_SILENT=true {py} "
                 "experiments/policy_improvement/phase2_diagnostics.py "
                 "--loadpath lossft_arm{arm}/{sub} --epoch final --tag {tag} "
                 "--skip-offline > linux_logs/{tag}.txt 2>&1".format(
                     py=PY, arm=arm, sub=SUB_T.format(seed), tag=tag))
        log("stage C: {} rc={}".format(tag, rc))
        if rc != 0:
            return False
    return True


# ---------------------------------------------------------------- stage D ---
def stage_d():
    """Mechanism diagnostics, registered priority order.

    Gradient interference is already complete and NEGATIVE; it is not repeated.
    Priority here is the stochastic-action question, which is the live hypothesis.
    """
    out = os.path.join(HERE, "ACTION_DISTRIBUTION.json")
    if os.path.exists(out):
        log("stage D: action distribution already complete")
        return True
    rc = gpu("stageD:action_distribution",
             "WANDB_MODE=disabled WANDB_SILENT=true {py} {s} "
             "--seeds 42,43,44,45 --n_conditions 256 --K 16 "
             "> linux_logs/action_distribution.txt 2>&1".format(
                 py=PY, s=os.path.join(HERE, "action_distribution.py")),
             max_wait=None)
    log("stage D: action distribution rc={}".format(rc))
    return rc == 0


def main():
    log("=== ORCHESTRATOR START (single GPU path) commit {} ===".format(
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()))
    log("foreign GPU MiB at start: {}".format(foreign_mib()))

    if not stage_b():
        log("STOP: stage B did not complete cleanly")
        return 1

    verdict = denominator_gate()
    if verdict is None:
        log("STOP: denominator gate could not be evaluated")
        return 1

    if verdict == "DENOMINATOR-NEGLIGIBLE":
        log("denominator NEGLIGIBLE -> skipping the four-seed B' branch, "
            "proceeding to stage C")
    else:
        log("denominator NON-NEGLIGIBLE -> running the four-seed B' branch FIRST")
        if not stage_bprime_full():
            log("STOP: B' branch did not complete cleanly")
            return 1

    if not stage_c():
        log("STOP: stage C did not complete cleanly")
        return 1

    if not stage_d():
        log("stage D incomplete (non-fatal); continuing to reporting")

    log("=== ORCHESTRATOR COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
