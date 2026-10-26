"""Submission-phase runner: one GPU-leased sub_eval process per group, re-launched
until every unit is VALID. Budget gate before each launch:
spent + projected(remaining units) <= CAP (20 GPU-h)."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../policy_improvement/loss_finetune/mechanism"))
sys.path.insert(0, HERE)
from gpu_lease import run_guarded, foreign_mib  # noqa: E402
from spend import spent, CAP  # noqa: E402
from units import unit_name, is_valid, CALLS  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
LOG = os.path.join(HERE, "logs/run.log")
ENV = ("cd {r} && export PYTHONPATH={r}:{r}/diffuser LD_LIBRARY_PATH=/home/jren313/miniconda3/envs/"
       "ecdiffuser-linux/lib:${{LD_LIBRARY_PATH-}} CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled && ").format(r=ROOT)


def log(m):
    line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), m)
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def units(g):
    sfx = "" if g.get("max_episodes") is None else "_k{}".format(g["max_episodes"])
    out = []
    for s in g["sets"].split(","):
        for R in str(g["R"]).split(","):
            for c in g["configs"].split(","):
                n, sol, p = c.split(":")
                out.append(unit_name(g["ckpt"], g["num_entity"], g["num_envs"], n, sol, p, R, s) + sfx)
    return out


def unit_cost_h(g, cfg):
    """Projected GPU-h for one unit (conservative, from measured phase-2 timings)."""
    n, sol, p = cfg.split(":"); calls = int(n) * CALLS[sol]
    neps = g.get("max_episodes") or 96; b = g["num_envs"]; nb = -(-neps // b)
    ms_call = {96: 55.0, 16: 9.5, 4: 7.0, 1: 6.0}.get(b, 55.0) * (2.0 if p == "fp32" else 1.0)
    sim_s_step = {96: 0.36, 16: 0.14, 4: 0.08, 1: 0.12}.get(b, 0.36)       # measured b16: 83 s / 600 steps
    return 1.15 * nb * 100 * (sim_s_step + calls * ms_call / 1e3) / 3600


def remaining_cost(groups):
    tot = 0.0
    for g in groups:
        cfgs = g["configs"].split(",")
        for u in units(g):
            if not is_valid(u):
                cfg = next(c for c in cfgs if "_{}{}_{}_".format(c.split(":")[1], c.split(":")[0], c.split(":")[2]) in u)
                tot += unit_cost_h(g, cfg)
        tot += 0.02 * (not all(is_valid(u) for u in units(g)))      # process start-up
    return tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("plan"); ap.add_argument("--max_launch", type=int, default=6)
    a = ap.parse_args()
    groups = json.load(open(a.plan))
    log("=== plan {} ({} groups) spent {:.4f}/{} projected {:.3f} ===".format(
        os.path.basename(a.plan), len(groups), spent(), CAP, remaining_cost(groups)))
    for gi, g in enumerate(groups):
        for k in range(a.max_launch):
            todo = [u for u in units(g) if not is_valid(u)]
            if not todo:
                break
            s, need = spent(), remaining_cost(groups[gi:])
            if s + need > CAP:
                log("STOP cap: spent {:.4f} + projected {:.3f} > {}".format(s, need, CAP)); return 2
            stage = "sub:{}:c{}:b{}:{}".format(g["ckpt"], g["num_entity"], g["num_envs"], g.get("tag", gi))
            log("[group {}/{} try {}] {} ({} units left, spent {:.4f})".format(
                gi + 1, len(groups), k + 1, stage, len(todo), s))
            cmd = ENV + ("{py} experiments/submission/sub_eval.py --ckpt {ckpt} --loadpath {lp} --epoch {ep} "
                         "--train_seed {ts} --num_entity {ne} --num_envs {b} --configs {cf} --R {R} --sets {S}{mx} "
                         ">> experiments/submission/logs/{name}.txt 2>&1").format(
                py=PY, ckpt=g["ckpt"], lp=g["loadpath"], ep=g["epoch"], ts=g["train_seed"], ne=g["num_entity"],
                b=g["num_envs"], cf=g["configs"], R=g["R"], S=g["sets"],
                mx="" if g.get("max_episodes") is None else " --max_episodes {}".format(g["max_episodes"]),
                name=stage.replace(":", "_"))
            rc = run_guarded(stage, cmd, log=log)
            bad = [u for u in units(g) if not is_valid(u)]
            log("  rc={} valid {}/{}".format(rc, len(units(g)) - len(bad), len(units(g))))
            if rc not in (0, 3, 4, 5, 7) and len(bad) == len(todo):
                log("  process failed without progress; see logs/{}.txt; stopping".format(stage.replace(":", "_")))
                return 1
            for _ in range(60):
                if not bad or foreign_mib() == 0:
                    break
                time.sleep(60)
    log("=== plan {} done; spent {:.4f} ===".format(os.path.basename(a.plan), spent()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
