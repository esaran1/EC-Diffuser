"""Contention-tolerant training runner with EXACT resume.

Each attempt runs under the global lease. If the gpu_guard stops a run on
foreign contention it writes state_contention_<s>.pt (full state); the next
attempt resumes from the latest one via EC_RESUME_FROM, which restores model,
EMA, optimizer, step, RNG and the exact data stream (verified against the
recorded generator state, else train.py refuses). A partial run WITHOUT a
contention checkpoint (hard kill) is quarantined and restarted fresh.

Usage: train_resumable.py --config config.X --prefix X --tag T [--cost 0.33]
"""
import argparse, glob, os, re, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
TF = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(TF), "loss_finetune", "mechanism"))
sys.path.insert(0, os.path.join(TF, "uwm"))
from gpu_lease import run_guarded, foreign_mib  # noqa: E402
from program_spend import spent  # noqa: E402

ROOT = "/home/jren313/EC-Diffuser-1"
PY = "/home/jren313/miniconda3/envs/ecdiffuser-linux/bin/python"
RUN = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
CAP = 15.0
ENV = ("cd {r} && export PYTHONPATH={r}:{r}/diffuser && export LD_LIBRARY_PATH="
       "/home/jren313/miniconda3/envs/ecdiffuser-linux/lib:${{LD_LIBRARY_PATH-}} && "
       "export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline && ").format(r=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cost", type=float, default=0.33)
    ap.add_argument("--reserve", type=float, default=0.0,
                    help="GPU-h to keep free for higher-priority work")
    ap.add_argument("--max_attempts", type=int, default=30)
    a = ap.parse_args()
    LOG = os.path.join(HERE, "train_{}.log".format(a.tag))

    def log(m):
        line = "[{}] {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), m)
        print(line, flush=True)
        open(LOG, "a").write(line + "\n")

    d = os.path.join(ROOT, "data/panda_push", a.prefix, RUN)
    final = os.path.join(d, "state_final.pt")
    for attempt in range(1, a.max_attempts + 1):
        if os.path.exists(final):
            open(os.path.join(d, "RUN_COMPLETE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            log("{} COMPLETE".format(a.tag))
            return 0
        s = spent()
        if s + a.cost + a.reserve > CAP:
            log("STOP cap: spent {:.4f} + {:.2f} + reserve {:.2f} > {}".format(s, a.cost, a.reserve, CAP))
            return 2
        conts = sorted(glob.glob(os.path.join(d, "state_contention_*.pt")),
                       key=lambda p: int(re.search(r"contention_(\d+)", p).group(1)))
        env_extra = ""
        if conts:
            label = os.path.basename(conts[-1])[len("state_"):-len(".pt")]
            env_extra = "EC_RESUME_FROM={} ".format(label)
            log("attempt {}: EXACT RESUME from {}".format(attempt, label))
        else:
            if os.path.isdir(d):
                q = os.path.join(ROOT, "data/panda_push/_contended_attempts",
                                 "{}_{}".format(a.tag, time.strftime("%m%d_%H%M%S")))
                os.makedirs(os.path.dirname(q), exist_ok=True)
                shutil.move(os.path.dirname(d), q)
                log("attempt {}: partial without resume point quarantined -> {}".format(attempt, q))
            log("attempt {}: fresh run".format(attempt))
        rc = run_guarded("train:{}:{}".format(a.tag, attempt),
                         ENV + env_extra + ("{py} diffuser/scripts/train.py --config {c} "
                                            "--num_entity 3 --rand_color --seed 42 "
                                            ">> linux_logs/train_{t}.txt 2>&1").format(
                                                py=PY, c=a.config, t=a.tag), log=log)
        log("attempt {} rc={} final={}".format(attempt, rc, os.path.exists(final)))
        if rc in (3, 4, 5):
            # run_guarded never launched the command (stable-clean wait timed
            # out, lease held by a live project process, or a foreign job raced
            # in after acquisition). Not a training failure: wait and retry.
            log("lease-level rc={} (command not run); retrying".format(rc))
            time.sleep(120)
            continue
        if rc != 0:
            # contention stops exit 0; nonzero is a real error (e.g. the exact-
            # resume verification refusing). Never loop on it.
            log("FAILED rc={} -- inspect linux_logs/train_{}.txt".format(rc, a.tag))
            return 1
    log("attempts exhausted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
