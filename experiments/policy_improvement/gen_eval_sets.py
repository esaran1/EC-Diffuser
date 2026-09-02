"""Phase 0a: generate frozen development evaluation scenario sets E3s/E4s.

Reuses record_episode_set() from experiments/scripts/isaacgym_control.py
VERBATIM so E3s/E4s are constructed exactly like E0/E1s/E2s.
E0/E1s/E2s are never regenerated - they are read and hashed only.
"""
import hashlib, json, os, pickle, subprocess, sys

import isaacgym  # noqa: F401  must precede torch
import numpy as np  # noqa: E402

sys.path.insert(0, "experiments/scripts")
import diffuser.utils as utils  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from isaacgym_control import Args, record_episode_set  # noqa: E402

OUT = "experiments/isaacgym_episode_sets"
EXISTING = {"E0": (20260820, f"{OUT}/replicate0_n96.pkl"),
            "E1s": (20261820, f"{OUT}/replicate1_n96.pkl"),
            "E2s": (20262820, f"{OUT}/replicate2_n96.pkl")}
NEW = {"E3s": (20263820, f"{OUT}/replicate3_n96.pkl"),
       "E4s": (20264820, f"{OUT}/replicate4_n96.pkl")}
N = 96

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def entry(name, seed, path, payload):
    return dict(eval_seed_id=name, gen_seed=seed, path=path,
                n_episodes=int(len(payload["init"])),
                cubes=3, horizon=100, num_envs_at_generation=16,
                init_sha256=sha(payload["init"]), goal_sha256=sha(payload["goal"]),
                payload_sha256=payload["sha256"],
                stored_seed=int(payload.get("seed", -1)))

def main():
    man = {"protocol": "five fixed DEVELOPMENT evaluation scenario sets",
           "note": ("evaluation-scenario seeds; NOT training seeds and NOT "
                    "repeated physics realizations"),
           "task": "3-cube PushCube", "N": N,
           "git_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"]).decode().strip(),
           "sets": []}

    for name, (seed, path) in EXISTING.items():          # read-only
        p = pickle.load(open(path, "rb"))
        assert len(p["init"]) == N, f"{name}: expected {N}"
        assert int(p["seed"]) == seed, f"{name}: seed mismatch"
        man["sets"].append({**entry(name, seed, path, p), "status": "preexisting"})
        print(f"{name}: preexisting, payload {p['sha256'][:16]}")

    need = {k: v for k, v in NEW.items() if not os.path.exists(v[1])}
    if need:
        args = Args()
        utils.set_global_device(args.device)
        env = setup_isaac_env(args)
        env.horizon = args.max_episode_length
        for name, (seed, path) in NEW.items():
            if os.path.exists(path):
                print(f"{name}: already exists, not regenerating")
            else:
                p = record_episode_set(env, N, seed=seed)
                assert len(p["init"]) == N
                with open(path, "wb") as fh:
                    pickle.dump(p, fh)
                print(f"{name}: generated seed={seed} payload {p['sha256'][:16]}")
    for name, (seed, path) in NEW.items():
        p = pickle.load(open(path, "rb"))
        man["sets"].append({**entry(name, seed, path, p), "status": "generated"})

    # distinctness: every set must differ from every other
    inits = {s["eval_seed_id"]: s["init_sha256"] for s in man["sets"]}
    assert len(set(inits.values())) == 5, f"duplicate scenario sets: {inits}"
    man["distinct_init_hashes"] = True

    with open("experiments/policy_improvement/eval_seed_manifest.json", "w") as fh:
        json.dump(man, fh, indent=1)
    print("\n=== MANIFEST ===")
    for s in man["sets"]:
        print(f"{s['eval_seed_id']:>4s} seed={s['gen_seed']} n={s['n_episodes']} "
              f"init={s['init_sha256'][:16]} goal={s['goal_sha256'][:16]} "
              f"payload={s['payload_sha256'][:16]} [{s['status']}]")

if __name__ == "__main__":
    main()
