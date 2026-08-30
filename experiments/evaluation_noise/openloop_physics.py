"""Open-loop physics repeatability. NO policy inference, NO DLP.

Restores identical serialized initial states, applies an identical fixed action
sequence, and repeats R times. Records RAW simulator state so the measurement is
of physics divergence, not representation noise.
"""
import argparse, json, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts")
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import Args, array_to_state_dict
import pickle

ACTION_SEED = 20260906


def raw_state(env):
    e = env.env if hasattr(env, "env") else env
    return {"q": utils.to_np(e._q).copy(), "qd": utils.to_np(e._qd).copy(),
            "root": utils.to_np(e._root_state).copy(),
            "eef": utils.to_np(e._eef_state).copy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=16)
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    idx = list(range(cli.episodes)); idx += [idx[-1]] * (env.num_envs - len(idx))
    rng = np.random.default_rng(ACTION_SEED)
    acts = [rng.uniform(-1, 1, (env.num_envs, 3)).astype(np.float32) for _ in range(cli.steps)]

    runs = []
    for r in range(cli.repeats):
        torch.manual_seed(999); np.random.seed(999)
        env.reset(set_init_states=array_to_state_dict(ep["init"][idx], ep["keys"], env.device),
                  set_goal_states=array_to_state_dict(ep["goal"][idx], ep["keys"], env.device))
        T = [raw_state(env)]
        for a in acts:
            env.step(a)
            T.append(raw_state(env))
        runs.append(T)
        print(f"  repeat {r} done", flush=True)

    ne = cli.episodes
    out = {"protocol": {"repeats": cli.repeats, "steps": cli.steps, "episodes": ne,
                        "action_seed": ACTION_SEED, "policy": "none (fixed action sequence)",
                        "state": "raw PhysX tensors (q, qd, root, eef) - no DLP"},
           "divergence": []}
    # cube root states: actor 1..n are the cubes (actor 0 is the franka)
    for t in range(cli.steps + 1):
        q = np.stack([runs[r][t]["q"][:ne] for r in range(cli.repeats)])
        eef = np.stack([runs[r][t]["eef"][:ne, :3] for r in range(cli.repeats)])
        root = np.stack([runs[r][t]["root"][:ne] for r in range(cli.repeats)])
        cube = root[:, :, 1:, :3]
        qd = np.stack([runs[r][t]["qd"][:ne] for r in range(cli.repeats)])

        def spread(x):
            # max pairwise deviation from the across-repeat mean
            return float(np.abs(x - x.mean(0, keepdims=True)).max()), \
                   float(np.sqrt(((x - x.mean(0, keepdims=True)) ** 2).mean()))
        mq, rq = spread(q); me, re_ = spread(eef); mc, rc = spread(cube); mv, rv = spread(qd)
        out["divergence"].append({"t": t, "dof_max": mq, "dof_rms": rq,
                                  "eef_max": me, "eef_rms": re_,
                                  "cube_max": mc, "cube_rms": rc,
                                  "dofvel_max": mv, "dofvel_rms": rv})
    json.dump(out, open("experiments/evaluation_noise/openloop_physics.json", "w"), indent=2)
    print("\nstep :   dof_max   eef_max  cube_max cube_rms")
    for t in [0, 1, 2, 5, 10, 20, 50, cli.steps]:
        d = out["divergence"][t]
        print(f"{t:5d} : {d['dof_max']:9.3e} {d['eef_max']:9.3e} {d['cube_max']:9.3e} {d['cube_rms']:9.3e}")
    print("\nwrote openloop_physics.json")


if __name__ == "__main__":
    main()
