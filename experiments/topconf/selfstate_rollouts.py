"""Phase I: cache RAW PHYSICAL STATE along NFE1/NFE2/NFE4 closed-loop rollouts,
then query all three policies offline at every cached state with matched noise.

No demo-action targets are used at self-induced states (protocol section 4).
We measure POLICY DISAGREEMENT, not correctness.
"""
import argparse, json, os, sys, time
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts"); sys.path.insert(0, "experiments/evaluation_noise")
from crn import CRNPolicyWrapper, CRN_BASE_SEED, derive_seed
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import ARMS, Args, array_to_state_dict, entity_positions, summarize_episode
import isaacgym_nfe_study as NS
import pickle

OUT = "experiments/topconf/selfstate"
STRIDE = 5   # cache every STRIDE-th decision to bound storage


def raw_phys(env):
    e = env.env if hasattr(env, "env") else env
    return (utils.to_np(e._q).copy(), utils.to_np(e._root_state).copy(),
            utils.to_np(e._eef_state).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--episodes", type=int, default=96)
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    args = Args(); utils.set_global_device(args.device)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep, _ = NS.get_episode_set(env, 0, cli.episodes)
    ARMS["flow"]["loadpath"] = f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{cli.seed}"
    keys = ep["keys"]; inits, goals = ep["init"], ep["goal"]
    nE = env.num_envs

    for nfe in cli.nfes:
        f = os.path.join(OUT, f"s{cli.seed}_nfe{nfe}_states.npz")
        if os.path.exists(f):
            print(f"[skip] {f}"); continue
        policy = NS.build_policy("flow", nfe, args)
        wrapped = CRNPolicyWrapper(policy)
        store = {k: [] for k in ["obs", "goal", "q", "root", "eef", "action",
                                 "episode", "step"]}
        succ = []
        t0 = time.time()
        for bs in range(0, cli.episodes, nE):
            idx = list(range(bs, min(bs + nE, cli.episodes)))
            idx += [idx[-1]] * (nE - len(idx))
            torch.manual_seed(CRN_BASE_SEED + bs); np.random.seed(CRN_BASE_SEED + bs)
            obs = env.reset(set_init_states=array_to_state_dict(inits[idx], keys, env.device),
                            set_goal_states=array_to_state_dict(goals[idx], keys, env.device))
            wrapped.new_batch(bs)
            cube_log = [[] for _ in range(nE)]; act_log = [[] for _ in range(nE)]
            eef_log = [[] for _ in range(nE)]; info_last = None
            for step in range(env.horizon):
                observation = obs["achieved_goal"].reshape(nE, -1)
                gl = obs["desired_goal"].reshape(nE, -1)
                _, samples = wrapped({0: observation, args.horizon - 1: gl},
                                     batch_size=1, verbose=False)
                a = samples.actions[:, 0]
                if step % STRIDE == 0:
                    q, root, eef = raw_phys(env)
                    store["obs"].append(observation.copy()); store["goal"].append(gl.copy())
                    store["q"].append(q); store["root"].append(root); store["eef"].append(eef)
                    store["action"].append(np.asarray(a)); store["step"].append(np.full(nE, step))
                    store["episode"].append(np.array(idx))
                obs, _, _, infos = env.step(a); info_last = infos
                st = entity_positions(env)
                for e in range(nE):
                    act_log[e].append(np.asarray(a[e], dtype=np.float64))
                    eef_log[e].append(st[e, 0, :3].copy()); cube_log[e].append(st[e, 1:, :2].copy())
            gs = np.asarray(env.goal_pos)
            for e in range(len(idx) if bs + nE <= cli.episodes else cli.episodes - bs):
                r = summarize_episode(episode=idx[e], actions=np.array(act_log[e]),
                                      eef=np.array(eef_log[e]), cubes=np.array(cube_log[e]),
                                      goal_cubes=gs[e], info=info_last[e], threshold=0.04)
                succ.append({"episode": int(idx[e]), "success": float(r["success"]),
                             "first_contact_step": float(r["first_contact_step"]),
                             "max_obj_dist": float(r["max_obj_dist"])})
            print(f"  nfe{nfe} batch {bs} done", flush=True)
        S = {k: np.concatenate(v, 0).astype(np.float32) if k not in ("episode", "step")
             else np.concatenate(v, 0).astype(np.int64) for k, v in store.items()}
        S["_success"] = np.array(json.dumps(succ))
        np.savez_compressed(f, **S)
        print(f"  -> wrote {f}  states={S['obs'].shape}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
