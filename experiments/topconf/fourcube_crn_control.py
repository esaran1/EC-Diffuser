"""4-cube NFE2 vs NFE4 at fixed H=100, R=3, seeds 42/43/44, with validated CRN.

Reuses the canonical CRN evaluator (experiments/evaluation_noise/crn_control.py
semantics) and the frozen 4-cube scenario set. NFE is the only treatment variable.
"""
import argparse, json, os, sys, time
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts"); sys.path.insert(0, "experiments/evaluation_noise")
from crn import CRNPolicyWrapper, CRN_BASE_SEED
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import (ARMS, Args as BaseArgs, array_to_state_dict,
                              entity_positions, summarize_episode)
import isaacgym_nfe_study as NS
import pickle

SET = "experiments/isaacgym_control/fourcube/episode_set_4cube.pkl"
OUT = "experiments/topconf/fourcube_r3"


class Args(BaseArgs):
    num_entity = 4
    max_episode_length = 100          # FIXED H, not the native 150


def evaluate_crn(wrapped, env, ep, args):
    keys = ep["keys"]; inits, goals = ep["init"], ep["goal"]
    n_ep = len(inits); nE = env.num_envs; recs = []; t0 = time.time()
    for bs in range(0, n_ep, nE):
        idx = list(range(bs, min(bs + nE, n_ep)))
        n_act = len(idx); idx += [idx[-1]] * (nE - n_act)
        torch.manual_seed(CRN_BASE_SEED + bs); np.random.seed(CRN_BASE_SEED + bs)
        obs = env.reset(set_init_states=array_to_state_dict(inits[idx], keys, env.device),
                        set_goal_states=array_to_state_dict(goals[idx], keys, env.device))
        wrapped.new_batch(bs)
        al = [[] for _ in range(nE)]; el = [[] for _ in range(nE)]
        cl = [[] for _ in range(nE)]; info_last = None
        for step in range(env.horizon):
            o = obs["achieved_goal"].reshape(nE, -1); g = obs["desired_goal"].reshape(nE, -1)
            _, s = wrapped({0: o, args.horizon - 1: g}, batch_size=1, verbose=False)
            a = s.actions[:, 0]
            obs, _, _, infos = env.step(a); info_last = infos
            st = entity_positions(env)
            for e in range(nE):
                al[e].append(np.asarray(a[e], dtype=np.float64))
                el[e].append(st[e, 0, :3].copy()); cl[e].append(st[e, 1:, :2].copy())
        gs = np.asarray(env.goal_pos)
        for e in range(n_act):
            recs.append(summarize_episode(episode=idx[e], actions=np.array(al[e]),
                        eef=np.array(el[e]), cubes=np.array(cl[e]), goal_cubes=gs[e],
                        info=info_last[e], threshold=0.04))
    return recs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--episodes", type=int, default=None)
    cli = ap.parse_args()
    args = Args(); utils.set_global_device(args.device)
    os.makedirs(OUT, exist_ok=True)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    assert env.num_objects == 4, f"expected 4 cubes, got {env.num_objects}"
    assert env.horizon == 100, f"expected H=100, got {env.horizon}"
    ep = pickle.load(open(SET, "rb"))
    if cli.episodes:
        ep = {**ep, "init": ep["init"][:cli.episodes], "goal": ep["goal"][:cli.episodes]}
    print(f"[set] {ep['sha256'][:16]} n={len(ep['init'])} cubes={ep['init'].shape[1]} H={env.horizon}",
          flush=True)
    # predeclared balanced order: arm order alternates with (rep + seed_index) parity
    plan = []
    for si, sd in enumerate(cli.seeds):
        for r in cli.reps:
            order = cli.steps if ((r + si) % 2 == 1) else cli.steps[::-1]
            for n in order:
                plan.append((sd, r, n))
    for sd, r, n in plan:
        f = os.path.join(OUT, f"4cube_H100_s{sd}_nfe{n}_rep{r}.json")
        if os.path.exists(f):
            print(f"[skip] {f}", flush=True); continue
        ARMS["flow"]["loadpath"] = f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{sd}"
        policy = NS.build_policy("flow", n, args)
        wrapped = CRNPolicyWrapper(policy)
        recs, el = evaluate_crn(wrapped, env, ep, args)
        summ = NS.summarize(recs, policy, f"s{sd}_nfe{n}_rep{r}", "flow", n, ep, el)
        summ.update(training_seed=sd, repeat=r, num_cubes=4, horizon=env.horizon,
                    crn_enabled=True, crn_base_seed=CRN_BASE_SEED, fixed_H=True)
        m = summ["measured_calls_per_plan"]
        if abs(m - float(n)) > 1e-6:
            summ["CALL_COUNT_MISMATCH"] = True
            print(f"  !! expected {n} calls, measured {m}", flush=True)
        json.dump({"summary": summ, "episodes": recs}, open(f, "w"), indent=2)
        print(f"  -> s{sd} nfe{n} rep{r}: success={summ['success_rate']:.4f} "
              f"({m:.1f} calls, {el:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
