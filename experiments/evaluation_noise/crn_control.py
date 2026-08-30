"""CRN-paired closed-loop evaluator. Canonical evaluate() logic, reproduced with
one addition: the CRN wrapper reseeds the RNG before each policy invocation,
keyed by (batch_start, decision index). Nothing else differs.
"""
import argparse, json, os, sys, time
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts"); sys.path.insert(0, "experiments/evaluation_noise")
from crn import CRNPolicyWrapper, CRN_BASE_SEED
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import (ARMS, Args, array_to_state_dict, entity_positions,
                              summarize_episode)
import isaacgym_nfe_study as NS
import pickle

OUT = "experiments/evaluation_noise/results"


def evaluate_crn(wrapped, env, episode_set, args):
    keys = episode_set["keys"]; inits, goals = episode_set["init"], episode_set["goal"]
    n_episodes = len(inits); n_envs = env.num_envs
    records = []; started = time.time()
    for batch_start in range(0, n_episodes, n_envs):
        batch = slice(batch_start, min(batch_start + n_envs, n_episodes))
        n_active = batch.stop - batch.start
        index = (list(range(batch.start, batch.stop)) + [batch.stop - 1] * (n_envs - n_active)
                 if n_active < n_envs else list(range(batch.start, batch.stop)))
        # env RNG is also seeded per batch so any reset draw is identical across arms
        torch.manual_seed(CRN_BASE_SEED + batch_start); np.random.seed(CRN_BASE_SEED + batch_start)
        obs = env.reset(set_init_states=array_to_state_dict(inits[index], keys, env.device),
                        set_goal_states=array_to_state_dict(goals[index], keys, env.device))
        wrapped.new_batch(batch_start)
        actions_log = [[] for _ in range(n_envs)]; eef_log = [[] for _ in range(n_envs)]
        cube_log = [[] for _ in range(n_envs)]; info_last = None
        step = 0
        while step < env.horizon:
            observation = obs["achieved_goal"].reshape(n_envs, -1)
            goal = obs["desired_goal"].reshape(n_envs, -1)
            conditions = {0: observation, args.horizon - 1: goal}
            _, samples = wrapped(conditions, batch_size=1, verbose=False)
            action = samples.actions[:, 0]
            obs, _, _, infos = env.step(action); info_last = infos
            state = entity_positions(env)
            for e in range(n_envs):
                actions_log[e].append(np.asarray(action[e], dtype=np.float64))
                eef_log[e].append(state[e, 0, :3].copy())
                cube_log[e].append(state[e, 1:, :2].copy())
            step += 1
        goal_state = np.asarray(env.goal_pos)
        for e in range(n_active):
            records.append(summarize_episode(
                episode=index[e], actions=np.array(actions_log[e]),
                eef=np.array(eef_log[e]), cubes=np.array(cube_log[e]),
                goal_cubes=goal_state[e], info=info_last[e], threshold=0.04))
    return records, time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 32])
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-crn", action="store_true", help="disable CRN (control condition)")
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    os.makedirs(OUT, exist_ok=True)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    episode_set, _ = NS.get_episode_set(env, cli.replicate, cli.episodes)
    print(f"[set] sha={episode_set['sha256'][:16]} n={len(episode_set['init'])} "
          f"CRN={'OFF' if cli.no_crn else 'ON'}", flush=True)

    for tseed in cli.seeds:
        ARMS["flow"]["loadpath"] = f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{tseed}"
        for steps in cli.steps:
            label = f"s{tseed}_nfe{steps}{cli.tag}"
            out = os.path.join(OUT, f"r{cli.replicate}_{label}.json")
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True); continue
            policy = NS.build_policy("flow", steps, args)
            wrapped = CRNPolicyWrapper(policy, enabled=not cli.no_crn)
            records, elapsed = evaluate_crn(wrapped, env, episode_set, args)
            summary = NS.summarize(records, policy, label, "flow", steps, episode_set, elapsed)
            summary.update(training_seed=tseed, crn_enabled=not cli.no_crn,
                           crn_base_seed=CRN_BASE_SEED, n_seeds_used=len(wrapped.seeds_used))
            m = summary["measured_calls_per_plan"]
            if abs(m - float(steps)) > 1e-6:
                summary["CALL_COUNT_MISMATCH"] = True
                print(f"  !! expected {steps} calls, measured {m}", flush=True)
            with open(out, "w") as h:
                json.dump({"summary": summary, "episodes": records}, h, indent=2)
            print(f"  -> {label}: success={summary['success_rate']:.4f} "
                  f"({m:.1f} calls, {elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
