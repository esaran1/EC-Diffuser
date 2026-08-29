"""Matched-NFE solver study. NO training, NO model/loss/data changes.

Only the numerical integration of dx/dt = v_theta(x,t) changes. Same frozen
episode set, same current states, goals, conditioning and the SAME x0 across
every solver and NFE within a given (episode, rollout step, seed).

Two error metrics per endpoint:
  A. ground-truth imagination error  -- chamfer_position vs the actual future
  B. numerical ODE error             -- latent distance to a high-accuracy
                                        RK4 reference from the same x0

NFE accounting: euler = steps, midpoint/heun = 2*steps, rk4 = 4*steps.
"""
import argparse, json, pickle, sys
import isaacgym  # noqa: F401  must precede torch
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
sys.path.insert(0, "experiments/scripts")
from latent_metric import training_stats, block_errors, chamfer_position
import solvers

import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24
# (method, n_steps) -> NFE.  Primary budgets: 2, 4, 8.
ARMS = [("euler", 2), ("euler", 4), ("euler", 8), ("euler", 16),
        ("midpoint", 1), ("midpoint", 2), ("midpoint", 4),
        ("heun", 1), ("heun", 2), ("heun", 4)]
# Reference ladder: convergence is checked between these two.
# Reference: RK4 with 64 steps (256 NFE). The Flow ODE does NOT converge to
# machine precision -- independent high-order schemes at NFE>=256 agree only to
# ~0.17 (see reference_uncertainty below), so the reference carries a stated
# uncertainty floor rather than a convergence claim. That floor is ~8x smaller
# than Euler@16's distance to it, so solver arms remain resolvable.
REF_MAIN = ("rk4", 64)              # 256 NFE
REF_CHECK = ("midpoint", 128)       # 256 NFE, INDEPENDENT scheme (not a finer RK4)
FLOW_SEEDS = {
    42: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"),
    43: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43"),
    44: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"),
}


class Args:
    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"; num_entity = 3; horizon = 5
    max_episode_length = 100; planning_only = True; push_t = False
    multiview = True; verbose = False; seed = 42; device = "cuda:0"
    preprocess_fns = []; push_t_num_color = 1


def canonical_euler(model, cond, x0, steps):
    """Verbatim flow_matching.py:358-373 loop, for the bit-identity check."""
    x = x0.clone(); model._apply_conditioning(x, cond)
    cmask = model._make_conditioning_mask(x, cond); dt = 1.0 / float(steps)
    for step in range(steps):
        t = x.new_full((x.shape[0],), float(step) / float(steps))
        v = model.model(x, cond, t * model.time_scale) * cmask.to(x.dtype)
        x = x + dt * v; model._apply_conditioning(x, cond)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--noise-seed", type=int, default=777)
    ap.add_argument("--ref-check-steps", type=int, default=2,
                    help="rollout steps on which to also run REF_CHECK")
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    raw = pickle.load(open("ecdiffuser-data/push_cubes/3C_randcolor/"
                           "panda_push_replay_buffer_dlp.pkl", "rb"))["observations"][:400]
    mu, sd = training_stats(raw.reshape(-1, 48, 10))

    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]
    print(f"[set] {ep['sha256'][:16]} n={len(ep['init'])}", flush=True)

    results, refdist, diags, validation, ref_chamfer = {}, {}, {}, {}, {}
    ref_conv = []

    for seed, (base, path) in FLOW_SEEDS.items():
        tag = f"flow_s{seed}"
        exp = utils.load_diffusion(base, args.dataset, path, epoch="latest", seed=args.seed,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval()
        norm = exp.dataset.normalizer

        rows = {a: [] for a in ARMS}; rd = {a: [] for a in ARMS}
        dg = {a: {"upd": [], "cos": [], "relv": []} for a in ARMS}

        idx = list(range(min(cli.episodes, env.num_envs)))
        idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen_noise = torch.Generator(device="cpu").manual_seed(cli.noise_seed)
        nE = min(cli.episodes, env.num_envs)

        for rs in range(cli.steps):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                       device=env.device, dtype=torch.float32),
                    args.horizon - 1: torch.as_tensor(norm.normalize(goal, "observations"),
                                                      device=env.device, dtype=torch.float32)}
            x0 = torch.randn((env.num_envs, args.horizon, model.transition_dim),
                             generator=gen_noise).to(env.device)

            with torch.no_grad():
                # --- Phase 3 validation, once per seed on the first step ---
                if rs == 0:
                    can = canonical_euler(model, cond, x0, 4)
                    dia, nfe_e, _ = solvers.integrate(model, cond, x0, "euler", 4)
                    dia2, _, _ = solvers.integrate(model, cond, x0, "euler", 4)
                    validation[tag] = {
                        "euler_bit_identity_max_abs_diff": float((dia - can).abs().max()),
                        "euler_determinism_max_abs_diff": float((dia - dia2).abs().max()),
                        "euler4_nfe_used": nfe_e}

                # --- high-accuracy reference ---
                ref, ref_nfe, _ = solvers.integrate(model, cond, x0, *REF_MAIN)
                ref_ob = norm.unnormalize(utils.to_np(ref)[:, :, model.action_dim:], "observations")
                ref_gen = ref_ob.reshape(env.num_envs, args.horizon, 2, N_PER_VIEW, 10)[:, 1, 0]
                if rs < cli.ref_check_steps:
                    ref2, ref2_nfe, _ = solvers.integrate(model, cond, x0, *REF_CHECK)
                    d = (ref2 - ref).flatten(1).norm(dim=1)[:nE]
                    ref_conv += [{"seed": seed, "step": rs, "d": float(v)} for v in d]

                per_arm_actions = {}
                for arm in ARMS:
                    meth, nst = arm
                    xe, nfe, d = solvers.integrate(model, cond, x0, meth, nst, collect=True)
                    if d["update_norm"]:
                        dg[arm]["upd"].append(float(np.mean(d["update_norm"])))
                    if d["cos_consecutive"]:
                        dg[arm]["cos"].append(float(np.mean(d["cos_consecutive"])))
                    if d["rel_v_change"]:
                        dg[arm]["relv"].append(float(np.mean(d["rel_v_change"])))
                    rd[arm].append(utils.to_np((xe - ref).flatten(1).norm(dim=1))[:nE])
                    ob = norm.unnormalize(utils.to_np(xe)[:, :, model.action_dim:], "observations")
                    per_arm_actions[arm] = norm.unnormalize(
                        utils.to_np(xe)[:, :, :model.action_dim], "actions")[:, 0]
                    rows[arm].append(ob.reshape(env.num_envs, args.horizon, 2, N_PER_VIEW, 10)[:, 1, 0])

            # advance with the same reference arm for every solver: one shared trajectory
            obs, _, _, _ = env.step(per_arm_actions[ARMS[0]])
            real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
            ref_chamfer.setdefault(tag, []).extend(
                [chamfer_position(ref_gen[e], real[e]) for e in range(nE)])
            for arm in ARMS:
                g = rows[arm][-1]
                rows[arm][-1] = [chamfer_position(g[e], real[e]) for e in range(nE)]
            print(f"  {tag} step {rs} done", flush=True)

        for arm in ARMS:
            meth, nst = arm
            key = f"{tag}_{meth}{nst}"
            flat = [r for chunk in rows[arm] for r in chunk]
            results[key] = {"seed": seed, "method": meth, "steps": nst,
                            "nfe": nst * solvers.NFE_PER_STEP[meth],
                            "chamfer": [float(v) for v in flat], "n": len(flat)}
            refdist[key] = [float(v) for c in rd[arm] for v in c]
            diags[key] = {
                "mean_update_norm": float(np.mean(dg[arm]["upd"])) if dg[arm]["upd"] else None,
                "mean_cos_consecutive": float(np.mean(dg[arm]["cos"])) if dg[arm]["cos"] else None,
                "mean_rel_v_change": float(np.mean(dg[arm]["relv"])) if dg[arm]["relv"] else None}
            print(f"    {key:24s} NFE={results[key]['nfe']:3d} "
                  f"chamfer={np.mean(flat):.5f} refdist={np.mean(refdist[key]):.5f}", flush=True)

    out = {"results": results, "reference_distance": refdist,
           "reference_chamfer": {k: [float(v) for v in vv] for k, vv in ref_chamfer.items()},
           "vector_field_diagnostics": diags, "implementation_validation": validation,
           "reference_convergence": ref_conv,
           "protocol": {"episode_set": ep["sha256"], "episodes": cli.episodes,
                        "rollout_steps": cli.steps, "noise_seed": cli.noise_seed,
                        "reference_main": {"method": REF_MAIN[0], "steps": REF_MAIN[1],
                                           "nfe": REF_MAIN[1] * solvers.NFE_PER_STEP[REF_MAIN[0]]},
                        "reference_check": {"method": REF_CHECK[0], "steps": REF_CHECK[1],
                                            "nfe": REF_CHECK[1] * solvers.NFE_PER_STEP[REF_CHECK[0]],
                                            "note": "independent scheme at equal NFE, not a finer RK4"},
                        "nfe_per_step": solvers.NFE_PER_STEP}}
    with open("experiments/loss_balance_audit/solver_study.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote experiments/loss_balance_audit/solver_study.json")


if __name__ == "__main__":
    main()
