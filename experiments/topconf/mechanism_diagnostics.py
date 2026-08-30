"""Why does NFE2 beat NFE1?  CPU/GPU diagnostic, NO training, NO simulator.

Uses the arm-neutral recorded replay protocol (recorded current/goal/next-state/
action; nothing model-authored). Decomposes the second Euler step into:

  A full      : x1 = x0 + h*v(x0);  x2 = x1 + h*v(x1)          (canonical NFE2)
  B act-only  : second step updates ONLY action coords
  C state-only: second step updates ONLY state coords
  D frozen-s  : state frozen at step-1 value, action recomputed from that state

Question: is the NFE1->NFE2 action improvement driven by action self-refinement
(B) or by state refinement feeding back through joint attention (C/D)?

These variants are DIAGNOSTIC PROBES, not valid generative processes.
"""
import argparse, json, pickle, sys
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
import solvers  # noqa: F401  (shared conditioning discipline reference)
import diffuser.utils as utils

DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
NPV = 24; HORIZON = 5; N_COND = 96; N_NOISE = 4
SAMPLE_SEED = 20260902; NOISE_SEED = 20260901
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}


def build_sample_set(sf):
    rng = np.random.default_rng(SAMPLE_SEED)
    ok = np.flatnonzero(np.asarray(sf) >= 1.0)
    eps = rng.choice(ok, N_COND, replace=False)
    ts = rng.integers(0, 100 - HORIZON, size=N_COND)
    return eps, ts


def main():
    raw = pickle.load(open(DATA, "rb"))
    obs, acts, goals, sf = raw["observations"], raw["actions"], raw["goals"], raw["info_goal_success_frac"]
    eps, ts = build_sample_set(sf)
    utils.set_global_device("cuda:0")
    out = {"protocol": {"n_cond": N_COND, "n_noise": N_NOISE, "sample_seed": SAMPLE_SEED,
                        "noise_seed": NOISE_SEED,
                        "targets": "recorded replay (arm-neutral); no model-authored targets",
                        "variants": {"A": "full 2-step Euler", "B": "2nd step updates action only",
                                     "C": "2nd step updates state only",
                                     "D": "state frozen at step1, action recomputed"}},
           "per_seed": {}}

    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", "panda_push", path, epoch="latest", seed=42,
                                   is_diffusion=True, override_dataset_path=DATA)
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        AD = model.action_dim
        cur, t1, gl, ta = obs[eps, ts], obs[eps, ts + 1], goals[eps, 99], acts[eps, ts]
        cond = {0: torch.as_tensor(norm.normalize(cur.reshape(N_COND, -1), "observations"),
                                   device="cuda:0", dtype=torch.float32),
                4: torch.as_tensor(norm.normalize(gl.reshape(N_COND, -1), "observations"),
                                   device="cuda:0", dtype=torch.float32)}
        gen = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
        acc = {k: [] for k in ["a1", "aA", "aB", "aC", "aD", "res_a", "res_s", "s1", "sA"]}
        with torch.no_grad():
            for _ in range(N_NOISE):
                z = torch.randn((N_COND, HORIZON, model.transition_dim), generator=gen).to("cuda:0")
                x = z.clone(); model._apply_conditioning(x, cond)
                cm = model._make_conditioning_mask(x, cond)

                # ---- step 1 (shared by everything) ----
                v0 = model.model(x, cond, x.new_zeros(N_COND) * model.time_scale) * cm.to(x.dtype)
                x1_h1 = x + 1.0 * v0                      # NFE1 endpoint (h=1)
                model._apply_conditioning(x1_h1, cond)
                x1 = x + 0.5 * v0                         # first half-step of NFE2
                model._apply_conditioning(x1, cond)

                # ---- step 2 variants (h=0.5, t=0.5) ----
                t_half = x.new_full((N_COND,), 0.5) * model.time_scale
                v1 = model.model(x1, cond, t_half) * cm.to(x1.dtype)

                xA = x1 + 0.5 * v1; model._apply_conditioning(xA, cond)          # full
                xB = x1.clone(); xB[:, :, :AD] = x1[:, :, :AD] + 0.5 * v1[:, :, :AD]
                model._apply_conditioning(xB, cond)                               # action only
                xC = x1.clone(); xC[:, :, AD:] = x1[:, :, AD:] + 0.5 * v1[:, :, AD:]
                model._apply_conditioning(xC, cond)                               # state only
                # D: take C's refined state, recompute the action from it (extra probe call)
                vD = model.model(xC, cond, t_half) * cm.to(xC.dtype)
                xD = xC.clone(); xD[:, :, :AD] = xC[:, :, :AD] + 0.5 * vD[:, :, :AD]
                model._apply_conditioning(xD, cond)

                def act(t): return norm.unnormalize(utils.to_np(t)[:, :, :AD], "actions")[:, 0]
                def st(t):
                    o = norm.unnormalize(utils.to_np(t)[:, :, AD:], "observations")
                    return o[:, 1].reshape(N_COND, 2, NPV, 10)[:, 0]
                acc["a1"].append(act(x1_h1)); acc["aA"].append(act(xA))
                acc["aB"].append(act(xB)); acc["aC"].append(act(xC)); acc["aD"].append(act(xD))
                acc["s1"].append(st(x1_h1)); acc["sA"].append(st(xA))
                # projected 1-step vs 2-step residuals in MODEL space
                d = (xA - x1_h1)
                acc["res_a"].append(utils.to_np(d[:, :, :AD]).reshape(N_COND, -1))
                acc["res_s"].append(utils.to_np(d[:, :, AD:]).reshape(N_COND, -1))

        A = {k: np.concatenate(v, 0) for k, v in acc.items()}
        tgt = np.repeat(ta, N_NOISE, axis=0) if False else np.tile(ta, (N_NOISE, 1))
        err = lambda a: float(np.linalg.norm(a - tgt, axis=-1).mean())
        # state error vs recorded next state
        from latent_metric import chamfer_position
        T1 = np.tile(t1[:, :NPV], (N_NOISE, 1, 1))
        serr = lambda s: float(np.mean([chamfer_position(s[i], T1[i]) for i in range(len(s))]))
        r = {"action_err": {"NFE1": err(A["a1"]), "A_full2": err(A["aA"]),
                            "B_action_only": err(A["aB"]), "C_state_only": err(A["aC"]),
                            "D_state_then_action": err(A["aD"])},
             "state_err": {"NFE1": serr(A["s1"]), "A_full2": serr(A["sA"])},
             "residual_norm": {"action": float(np.linalg.norm(A["res_a"], axis=1).mean()),
                               "state": float(np.linalg.norm(A["res_s"], axis=1).mean())},
             "action_move_1to2": float(np.linalg.norm(A["aA"] - A["a1"], axis=-1).mean())}
        out["per_seed"][str(seed)] = r
        print(f"\n=== seed {seed} ===")
        print(f"  action err  NFE1={r['action_err']['NFE1']:.5f}  full2={r['action_err']['A_full2']:.5f}"
              f"  B(act-only)={r['action_err']['B_action_only']:.5f}"
              f"  C(state-only)={r['action_err']['C_state_only']:.5f}"
              f"  D(state->act)={r['action_err']['D_state_then_action']:.5f}")
        print(f"  state err   NFE1={r['state_err']['NFE1']:.5f}  full2={r['state_err']['A_full2']:.5f}")
        print(f"  |1step-2step| action={r['residual_norm']['action']:.4f} state={r['residual_norm']['state']:.4f}")

    json.dump(out, open("experiments/topconf/mechanism_diagnostics.json", "w"), indent=2)
    print("\nwrote mechanism_diagnostics.json")


if __name__ == "__main__":
    sys.path.insert(0, "experiments/loss_balance_audit")
    main()
