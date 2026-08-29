"""NO-TRAINING solver/reference audit. Raw model space only -- no DLP decode,
no Hungarian, no Chamfer for any convergence number.

Phases: raw-space convergence (2), conditioning audit (3/4), time audit (5),
local regularity (7), RK intermediate OOD (8), frozen-field control (9).
"""
import json, pickle, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit"); sys.path.insert(0, "experiments/scripts")
import solvers
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

NSUB = 8  # small fixed subset for the expensive high-NFE ladders


class Args:
    env_config_dir="env_config/generalization_num_cubes"; dataset="panda_push"
    num_entity=3; horizon=5; max_episode_length=100; planning_only=True
    push_t=False; multiview=True; verbose=False; seed=42; device="cuda:0"
    preprocess_fns=[]; push_t_num_color=1


def free_mask(model, x, cond):
    """True on UNCONDITIONED coordinates only."""
    return model._make_conditioning_mask(x, cond)


def norms(a, b, m):
    """mean-abs / RMS / max-abs over masked coords, per sample then averaged."""
    d = (a - b).abs() * m.to(a.dtype)
    n = m.sum(dim=(1, 2)).to(a.dtype)
    mean = (d.sum(dim=(1, 2)) / n).mean().item()
    rms = ((d.square().sum(dim=(1, 2)) / n).sqrt()).mean().item()
    mx = d.flatten(1).max(dim=1)[0].mean().item()
    return {"mean_abs": mean, "rms": rms, "max_abs": mx}


def split_masks(model, x, cond):
    m = free_mask(model, x, cond)
    ma = torch.zeros_like(m); ma[:, :, :model.action_dim] = True; ma &= m
    mo = torch.zeros_like(m); mo[:, :, model.action_dim:] = True; mo &= m
    return m, ma, mo


def main():
    out = {}
    args = Args(); utils.set_global_device(args.device)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]
    exp = utils.load_diffusion("data", args.dataset,
        "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42", epoch="latest", seed=42,
        is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                 "panda_push_replay_buffer_dlp.pkl")
    model = exp.ema; model.eval(); norm = exp.dataset.normalizer

    idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
    obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                    set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
    o = obs["achieved_goal"].reshape(env.num_envs, -1)
    g = obs["desired_goal"].reshape(env.num_envs, -1)
    cond = {0: torch.as_tensor(norm.normalize(o, "observations"), device=env.device, dtype=torch.float32),
            4: torch.as_tensor(norm.normalize(g, "observations"), device=env.device, dtype=torch.float32)}
    cond = {k: v[:NSUB] for k, v in cond.items()}
    x0 = torch.randn((NSUB, 5, model.transition_dim),
                     generator=torch.Generator(device="cpu").manual_seed(777)).to(env.device)

    m_all, m_act, m_obs = split_masks(model, x0, cond)
    out["masks"] = {"free_coords": int(m_all.sum()), "action_free": int(m_act.sum()),
                    "obs_free": int(m_obs.sum()), "total": int(m_all.numel()),
                    "conditioned": int((~m_all).sum())}
    print("[masks]", out["masks"], flush=True)

    with torch.no_grad():
        # ---------- PHASE 3/4: conditioning idempotence + free-only equivalence ----------
        # Variant B: integrate ONLY free coords, holding conditioned coords analytically fixed.
        def euler_freeonly(steps):
            x = x0.clone(); model._apply_conditioning(x, cond)
            frozen = x.clone()
            dt = 1.0 / steps
            for k in range(steps):
                t = x.new_full((x.shape[0],), k * dt)
                v = model.model(x, cond, t * model.time_scale)
                x = torch.where(m_all, x + dt * v, frozen)   # analytic hold
            return x
        a = solvers.integrate(model, cond, x0, "euler", 16)[0]
        b = euler_freeonly(16)
        out["conditioning_variants"] = {
            "A_projected_vs_B_analytic_hold_maxabs": float((a - b).abs().max()),
            "conditioned_coords_move_under_A": float(
                (a * (~m_all).to(a.dtype) - x0 * (~m_all).to(a.dtype)).abs().max()
                if False else (solvers.integrate(model, cond, x0, "euler", 16)[0][~m_all]
                               - model._apply_conditioning(x0.clone(), cond)[~m_all]).abs().max())}
        print("[cond variants]", out["conditioning_variants"], flush=True)

        # ---------- PHASE 5: time variable audit ----------
        ta = {}
        for meth, st in [("euler", 4), ("midpoint", 2), ("heun", 2), ("rk4", 1)]:
            ts, dt = [], 1.0 / st
            for k in range(st):
                t = k * dt
                if meth == "euler": ts += [t]
                elif meth == "midpoint": ts += [t, t + 0.5 * dt]
                elif meth == "heun": ts += [t, t + dt]
                elif meth == "rk4": ts += [t, t + 0.5 * dt, t + 0.5 * dt, t + dt]
            ta[f"{meth}@{st}"] = {"t_values": ts, "dt": dt,
                                  "scaled": [t * model.time_scale for t in ts]}
        # is the embedding quantized? feed nearby t and compare
        emb = model.model.time_mlp[0]
        probe = torch.tensor([0.5, 0.5 + 1e-7, 0.5 + 1e-6, 0.5 + 1e-5], device=env.device) * model.time_scale
        e = emb(probe)
        ta["embedding_sensitivity_vs_t=0.5"] = [float((e[i] - e[0]).abs().max()) for i in range(4)]
        ta["time_scale"] = float(model.time_scale)
        ta["max_embed_frequency_per_unit_t"] = float(model.time_scale * 1.0)
        ta["fastest_embed_period_in_t"] = float(2 * np.pi / model.time_scale)
        out["time_audit"] = ta
        print("[time] fastest period in t =", ta["fastest_embed_period_in_t"], flush=True)

        # ---------- PHASE 2: RAW-SPACE CONVERGENCE ----------
        conv = {}
        ladders = {"euler": [16, 32, 64, 128, 256, 512],
                   "midpoint": [8, 16, 32, 64, 128, 256],
                   "rk4": [4, 8, 16, 32, 64, 128]}
        endpoints = {}
        for meth, steps_list in ladders.items():
            endpoints[meth] = {}
            for st in steps_list:
                xe, nfe, _ = solvers.integrate(model, cond, x0, meth, st)
                endpoints[meth][nfe] = xe
                print(f"  [conv] {meth}@{st} NFE={nfe}", flush=True)
        for meth in ladders:
            ks = sorted(endpoints[meth])
            conv[meth] = []
            for i in range(1, len(ks)):
                a_, b_ = endpoints[meth][ks[i]], endpoints[meth][ks[i - 1]]
                conv[meth].append({"nfe": ks[i], "prev_nfe": ks[i - 1],
                                   "all_free": norms(a_, b_, m_all),
                                   "action": norms(a_, b_, m_act),
                                   "observation": norms(a_, b_, m_obs)})
        # cross-scheme agreement at the finest affordable settings
        cross = {}
        for p, q in [("euler", "midpoint"), ("euler", "rk4"), ("midpoint", "rk4")]:
            kp, kq = max(endpoints[p]), max(endpoints[q])
            cross[f"{p}@{kp}NFE_vs_{q}@{kq}NFE"] = {
                "all_free": norms(endpoints[p][kp], endpoints[q][kq], m_all),
                "action": norms(endpoints[p][kp], endpoints[q][kq], m_act),
                "observation": norms(endpoints[p][kp], endpoints[q][kq], m_obs)}
        out["raw_convergence"] = conv; out["cross_scheme"] = cross

        # ---------- PHASE 7: local regularity ----------
        reg = {"state": {}, "time": {}}
        x = x0.clone(); model._apply_conditioning(x, cond)
        dt = 1 / 16
        for k in range(8):  # walk to mid-trajectory
            t = x.new_full((x.shape[0],), k * dt)
            v = model.model(x, cond, t * model.time_scale) * m_all.to(x.dtype)
            x = x + dt * v; model._apply_conditioning(x, cond)
        t_mid = x.new_full((x.shape[0],), 0.5)
        v0 = model.model(x, cond, t_mid * model.time_scale)
        gen = torch.Generator(device="cpu").manual_seed(11)
        d = torch.randn(x.shape, generator=gen).to(x.device) * m_all.to(x.dtype)
        d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1)
        for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
            xp = x + eps * d; model._apply_conditioning(xp, cond)
            vp = model.model(xp, cond, t_mid * model.time_scale)
            reg["state"][str(eps)] = float(((vp - v0).flatten(1).norm(dim=1)
                                            / (eps * d.flatten(1).norm(dim=1))).mean())
        for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
            vt = model.model(x, cond, (t_mid + eps) * model.time_scale)
            reg["time"][str(eps)] = float(((vt - v0).flatten(1).norm(dim=1) / eps).mean())
        out["local_regularity"] = reg
        print("[regularity] state", reg["state"], "\n             time", reg["time"], flush=True)

        # ---------- PHASE 8: RK intermediate-state OOD ----------
        ref_path = {}
        x = x0.clone(); model._apply_conditioning(x, cond)
        for k in range(16):
            t = x.new_full((x.shape[0],), k / 16)
            ref_path[k / 16] = x.clone()
            v = model.model(x, cond, t * model.time_scale) * m_all.to(x.dtype)
            x = x + (1 / 16) * v; model._apply_conditioning(x, cond)
        ref_path[1.0] = x.clone()

        def nearest_ref(state, t):
            kk = min(ref_path, key=lambda z: abs(z - t))
            return float((state - ref_path[kk]).flatten(1).norm(dim=1).mean())

        ood = {}
        for meth, st in [("euler", 4), ("midpoint", 2), ("heun", 2), ("rk4", 1)]:
            recs = []
            xx = x0.clone(); model._apply_conditioning(xx, cond)
            dtl = 1.0 / st
            for k in range(st):
                t = k * dtl
                def ev(state, tt, lab):
                    vv = model.model(state, cond, state.new_full((state.shape[0],), tt) * model.time_scale)
                    recs.append({"stage": lab, "t": tt,
                                 "state_norm": float(state.flatten(1).norm(dim=1).mean()),
                                 "dist_to_euler16_path": nearest_ref(state, tt),
                                 "abs_max_feature": float(state.abs().max()),
                                 "frac_outside_pm1": float((state.abs() > 1.0).float().mean()),
                                 "velocity_norm": float(vv.flatten(1).norm(dim=1).mean())})
                    return vv * m_all.to(state.dtype)
                if meth == "euler":
                    v1 = ev(xx, t, "k1"); xx = xx + dtl * v1
                elif meth == "midpoint":
                    v1 = ev(xx, t, "k1"); xm = xx + 0.5 * dtl * v1; model._apply_conditioning(xm, cond)
                    v2 = ev(xm, t + 0.5 * dtl, "k2"); xx = xx + dtl * v2
                elif meth == "heun":
                    v1 = ev(xx, t, "k1"); xe = xx + dtl * v1; model._apply_conditioning(xe, cond)
                    v2 = ev(xe, t + dtl, "k2"); xx = xx + dtl * 0.5 * (v1 + v2)
                elif meth == "rk4":
                    k1 = ev(xx, t, "k1"); xa = xx + .5 * dtl * k1; model._apply_conditioning(xa, cond)
                    k2 = ev(xa, t + .5 * dtl, "k2"); xb = xx + .5 * dtl * k2; model._apply_conditioning(xb, cond)
                    k3 = ev(xb, t + .5 * dtl, "k3"); xc = xx + dtl * k3; model._apply_conditioning(xc, cond)
                    k4 = ev(xc, t + dtl, "k4"); xx = xx + dtl * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                model._apply_conditioning(xx, cond)
            ood[f"{meth}@{st}"] = recs
        out["rk_intermediate_ood"] = ood

    with open("experiments/loss_balance_audit/solver_reference_audit.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote solver_reference_audit.json")


if __name__ == "__main__":
    main()
