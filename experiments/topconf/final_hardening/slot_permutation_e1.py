"""E1: does scenario->GPU-slot assignment change fixed-policy outcomes?

ONLY intended change: which env slot executes scenario i.

Policy noise is made SCENARIO-KEYED (not slot-keyed). The canonical sampler draws
one (batch, H, D) tensor and row k goes to slot k, so a naive permutation would
change slot AND noise together. Here we draw the batch noise in SCENARIO order
under the same CRN seed, then scatter rows to the slots those scenarios occupy.
The marginal distribution is unchanged; only the coupling to slots changes.
"""
import argparse, hashlib, json, os, sys, time
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts"); sys.path.insert(0, "experiments/evaluation_noise")
from crn import CRN_BASE_SEED, derive_seed
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import (ARMS, Args, array_to_state_dict, entity_positions,
                              summarize_episode)
import isaacgym_nfe_study as NS
import pickle

OUT = "experiments/topconf/final_hardening/e1_results"
PERM_SEED = 20260915
N_PERMS = 2          # non-identity; plus identity = 3 runs (frozen design)


def make_perms(n_envs, k):
    """Frozen, predeclared within-batch slot permutations."""
    rng = np.random.default_rng(PERM_SEED)
    out = [("identity", np.arange(n_envs))]
    for j in range(k):
        p = rng.permutation(n_envs)
        while np.array_equal(p, np.arange(n_envs)):
            p = rng.permutation(n_envs)
        out.append((f"perm{j+1}", p))
    return out


def euler_scenario_keyed(model, cond, z, nfe):
    """Canonical Euler loop with externally supplied noise (bit-identical form)."""
    x = z.clone(); model._apply_conditioning(x, cond)
    cm = model._make_conditioning_mask(x, cond)
    for k in range(nfe):
        t = x.new_full((x.shape[0],), k / nfe)
        x = x + (1.0 / nfe) * (model.model(x, cond, t * model.time_scale) * cm.to(x.dtype))
        model._apply_conditioning(x, cond)
    return x


def run(perm, env, ep, args, model, norm, nfe, log_phys=False):
    """perm[slot] = position-within-batch of the scenario placed in that slot."""
    keys = ep["keys"]; inits, goals = ep["init"], ep["goal"]
    n_ep = len(inits); nE = env.num_envs
    recs = {}; phys = []
    for bs in range(0, n_ep, nE):
        base = list(range(bs, min(bs + nE, n_ep)))
        n_act = len(base); base += [base[-1]] * (nE - n_act)
        base = np.array(base)
        # slot s executes scenario base[perm[s]]
        slot_to_scn = base[perm]
        torch.manual_seed(CRN_BASE_SEED + bs); np.random.seed(CRN_BASE_SEED + bs)
        obs = env.reset(set_init_states=array_to_state_dict(inits[slot_to_scn], keys, env.device),
                        set_goal_states=array_to_state_dict(goals[slot_to_scn], keys, env.device))
        al = {int(s): [] for s in slot_to_scn}; el = {int(s): [] for s in slot_to_scn}
        cl = {int(s): [] for s in slot_to_scn}; info_last = None
        for step in range(env.horizon):
            o = obs["achieved_goal"].reshape(nE, -1); g = obs["desired_goal"].reshape(nE, -1)
            cond = {0: torch.as_tensor(norm.normalize(o, "observations"), device=env.device, dtype=torch.float32),
                    args.horizon - 1: torch.as_tensor(norm.normalize(g, "observations"), device=env.device, dtype=torch.float32)}
            # --- SCENARIO-KEYED noise: draw in scenario order, scatter to slots ---
            s = derive_seed(CRN_BASE_SEED, bs, step)
            torch.manual_seed(s); torch.cuda.manual_seed_all(s)
            z_scn = torch.randn((nE, args.horizon, model.transition_dim), device=env.device)
            z = torch.empty_like(z_scn)
            z[np.arange(nE)] = z_scn[perm]          # slot s gets the row of its scenario
            with torch.no_grad():
                x = euler_scenario_keyed(model, cond, z, nfe)
            a = norm.unnormalize(utils.to_np(x)[:, :, :model.action_dim], "actions")[:, 0]
            obs, _, _, infos = env.step(a); info_last = infos
            st = entity_positions(env)
            for slot in range(nE):
                scn = int(slot_to_scn[slot])
                al[scn].append(np.asarray(a[slot], dtype=np.float64))
                el[scn].append(st[slot, 0, :3].copy()); cl[scn].append(st[slot, 1:, :2].copy())
            if log_phys and step in (0, 10, 50, 99):
                phys.append({"batch": bs, "step": step,
                             "cubes": st[:, 1:, :2].copy().tolist(),
                             "slot_to_scn": slot_to_scn.tolist()})
        gs = np.asarray(env.goal_pos)
        assert n_act == nE, f"partial batch unsupported in E1 (n_act={n_act}, nE={nE})"
        for slot in range(nE):
            scn = int(slot_to_scn[slot])
            assert scn not in recs, f"duplicate scenario {scn} - collation bug"
            recs[scn] = summarize_episode(episode=scn, actions=np.array(al[scn]),
                          eef=np.array(el[scn]), cubes=np.array(cl[scn]),
                          goal_cubes=gs[slot], info=info_last[slot], threshold=0.04)
    return [recs[k] for k in sorted(recs)], phys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--only", default=None)
    cli = ap.parse_args()
    args = Args(); utils.set_global_device(args.device)
    os.makedirs(OUT, exist_ok=True)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep, _ = NS.get_episode_set(env, 0, cli.episodes)
    ARMS["flow"]["loadpath"] = f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{cli.seed}"
    policy = NS.build_policy("flow", cli.nfe, args)
    model = policy.diffusion_model; norm = policy.normalizer
    # ---- section 7 provenance assertions ----
    import hashlib as _h
    ck = f"data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{cli.seed}/state_400000.pt"
    ck_sha = _h.sha256(open(ck, "rb").read()).hexdigest()
    EXPECT = {42: "861dc34434474455a25dc3a15ea4e1754066202df538364cf41114b42f4fcc3b"}
    assert ck_sha == EXPECT[cli.seed], f"checkpoint hash mismatch: {ck_sha[:16]}"
    assert ep["sha256"] == "35144910b1471b7b0d50d17da18b01db0b5e61e21d7e16e4ef1aa266ee80d511", "scenario-set hash mismatch"
    assert env.num_objects == 3, f"num_objects={env.num_objects}"
    assert env.horizon == 100, f"horizon={env.horizon}"
    assert env.num_envs == 16, f"num_envs={env.num_envs}"
    assert len(ep["init"]) == cli.episodes == 96, "episode count"
    print(f"[provenance] ckpt {ck_sha[:16]} set {ep['sha256'][:16]} "
          f"objs={env.num_objects} H={env.horizon} num_envs={env.num_envs} n={len(ep['init'])}", flush=True)

    perms = make_perms(env.num_envs, N_PERMS)
    meta = {"perm_seed": PERM_SEED, "n_envs": int(env.num_envs), "nfe": cli.nfe,
            "training_seed": cli.seed, "episode_set_sha": ep["sha256"],
            "permutations": {n: p.tolist() for n, p in perms},
            "perm_hashes": {n: hashlib.sha256(str(p.tolist()).encode()).hexdigest()[:16]
                            for n, p in perms}}
    json.dump(meta, open(f"{OUT}/e1_design.json", "w"), indent=2)
    print(json.dumps(meta["perm_hashes"], indent=1), flush=True)
    for name, p in perms:
        if cli.only and name != cli.only: continue
        f = f"{OUT}/e1_{name}.json"
        if os.path.exists(f):
            print(f"[skip] {f}", flush=True); continue
        calls = [0]
        hk = model.model.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
        t0 = time.time()
        recs, phys = run(p, env, ep, args, model, norm, cli.nfe, log_phys=(name != "identity"))
        el = time.time() - t0
        hk.remove()
        n_plans = 6 * env.horizon           # 6 batches x horizon decisions
        cpp = calls[0] / n_plans
        assert abs(cpp - cli.nfe) < 1e-9, f"calls/plan {cpp} != NFE {cli.nfe}"
        assert len(recs) == 96, f"expected 96 scenarios, got {len(recs)}"
        assert sorted(r["episode"] for r in recs) == list(range(96)), "scenario id mismatch"
        succ = float(np.mean([r["success"] for r in recs]))
        json.dump({"permutation": name, "perm": p.tolist(), "success_rate": succ,
                   "calls_per_plan": cpp, "checkpoint_sha256": ck_sha,
                   "episode_set_sha256": ep["sha256"],
                   "perm_sha256": _h.sha256(str(p.tolist()).encode()).hexdigest(),
                   "n": len(recs), "wall_s": el, "episodes": recs, "phys": phys[:8]},
                  open(f, "w"), indent=2)
        print(f"  -> {name}: success={succ:.4f} n={len(recs)} ({el:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
