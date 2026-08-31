"""Phase I core: policy disagreement as a function of WHICH state distribution
the query states come from.

D_demo  = E[||pi_1(s) - pi_2(s)||]  for s ~ recorded replay
D_self1 = same, for s ~ states visited by the NFE1 policy
D_self2 = same, for s ~ states visited by the NFE2 policy
D_self4 = same, for s ~ states visited by the NFE4 policy

Matched Flow noise across policies at every query (CRN). No simulator stepping.
No demo-action targets at self-induced states (protocol section 4): this measures
policy SENSITIVITY TO NFE, never correctness.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
import diffuser.utils as utils
import pickle

OUT = "experiments/topconf/selfstate"
DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
NPV = 24; HORIZON = 5
QUERY_SEED = 20260910


def euler(model, cond, z, n):
    x = z.clone(); model._apply_conditioning(x, cond)
    cm = model._make_conditioning_mask(x, cond)
    for k in range(n):
        t = x.new_full((x.shape[0],), k / n)
        x = x + (1.0 / n) * (model.model(x, cond, t * model.time_scale) * cm.to(x.dtype))
        model._apply_conditioning(x, cond)
    return x


def query(model, norm, obs, goal, nfes, dev, bs=96):
    """Return {nfe: actions} at the given states, with SHARED noise per state."""
    out = {n: [] for n in nfes}
    g = torch.Generator(device="cpu").manual_seed(QUERY_SEED)
    for i in range(0, len(obs), bs):
        o = obs[i:i + bs]; gl = goal[i:i + bs]
        cond = {0: torch.as_tensor(norm.normalize(o, "observations"), device=dev, dtype=torch.float32),
                HORIZON - 1: torch.as_tensor(norm.normalize(gl, "observations"), device=dev, dtype=torch.float32)}
        z = torch.randn((len(o), HORIZON, model.transition_dim), generator=g).to(dev)
        with torch.no_grad():
            for n in nfes:
                x = euler(model, cond, z, n)     # SAME z for every nfe
                out[n].append(norm.unnormalize(utils.to_np(x)[:, :, :model.action_dim],
                                               "actions")[:, 0])
    return {n: np.concatenate(v, 0) for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-states", type=int, default=2000)
    cli = ap.parse_args()
    dev = "cuda:0"; utils.set_global_device(dev)
    exp = utils.load_diffusion("data", "panda_push",
        f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{cli.seed}", epoch="latest", seed=42,
        is_diffusion=True, override_dataset_path=DATA)
    model = exp.ema; model.eval(); norm = exp.dataset.normalizer
    rng = np.random.default_rng(QUERY_SEED)
    res = {"protocol": {"query_seed": QUERY_SEED, "shared_noise_across_nfe": True,
                        "note": "disagreement only; no demo targets at self-induced states"}}

    # ---- demo/replay state distribution ----
    raw = pickle.load(open(DATA, "rb"))
    o_all, g_all, sf = raw["observations"], raw["goals"], raw["info_goal_success_frac"]
    ok = np.flatnonzero(np.asarray(sf) >= 1.0)
    de = rng.choice(ok, cli.max_states // 2, replace=True)
    dt = rng.integers(0, 95, size=len(de))
    demo_obs = o_all[de, dt].reshape(len(de), -1)
    demo_goal = g_all[de, 99].reshape(len(de), -1)
    dists = {"demo": (demo_obs, demo_goal, None)}

    # ---- self-induced state distributions ----
    for n in cli.nfes:
        f = os.path.join(OUT, f"s{cli.seed}_nfe{n}_states.npz")
        if not os.path.exists(f):
            print(f"[warn] missing {f}"); continue
        Z = np.load(f, allow_pickle=True)
        m = len(Z["obs"])
        sel = rng.choice(m, min(cli.max_states, m), replace=False)
        dists[f"self{n}"] = (Z["obs"][sel], Z["goal"][sel], Z["step"][sel])

    print(f"{'distribution':>14s} {'n':>6s} " + " ".join(f"{'D'+str(a)+str(b):>9s}"
          for a, b in [(1,2),(1,4),(2,4)]))
    for name, (o, g, stp) in dists.items():
        A = query(model, norm, o, g, cli.nfes, dev)
        d = {}
        for a, b in [(1, 2), (1, 4), (2, 4)]:
            if a in A and b in A:
                d[f"{a}-{b}"] = float(np.linalg.norm(A[a] - A[b], axis=-1).mean())
        mag = float(np.linalg.norm(A[cli.nfes[0]], axis=-1).mean())
        res[name] = {"n_states": int(len(o)), "disagreement": d, "action_magnitude": mag}
        print(f"{name:>14s} {len(o):6d} " + " ".join(f"{d.get(f'{a}-{b}', float('nan')):9.5f}"
              for a, b in [(1,2),(1,4),(2,4)]))
        # stratify self-distributions by phase
        if stp is not None:
            for lab, msk in [("early(step<20)", stp < 20), ("mid(20-60)", (stp >= 20) & (stp < 60)),
                             ("late(>=60)", stp >= 60)]:
                if msk.sum() < 20: continue
                Am = {k: v[msk] for k, v in A.items()}
                dd = float(np.linalg.norm(Am[1] - Am[2], axis=-1).mean())
                res[name][f"D12_{lab}"] = dd
                print(f"{'  '+lab:>14s} {int(msk.sum()):6d} {dd:9.5f}")
    json.dump(res, open(f"experiments/topconf/selfstate_disagreement_s{cli.seed}.json", "w"), indent=2)
    print(f"\nwrote selfstate_disagreement_s{cli.seed}.json")


if __name__ == "__main__":
    main()
