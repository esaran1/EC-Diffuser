"""Offline sweet-spot search over state-leading paths (CPU).

SELECTION RULE (fixed before results): among alpha in {0.25,0.5,0.75},
eligible iff state_err <= 1.02 x diagonal state_err (UWM, NFE 4). The eligible
path with the lowest executed-action error goes to closed loop; a second only
if it ALSO beats the diagonal's executed-action error. If none is eligible,
the intermediate with the best action error among state_err <= 1.05x goes,
flagged.
"""
import json, sys, numpy as np, torch
sys.path.insert(0, "diffuser"); sys.path.insert(0, "experiments/policy_improvement/tal_followup/uwm")
from diffuser.utils.arrays import set_global_device; set_global_device("cpu")
from diffuser.utils.serialization import load_config
from two_time_sampler import two_time_sample, schedule_points
torch.set_num_threads(24)
R = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load(prefix):
    mc = load_config("data/panda_push/" + prefix, R, "model_config.pkl")
    dc = load_config("data/panda_push/" + prefix, R, "diffusion_config.pkl")
    m = mc._class(**mc._dict); w = dc._class(m, **dc._dict); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt" % (prefix, R),
                                               map_location="cpu")["ema"]); return w.eval()


dsc = load_config("data/panda_push/uwm_ctl", R, "dataset_config.pkl")
dsc._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
ds = dsc(seed=42)
idx = np.random.RandomState(20260930).choice(len(ds), 64, replace=False)
X1 = torch.stack([torch.as_tensor(ds[int(i)].trajectories) for i in idx]).float()
C = {k: torch.stack([torch.as_tensor(ds[int(i)].conditions[k]) for i in idx]).float()
     for k in ds[int(idx[0])].conditions}


@torch.no_grad()
def run(w, path, steps, seed):
    B = X1.shape[0]; ad = 3
    x1 = X1.clone(); w._apply_conditioning(x1, C); mask = w._make_conditioning_mask(x1, C)
    g = torch.Generator().manual_seed(seed); x0 = torch.randn(x1.shape, generator=g)
    def vel(x, ts, ta):
        return w._call_two_time_model(x, C, torch.full((B,), ts), torch.full((B,), ta))
    def cond(x):
        w._apply_conditioning(x, C); return x
    out = two_time_sample(vel, x0, ad, path=path, steps=steps, apply_cond=cond, cond_mask=mask)
    om = mask[:, :, ad:].float()
    ae = (out[:, :, :ad] - x1[:, :, :ad]).abs()
    se_t = [((out[:, t, ad:] - x1[:, t, ad:]).abs() * om[:, t]).sum() / om[:, t].sum()
            for t in range(x1.shape[1]) if om[:, t].sum() > 0]
    return {"act0": ae[:, 0].mean().item(), "act_all": ae.mean().item(),
            "state": (((out[:, :, ad:] - x1[:, :, ad:]).abs() * om).sum() / om.sum()).item(),
            "state_per_step": [s.item() for s in se_t]}


@torch.no_grad()
def oracle_inverse(w, steps, seed):
    """Deployment-UNAVAILABLE reference: state held at the clean target
    (t_state = 1), action integrated from noise. The 'fully inverse' corner."""
    B = X1.shape[0]; ad = 3
    x1 = X1.clone(); w._apply_conditioning(x1, C)
    g = torch.Generator().manual_seed(seed); x = x1.clone()
    x[..., :ad] = torch.randn(x1[..., :ad].shape, generator=g)
    for k in range(steps):
        ta = k / steps
        v = w._call_two_time_model(x, C, torch.full((B,), 1.0), torch.full((B,), ta))
        x[..., :ad] = x[..., :ad] + v[..., :ad] / steps
    return (x[:, 0, :ad] - x1[:, 0, :ad]).abs().mean().item()


res = {}
OUT = "experiments/policy_improvement/tal_followup/uwm/SWEET_SPOT.json"
models = {}
for steps, arm, prefix in ((4, "uwm10k", "uwm_uwm_ext"), (4, "ctl10k", "uwm_ctl_ext"),
                           (8, "uwm10k", "uwm_uwm_ext"), (8, "ctl10k", "uwm_ctl_ext")):
    if arm not in models:
        models[arm] = load(prefix)
    w = models[arm]; res.setdefault(arm, {})
    for a in ALPHAS:
        rs = [run(w, "lead%.2f" % a, steps, s) for s in (1, 2, 3)]
        r = {k: float(np.mean([x[k] for x in rs])) if k != "state_per_step"
             else list(np.mean([x[k] for x in rs], 0)) for k in rs[0]}
        res[arm]["nfe%d_a%.2f" % (steps, a)] = r
        print("%s NFE%d alpha %.2f  act@0 %.4f act_all %.4f state %.4f" % (
            arm, steps, a, r["act0"], r["act_all"], r["state"]), flush=True)
    res[arm]["nfe%d_oracle_inverse_act0" % steps] = float(np.mean([oracle_inverse(w, steps, s) for s in (1, 2, 3)]))
    print("%s NFE%d oracle-state inverse act@0 %.4f" % (arm, steps, res[arm]["nfe%d_oracle_inverse_act0" % steps]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

for arm in res:
    print("\n%s" % arm)
    for steps in (4, 8):
        d = res[arm]["nfe%d_a0.00" % steps]
        print("  NFE %d   alpha  act@0    act_all  state    state/diag  per-step state" % steps)
        for a in ALPHAS:
            r = res[arm]["nfe%d_a%.2f" % (steps, a)]
            print("          %.2f   %.4f   %.4f   %.4f   %.3f       %s" % (
                a, r["act0"], r["act_all"], r["state"], r["state"] / d["state"],
                " ".join("%.3f" % x for x in r["state_per_step"])))
        print("          oracle-state (fully inverse, NOT deployable) act@0 %.4f" % res[arm]["nfe%d_oracle_inverse_act0" % steps])

u = res["uwm10k"]; d = u["nfe4_a0.00"]
cand = [a for a in (0.25, 0.5, 0.75) if u["nfe4_a%.2f" % a]["state"] <= 1.02 * d["state"]]
flag = None
if not cand:
    cand = [a for a in (0.25, 0.5, 0.75) if u["nfe4_a%.2f" % a]["state"] <= 1.05 * d["state"]]; flag = "relaxed to 1.05x"
cand.sort(key=lambda a: u["nfe4_a%.2f" % a]["act0"])
chosen = cand[:1]
if len(cand) > 1 and u["nfe4_a%.2f" % cand[1]]["act0"] < d["act0"]:
    chosen.append(cand[1])
res["selection"] = {"eligible_sorted": cand, "chosen_for_closed_loop": chosen, "flag": flag,
                    "diag_act0": d["act0"], "diag_state": d["state"]}
json.dump(res, open(OUT, "w"), indent=1)
print("\nSELECTION:", res["selection"])
