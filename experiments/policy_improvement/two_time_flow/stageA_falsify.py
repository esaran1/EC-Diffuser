"""Stage A falsification suite for the action-side interior minimum.

  §2 pre-FT vs post-FT 5x5 surfaces (is the U-shape learned or architectural?)
  §3 dense boundary strip near t_state=1 (endpoint artifact?)
  §4 reconstructed clean-action error  x1_hat = x_t + (1-t) v_hat
  §5 larger sample + bootstrap CIs, sign consistency across batches
  §7 conditioning-geometry audit (is the U-shape a scaling artifact?)
"""
import argparse, json, os, pickle, sys

import isaacgym  # noqa: F401  MUST precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
sys.path.insert(0, "experiments/scripts")
sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
from isaacgym_control import Args  # noqa: E402
from twotime_denoiser import TwoTimePINTDenoiser  # noqa: E402
from twotime_flow import TwoTimeConditionalFlowMatching  # noqa: E402

OUT = "experiments/policy_improvement/two_time_flow/stageA"
CANON = "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
FT = "experiments/policy_improvement/two_time_flow/stage1/ft_state.pt"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
STRIP = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
STRIP_TA = [0.25, 0.5, 0.75]


def build(a):
    exp = utils.load_diffusion("data", a.dataset, CANON, epoch="latest",
        seed=a.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    base = exp.diffusion
    mc = pickle.load(open(f"data/panda_push/{CANON}/model_config.pkl", "rb"))
    md = dict(mc._dict) if hasattr(mc, "_dict") else dict(mc)
    kw = {k: v for k, v in md.items() if k != "device"}
    net = TwoTimePINTDenoiser(**kw).to(a.device)
    net.load_state_dict(base.model.state_dict(), strict=True)
    tt = TwoTimeConditionalFlowMatching(
        net, horizon=base.horizon, observation_dim=base.observation_dim,
        action_dim=base.action_dim, loss_type=base.loss_type,
        action_weight=float(base.loss_weight_matrix[0, 0]), loss_discount=1,
        loss_weights=None, time_scale=base.time_scale, p_diag=0.5).to(a.device)
    return exp, base, tt


@torch.no_grad()
def cell(model, dl_batches, dev, ad, t_s, t_a, seed=99):
    """Per-batch velocity error AND reconstructed-action error at one cell."""
    va, vs, ra = [], [], []
    for i, b in enumerate(dl_batches):
        x1 = b["traj"].to(dev)
        c = {k: v.to(dev) for k, v in b["cond"].items()}
        torch.manual_seed(seed + i)
        x0 = torch.randn_like(x1)
        x1l = x1.clone()
        model._apply_conditioning(x1l, c); model._apply_conditioning(x0, c)
        B = x1.shape[0]
        ts = torch.full((B,), float(t_s), device=dev, dtype=x1.dtype)
        ta = torch.full((B,), float(t_a), device=dev, dtype=x1.dtype)
        xt = model.corrupt(x1l, x0, ts, ta); model._apply_conditioning(xt, c)
        pred = model._call_two_time_model(xt, c, ts, ta)
        targ = x1l - x0
        m = model._make_conditioning_mask(xt, c)
        err = (pred - targ).abs()
        am, om = m[:, :, :ad], m[:, :, ad:]
        va.append(float((err[:, :, :ad] * am).sum() / am.sum().clamp(min=1)))
        vs.append(float((err[:, :, ad:] * om).sum() / om.sum().clamp(min=1)))
        # §4 reconstruction: x1_hat = x_t + (1-t)*v_hat, degenerate at t=1
        if t_a < 1.0:
            x1h = xt[:, :, :ad] + (1.0 - float(t_a)) * pred[:, :, :ad]
            rerr = (x1h - x1l[:, :, :ad]).abs()
            ra.append(float((rerr * am).sum() / am.sum().clamp(min=1)))
    return va, vs, (ra if ra else None)


def boot(v, n=4000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2: return float(v.mean()), None, None
    r = np.random.default_rng(seed)
    m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=40)   # 40*8 = 320 samples/cell
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    a = Args(); utils.set_global_device(a.device)
    exp, base, tt = build(a)
    dev, ad, ds = a.device, base.action_dim, exp.dataset

    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batches, it = [], iter(dl)
    for _ in range(cli.batches):
        b = next(it)
        batches.append({"traj": b.trajectories, "cond": dict(b.conditions)})
    print(f"[data] {len(batches)} batches x 8 = {len(batches)*8} samples per cell", flush=True)

    res = {"n_samples_per_cell": len(batches) * 8}

    # ---------- §7 conditioning geometry (no data needed) ----------
    geo = {}
    with torch.no_grad():
        for ta_fix in (0.25, 0.5, 0.75):
            rows = []
            e_a = tt.model.time_mlp(torch.full((1,), ta_fix * tt.time_scale, device=dev))
            for ts in GRID + [0.9, 0.95, 0.99]:
                e_s = tt.model.time_mlp(torch.full((1,), ts * tt.time_scale, device=dev))
                w = tt.model.adaln_state_weight
                comb = w * e_s + (1 - w) * e_a
                e_diag = tt.model.time_mlp(torch.full((1,), ta_fix * tt.time_scale, device=dev))
                rows.append(dict(t_state=ts,
                    norm_e_state=float(e_s.norm()), norm_e_action=float(e_a.norm()),
                    norm_combined=float(comb.norm()),
                    cos_to_diagonal=float(torch.nn.functional.cosine_similarity(comb, e_diag, dim=1))))
            geo[f"t_action={ta_fix}"] = rows
    res["conditioning_geometry"] = geo
    print("[geo] conditioning-geometry audit done", flush=True)

    # ---------- §2 PRE-FT surface ----------
    pre = {}
    for ts in GRID:
        for ta in GRID:
            va, vs, ra = cell(tt, batches, dev, ad, ts, ta)
            pre[f"{ts}_{ta}"] = dict(t_state=ts, t_action=ta,
                action_vel=float(np.mean(va)), state_vel=float(np.mean(vs)),
                action_recon=(float(np.mean(ra)) if ra else None),
                action_vel_batches=va)
        print(f"[pre-FT] row t_s={ts} done", flush=True)
    res["pre_ft_surface"] = pre

    # ---------- load fine-tuned weights, POST-FT surface ----------
    if os.path.exists(FT):
        sd = torch.load(FT, map_location=dev)
        tt.model.load_state_dict(sd["model"], strict=True)
        post = {}
        for ts in GRID:
            for ta in GRID:
                va, vs, ra = cell(tt, batches, dev, ad, ts, ta)
                post[f"{ts}_{ta}"] = dict(t_state=ts, t_action=ta,
                    action_vel=float(np.mean(va)), state_vel=float(np.mean(vs)),
                    action_recon=(float(np.mean(ra)) if ra else None),
                    action_vel_batches=va)
            print(f"[post-FT] row t_s={ts} done", flush=True)
        res["post_ft_surface"] = post

        # ---------- §3 boundary strip (post-FT) ----------
        strip = {}
        for ta in STRIP_TA:
            for ts in STRIP:
                va, vs, ra = cell(tt, batches, dev, ad, ts, ta)
                m, lo, hi = boot(va)
                strip[f"{ts}_{ta}"] = dict(t_state=ts, t_action=ta,
                    action_vel=m, ci_lo=lo, ci_hi=hi,
                    action_recon=(float(np.mean(ra)) if ra else None))
            print(f"[strip] t_a={ta} done", flush=True)
        res["boundary_strip"] = strip
    else:
        res["post_ft_surface"] = None
        print(f"[warn] no fine-tuned weights at {FT}", flush=True)

    json.dump(res, open(f"{OUT}/stageA.json", "w"), indent=1)
    print("wrote", f"{OUT}/stageA.json")


if __name__ == "__main__":
    main()
