"""Stage-1 two-time gate: smoke fine-tune from canonical Flow + interior probes.

Predeclared, hard-capped. Aborts if projected cost would breach the cap.
Nothing here modifies loss weighting, terminal-action target, first-action
weight, horizon, data, architecture, or optimizer semantics.
"""
import argparse, json, os, sys, time

import isaacgym  # noqa: F401  MUST precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
sys.path.insert(0, "experiments/scripts")
sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
from isaacgym_control import Args  # noqa: E402
from twotime_denoiser import TwoTimePINTDenoiser  # noqa: E402
from twotime_flow import TwoTimeConditionalFlowMatching  # noqa: E402

OUT = "experiments/policy_improvement/two_time_flow/stage1"
CANON = "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CAP_GPU_H = 0.15


def build(args, p_diag):
    """Load canonical Flow, then swap in the zero-parameter two-time modules."""
    exp = utils.load_diffusion("data", args.dataset, CANON, epoch="latest",
        seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    base = exp.diffusion
    src = base.model
    # Read the ACTUAL architecture from the saved config; never hardcode.
    import pickle
    mc = pickle.load(open(
        f"data/panda_push/{CANON}/model_config.pkl", "rb"))
    md = dict(mc._dict) if hasattr(mc, "_dict") else dict(mc)
    # Pass the saved config through verbatim. NOTE the constructor uses
    # singular n_head / n_layer / block_size -- plural names would be
    # swallowed by **kwargs and silently replaced by defaults.
    kw = {k: v for k, v in md.items() if k != "device"}
    print(f"[arch from config] {kw.get('hidden_dim')}d {kw.get('n_layer')}L "
          f"heads={kw.get('n_head')} feat={kw.get('features_dim')} "
          f"block={kw.get('block_size')}", flush=True)
    tt_net = TwoTimePINTDenoiser(**kw).to(args.device)
    tt_net.load_state_dict(src.state_dict(), strict=True)   # zero new params
    tt = TwoTimeConditionalFlowMatching(
        tt_net, horizon=base.horizon, observation_dim=base.observation_dim,
        action_dim=base.action_dim, loss_type=base.loss_type,
        action_weight=float(base.loss_weight_matrix[0, 0]),
        loss_discount=1, loss_weights=None, time_scale=base.time_scale,
        p_diag=p_diag).to(args.device)
    return exp, base, tt


@torch.no_grad()
def replay_errors(model, ds, dev, ad, t_s, t_a, nb, seed=99):
    """Per-block |pred - target| at a FIXED (t_s,t_a) cell, free coords only."""
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    it = iter(dl); ae, se = [], []
    for i in range(nb):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        x1 = b.trajectories.to(dev)
        c = {k: v.to(dev) for k, v in b.conditions.items()}
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
        ae.append(float((err[:, :, :ad] * am).sum() / am.sum().clamp(min=1)))
        se.append(float((err[:, :, ad:] * om).sum() / om.sum().clamp(min=1)))
    return float(np.mean(ae)), float(np.mean(se))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--p-diag", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--grid-batches", type=int, default=6)
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    a = Args(); utils.set_global_device(a.device)
    dev = a.device
    exp, base, tt = build(a, cli.p_diag)
    ds = exp.dataset; ad = base.action_dim
    res = {"p_diag": cli.p_diag, "steps": cli.steps, "lr": cli.lr,
           "batch": cli.batch, "accum": cli.accum}

    # ---- CONTROL: INIT-DIAGONAL must reproduce CANONICAL bit-exactly ----
    torch.manual_seed(7)
    xprobe = torch.randn(4, base.horizon, base.transition_dim, device=dev)
    tprobe = torch.rand(4, device=dev)
    with torch.no_grad():
        p_can = base._call_model(xprobe, {}, tprobe * base.time_scale)
        p_two = tt._call_two_time_model(xprobe, {}, tprobe, tprobe)
    res["init_diagonal_bitexact"] = bool(torch.equal(p_can, p_two))
    res["init_diagonal_maxdiff"] = float((p_can - p_two).abs().max())
    print(f"[CONTROL] INIT-DIAGONAL bit-exact={res['init_diagonal_bitexact']} "
          f"maxdiff={res['init_diagonal_maxdiff']:.3e}", flush=True)
    if not res["init_diagonal_bitexact"]:
        json.dump(res, open(f"{OUT}/stage1.json", "w"), indent=1)
        raise SystemExit("GATE FAIL: retrofit is not bit-exact at init")

    # ---- pre-fine-tune reference on the diagonal ----
    res["pre_ft"] = {}
    for tv in (0.25, 0.5, 0.75):
        ae, se = replay_errors(tt, ds, dev, ad, tv, tv, cli.grid_batches)
        res["pre_ft"][f"diag_t{tv}"] = dict(action=ae, state=se)
    print(f"[PRE-FT diagonal] {res['pre_ft']}", flush=True)

    # ---- Stage 1 fine-tune (throughput probe folded into first 100 steps) ----
    dl = torch.utils.data.DataLoader(ds, batch_size=cli.batch, shuffle=True,
                                     num_workers=1, generator=torch.Generator().manual_seed(20260905))
    it = iter(dl)
    opt = torch.optim.Adam(tt.parameters(), lr=cli.lr, betas=(0.9, 0.999))
    tt.train(); hist = []
    t0 = time.time()
    for step in range(cli.steps):
        opt.zero_grad(set_to_none=True)
        agg = 0.0; info_acc = {}
        for _ in range(cli.accum):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
            loss, info = tt.loss(x, c)
            if not torch.isfinite(loss):
                raise SystemExit(f"GATE FAIL: non-finite loss at step {step}")
            (loss / cli.accum).backward()
            agg += float(loss) / cli.accum
            for k, v in info.items():
                info_acc[k] = info_acc.get(k, 0.0) + float(v) / cli.accum
        opt.step()
        if step % 50 == 0 or step == cli.steps - 1:
            hist.append(dict(step=step, loss=agg, **info_acc))
            print(f"  step {step:4d} loss {agg:.4f} act {info_acc.get('action_loss',0):.4f} "
                  f"obs {info_acc.get('observation_loss',0):.4f} "
                  f"diag {info_acc.get('frac_diag',0):.3f}", flush=True)
        if step == 99:   # folded throughput probe
            rate = (time.time() - t0) / 100
            proj = (cli.steps * rate + 0.065 * 3600) / 3600
            print(f"[PROBE] {rate:.4f} s/step -> projected total {proj:.3f} GPU-h", flush=True)
            res["s_per_step"] = rate; res["projected_gpu_h"] = proj
            if proj > CAP_GPU_H:
                json.dump(res, open(f"{OUT}/stage1.json", "w"), indent=1)
                raise SystemExit(f"STOP: projected {proj:.3f} exceeds cap {CAP_GPU_H}")
    res["train_seconds"] = time.time() - t0
    res["train_history"] = hist
    tt.eval()
    # persist the fine-tuned weights so Stage A can do pre-vs-post
    torch.save({"model": tt.model.state_dict(), "steps": cli.steps,
                "p_diag": cli.p_diag, "lr": cli.lr},
               f"{OUT}/ft_state.pt")
    print(f"[saved] {OUT}/ft_state.pt", flush=True)

    # ---- diagonal retention after fine-tune ----
    res["post_ft"] = {}
    for tv in (0.25, 0.5, 0.75):
        ae, se = replay_errors(tt, ds, dev, ad, tv, tv, cli.grid_batches)
        res["post_ft"][f"diag_t{tv}"] = dict(action=ae, state=se)
    print(f"[POST-FT diagonal] {res['post_ft']}", flush=True)
    with torch.no_grad():
        p_two2 = tt._call_two_time_model(xprobe, {}, tprobe, tprobe)
    res["post_ft_diag_drift_vs_canonical"] = float((p_can - p_two2).abs().max())

    # ---- PRIMARY GATE: interior 5x5 surface ----
    surf = {}
    for ts in GRID:
        for ta in GRID:
            ae, se = replay_errors(tt, ds, dev, ad, ts, ta, cli.grid_batches)
            surf[f"{ts}_{ta}"] = dict(t_state=ts, t_action=ta, action=ae, state=se)
            print(f"  cell t_s={ts:.2f} t_a={ta:.2f} action={ae:.4f} state={se:.4f}", flush=True)
    res["surface"] = surf
    json.dump(res, open(f"{OUT}/stage1.json", "w"), indent=1)
    print("\nwrote", f"{OUT}/stage1.json")


if __name__ == "__main__":
    main()
