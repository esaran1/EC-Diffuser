"""§3 MeanFlow failure diagnostics on the SAVED 20k checkpoint. No retraining.

Tests the OMP (arXiv:2512.19347) gradient-starvation / velocity-magnitude
mechanism, plus representation-collapse and directional diagnostics, against
canonical Standard Flow on identical replay batches.
"""
import json, sys
import isaacgym  # noqa: F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402
sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args  # noqa: E402

OUT = "experiments/policy_improvement/two_time_flow/meanflow_failure_diag.json"
MF = "imf_viability/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
SF = "phase3_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
NB, BS = 24, 8


def load(path, epoch):
    a = Args(); utils.set_global_device(a.device)
    e = utils.load_diffusion("data", a.dataset, path, epoch=epoch, seed=a.seed,
        is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    return a, e


def eff_rank(M, eps=1e-12):
    """Entropy-based effective rank of the singular spectrum."""
    M = M - M.mean(0, keepdim=True)
    s = torch.linalg.svdvals(M.float())
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum())), s


def main():
    res = {}
    args, exp = load(MF, "latest")
    dev = args.device
    ds = exp.dataset
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0)

    # ---- collect paired velocity stats for MeanFlow ----
    m = exp.ema
    ad = m.action_dim
    rows = {"pred_norm": [], "targ_norm": [], "cos": [], "ratio": []}
    reprs = []
    it = iter(dl)
    with torch.no_grad():
        pass
    for i in range(NB):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
        torch.manual_seed(1000 + i)
        loss, info, det = m._compute_meanflow_loss(x, c, return_details=True)
        pred = det["compound_velocity"].detach()
        targ = det["target_velocity"].detach()
        mask = det["conditioning_mask"]
        # flatten over free coords only
        pf = (pred * mask).reshape(pred.shape[0], -1)
        tf = (targ * mask).reshape(targ.shape[0], -1)
        pn = pf.norm(dim=1); tn = tf.norm(dim=1)
        cs = torch.nn.functional.cosine_similarity(pf, tf, dim=1)
        rows["pred_norm"] += pn.tolist(); rows["targ_norm"] += tn.tolist()
        rows["cos"] += cs.tolist(); rows["ratio"] += (pn / tn.clamp(min=1e-9)).tolist()
        reprs.append(pf.cpu())

    a = {k: np.array(v) for k, v in rows.items()}
    er, sv = eff_rank(torch.cat(reprs))
    res["meanflow"] = dict(
        pred_norm_mean=float(a["pred_norm"].mean()), targ_norm_mean=float(a["targ_norm"].mean()),
        norm_ratio_mean=float(a["ratio"].mean()), norm_ratio_median=float(np.median(a["ratio"])),
        cos_mean=float(a["cos"].mean()), cos_median=float(np.median(a["cos"])),
        angular_err_deg=float(np.degrees(np.arccos(np.clip(a["cos"], -1, 1))).mean()),
        output_effective_rank=er, n_samples=len(a["cos"]))

    # ---- OMP gradient-starvation test: bin cosine by target magnitude ----
    q = np.quantile(a["targ_norm"], [0, .25, .5, .75, 1.0])
    bins = []
    for lo, hi in zip(q[:-1], q[1:]):
        sel = (a["targ_norm"] >= lo) & (a["targ_norm"] <= hi)
        if sel.sum() < 3: continue
        bins.append(dict(targ_norm_lo=float(lo), targ_norm_hi=float(hi), n=int(sel.sum()),
                         cos_mean=float(a["cos"][sel].mean()),
                         norm_ratio_mean=float(a["ratio"][sel].mean())))
    res["meanflow_magnitude_bins"] = bins

    # ---- MVP/IVC boundary test (arXiv:2602.13810, Eq. 15) ----
    # Their claim: L_MF is blind to the boundary, so u_theta(z,t,t) may carry a
    # persistent bias C/(r-t). The checkable quantity for OUR code: at r==t the
    # average velocity must equal the instantaneous velocity. Our implementation
    # parameterizes v(z,t) := u(z,t,t) (boundary iMF), so we test how far
    # u(z,t,t) is from the flow-matching target x1 - z.
    bnd = {"resid": [], "cos": [], "ratio": []}
    it = iter(dl)
    with torch.no_grad():
        for i in range(NB):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
            torch.manual_seed(2000 + i)
            x1l = x.clone(); z = torch.randn_like(x)
            for tt, val in c.items():
                x1l[:, tt, ad:] = val; z[:, tt, ad:] = val
            t = torch.rand(x.shape[0], device=dev)
            zt = (1 - t.view(-1,1,1)) * z + t.view(-1,1,1) * x1l
            # boundary call: r == t  (interval zero)
            u_tt = m._call_interval_model(zt, c, t * m.time_scale,
                                          torch.zeros_like(t))
            tgt = z - x1l   # our convention: target = noise - data
            msk = m._make_conditioning_mask(zt, c)
            uf = (u_tt * msk).reshape(x.shape[0], -1)
            tf = (tgt * msk).reshape(x.shape[0], -1)
            bnd["resid"] += (uf - tf).norm(dim=1).tolist()
            bnd["cos"] += torch.nn.functional.cosine_similarity(uf, tf, dim=1).tolist()
            bnd["ratio"] += (uf.norm(dim=1) / tf.norm(dim=1).clamp(min=1e-9)).tolist()
    bb = {k: np.array(v) for k, v in bnd.items()}
    res["meanflow_boundary_r_eq_t"] = dict(
        expected="u(z,t,t) should equal the instantaneous target (noise - data)",
        residual_norm_mean=float(bb["resid"].mean()),
        cos_mean=float(bb["cos"].mean()),
        norm_ratio_mean=float(bb["ratio"].mean()), n=len(bb["cos"]))

    # ---- Standard Flow comparison on identical batches ----
    args2, exp2 = load(SF, "final")
    m2 = exp2.ema
    rows2 = {"pred_norm": [], "targ_norm": [], "cos": [], "ratio": []}
    reprs2 = []
    it = iter(dl)
    for i in range(NB):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
        torch.manual_seed(1000 + i)
        loss, info, det = m2._compute_flow_loss(x, c, return_details=True)
        pred = det["prediction"].detach() if "prediction" in det else None
        targ = det["target_velocity"].detach()
        mask = det["conditioning_mask"]
        if pred is None:
            pred = m2._call_model(det["xt"], c, det["model_time"]).detach()
        pf = (pred * mask).reshape(pred.shape[0], -1)
        tf = (targ * mask).reshape(targ.shape[0], -1)
        pn = pf.norm(dim=1); tn = tf.norm(dim=1)
        cs = torch.nn.functional.cosine_similarity(pf, tf, dim=1)
        rows2["pred_norm"] += pn.tolist(); rows2["targ_norm"] += tn.tolist()
        rows2["cos"] += cs.tolist(); rows2["ratio"] += (pn / tn.clamp(min=1e-9)).tolist()
        reprs2.append(pf.cpu())
    a2 = {k: np.array(v) for k, v in rows2.items()}
    er2, _ = eff_rank(torch.cat(reprs2))
    res["standard_flow"] = dict(
        pred_norm_mean=float(a2["pred_norm"].mean()), targ_norm_mean=float(a2["targ_norm"].mean()),
        norm_ratio_mean=float(a2["ratio"].mean()), norm_ratio_median=float(np.median(a2["ratio"])),
        cos_mean=float(a2["cos"].mean()), cos_median=float(np.median(a2["cos"])),
        angular_err_deg=float(np.degrees(np.arccos(np.clip(a2["cos"], -1, 1))).mean()),
        output_effective_rank=er2, n_samples=len(a2["cos"]))
    q2 = np.quantile(a2["targ_norm"], [0, .25, .5, .75, 1.0])
    b2 = []
    for lo, hi in zip(q2[:-1], q2[1:]):
        sel = (a2["targ_norm"] >= lo) & (a2["targ_norm"] <= hi)
        if sel.sum() < 3: continue
        b2.append(dict(targ_norm_lo=float(lo), targ_norm_hi=float(hi), n=int(sel.sum()),
                       cos_mean=float(a2["cos"][sel].mean()),
                       norm_ratio_mean=float(a2["ratio"][sel].mean())))
    res["standard_flow_magnitude_bins"] = b2

    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
