"""Small action VAE (CPU). Version A: latent = action dim (3). Version B: 2.

The terminal stop target is 1/5 of the action rows EC-Diffuser trains on, so
it is mixed in at that frequency. Deterministic encoding = posterior mean.

QUALITY GATE (fixed before training):
  - per-dim RMSE <= 0.05 * that dim's std   (>= 99.75% variance per dim)
  - terminal stop action |error|_inf <= 0.02
  - no collapsed latent dim: KL_d > 0.05 nats and Var(mu_d) > 1e-3
  - magnitude and direction preserved (reported)
"""
import argparse, json, os, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))


class ActionVAE(nn.Module):
    def __init__(self, a_dim=3, z_dim=3, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(a_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, a_dim))
        self.z_dim = z_dim

    def encode(self, a):
        mu, logv = self.enc(a).chunk(2, -1)
        return mu, logv.clamp(-12, 6)

    def forward(self, a):
        mu, logv = self.encode(a)
        z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        return self.dec(z), mu, logv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=30000)
    a = ap.parse_args()
    torch.manual_seed(20260928); np.random.seed(20260928); torch.set_num_threads(8)
    d = np.load(os.path.join(HERE, "actions.npz"))
    A = torch.tensor(d["actions"]); T = torch.tensor(d["terminal"])
    n = len(A); perm = torch.randperm(n); ntr = int(0.95 * n)
    tr, va = A[perm[:ntr]], A[perm[ntr:]]
    m = ActionVAE(3, a.z); opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    for it in range(a.iters):
        idx = torch.randint(0, ntr, (1024,))
        x = tr[idx]
        k = torch.rand(1024) < 0.2                     # terminal row frequency in EC-Diffuser
        x = torch.where(k[:, None], T.expand(1024, 3), x)
        rec, mu, logv = m(x)
        kl = 0.5 * (mu.pow(2) + logv.exp() - 1 - logv).sum(-1).mean()
        loss = (rec - x).pow(2).sum(-1).mean() + a.beta * kl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 5000 == 0:
            print("it %5d loss %.5f kl %.3f" % (it, loss.item(), kl.item()), flush=True)
    m.eval()
    with torch.no_grad():
        mu, logv = m.encode(va); rec = m.dec(mu)
        std = va.std(0); rmse = (rec - va).pow(2).mean(0).sqrt()
        kl_d = (0.5 * (mu.pow(2) + logv.exp() - 1 - logv)).mean(0)
        var_mu = mu.var(0)
        tmu, _ = m.encode(T); trec = m.dec(tmu); terr = (trec - T).abs().max().item()
        cos = nn.functional.cosine_similarity(rec, va, dim=-1)
        mag = (rec.norm(dim=-1) - va.norm(dim=-1)).abs()
        # latent limits over real + terminal (for [-1,1] normalization in EC-Diffuser)
        mu_all, _ = m.encode(torch.cat([A, T])); zmin = mu_all.min(0).values; zmax = mu_all.max(0).values
    rel = (rmse / std)
    gate = {"per_dim_rel_rmse_ok": bool((rel <= 0.05).all()),
            "terminal_ok": terr <= 0.02,
            "active_dims_ok": bool(((kl_d > 0.05) & (var_mu > 1e-3)).all())}
    gate["PASS"] = all(gate.values())
    res = {"z": a.z, "beta": a.beta, "rel_rmse_per_dim": rel.tolist(), "rmse_per_dim": rmse.tolist(),
           "terminal_abs_err": terr, "terminal_recon": trec.tolist(), "kl_per_dim": kl_d.tolist(),
           "var_mu_per_dim": var_mu.tolist(), "cos_p01": float(cos.quantile(0.01)),
           "cos_mean": float(cos.mean()), "mag_err_p99": float(mag.quantile(0.99)),
           "z_min": zmin.tolist(), "z_max": zmax.tolist(), "gate": gate}
    torch.save({"state_dict": m.state_dict(), "z_dim": a.z, "z_min": zmin, "z_max": zmax,
                "terminal": T, "result": res}, os.path.join(HERE, "action_vae_z%d.pt" % a.z))
    json.dump(res, open(os.path.join(HERE, "action_vae_z%d.json" % a.z), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("rel_rmse_per_dim", "terminal_abs_err", "kl_per_dim",
                                           "var_mu_per_dim", "cos_p01", "mag_err_p99", "gate")}, indent=1))


if __name__ == "__main__":
    main()
