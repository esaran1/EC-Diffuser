"""Stage G register diagnostic (directive section 35).

Asks ONE question: is there evidence that EC-Diffuser's physical state/action
tokens are being used as generic computational scratch space, in the sense of
Darcet et al. 2024?

Methodology follows ../literature/REGISTERS_AUDIT.md, including its warnings:

  - the signal is BIMODALITY of the token-norm histogram, not high variance.
    A percentile rule applied to a unimodal distribution manufactures
    "outliers" that do not exist. We therefore test for bimodality explicitly
    and report "no artifact" when absent.
  - norms must be compared WITHIN a homogeneous token population. Action and
    DLP tokens have different roles and will differ in norm for boring
    reasons; a cross-type ranking would be confident and meaningless.
  - diffusion time t is an axis with no ViT analogue, so everything is
    measured per-t.

The prior is NEGATIVE: EC-Diffuser is ~60M params / 8 blocks / 49 tokens per
step, far below the >=ViT-Large, long-training, redundant-background regime
where the phenomenon was observed -- and it already has a global-aggregation
slot (the action token).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, ".")

T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def bimodality(x, bins=60, min_secondary_frac=0.01):
    """Evidence for a SUBSTANTIAL secondary high-norm mode.

    Darcet et al.'s signature is ~2% of tokens at ~10x the main-mode norm,
    read off a VISIBLE valley. An earlier version of this function scored any
    nonzero far bin as a second peak, which in a long right tail always yields
    depth 1.0 -- a sparse-tail artifact, not bimodality. It is fixed here to
    require the secondary mode to hold at least `min_secondary_frac` of the
    mass before it counts at all.

    Returns (depth, detail). depth == 0 means no qualifying secondary mode.
    """
    h, edges = np.histogram(x, bins=bins)
    total = h.sum()
    if total == 0:
        return 0.0, {}
    i_max = int(np.argmax(h))
    floor = max(1, int(min_secondary_frac * total))
    best = (0.0, None, None)
    for j in range(len(h)):
        if abs(j - i_max) < 3 or h[j] < floor:
            continue                      # secondary mode must be substantial
        lo, hi = sorted((i_max, j))
        valley = h[lo:hi + 1].min()
        peak2 = h[j]
        depth = (min(h[i_max], peak2) - valley) / float(min(h[i_max], peak2))
        if depth > best[0]:
            best = (depth, j, valley)
    depth, j, valley = best
    return float(depth), {
        "peak1_bin": i_max, "peak1_count": int(h[i_max]),
        "peak1_edge": float(edges[i_max]),
        "peak2_bin": j, "peak2_count": int(h[j]) if j is not None else None,
        "peak2_edge": float(edges[j]) if j is not None else None,
        "valley_count": int(valley) if valley is not None else None,
        "secondary_mass_floor": floor, "n": int(total),
        "max_over_mean": float(x.max() / max(x.mean(), 1e-9)),
        "p99_over_mean": float(np.percentile(x, 99) / max(x.mean(), 1e-9)),
    }


@torch.no_grad()
def token_norms(model, x, cond, t_val, device):
    """Final-block token hidden-state norms, split by token population.

    Captures the transformer output BEFORE the split decoders, which is the
    analogue of the ViT "output token" the paper measures.
    """
    bs, T, f = x.size()
    ad = model.action_dim
    actions = x[:, :, :ad]
    x_particles = x[:, :, ad:].view(bs, T, -1, model.features_dim)

    action_particle = model.action_projection(actions) + model.action_encoding.repeat(bs, T, 1)
    sp = model.particle_projection(x_particles)
    if model.multiview:
        n_p = sp.size(2) // 2
        v1 = sp[:, :, :n_p, :] + model.view1_encoding.repeat(bs, T, n_p, 1)
        v2 = sp[:, :, n_p:, :] + model.view2_encoding.repeat(bs, T, n_p, 1)
        sp = torch.cat([v1, v2], dim=2)
    else:
        sp = sp + model.particle_encoding.repeat(bs, T, sp.size(2), 1)

    x_cat = torch.cat([action_particle.unsqueeze(2), sp], dim=2)
    t = torch.full((bs,), t_val * model_time_scale(model), device=device,
                   dtype=x.dtype)
    t_embed = model._time_embedding(t, None)
    x_proj = (x_cat + t_embed[:, None, None, :]).permute(0, 2, 1, 3)
    out = model.particle_transformer(x_proj, action_particle, t_embed)
    out = out.permute(0, 2, 1, 3)               # [bs, T, 1+P, D]

    norms = out.norm(dim=-1)                    # [bs, T, 1+P]
    return {"action": norms[:, :, 0].flatten().cpu().numpy(),
            "particles": norms[:, :, 1:].flatten().cpu().numpy(),
            "per_particle_mean": norms[:, :, 1:].mean(dim=(0, 1)).cpu().numpy()}


def model_time_scale(model):
    return 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savepath", required=True)
    ap.add_argument("--epoch", default="final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diffuser.utils.serialization import load_config
    ds_cfg = load_config(a.savepath, "dataset_config.pkl")
    ds_cfg._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
    ds = ds_cfg(seed=42)
    mcfg = load_config(a.savepath, "model_config.pkl")
    dcfg = load_config(a.savepath, "diffusion_config.pkl")
    diffusion = dcfg(mcfg())
    ck = torch.load(os.path.join(a.savepath, "state_{}.pt".format(a.epoch)),
                    map_location="cpu")
    diffusion.load_state_dict(ck["ema"])
    diffusion.to(a.device).eval()
    model = diffusion.model

    rng = np.random.RandomState(20260916)
    idx = rng.choice(len(ds), size=a.n_batch, replace=False)
    xs, conds = [], {}
    for i in idx:
        b = ds[int(i)]
        xs.append(torch.as_tensor(b.trajectories))
        for k, v in b.conditions.items():
            conds.setdefault(k, []).append(torch.as_tensor(v))
    x1 = torch.stack(xs).float().to(a.device)
    cond = {k: torch.stack(v).float().to(a.device) for k, v in conds.items()}

    g = torch.Generator(device="cpu").manual_seed(20260916)
    x0 = torch.randn(x1.shape, generator=g).to(a.device)

    payload = {"savepath": a.savepath, "epoch": a.epoch,
               "checkpoint_step": int(ck.get("step", -1)), "per_t": {}}
    print("\n  t     pop         n      mean     sd      p99    max   bimodality")
    for t_val in T_GRID:
        xt = (1.0 - t_val) * x0 + t_val * x1
        diffusion._apply_conditioning(xt, cond)
        nm = token_norms(model, xt, cond, t_val, a.device)
        rec = {}
        for pop in ("action", "particles"):
            v = nm[pop]
            depth, detail = bimodality(v)
            rec[pop] = {"n": int(v.size), "mean": float(v.mean()),
                        "sd": float(v.std()), "p99": float(np.percentile(v, 99)),
                        "max": float(v.max()), "bimodality_depth": depth,
                        "bimodality_detail": detail}
            print("  {:.2f}  {:<10} {:>6} {:>9.3f} {:>7.3f} {:>8.3f} {:>7.3f} {:>8.3f}"
                  .format(t_val, pop, v.size, v.mean(), v.std(),
                          np.percentile(v, 99), v.max(), depth))
        # ratio of the action token's norm to the particle population
        rec["action_over_particle_mean"] = (
            rec["action"]["mean"] / max(rec["particles"]["mean"], 1e-9))
        # persistent per-particle outliers (within the homogeneous population)
        ppm = nm["per_particle_mean"]
        rec["per_particle_mean"] = ppm.tolist()
        rec["per_particle_max_over_median"] = float(
            ppm.max() / max(np.median(ppm), 1e-9))
        payload["per_t"]["{:.2f}".format(t_val)] = rec

    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
