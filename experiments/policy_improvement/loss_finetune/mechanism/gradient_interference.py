"""Mechanism Test 3: gradient interference between loss components.

Question: does the terminal zero-action objective produce a gradient that
CONFLICTS, in shared parameters, with the executed-action objective? If so, that
is a plausible mechanism for why masking it can help control without improving
one-step action accuracy.

Decomposes the weighted flow loss into four components over the SAME batch, the
SAME flow times and the SAME noise:

    L_exec   executed action, t=0
    L_mid    actions t=1..H-2
    L_term   terminal placeholder action, t=H-1
    L_state  state/observation prediction

and reports cos(g_term, g_exec), cos(g_term, g_mid), cos(g_term, g_state) plus
the interpretable projection dot(g_term, g_exec) / ||g_exec||^2.

IMPORTANT SCOPE. This is an optimization-geometry diagnostic in a shared
parameter space. It does NOT revive the withdrawn claim that raw state/action
gradient MAGNITUDES are semantically comparable -- directional alignment between
gradients of different objectives in the same parameters is a different and
legitimate question. Magnitudes are reported only as norms, never compared across
components as "importance".

PARAMETER GROUPS ARE PREDECLARED (before any result was inspected):
    full        every trainable parameter
    trunk       model.particle_transformer.*
    action_head model.action_projection.* + model.action_decoder.*
    time_cond   model.time_mlp.*
These map directly onto the module structure of AdaLNPINTDenoiser; no search over
layers was performed.

Each component uses the SAME weights from `loss_weight_matrix` and the SAME
conditioning mask as the production loss, restricted to its slice, and the SAME
denominator. That keeps every component on the production objective's scale, so
the only thing varying between them is WHICH coordinates contribute.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

GROUPS = {
    "full": lambda n: True,
    "trunk": lambda n: n.startswith("model.particle_transformer."),
    "action_head": lambda n: (n.startswith("model.action_projection.")
                              or n.startswith("model.action_decoder.")),
    "time_cond": lambda n: n.startswith("model.time_mlp."),
}

REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")


def build(config_name, seed, device):
    sys.argv = ["t", "--config", config_name, "--num_entity", "3", "--rand_color",
                "--seed", str(seed), "--device", device]
    import diffuser.utils as utils
    from diffuser.utils.args import ArgsParser
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk
    args = ArgsParser().parse_args("diffusion")
    with open(REF_CFG, "rb") as fh:
        ref = pickle.load(fh)
    obs_dim, act_dim = int(ref._dict["observation_dim"]), int(ref._dict["action_dim"])
    mc = utils.Config(
        args.model, savepath=None, verbose=False,
        features_dim=args.features_dim, action_dim=act_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview, device=device)
    dc = utils.Config(args.diffusion, savepath=None, verbose=False, device=device,
                      **dwk(args, obs_dim, act_dim))
    return args, dc(mc()).to(device)


def component_losses(diff, x1, cond, x0, t, torch):
    """Weighted loss restricted to each temporal/semantic slice.

    Uses the production weights, mask and denominator so all four components live
    on the same scale as the real objective.
    """
    diff._validate_trajectory(x1)
    x1l = x1.clone()
    x0l = x0.clone()
    diff._apply_conditioning(x0l, cond)
    diff._apply_conditioning(x1l, cond)
    bt = t.view(-1, 1, 1)
    xt = (1.0 - bt) * x0l + bt * x1l
    diff._apply_conditioning(xt, cond)
    target = x1l - x0l
    pred = diff._call_model(xt, cond, t * diff.time_scale)
    err = torch.abs(pred - target)
    if diff.loss_type == "l2":
        err = err.square()

    cmask = diff._make_conditioning_mask(x1l, cond)
    w = diff.loss_weight_matrix.to(device=x1.device, dtype=x1.dtype).unsqueeze(0)
    aw = w * cmask.to(x1.dtype)
    denom = cmask.sum().to(x1.dtype)
    ad, H = diff.action_dim, diff.horizon

    def slice_loss(sl):
        m = torch.zeros_like(aw)
        m[sl] = 1.0
        return (err * aw * m).sum() / denom

    return {
        "exec": slice_loss((slice(None), 0, slice(0, ad))),
        "mid": slice_loss((slice(None), slice(1, H - 1), slice(0, ad))),
        "term": slice_loss((slice(None), H - 1, slice(0, ad))),
        "state": slice_loss((slice(None), slice(None), slice(ad, None))),
    }


def flat_grad(loss, named_params, keep, torch):
    grads = torch.autograd.grad(loss, [p for _, p in named_params],
                                retain_graph=True, allow_unused=True)
    out = []
    for (n, p), g in zip(named_params, grads):
        if not keep(n):
            continue
        out.append(torch.zeros_like(p).flatten() if g is None else g.flatten())
    return torch.cat(out).double() if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="canonical",
                    help="'canonical' or a path to state_final.pt")
    ap.add_argument("--config", default="config.pandapush_lossft_armA")
    ap.add_argument("--n_batches", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="canonical")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "" if cli.device == "cpu" else "0")
    import torch
    import diffuser.utils as utils

    args, diff = build(cli.config, cli.seed, cli.device)

    ckpt = (os.path.join("data/panda_push/flow",
                         "3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt")
            if cli.checkpoint == "canonical" else cli.checkpoint)
    data = torch.load(ckpt, map_location=cli.device)
    diff.load_state_dict(data["model"])
    diff.rebuild_loss_weight_matrix()
    diff.eval()

    dataset_config = utils.Config(
        args.loader, savepath=None, verbose=False,
        dataset_path=args.dataset_path, dataset_name=args.dataset,
        horizon=args.horizon, obs_only=args.obs_only, action_only=args.action_only,
        normalizer=args.normalizer, particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns, use_padding=args.use_padding,
        max_path_length=args.max_path_length, overfit=args.overfit,
        single_view=(args.input_type == "dlp" and not args.multiview))
    dataset = dataset_config()

    # PREDECLARED fixed batches: deterministic generator, fixed noise and times,
    # identical across every component compared.
    gen = torch.Generator().manual_seed(20260913)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cli.batch_size,
                                         shuffle=True, num_workers=0, generator=gen)
    noise_gen = torch.Generator().manual_seed(20260913)

    named = [(n, p) for n, p in diff.named_parameters() if p.requires_grad]
    pairs = [("term", "exec"), ("term", "mid"), ("term", "state"), ("exec", "mid")]
    acc = {g: {p: [] for p in pairs} for g in GROUPS}
    proj = {g: [] for g in GROUPS}
    norms = {k: [] for k in ("exec", "mid", "term", "state")}

    it = iter(loader)
    for bi in range(cli.n_batches):
        batch = next(it)
        x1 = batch.trajectories.to(cli.device).float()
        cond = {int(k): v.to(cli.device).float() for k, v in batch.conditions.items()}
        x0 = torch.randn(x1.shape, generator=noise_gen).to(cli.device)
        t = torch.rand(x1.shape[0], generator=noise_gen).to(cli.device)

        diff.zero_grad(set_to_none=True)
        comps = component_losses(diff, x1, cond, x0, t, torch)
        for k, v in comps.items():
            norms[k].append(float(v.item()))

        for gname, keep in GROUPS.items():
            g = {k: flat_grad(v, named, keep, torch) for k, v in comps.items()}
            for a, b in pairs:
                ga, gb = g[a], g[b]
                na, nb = float(ga.norm()), float(gb.norm())
                acc[gname][(a, b)].append(
                    float((ga @ gb) / (na * nb)) if na > 0 and nb > 0 else float("nan"))
            if float(g["exec"].norm()) > 0:
                proj[gname].append(float((g["term"] @ g["exec"])
                                         / (g["exec"] @ g["exec"])))
        if (bi + 1) % 4 == 0:
            print("  batch {}/{}".format(bi + 1, cli.n_batches), flush=True)

    def summarize(vals):
        a = np.asarray([v for v in vals if v == v])
        if a.size == 0:
            return None
        return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
                "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
                "min": float(a.min()), "max": float(a.max()),
                "frac_negative": float((a < 0).mean())}

    result = {"tag": cli.tag, "checkpoint": ckpt, "config": cli.config,
              "n_batches": cli.n_batches, "batch_size": cli.batch_size,
              "component_loss_means": {k: float(np.mean(v)) for k, v in norms.items()},
              "cosines": {g: {"{}_vs_{}".format(a, b): summarize(acc[g][(a, b)])
                              for a, b in pairs} for g in GROUPS},
              "projection_term_on_exec": {g: summarize(proj[g]) for g in GROUPS}}

    print("\n=== cos(g_term, g_exec) by parameter group ===")
    for g in GROUPS:
        s = result["cosines"][g]["term_vs_exec"]
        print("  {:<12} mean {:+.4f}  median {:+.4f}  IQR [{:+.4f},{:+.4f}]  frac<0 {:.2f}"
              .format(g, s["mean"], s["median"], s["q25"], s["q75"], s["frac_negative"]))
    print("\n=== other pairs (full) ===")
    for a, b in pairs:
        s = result["cosines"]["full"]["{}_vs_{}".format(a, b)]
        print("  cos({:<5},{:<5}) mean {:+.4f}  frac<0 {:.2f}".format(
            a, b, s["mean"], s["frac_negative"]))

    out = cli.out or ("experiments/policy_improvement/loss_finetune/mechanism/"
                      "GRADIENT_INTERFERENCE_{}.json".format(cli.tag))
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
