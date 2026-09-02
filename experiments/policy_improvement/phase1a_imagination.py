"""Phase 1A: Flow imagination at NFE 1/2/4/10/15/32 + partial-denoising figure.

EMA deployment weights. Identical conditions and identical latent noise across
NFE (valid: same shape, same seed). Neutral replay reference; no policy-authored
targets. Gaussian included as a cheap qualitative positive control.
"""
import argparse, json, os, pickle

import isaacgym  # noqa: F401  must precede torch
import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.configuration import flow_sampling_kwargs  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from dlp_utils import extract_dlp_features_with_bg, get_recon_from_dlps  # noqa: E402
import sys; sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args, array_to_state_dict  # noqa: E402

OUT = "experiments/policy_improvement/phase1a"
NFES = [1, 2, 4, 10, 15, 32]
PPV = 24            # particles per view
NOISE_SEED = 20260902

FLOW = dict(loadbase="data",
            loadpath="flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42")
GAUSS = dict(loadbase="ecdiffuser-data/pretrained_models",
             loadpath="diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100")


def decode_view(particles, z_bg, model, device):
    """Verbatim from experiments/scripts/dlp_imagination.py:37 (proven path)."""
    particles = torch.as_tensor(particles, device=device, dtype=torch.float32)
    if particles.dim() == 2:
        particles = particles.unsqueeze(0)
    z_bg = torch.as_tensor(z_bg, device=device, dtype=torch.float32)
    if z_bg.dim() == 1:
        z_bg = z_bg.unsqueeze(0)
    # get_recon_from_dlps already returns HWC uint8 in [0,255] (dlp_utils.py:305-306).
    # Do NOT rescale or clip to [0,1] - that renders everything black.
    return get_recon_from_dlps(particles, z_bg, model, device)


def load(spec, args):
    exp = utils.load_diffusion(
        spec["loadbase"], args.dataset, spec["loadpath"], epoch="latest",
        seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    return exp.ema, exp                      # EMA deployment weights


def sample(diff, cond, nfe, gen, chain=False):
    """One conditional_sample with fixed noise, matched across NFE."""
    torch.manual_seed(NOISE_SEED)            # identical z across NFE
    kw = dict(verbose=False)
    if hasattr(diff, "n_solver_steps"):
        kw["n_steps"] = nfe
    if chain:
        kw["return_chain"] = True
    return diff.conditional_sample(cond, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    args = Args()
    utils.set_global_device(args.device)
    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length
    dev, model, H = env.device, env.latent_rep_model, args.horizon

    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]; nE = env.num_envs
    obs = env.reset(set_init_states=array_to_state_dict(ep["init"][:nE], keys, dev),
                    set_goal_states=array_to_state_dict(ep["goal"][:nE], keys, dev))
    def _np(v):
        return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    observation = _np(obs["achieved_goal"]).reshape(nE, -1)
    goal = _np(obs["desired_goal"]).reshape(nE, -1)
    front = env.env.obs_dict["media"][:, 0]
    _, z_bg = extract_dlp_features_with_bg(front, model, dev)

    flow, fexp = load(FLOW, args)
    norm = fexp.dataset.normalizer
    nobs = torch.as_tensor(norm.normalize(observation, "observations"),
                           device=dev, dtype=torch.float32)
    ngoal = torch.as_tensor(norm.normalize(goal, "observations"),
                            device=dev, dtype=torch.float32)
    cond = {0: nobs, H - 1: ngoal}

    # ---- A. NFE sweep -------------------------------------------------------
    gen = torch.Generator(device="cpu")
    imag = {}
    for nfe in NFES:
        s = sample(flow, cond, nfe, gen)
        x = s.trajectories if hasattr(s, "trajectories") else s
        x = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)
        o = norm.unnormalize(x[..., flow.action_dim:], "observations")
        imag[nfe] = o.reshape(len(o), H, 2, PPV, 10)
        print(f"NFE {nfe}: sampled {o.shape}")

    gs, gexp = load(GAUSS, args)
    torch.manual_seed(NOISE_SEED)
    gsamp = gs.conditional_sample(cond, verbose=False)
    gx = gsamp.trajectories if hasattr(gsamp, "trajectories") else gsamp
    gx = np.asarray(gx.detach().cpu() if torch.is_tensor(gx) else gx)
    go = norm.unnormalize(gx[..., gs.action_dim:], "observations")
    gaussian = go.reshape(len(go), H, 2, PPV, 10)
    print("gaussian control sampled")

    # true current frame as the DLP-decoder reference
    cur = observation.reshape(nE, 2, PPV, 10)
    gl = goal.reshape(nE, 2, PPV, 10)

    for e in range(cli.episodes):
        rows = ["DLP ref"] + [f"Flow NFE{n}" for n in NFES] + ["Gaussian"]
        fig, ax = plt.subplots(len(rows), H, figsize=(2.5 * H, 2.5 * len(rows)))
        for c in range(H):
            lab = "current (cond)" if c == 0 else ("GOAL CONDITION" if c == H - 1
                                                   else f"predicted t={c}")
            ax[0, c].set_title(lab, fontsize=11)
        for c in range(H):
            src = cur[e, 0] if c == 0 else (gl[e, 0] if c == H - 1 else None)
            ax[0, c].imshow(decode_view(src, z_bg[e], model, dev)) if src is not None \
                else ax[0, c].axis("off")
            ax[0, c].axis("off")
        for r, n in enumerate(NFES, start=1):
            for c in range(H):
                ax[r, c].imshow(decode_view(imag[n][e, c, 0], z_bg[e], model, dev))
                ax[r, c].axis("off")
        for c in range(H):
            ax[len(rows) - 1, c].imshow(decode_view(gaussian[e, c, 0], z_bg[e], model, dev))
            ax[len(rows) - 1, c].axis("off")
        for r, nm in enumerate(rows):
            ax[r, 0].set_ylabel(nm)
            ax[r, 0].text(-0.16, 0.5, nm, transform=ax[r, 0].transAxes,
                          fontsize=12, va="center", ha="right", fontweight="bold")
        fig.suptitle(f"Flow imagination vs NFE - episode {e} (EMA, identical z, row4 = GOAL CONDITION)",
                     fontsize=14)
        fig.tight_layout()
        p = f"{OUT}/imagination_nfe_ep{e}.png"
        fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)

    # ---- B. partial denoising ----------------------------------------------
    s = sample(flow, cond, 32, gen, chain=True)
    ch = None
    for attr in ("chain", "diffusion", "chains"):
        if hasattr(s, attr):
            ch = getattr(s, attr); break
    if ch is None and isinstance(s, (tuple, list)):
        for el in s:
            if isinstance(el, (list, tuple)) and len(el) > 2:
                ch = el; break
    if torch.is_tensor(ch):                       # [B, steps, H, D] -> list
        ch = [ch[:, i] for i in range(ch.shape[1])]
    if ch is None:
        print("WARNING: no chain returned; skipping partial-denoising figure")
    else:
        ch = [np.asarray(c.detach().cpu()) for c in ch]
        picks = [0, len(ch) // 4, len(ch) // 2, 3 * len(ch) // 4, len(ch) - 1]
        labs = ["initial noise", "25%", "50%", "75%", "final"]
        for e in range(min(2, cli.episodes)):
            fig, ax = plt.subplots(H, len(picks), figsize=(2.5 * len(picks), 2.5 * H))
            for ci, (pi, lb) in enumerate(zip(picks, labs)):
                o = norm.unnormalize(ch[pi][..., flow.action_dim:], "observations")
                o = o.reshape(len(o), H, 2, PPV, 10)
                for t in range(H):
                    ax[t, ci].imshow(decode_view(o[e, t, 0], z_bg[e], model, dev))
                    ax[t, ci].axis("off")
                    if t == 0: ax[t, ci].set_title(lb, fontsize=12)
            for t in range(H):
                nm = "t=0 (cond)" if t == 0 else ("GOAL CONDITION" if t == H - 1 else f"t={t}")
                ax[t, 0].text(-0.16, 0.5, nm, transform=ax[t, 0].transAxes,
                              fontsize=11, va="center", ha="right", fontweight="bold")
            fig.suptitle(f"Partial denoising, NFE32 - episode {e}", fontsize=14)
            fig.tight_layout()
            p = f"{OUT}/partial_denoising_ep{e}.png"
            fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)

    np.savez(f"{OUT}/imagination_raw.npz",
             **{f"nfe{n}": imag[n] for n in NFES}, gaussian=gaussian, cur=cur, goal=gl)
    print("done")


if __name__ == "__main__":
    main()
