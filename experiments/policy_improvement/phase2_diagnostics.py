"""Phase 2 20k diagnostics: LIVE vs EMA losses/replay, five-seed control, imagination.

Predeclared. No checkpoint selection by control performance.
"""
import argparse, json, os, pickle, time

import isaacgym  # noqa: F401
import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from dlp_utils import extract_dlp_features_with_bg, get_recon_from_dlps  # noqa: E402
import sys; sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args, array_to_state_dict, entity_positions  # noqa: E402

OUT = "experiments/policy_improvement/phase2"
SETS = [("E0", "replicate0_n96.pkl"), ("E1s", "replicate1_n96.pkl"),
        ("E2s", "replicate2_n96.pkl"), ("E3s", "replicate3_n96.pkl"),
        ("E4s", "replicate4_n96.pkl")]
NFES = [1, 2, 4, 10, 15]
PPV, NOISE_SEED, SUCCESS_THRESH = 24, 20260902, 0.04


def decode_view(p, z_bg, model, device):
    p = torch.as_tensor(p, device=device, dtype=torch.float32)
    if p.dim() == 2: p = p.unsqueeze(0)
    z = torch.as_tensor(z_bg, device=device, dtype=torch.float32)
    if z.dim() == 1: z = z.unsqueeze(0)
    return get_recon_from_dlps(p, z, model, device)


def loss_probe(diff, ds, dev, n=30):
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    it = iter(dl); rec = []
    for _ in range(n):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
        diff.zero_grad(set_to_none=True)
        loss, info = diff.loss(x, c)
        loss.backward()
        ap = [p for nm, p in diff.named_parameters() if "action" in nm]
        g = float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in ap
                                 if p.grad is not None)).item())
        rec.append((float(loss), float(info.get("action_loss", np.nan)),
                    float(info.get("observation_loss", np.nan)), g))
    a = np.array(rec)
    return dict(objective=float(a[:, 0].mean()), action_loss=float(a[:, 1].mean()),
                state_loss=float(a[:, 2].mean()), action_grad=float(a[:, 3].mean()),
                action_grad_alive=bool(a[:, 3].mean() > 0))


def replay_error(diff, ds, dev, nfe=4, n_batches=12):
    """Neutral replay: condition on real (obs, goal) from data; compare to real next."""
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    ae, se = [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= n_batches: break
            x = b.trajectories.to(dev); c = {k: v.to(dev) for k, v in b.conditions.items()}
            torch.manual_seed(NOISE_SEED)
            kw = dict(verbose=False)
            if hasattr(diff, "n_solver_steps"): kw["n_steps"] = nfe
            s = diff.conditional_sample(c, **kw)
            xs = s.trajectories if hasattr(s, "trajectories") else s
            ad = diff.action_dim
            ae.append(float((xs[:, :, :ad] - x[:, :, :ad]).abs().mean()))
            se.append(float((xs[:, 1:4, ad:] - x[:, 1:4, ad:]).abs().mean()))
    return float(np.mean(ae)), float(np.mean(se))


def control(diff, env, args, ep, dev, norm):
    """Closed-loop on one frozen scenario set. success = max_obj_dist <= 0.04."""
    keys = ep["keys"]; nE = env.num_envs; n_ep = len(ep["init"]); succ = []
    for bs in range(0, n_ep, nE):
        idx = list(range(bs, min(bs + nE, n_ep))); na = len(idx)
        idx += [idx[-1]] * (nE - na)
        torch.manual_seed(NOISE_SEED + bs)
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, dev),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, dev))
        for _ in range(env.horizon):
            o = np.asarray(obs["achieved_goal"]).reshape(nE, -1)
            g = np.asarray(obs["desired_goal"]).reshape(nE, -1)
            c = {0: torch.as_tensor(norm.normalize(o, "observations"), device=dev, dtype=torch.float32),
                 args.horizon - 1: torch.as_tensor(norm.normalize(g, "observations"), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                kw = dict(verbose=False)
                if hasattr(diff, "n_solver_steps"): kw["n_steps"] = 4
                s = diff.conditional_sample(c, **kw)
            xs = s.trajectories if hasattr(s, "trajectories") else s
            a = norm.unnormalize(xs[:, 0, :diff.action_dim].cpu().numpy(), "actions")
            obs, _, _, infos = env.step(torch.as_tensor(a, device=dev, dtype=torch.float32))
        st = entity_positions(env)
        gp = np.asarray(ep["goal"][idx])
        d = np.linalg.norm(st[:, 1:, :2] - gp, axis=-1).max(axis=1)
        succ.extend((d[:na] <= SUCCESS_THRESH).astype(float).tolist())
    return float(np.mean(succ)), len(succ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--loadpath", default="imf_viability/3C_dlp_adalnpint_randcolor_H5_T4_seed42")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--only-sets", nargs="*", default=None,
                    help="restrict control eval to these set ids")
    ap.add_argument("--skip-offline", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="merge control results into existing json")
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    args = Args(); utils.set_global_device(args.device)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    exp = utils.load_diffusion("data", args.dataset, cli.loadpath, epoch=cli.epoch,
                               seed=args.seed, is_diffusion=True,
                               override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    dev = args.device; norm = exp.dataset.normalizer
    res = {"loadpath": cli.loadpath, "epoch": str(exp.epoch)}

    for tag, m in ((("LIVE", exp.diffusion), ("EMA", exp.ema))
                   if not cli.skip_offline else ()):
        res[tag] = loss_probe(m, exp.dataset, dev)
        ae, se = replay_error(m, exp.dataset, dev)
        res[tag].update(replay_action_error=ae, replay_state_error=se)
        print(f"{tag}: {res[tag]}", flush=True)

    # imagination panel, EMA and LIVE, identical conditions
    ep0 = pickle.load(open(f"experiments/isaacgym_episode_sets/{SETS[0][1]}", "rb"))
    nE = env.num_envs
    obs = env.reset(set_init_states=array_to_state_dict(ep0["init"][:nE], ep0["keys"], dev),
                    set_goal_states=array_to_state_dict(ep0["goal"][:nE], ep0["keys"], dev))
    o = np.asarray(obs["achieved_goal"]).reshape(nE, -1)
    g = np.asarray(obs["desired_goal"]).reshape(nE, -1)
    cond = {0: torch.as_tensor(norm.normalize(o, "observations"), device=dev, dtype=torch.float32),
            args.horizon - 1: torch.as_tensor(norm.normalize(g, "observations"), device=dev, dtype=torch.float32)}
    front = env.env.obs_dict["media"][:, 0]
    _, z_bg = extract_dlp_features_with_bg(front, env.latent_rep_model, dev)
    for tag, m in ((("EMA", exp.ema), ("LIVE", exp.diffusion))
                   if not cli.skip_offline else ()):
        imag = {}
        for nfe in NFES:
            torch.manual_seed(NOISE_SEED)
            kw = dict(verbose=False)
            if hasattr(m, "n_solver_steps"): kw["n_steps"] = nfe
            with torch.no_grad(): s = m.conditional_sample(cond, **kw)
            xs = s.trajectories if hasattr(s, "trajectories") else s
            oo = norm.unnormalize(np.asarray(xs.cpu())[..., m.action_dim:], "observations")
            imag[nfe] = oo.reshape(len(oo), args.horizon, 2, PPV, 10)
        for e in range(2):
            fig, ax = plt.subplots(len(NFES), args.horizon,
                                   figsize=(2.5 * args.horizon, 2.5 * len(NFES)))
            for r, nfe in enumerate(NFES):
                for c in range(args.horizon):
                    ax[r, c].imshow(decode_view(imag[nfe][e, c, 0], z_bg[e], env.latent_rep_model, dev))
                    ax[r, c].axis("off")
                    if r == 0:
                        ax[r, c].set_title("current (cond)" if c == 0 else
                                           ("GOAL CONDITION" if c == args.horizon - 1
                                            else f"predicted t={c}"), fontsize=11)
                ax[r, 0].text(-0.16, 0.5, f"MF NFE{nfe}", transform=ax[r, 0].transAxes,
                              fontsize=12, va="center", ha="right", fontweight="bold")
            fig.suptitle(f"MeanFlow 20k imagination ({tag}) - episode {e}", fontsize=14)
            fig.tight_layout()
            p = f"{OUT}/mf20k_imagination_{tag}_ep{e}.png"
            fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p, flush=True)

    if not cli.skip_control:
        sys.path.insert(0, "diffuser")
        from diffuser.utils import gpu_guard
        res["control"] = {}
        me = os.getpid()
        todo = [(n, f) for n, f in SETS
                if cli.only_sets is None or n in cli.only_sets]
        for name, fn in todo:
            pre = gpu_guard.snapshot()
            foreign = [a for a in pre["compute_apps"]
                       if a["pid"] != me and a["used_mib"] >= 1000]
            if foreign:
                res["control"][name] = dict(status="ABORTED_CONTENDED", foreign=foreign,
                                            pre_snapshot=pre)
                print(f"control {name}: ABORTED - foreign GPU process present: {foreign}",
                      flush=True)
                continue
            ep = pickle.load(open(f"experiments/isaacgym_episode_sets/{fn}", "rb"))
            t0 = time.time()
            sr, n = control(exp.ema, env, args, ep, dev, norm)
            post = gpu_guard.snapshot()
            foreign_post = [a for a in post["compute_apps"]
                            if a["pid"] != me and a["used_mib"] >= 1000]
            res["control"][name] = dict(
                success=sr, n=n, seconds=time.time() - t0,
                status="VALID" if not foreign_post else "INVALID_CONTENDED_DURING",
                pre_snapshot=pre, post_snapshot=post, foreign_during=foreign_post)
            flag = "" if not foreign_post else "  !! CONTENDED DURING RUN - INVALID"
            print(f"control {name}: success={sr:.4f} (n={n}, {time.time()-t0:.0f}s)"
                  f"{flag}", flush=True)
        if cli.merge and os.path.exists(f"{OUT}/phase2_20k_diagnostics.json"):
            prev = json.load(open(f"{OUT}/phase2_20k_diagnostics.json"))
            merged = dict(prev.get("control", {}))
            merged.update(res["control"])          # new results win
            res = {**prev, **res, "control": merged}
        valid = [res["control"][k]["success"] for k, _ in SETS
                 if res["control"].get(k, {}).get("status") == "VALID"]
        res["n_valid_sets"] = len(valid)
        if len(valid) >= 2:
            res["control_mean"] = float(np.mean(valid))
            res["control_sd"] = float(np.std(valid, ddof=1))
        else:
            res["control_mean"] = res["control_sd"] = None
    json.dump(res, open(f"{OUT}/phase2_20k_diagnostics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
