"""First-step contamination persistence (directive section 27).

How long does the single contaminated first Adam update leave a measurable
trajectory difference?

Compares, at every checkpoint both protocols share:

    A_clean_s  vs  A_legacy_s
    B_clean_s  vs  B_legacy_s

Metrics: parameter relative RMS, EMA relative RMS, frozen-bank output RMS, and
the executed t=0 action disagreement. The clean and legacy arms are otherwise
identical -- same canonical source, same seed, same dataloader seed, same
objective -- so any difference is attributable to the first-step contamination
and its downstream amplification.

For scale, the within-protocol treatment effect (A vs B at the same step and
protocol) is reported alongside, so persistence can be read against the size of
the intervention itself.

Zero GPU: parameters come from checkpoints; outputs are evaluated on CPU against
the frozen batch already used by every other offline comparison in this program.
"""
import json
import os
import pickle
import sys

SUB_T = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")
FROZEN = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"
LABELS = ["early10", "500", "1000", "1500", "final"]
SEEDS = [42, 43, 44, 45]


def ckpt(protocol, arm, seed, label):
    root = "data/panda_push/lossft_arm{}{}".format(
        arm, "clean" if protocol == "clean" else "")
    return os.path.join(root, SUB_T.format(seed), "state_{}.pt".format(label))


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    import diffuser.utils as utils
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk
    from diffuser.utils.args import ArgsParser

    sys.argv = ["t", "--config", "config.pandapush_lossft_armB_seed42",
                "--num_entity", "3", "--rand_color", "--seed", "42", "--device", "cpu"]
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
        max_particles=args.max_particles, multiview=args.multiview, device="cpu")
    dc = utils.Config(args.diffusion, savepath=None, verbose=False, device="cpu",
                      **dwk(args, obs_dim, act_dim))
    model = dc(mc()).to("cpu").float()

    cache = torch.load(FROZEN, map_location="cpu")
    x1 = cache["x1"].float()
    cond = {int(k): v.float() for k, v in cache["cond"].items()}
    x0 = cache["x0"].float()
    t = cache["t"].float()

    def outputs(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            _, _, det = model._compute_flow_loss(x1, cond, x0=x0, t=t,
                                                 return_details=True)
            return model._call_model(det["xt"], cond,
                                     det["time"] * model.time_scale).double()

    def rel_rms(sd_a, sd_b):
        num = den = 0.0
        for k, v in sd_a.items():
            if k == "loss_weight_matrix" or not torch.is_tensor(v) \
                    or not v.is_floating_point():
                continue
            d = sd_b[k].double() - v.double()
            num += float((d ** 2).sum())
            den += float((v.double() ** 2).sum())
        return (num / den) ** 0.5 if den else None

    results, missing = {}, []
    for seed in SEEDS:
        entry = {}
        for label in LABELS:
            row = {}
            for arm in ("A", "B"):
                pl = ckpt("legacy", arm, seed, label)
                pc = ckpt("clean", arm, seed, label)
                if not (os.path.exists(pl) and os.path.exists(pc)):
                    missing.append("{}{} {}".format(arm, seed, label))
                    continue
                L = torch.load(pl, map_location="cpu")
                C = torch.load(pc, map_location="cpu")
                oL, oC = outputs(L["ema"]), outputs(C["ema"])
                ad = act_dim
                row[arm] = {
                    "param_rel_rms_clean_vs_legacy": rel_rms(L["model"], C["model"]),
                    "ema_rel_rms_clean_vs_legacy": rel_rms(L["ema"], C["ema"]),
                    "output_rms_clean_vs_legacy": float(
                        ((oC - oL) ** 2).mean().sqrt()),
                    "t0_action_rms_clean_vs_legacy": float(
                        ((oC[:, 0, :ad] - oL[:, 0, :ad]) ** 2).mean().sqrt()),
                }
                del L, C
            # treatment-effect scale at the same step, within each protocol
            for proto in ("legacy", "clean"):
                pa = ckpt(proto, "A", seed, label)
                pb = ckpt(proto, "B", seed, label)
                if os.path.exists(pa) and os.path.exists(pb):
                    PA = torch.load(pa, map_location="cpu")
                    PB = torch.load(pb, map_location="cpu")
                    row["{}_A_vs_B_param_rel_rms".format(proto)] = rel_rms(
                        PA["model"], PB["model"])
                    del PA, PB
            if row:
                entry[label] = row
        if entry:
            results[seed] = entry

    if missing:
        print("missing checkpoint pairs ({}): {}".format(
            len(missing), ", ".join(missing[:10])))
    if not results:
        print("no clean/legacy checkpoint pairs available yet")
        return

    print("=" * 96)
    print("FIRST-STEP CONTAMINATION PERSISTENCE (clean vs legacy, same seed/arm)")
    print("=" * 96)
    for seed in sorted(results):
        print("\nseed {}".format(seed))
        print("  %-9s %-6s %14s %14s %14s | %14s" % (
            "step", "arm", "param relRMS", "ema relRMS", "t0 out RMS",
            "A-vs-B param"))
        for label in LABELS:
            if label not in results[seed]:
                continue
            row = results[seed][label]
            for arm in ("A", "B"):
                if arm not in row:
                    continue
                r = row[arm]
                ab = row.get("legacy_A_vs_B_param_rel_rms")
                print("  %-9s %-6s %14.4e %14.4e %14.4e | %14s" % (
                    label, arm, r["param_rel_rms_clean_vs_legacy"],
                    r["ema_rel_rms_clean_vs_legacy"],
                    r["t0_action_rms_clean_vs_legacy"],
                    "{:.4e}".format(ab) if ab else "n/a"))

    out = ("experiments/policy_improvement/loss_finetune/clean_first_step/"
           "FIRST_STEP_PERSISTENCE.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
