"""Mechanism: deterministic output-drift analysis (directive section 26).

For each A/B seed pair, measure how much the policy MAPPING changed, on an
identical neutral input bank with identical noise and flow times, and compare
that to how much each arm moved from the canonical starting point.

Question: does B change only tiny action values, or does it systematically alter
the policy map despite near-identical ground-truth error?

Reports, per seed:
    t0 action RMS difference        (executed action)
    t1..H-2 action RMS difference   (mid-horizon)
    terminal action RMS difference  (the masked placeholder, reported separately)
    state/observation RMS difference
    mean cosine between A and B action outputs at t=0
each also computed for A-vs-canonical and B-vs-canonical, so the A-B change can
be read against the size of the fine-tuning displacement itself.

CPU only; uses the frozen batch already used by the objective tests so the inputs
are identical to every other offline comparison in this program.
"""
import json
import os
import pickle
import sys

SUB_TMPL = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")
CANONICAL = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
             "state_400000.pt")
FROZEN = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"


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
    ad, H = act_dim, model.horizon

    def outputs(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            _, _, det = model._compute_flow_loss(x1, cond, x0=x0, t=t,
                                                 return_details=True)
            return model._call_model(det["xt"], cond,
                                     det["time"] * model.time_scale).double()

    def rms(a, b, sl):
        d = (a[sl] - b[sl])
        return float((d ** 2).mean().sqrt())

    canon = outputs(torch.load(CANONICAL, map_location="cpu")["model"])
    results = {}
    for seed in (42, 43, 44, 45):
        sub = SUB_TMPL.format(seed)
        oA = outputs(torch.load(
            "data/panda_push/lossft_armA/{}/state_final.pt".format(sub),
            map_location="cpu")["ema"])
        oB = outputs(torch.load(
            "data/panda_push/lossft_armB/{}/state_final.pt".format(sub),
            map_location="cpu")["ema"])

        slices = {
            "t0_action": (slice(None), 0, slice(0, ad)),
            "mid_action": (slice(None), slice(1, H - 1), slice(0, ad)),
            "terminal_action": (slice(None), H - 1, slice(0, ad)),
            "state": (slice(None), slice(None), slice(ad, None)),
        }
        entry = {}
        for name, sl in slices.items():
            entry[name] = {
                "A_vs_B": rms(oA, oB, sl),
                "A_vs_canonical": rms(oA, canon, sl),
                "B_vs_canonical": rms(oB, canon, sl),
            }
        a0 = oA[:, 0, :ad]
        b0 = oB[:, 0, :ad]
        cos = torch.nn.functional.cosine_similarity(a0, b0, dim=-1)
        entry["t0_action_cosine_A_vs_B"] = {
            "mean": float(cos.mean()), "min": float(cos.min())}
        results[seed] = entry

        print("seed {}: t0 A-B {:.5e} (A-canon {:.5e}, B-canon {:.5e})  "
              "terminal A-B {:.5e}  state A-B {:.5e}  t0cos {:.6f}".format(
                  seed, entry["t0_action"]["A_vs_B"],
                  entry["t0_action"]["A_vs_canonical"],
                  entry["t0_action"]["B_vs_canonical"],
                  entry["terminal_action"]["A_vs_B"],
                  entry["state"]["A_vs_B"],
                  entry["t0_action_cosine_A_vs_B"]["mean"]), flush=True)

    out = ("experiments/policy_improvement/loss_finetune/mechanism/"
           "OUTPUT_DRIFT.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("written", out)


if __name__ == "__main__":
    main()
