"""Stage B analysis: how far does the denominator-only change move the model,
RELATIVE to the masking intervention itself?

Predeclared gate (directive section 10), evaluated at step 500:

    R_param  = ||theta_B' - theta_B|| / ||theta_B - theta_A||   < 0.05
    R_output = ||f_B'    - f_B||     / ||f_B    - f_A||         < 0.05
    executed-action B vs B' difference < 5% of the B vs A difference
    no clipping / optimizer mechanism invalidating the constant-scale argument

The B vs A reference is taken at the SAME step from the accepted scientific runs,
so numerator and denominator of each ratio are matched in training length.

Note on coverage: the accepted A and B runs saved checkpoints at 500/1000/1500,
so **step 500 is the only point with a matched A-vs-B reference** -- which is
exactly where the predeclared gate is evaluated. Steps 10/100/250 report the
absolute B'-vs-B divergence trajectory (useful for showing it is not growing
explosively) but no ratio, because no matched reference exists there. The gate is
not weakened by this; it was always specified at step 500.

Zero GPU: parameter norms are read from checkpoints; outputs are evaluated on CPU
against the frozen batch already used for the objective tests.
"""
import hashlib
import json
import os
import pickle
import sys

PROBE = "experiments/policy_improvement/loss_finetune/mechanism/probe_runs"
SCI = "data/panda_push"
SUB = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
FROZEN = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"
REF_CFG = os.path.join(SCI, "lossft_armA", SUB, "diffusion_config.pkl")
CANONICAL = os.path.join(SCI, "flow", SUB, "state_400000.pt")
STEPS = (10, 100, 250, 500)


def param_vector_diff(sd_a, sd_b, torch):
    """Relative-free absolute L2 distance over trainable float params (no buffers)."""
    total = 0.0
    for k, v in sd_a.items():
        if k == "loss_weight_matrix" or not torch.is_tensor(v) or not v.is_floating_point():
            continue
        d = (sd_b[k].double() - v.double())
        total += float((d ** 2).sum())
    return total ** 0.5


def param_norm(sd, torch):
    total = 0.0
    for k, v in sd.items():
        if k == "loss_weight_matrix" or not torch.is_tensor(v) or not v.is_floating_point():
            continue
        total += float((v.double() ** 2).sum())
    return total ** 0.5


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    import diffuser.utils as utils
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk

    bdir = os.path.join(PROBE, "pandapush_lossft_armB_seed42")
    pdir = os.path.join(PROBE, "pandapush_lossft_armBprime_seed42")
    for d in (bdir, pdir):
        if not os.path.exists(os.path.join(d, "PROBE_COMPLETE.json")):
            raise SystemExit("probe not complete: {}".format(d))

    # --- scientific B vs A reference at matched steps ------------------------
    sci_B = os.path.join(SCI, "lossft_armB", SUB)
    sci_A = os.path.join(SCI, "lossft_armA", SUB)

    results = {"steps": {}, "notes": []}

    # Build one model to evaluate outputs through the production path.
    sys.argv = ["t", "--config", "config.pandapush_lossft_armB_seed42",
                "--num_entity", "3", "--rand_color", "--seed", "42", "--device", "cpu"]
    from diffuser.utils.args import ArgsParser
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
    ad = act_dim

    def outputs(state_dict):
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            _, _, det = model._compute_flow_loss(x1, cond, x0=x0, t=t,
                                                 return_details=True)
            pred = model._call_model(det["xt"], cond, det["time"] * model.time_scale)
        return pred.detach().double()

    for step in STEPS:
        bp = os.path.join(bdir, "state_probe{}.pt".format(step))
        pp = os.path.join(pdir, "state_probe{}.pt".format(step))
        if not (os.path.exists(bp) and os.path.exists(pp)):
            continue
        B = torch.load(bp, map_location="cpu")
        P = torch.load(pp, map_location="cpu")

        entry = {"step": step}
        # parameter divergence B' vs B
        d_bp = param_vector_diff(B["model"], P["model"], torch)
        d_bp_ema = param_vector_diff(B["ema"], P["ema"], torch)
        entry["abs_param_diff_Bprime_vs_B"] = d_bp
        entry["abs_param_diff_Bprime_vs_B_ema"] = d_bp_ema
        entry["param_norm_B"] = param_norm(B["model"], torch)
        entry["rel_rms_Bprime_vs_B"] = d_bp / entry["param_norm_B"]

        # scientific reference: B vs A at the same step, from the accepted runs
        sb = os.path.join(sci_B, "state_{}.pt".format(step))
        sa = os.path.join(sci_A, "state_{}.pt".format(step))
        if os.path.exists(sb) and os.path.exists(sa):
            SB = torch.load(sb, map_location="cpu")
            SA = torch.load(sa, map_location="cpu")
            d_ba = param_vector_diff(SA["model"], SB["model"], torch)
            entry["abs_param_diff_B_vs_A"] = d_ba
            entry["R_param"] = d_bp / d_ba if d_ba > 0 else None
            # outputs
            oB, oP = outputs(B["model"]), outputs(P["model"])
            oSA, oSB = outputs(SA["model"]), outputs(SB["model"])
            num = float(((oP - oB) ** 2).sum().sqrt())
            den = float(((oSB - oSA) ** 2).sum().sqrt())
            entry["abs_output_diff_Bprime_vs_B"] = num
            entry["abs_output_diff_B_vs_A"] = den
            entry["R_output"] = num / den if den > 0 else None
            # executed action t=0 only
            nA = float(((oP[:, 0, :ad] - oB[:, 0, :ad]) ** 2).sum().sqrt())
            dA = float(((oSB[:, 0, :ad] - oSA[:, 0, :ad]) ** 2).sum().sqrt())
            entry["abs_t0_diff_Bprime_vs_B"] = nA
            entry["abs_t0_diff_B_vs_A"] = dA
            entry["R_t0"] = nA / dA if dA > 0 else None
            del SA, SB
        results["steps"][step] = entry
        del B, P
        print("step {:>4}: |B'-B|={:.4e}  |B-A|={:.4e}  R_param={}  R_output={}  R_t0={}"
              .format(step, entry["abs_param_diff_Bprime_vs_B"],
                      entry.get("abs_param_diff_B_vs_A", float("nan")),
                      "{:.4f}".format(entry["R_param"]) if entry.get("R_param") else "n/a",
                      "{:.4f}".format(entry["R_output"]) if entry.get("R_output") else "n/a",
                      "{:.4f}".format(entry["R_t0"]) if entry.get("R_t0") else "n/a"),
              flush=True)

    # --- predeclared gate at step 500 ---------------------------------------
    final = results["steps"].get(500, {})
    gate = {
        "R_param<0.05": (final.get("R_param") is not None and final["R_param"] < 0.05),
        "R_output<0.05": (final.get("R_output") is not None and final["R_output"] < 0.05),
        "R_t0<0.05": (final.get("R_t0") is not None and final["R_t0"] < 0.05),
        "no_clipping": True,   # max_grad_norm=None verified in DENOMINATOR_THEORY.md
    }
    gate["ALL_PASS"] = all(gate.values())
    results["gate"] = gate
    results["verdict"] = ("DENOMINATOR-NEGLIGIBLE" if gate["ALL_PASS"]
                          else "DENOMINATOR-NOT-NEGLIGIBLE")

    print("\nPREDECLARED GATE AT STEP 500:")
    for k, v in gate.items():
        print("  {:<18} {}".format(k, v))
    print("\n  VERDICT: {}".format(results["verdict"]))

    out = os.path.join("experiments/policy_improvement/loss_finetune/mechanism",
                       "BPRIME_PROBE_DIVERGENCE.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print("written", out)


if __name__ == "__main__":
    main()
