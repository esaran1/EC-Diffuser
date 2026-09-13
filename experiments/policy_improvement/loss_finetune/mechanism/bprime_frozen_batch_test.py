"""Real-path B vs B' frozen-batch test (directive section 7).

Drives the SAME production construction path train.py uses
(ArgsParser -> diffusion_wrapper_kwargs -> utils.Config), loads the shared
canonical checkpoint, rebuilds the loss matrix exactly as train.py does, and on
ONE frozen batch with frozen noise and frozen time requires:

  forward predictions       BIT IDENTICAL
  unweighted losses         IDENTICAL
  numerators                IDENTICAL
  mask structure            IDENTICAL (terminal contribution = 0 in both)
  weighted total ratio      B'/B == 46464/46560 within numerical tolerance

If they differ in any other way, STOP before GPU.

CPU only.
"""
import hashlib
import json
import os
import pickle
import sys

C_EXPECTED = 46464.0 / 46560.0
REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")
CANONICAL = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
             "state_400000.pt")
FROZEN = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"


def build(config_name, seed, torch, utils, dwk):
    sys.argv = ["train.py", "--config", config_name,
                "--num_entity", "3", "--rand_color",
                "--seed", str(seed), "--device", "cpu"]
    from diffuser.utils.args import ArgsParser
    args = ArgsParser().parse_args("diffusion")
    with open(REF_CFG, "rb") as fh:
        ref = pickle.load(fh)
    obs_dim = int(ref._dict["observation_dim"])
    act_dim = int(ref._dict["action_dim"])
    mc = utils.Config(
        args.model, savepath=None, verbose=False,
        features_dim=args.features_dim, action_dim=act_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview, device="cpu")
    dc = utils.Config(args.diffusion, savepath=None, verbose=False, device="cpu",
                      **dwk(args, obs_dim, act_dim))
    diff = dc(mc()).to("cpu").float()
    diff.load_state_dict(torch.load(CANONICAL, map_location="cpu")["model"])
    diff.rebuild_loss_weight_matrix()
    diff.eval()
    return args, diff


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    import diffuser.utils as utils
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk
    from diffuser.utils.objective_guard import (describe_objective,
                                                verify_terminal_mask_effective)

    cache = torch.load(FROZEN, map_location="cpu")
    x1 = cache["x1"].float()
    cond = {int(k): v.float() for k, v in cache["cond"].items()}
    x0 = cache["x0"].float()
    t = cache["t"].float()
    cond_ts = tuple(sorted(cond))
    batch = x1.shape[0]

    arms = {}
    for name, cfg in (("B", "config.pandapush_lossft_armB_seed42"),
                      ("Bprime", "config.pandapush_lossft_armBprime_seed42")):
        args, diff = build(cfg, 42, torch, utils, dwk)
        obj = describe_objective(diff)
        probe = verify_terminal_mask_effective(diff, cond_ts, batch)
        with torch.no_grad():
            loss, info, det = diff._compute_flow_loss(x1, cond, x0=x0, t=t,
                                                      return_details=True)
            pred = diff._call_model(det["xt"], cond, det["time"] * diff.time_scale)
            err = torch.abs(pred - det["target_velocity"])
            if diff.loss_type == "l2":
                err = err.square()
            cmask = det["conditioning_mask"]
            aw = diff.loss_weight_matrix.unsqueeze(0) * cmask.to(x1.dtype)
            numerator = float((err * aw).sum().item())
            ad, H = diff.action_dim, diff.horizon
            term = float((err[:, H - 1, :ad] * aw[:, H - 1, :ad]).sum().item())
        arms[name] = {
            "objective": obj, "probe": probe,
            "pred_sha": hashlib.sha256(
                pred.detach().to(torch.float64).numpy().tobytes()).hexdigest()[:16],
            "total_loss": float(loss.item()),
            "numerator": numerator,
            "terminal_numerator": term,
            "unweighted_action": float(info["action_loss"].item()),
            "unweighted_observation": float(info["observation_loss"].item()),
            "mask_sum": int(cmask.sum().item()),
        }
        del diff

    B, Bp = arms["B"], arms["Bprime"]
    fails = []

    def chk(name, ok, detail=""):
        if not ok:
            fails.append("{}: {}".format(name, detail))
        print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                                   "" if ok else "  <- " + str(detail)))

    chk("forward predictions BIT IDENTICAL", B["pred_sha"] == Bp["pred_sha"],
        (B["pred_sha"], Bp["pred_sha"]))
    chk("numerators IDENTICAL", B["numerator"] == Bp["numerator"],
        (B["numerator"], Bp["numerator"]))
    chk("unweighted action loss IDENTICAL",
        B["unweighted_action"] == Bp["unweighted_action"])
    chk("unweighted observation loss IDENTICAL",
        B["unweighted_observation"] == Bp["unweighted_observation"])
    chk("both mask the terminal action",
        B["probe"]["terminal_action_masked"] and Bp["probe"]["terminal_action_masked"])
    chk("terminal numerator contribution is zero in both",
        B["terminal_numerator"] == 0.0 and Bp["terminal_numerator"] == 0.0,
        (B["terminal_numerator"], Bp["terminal_numerator"]))
    chk("loss weight matrices identical",
        B["objective"]["loss_weight_matrix_sha16"]
        == Bp["objective"]["loss_weight_matrix_sha16"])
    chk("conditioning mask sums identical (same structure)",
        B["mask_sum"] == Bp["mask_sum"], (B["mask_sum"], Bp["mask_sum"]))
    chk("B effective denominator == 46464",
        B["probe"]["effective_loss_denominator"] == 46464,
        B["probe"]["effective_loss_denominator"])
    chk("B' effective denominator == 46560 (A's value)",
        Bp["probe"]["effective_loss_denominator"] == 46560,
        Bp["probe"]["effective_loss_denominator"])

    # Tolerance is set by float32, not by the algebra. The numerator is verified
    # bit-identical above, so the only error is rounding in the final division;
    # float32 eps is 1.19e-7, and a few eps is the right scale. A tolerance of
    # 1e-9 would be tighter than the arithmetic can represent.
    F32_EPS = 2.0 ** -23
    ratio = Bp["total_loss"] / B["total_loss"]
    chk("B'/B total loss ratio == 46464/46560 (within float32 rounding)",
        abs(ratio - C_EXPECTED) < 5 * F32_EPS,
        "{:.12f} vs {:.12f}  (|diff| {:.3e} = {:.2f} x f32 eps)".format(
            ratio, C_EXPECTED, abs(ratio - C_EXPECTED),
            abs(ratio - C_EXPECTED) / F32_EPS))

    print("\n  B  total loss {:.12f}   denominator {}".format(
        B["total_loss"], B["probe"]["effective_loss_denominator"]))
    print("  B' total loss {:.12f}   denominator {}".format(
        Bp["total_loss"], Bp["probe"]["effective_loss_denominator"]))
    print("  ratio {:.12f}   expected c = {:.12f}   |diff| {:.3e}".format(
        ratio, C_EXPECTED, abs(ratio - C_EXPECTED)))

    out = {"arms": arms, "ratio": ratio, "c_expected": C_EXPECTED,
           "verdict": "FAIL" if fails else "PASS", "failures": fails}
    d = "experiments/policy_improvement/loss_finetune/mechanism"
    with open(os.path.join(d, "BPRIME_FROZEN_BATCH.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print("\n  VERDICT: {}".format(out["verdict"]))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
