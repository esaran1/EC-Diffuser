"""Paired A/B verification for one replication seed.

`--stage preflight` (zero GPU, before training):
  Builds BOTH arms through the production path (ArgsParser ->
  diffusion_wrapper_kwargs -> utils.Config), loads the shared canonical
  checkpoint, rebuilds the loss matrix as train.py does, and requires:

    COMMON INVARIANTS   model/EMA hashes, forward prediction, unweighted
                        semantic losses, architecture, observation weighting,
                        nonterminal action weighting, lambda, LR, warmup,
                        step_start_ema, dataloader seed, budget
    INTENDED DIFFERENCE only the terminal-action mask, and the inherited
                        active-coordinate denominator 46560 -> 46464
                        (ratio ~1.002066)

  If any common invariant fails, the pair is stopped BEFORE training.

`--stage postrun` (zero GPU, after both arms finish):
  Requires the trained A and B weights to actually differ, both finals to be at
  internal step 2000, and the loss matrices to remain identical (for A vs B the
  matrices ARE identical by design -- masking acts on the conditioning mask, not
  the weight matrix, so a matrix difference here would indicate contamination).

Exit 0 = pass, 1 = fail.
"""
import argparse
import hashlib
import json
import os
import sys

SUB_TMPL = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
CANONICAL = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
             "state_400000.pt")
FROZEN_BATCH = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"
REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")


def build(arm, seed, dataloader_seed, torch, utils, dwk):
    """Construct one arm exactly as train.py does, including the post-load rebuild."""
    sys.argv = ["train.py", "--config", "config.pandapush_lossft_arm{}".format(arm),
                "--num_entity", "3", "--rand_color",
                "--seed", str(seed), "--dataloader_seed", str(dataloader_seed),
                "--device", "cpu"]
    from diffuser.utils.args import ArgsParser
    args = ArgsParser().parse_args("diffusion")

    import pickle
    with open(REF_CFG, "rb") as fh:
        ref = pickle.load(fh)
    obs_dim = int(ref._dict["observation_dim"])
    act_dim = int(ref._dict["action_dim"])

    model_config = utils.Config(
        args.model, savepath=None, verbose=False,
        features_dim=args.features_dim, action_dim=act_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview, device="cpu")
    diffusion_config = utils.Config(
        args.diffusion, savepath=None, verbose=False, device="cpu",
        **dwk(args, obs_dim, act_dim))
    diff = diffusion_config(model_config()).to("cpu").float()

    data = torch.load(CANONICAL, map_location="cpu")
    diff.load_state_dict(data["model"])
    diff.rebuild_loss_weight_matrix()          # exactly as train.py does
    diff.eval()
    return args, diff


def phash(module, torch):
    d = hashlib.sha256()
    for n, t in sorted(module.named_parameters()):
        d.update(n.encode())
        d.update(t.detach().cpu().to(torch.float64).numpy().tobytes())
    return d.hexdigest()[:16]


def preflight(cli):
    import torch
    import diffuser.utils as utils
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk
    from diffuser.utils.objective_guard import (describe_objective,
                                                verify_terminal_mask_effective)

    cache = torch.load(FROZEN_BATCH, map_location="cpu")
    x1 = cache["x1"].float()
    cond = {int(k): v.float() for k, v in cache["cond"].items()}
    x0 = cache["x0"].float()
    t = cache["t"].float()
    cond_ts = tuple(sorted(cond))

    info = {}
    fails = []
    for arm in ("A", "B"):
        args, diff = build(arm, cli.seed, cli.dataloader_seed, torch, utils, dwk)
        obj = describe_objective(diff)
        probe = verify_terminal_mask_effective(diff, cond_ts, x1.shape[0])
        with torch.no_grad():
            loss, li, det = diff._compute_flow_loss(x1, cond, x0=x0, t=t,
                                                    return_details=True)
            pred = diff._call_model(det["xt"], cond, det["time"] * diff.time_scale)
        info[arm] = {
            "objective": obj, "probe": probe,
            "param_sha": phash(diff, torch),
            "pred_sha": hashlib.sha256(
                pred.detach().to(torch.float64).numpy().tobytes()).hexdigest()[:16],
            "total_loss": float(loss.item()),
            "unweighted_action": float(li["action_loss"].item()),
            "unweighted_observation": float(li["observation_loss"].item()),
            "args": {"seed": args.seed, "dataloader_seed": args.dataloader_seed,
                     "lr": float(args.learning_rate),
                     "warmup": int(args.lr_warmup_steps),
                     "step_start_ema": int(args.step_start_ema),
                     "n_train_steps": int(args.n_train_steps),
                     "action_weight": float(args.action_weight),
                     "warm_start": args.warm_start_checkpoint},
        }
        del diff

    A, B = info["A"], info["B"]

    def chk(name, ok, detail=""):
        if not ok:
            fails.append("{}: {}".format(name, detail))
        print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                                   "" if ok else "  <- " + str(detail)))

    # --- COMMON INVARIANTS --------------------------------------------------
    chk("same initial weights", A["param_sha"] == B["param_sha"],
        (A["param_sha"], B["param_sha"]))
    chk("same forward prediction", A["pred_sha"] == B["pred_sha"],
        (A["pred_sha"], B["pred_sha"]))
    chk("same canonical source", A["args"]["warm_start"] == B["args"]["warm_start"]
        == CANONICAL, (A["args"]["warm_start"], B["args"]["warm_start"]))
    for k in ("seed", "dataloader_seed", "lr", "warmup", "step_start_ema",
              "n_train_steps", "action_weight"):
        chk("same {}".format(k), A["args"][k] == B["args"][k],
            (A["args"][k], B["args"][k]))
    chk("same unweighted observation loss",
        A["unweighted_observation"] == B["unweighted_observation"],
        (A["unweighted_observation"], B["unweighted_observation"]))
    chk("observation weighting identical",
        A["objective"]["observation_block_sum"] == B["objective"]["observation_block_sum"],
        (A["objective"]["observation_block_sum"], B["objective"]["observation_block_sum"]))
    chk("lambda_action identical and 1.0",
        A["objective"]["lambda_action"] == B["objective"]["lambda_action"] == 1.0,
        (A["objective"]["lambda_action"], B["objective"]["lambda_action"]))
    chk("action weight matrix identical (mask is not a weight change)",
        A["objective"]["loss_weight_matrix_sha16"] == B["objective"]["loss_weight_matrix_sha16"],
        (A["objective"]["loss_weight_matrix_sha16"], B["objective"]["loss_weight_matrix_sha16"]))

    # --- INTENDED DIFFERENCE ------------------------------------------------
    chk("A terminal action ACTIVE", A["objective"]["mask_terminal_action"] is False
        and A["probe"]["terminal_action_masked"] is False, A["probe"])
    chk("B terminal action MASKED", B["objective"]["mask_terminal_action"] is True
        and B["probe"]["terminal_action_masked"] is True, B["probe"])
    dA = A["probe"]["loss_denominator"]
    dB = B["probe"]["loss_denominator"]
    chk("denominator A > B (supervision removed)", dA > dB, (dA, dB))
    ratio = dA / dB
    chk("renormalization ratio ~1.002066", abs(ratio - 1.0020661157024793) < 1e-9,
        ratio)
    chk("A/B unweighted action losses differ (different active set)",
        A["unweighted_action"] != B["unweighted_action"],
        (A["unweighted_action"], B["unweighted_action"]))

    print("  denominators: A={} B={} ratio={:.9f}".format(dA, dB, ratio))
    out = {"stage": "preflight", "seed": cli.seed,
           "dataloader_seed": cli.dataloader_seed,
           "A": A, "B": B, "denominator_ratio": ratio,
           "verdict": "FAIL" if fails else "PASS", "failures": fails}
    return out, fails


def postrun(cli):
    import torch
    sub = SUB_TMPL.format(cli.seed)
    fails = []
    st = {}
    for arm in ("A", "B"):
        p = os.path.join("data/panda_push", "lossft_arm{}".format(arm), sub,
                         "state_final.pt")
        if not os.path.exists(p):
            fails.append("missing final checkpoint for arm {}".format(arm))
            continue
        st[arm] = torch.load(p, map_location="cpu")
    if fails:
        return {"stage": "postrun", "verdict": "FAIL", "failures": fails}, fails

    def chk(name, ok, detail=""):
        if not ok:
            fails.append("{}: {}".format(name, detail))
        print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                                   "" if ok else "  <- " + str(detail)))

    for arm in ("A", "B"):
        chk("arm {} final internal step == 2000".format(arm),
            int(st[arm]["step"]) == 2000, st[arm]["step"])

    num = den = 0.0
    for k, v in st["A"]["model"].items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        if k == "loss_weight_matrix":
            continue
        d = st["B"]["model"][k].double() - v.double()
        num += float((d ** 2).sum())
        den += float((v.double() ** 2).sum())
    drift = (num / den) ** 0.5
    chk("A and B trained weights differ", drift > 0, drift)

    mA = float(st["A"]["model"]["loss_weight_matrix"].sum())
    mB = float(st["B"]["model"]["loss_weight_matrix"].sum())
    chk("loss matrices identical (A vs B differ by mask, not weights)",
        mA == mB, (mA, mB))

    print("  A->B relative RMS drift = {:.6e}".format(drift))
    return ({"stage": "postrun", "seed": cli.seed, "drift": drift,
             "matrix_sums": {"A": mA, "B": mB},
             "verdict": "FAIL" if fails else "PASS", "failures": fails}, fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataloader_seed", type=int, required=True)
    ap.add_argument("--stage", choices=("preflight", "postrun"), required=True)
    cli = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("[verify_pair] seed={} dataloader_seed={} stage={}".format(
        cli.seed, cli.dataloader_seed, cli.stage))

    out, fails = preflight(cli) if cli.stage == "preflight" else postrun(cli)

    d = "experiments/policy_improvement/loss_finetune/replication"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "PAIRING_seed{}_{}.json".format(cli.seed, cli.stage)),
              "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)

    print("[verify_pair] VERDICT: {}".format(out["verdict"]))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
