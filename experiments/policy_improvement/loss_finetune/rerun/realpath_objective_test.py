"""Real-path end-to-end objective test + one-step divergence sentinel.

This is the test the failed attempt was missing.

The old `verify_pairing.py` built the diffusion wrapper DIRECTLY, passing
`mask_terminal_action` and `lambda_action` as explicit kwargs -- thereby
bypassing `diffusion_wrapper_kwargs`, the very function that was dropping them.
It verified the configs parsed correctly, not that the values reached the model.
It reported PAIRED, correctly for what it tested and uselessly for what mattered.

This script instead drives the SAME production path `train.py` uses:

    ArgsParser().parse_args('diffusion')  ->  diffusion_wrapper_kwargs(...)
      ->  utils.Config(args.diffusion, ...)  ->  wrapper instance

and then asserts, on ONE frozen batch with ONE frozen noise draw and ONE frozen
time draw:

  PREDICTIONS  identical across A/B/C/D   (same weights, same input)
  OBJECTIVE    differs exactly as intended (different weighting/masking)

That distinction -- same model output, different objective -- is the whole point.

It then performs the one-step divergence sentinel (directive section 9): one real
optimizer update per arm from identical state, requiring B/C/D to stop being
byte-identical afterwards. If they remain identical, the objective is still not
reaching optimization and no 2,000-step run should be launched.

Runs on CPU. Invoked per-arm as a subprocess because ArgsParser reads argv and
the config modules are imported by name.
"""
import argparse
import copy
import hashlib
import json
import os
import sys


def hash_params(module):
    """Order-stable hash of TRAINABLE parameters only.

    Deliberately excludes buffers. `loss_weight_matrix` is a registered buffer
    that differs across arms by construction, so hashing the full state_dict
    would make C and D look divergent purely because their weight matrices
    differ -- the very thing the sentinel is supposed to test independently.
    The sentinel must show that different objectives move the WEIGHTS.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().to(torch_float64).numpy().tobytes())
    return digest.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--batch_cache", required=True)
    parser.add_argument("--ref_config", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    cli = parser.parse_args()

    # ArgsParser consumes sys.argv; hand it exactly what the production runner
    # passes. The mode-selecting flags matter: without --num_entity/--rand_color
    # the parser falls back to a different mode whose architecture is 256-dim,
    # not the canonical 512-dim, and the canonical weights will not load.
    sys.argv = [
        "train.py",
        "--config", "config.pandapush_lossft_arm{}".format(cli.arm),
        "--num_entity", "3", "--rand_color", "--seed", "42",
        "--device", "cpu",
    ]

    global torch_float64
    import torch
    torch_float64 = torch.float64

    from diffuser.utils.args import ArgsParser
    from diffuser.configuration import diffusion_wrapper_kwargs
    import diffuser.utils as utils
    from diffuser.utils.objective_guard import describe_objective, verify_terminal_mask_effective

    args = ArgsParser().parse_args("diffusion")

    # --- PRODUCTION PATH: identical to train.py ----------------------------
    # Dimensions come from the config Arm A actually trained with, not from a
    # guess. (An early draft of this script assumed action_dim=4; the real
    # value is 3. Guessing dimensions is how the original defect survived.)
    import pickle
    with open(cli.ref_config, "rb") as handle:
        ref = pickle.load(handle)
    observation_dim = int(ref._dict["observation_dim"])
    action_dim = int(ref._dict["action_dim"])

    model_config = utils.Config(
        args.model,
        savepath=None,
        features_dim=args.features_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        n_head=args.n_heads,
        n_layer=args.n_layers,
        dropout=args.dropout,
        block_size=args.horizon,
        positional_bias=args.positional_bias,
        max_particles=args.max_particles,
        multiview=args.multiview,
        device="cpu",
        verbose=False,
    )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=None,
        device="cpu",
        verbose=False,
        **diffusion_wrapper_kwargs(args, observation_dim, action_dim)
    )
    # Fail loudly if the parsed architecture drifts from what Arm A trained.
    # A silent fallback to a different mode gives a 256-dim model that cannot
    # load the canonical 512-dim weights; better to say so than to limp on.
    ref_model_cfg = os.path.join(os.path.dirname(cli.ref_config), "model_config.pkl")
    with open(ref_model_cfg, "rb") as handle:
        ref_model = pickle.load(handle)
    for key in ("hidden_dim", "projection_dim", "n_head", "n_layer", "block_size",
                "features_dim", "action_dim", "multiview"):
        want = ref_model._dict[key]
        got = model_config._dict[key]
        if want != got:
            raise SystemExit(
                "architecture mismatch vs Arm A on {}: reference {!r}, parsed {!r}".format(
                    key, want, got))

    model = model_config()
    diffusion = diffusion_config(model)
    diffusion = diffusion.to("cpu").float()

    # --- identical canonical weights for every arm -------------------------
    canonical = torch.load(cli.canonical, map_location="cpu")
    diffusion.load_state_dict(canonical["model"])
    # Mirror the production warm-start exactly: loss_weight_matrix is a buffer,
    # so the load above restores the canonical lambda=1 matrix over the
    # configured one. train.py rebuilds it here; so must this test, or the test
    # would not be exercising the real path.
    rebuilt_before, rebuilt_after = diffusion.rebuild_loss_weight_matrix()
    diffusion.eval()

    objective = describe_objective(diffusion)
    objective["matrix_sum_restored_from_checkpoint"] = rebuilt_before
    objective["matrix_sum_after_rebuild"] = rebuilt_after
    mask_probe = verify_terminal_mask_effective(diffusion)

    # --- identical frozen batch / noise / time -----------------------------
    cache = torch.load(cli.batch_cache, map_location="cpu")
    x1 = cache["x1"].float()
    cond = {int(k): v.float() for k, v in cache["cond"].items()}
    x0 = cache["x0"].float()
    t = cache["t"].float()

    with torch.no_grad():
        loss, info, details = diffusion._compute_flow_loss(
            x1, cond, x0=x0, t=t, return_details=True)
        prediction = diffusion._call_model(
            details["xt"], cond, details["time"] * diffusion.time_scale)

        # Decompose the objective into the pieces the directive asks for.
        error = torch.abs(prediction - details["target_velocity"])
        if diffusion.loss_type == "l2":
            error = error.square()
        cmask = details["conditioning_mask"]
        weights = diffusion.loss_weight_matrix.unsqueeze(0)
        active = weights * cmask.to(x1.dtype)
        ad = diffusion.action_dim
        H = diffusion.horizon
        denom = cmask.sum()

        weighted_action = (error[:, :, :ad] * active[:, :, :ad]).sum() / denom
        weighted_obs = (error[:, :, ad:] * active[:, :, ad:]).sum() / denom
        term_contrib = (error[:, H - 1, :ad] * active[:, H - 1, :ad]).sum() / denom

    result = {
        "arm": cli.arm,
        "objective": objective,
        "terminal_mask_probe": mask_probe,
        "prediction_sha16": hashlib.sha256(
            prediction.detach().to(torch.float64).numpy().tobytes()).hexdigest()[:16],
        "model_params_sha16_before": hash_params(diffusion),
        "losses": {
            "total_loss": float(loss.item()),
            "unweighted_action_loss": float(info["action_loss"].item()),
            "unweighted_observation_loss": float(info["observation_loss"].item()),
            "unweighted_flow_loss": float(info["unweighted_flow_loss"].item()),
            "weighted_action_contribution": float(weighted_action.item()),
            "weighted_observation_contribution": float(weighted_obs.item()),
            "terminal_action_contribution": float(term_contrib.item()),
            "conditioning_mask_denominator": float(denom.item()),
        },
    }

    # --- ONE-STEP DIVERGENCE SENTINEL (directive section 9) ----------------
    # Fresh Adam, identical start, exactly one optimizer update, then hash.
    sentinel = copy.deepcopy(diffusion)
    sentinel.train()
    optimizer = torch.optim.Adam(sentinel.parameters(), lr=cli.lr)
    step_loss, _ = sentinel._compute_flow_loss(x1, cond, x0=x0, t=t)
    optimizer.zero_grad()
    step_loss.backward()

    grad_sq = 0.0
    action_grad_sq = 0.0
    for name, p in sentinel.named_parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.detach().pow(2).sum().item())
            if "action" in name:
                action_grad_sq += float(p.grad.detach().pow(2).sum().item())
    optimizer.step()

    result["one_step"] = {
        "loss_before_update": float(step_loss.item()),
        "grad_norm": grad_sq ** 0.5,
        "action_grad_norm": action_grad_sq ** 0.5,
        "model_params_sha16_after_one_step": hash_params(sentinel),
    }

    with open(cli.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
