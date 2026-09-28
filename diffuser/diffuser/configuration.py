"""Dependency-free construction helpers shared by training and planning entrypoints."""


def diffusion_wrapper_kwargs(args, observation_dim, action_dim):
    """Return the exact wrapper arguments represented by parsed training config."""
    kwargs = {
        "horizon": args.horizon,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "n_timesteps": args.n_diffusion_steps,
        "loss_type": args.loss_type,
        "clip_denoised": args.clip_denoised,
        "predict_epsilon": args.predict_epsilon,
        "action_weight": args.action_weight,
        "loss_weights": args.loss_weights,
        "loss_discount": args.loss_discount,
        "obs_only": args.obs_only,
        "action_only": args.action_only,
    }
    if hasattr(args, "time_scale"):
        kwargs["time_scale"] = args.time_scale
    # Loss-screen knobs. These MUST be forwarded explicitly: this function is a
    # whitelist, so anything omitted silently falls back to the wrapper's
    # constructor default. Omitting them made every loss arm train arm A's
    # objective and produce byte-identical checkpoints.
    # Two-time / UWM knobs (tal_followup program). Same whitelist hazard: if
    # p_diag were omitted here, a "faithful UWM" arm would silently train with
    # the wrapper default and be indistinguishable from the control.
    for _k in ("mask_terminal_action", "lambda_action",
               "denominator_ignores_terminal_mask",
               "p_diag", "adaln_state_weight", "action_vae_path",
               "prev_action_sigma"):
        if hasattr(args, _k):
            kwargs[_k] = getattr(args, _k)
    return kwargs


def flow_sampling_kwargs(diffusion_model, configured_steps):
    """Return a planning step override only for flow-matching wrappers."""
    if hasattr(diffusion_model, "n_solver_steps"):
        return {"n_steps": configured_steps}
    return {}
