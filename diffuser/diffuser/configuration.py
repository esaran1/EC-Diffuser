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
    return kwargs


def flow_sampling_kwargs(diffusion_model, configured_steps):
    """Return a planning step override only for flow-matching wrappers."""
    if hasattr(diffusion_model, "n_solver_steps"):
        return {"n_steps": configured_steps}
    return {}
