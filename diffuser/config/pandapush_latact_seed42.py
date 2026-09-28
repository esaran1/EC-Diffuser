"""Same-dim LATENT-ACTION arm (Tal: fully latent EC-Diffuser), 5k from canonical.

Matched to the diagonal control's first leg (canonical -> 5k, dataloader_seed
20260916, clean first step, same LR/warmup/EMA). The ONLY change: the action
channel is a frozen same-dim (z=3) VAE latent, gated in
experiments/policy_improvement/tal_followup/action_latent/action_vae_z3.json.
"""
from config.pandapush_uwm_ctl_seed42 import base as _base
from config.pandapush_uwm_ctl_seed42 import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
_VAE = "experiments/policy_improvement/tal_followup/action_latent/action_vae_z3.pt"
for k in ("p_diag", "adaln_state_weight"):
    base["diffusion"].pop(k, None)
base["diffusion"].update(
    prefix="latact/",
    model="models.AdaLNPINTDenoiser",
    diffusion="models.LatentActionFlowMatching",
    loader="datasets.LatentActionGoalDataset",
    action_vae_path=_VAE,
)
