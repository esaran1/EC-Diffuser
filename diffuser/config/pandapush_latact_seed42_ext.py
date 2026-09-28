"""Latent-action arm 5k->10k, mirroring the control extension exactly."""
from config.pandapush_latact_seed42 import base as _base
from config.pandapush_latact_seed42 import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
base["diffusion"].update(
    prefix="latact_ext/",
    warm_start_checkpoint="data/panda_push/latact/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_final.pt",
    n_train_steps=5000, dataloader_seed=20260917)
