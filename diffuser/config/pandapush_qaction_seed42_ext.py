"""Direct-action arm, 5k->10k extension. Mirrors the diagonal control's
trajectory exactly (canonical -> 5k with dataloader_seed 20260916 -> +5k with
dataloader_seed 20260917), so qaction 10k is matched to control 10k in source,
update count, data streams and optimizer schedule. Only the action pathway
differs."""
from config.pandapush_qaction_seed42 import base as _base
from config.pandapush_qaction_seed42 import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
base["diffusion"].update(
    prefix="qaction_ext/",
    warm_start_checkpoint=(
        "data/panda_push/qaction/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_final.pt"),
    n_train_steps=5000,
    dataloader_seed=20260917,
)
