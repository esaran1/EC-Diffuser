"""UWM faithful arm, section-23 authorized extension 5000 -> 10000 steps.

Identical to pandapush_uwm_uwm_seed42 except n_train_steps. Warm-starts from
the arm's OWN 5000-step checkpoint so the optimizer budget is cumulative rather
than restarted.
"""
from config.pandapush_uwm_uwm_seed42 import base as _base
from config.pandapush_uwm_uwm_seed42 import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
base["diffusion"].update(
    prefix="uwm_uwm_ext/",
    warm_start_checkpoint=(
        "data/panda_push/uwm_uwm/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_final.pt"),
    n_train_steps=5000,
    dataloader_seed=20260917,
)
