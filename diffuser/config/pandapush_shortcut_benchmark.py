"""1,001-step single-GPU Shortcut timing benchmark using canonical PandaPush architecture/data."""

from config.pandapush_flow_smoke import base as flow_base
from config.pandapush_flow_smoke import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    model="models.IntervalAdaLNPINTDenoiser",
    diffusion="models.ShortcutModel",
    prefix="shortcut_benchmark/",
    n_steps_per_epoch=1001,
    n_train_steps=1001,
    save_freq=1000,
    n_saves=1,
    eval_freq=0,
)
