"""Bounded single-GPU Shortcut smoke using canonical PandaPush architecture/data."""

from config.pandapush_flow_smoke import base as flow_base
from config.pandapush_flow_smoke import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    model="models.IntervalAdaLNPINTDenoiser",
    diffusion="models.ShortcutModel",
    prefix="shortcut_smoke/",
)
