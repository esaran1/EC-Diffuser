"""UWM faithful independent-time arm (seed 42).

Differs from the sibling arm in EXACTLY ONE value: p_diag = 0.0.
See pandapush_uwm_base.py.
"""
from config.pandapush_uwm_base import base as _base
from config.pandapush_uwm_base import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
base["diffusion"].update(
    prefix="uwm_uwm/",
    p_diag=0.0,
)
