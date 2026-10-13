"""Looped EC-Diffuser (Looped-DiT adaptation), 5k from canonical, matched to the
control's first leg. Blocks 4-7 looped x2, XSA in looped blocks, Final+Mean
deep supervision. No new parameters."""
from config.pandapush_uwm_ctl_seed42 import base as _base
from config.pandapush_uwm_ctl_seed42 import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
for k in ("p_diag", "adaln_state_weight"):
    base["diffusion"].pop(k, None)
base["diffusion"].update(prefix="looped/", model="models.LoopedPINTDenoiser",
                         diffusion="models.LoopedFlowMatching",
                         loop_start=4, loop_end=8, n_loops=2, loop_xsa=True)
