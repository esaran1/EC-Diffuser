"""A2A-style previous-action source (arXiv:2602.07322), 5k from canonical,
matched to the control's first leg. Only change: the action channel's flow
source is the previously executed action + 0.1*N(0,I)."""
from config.pandapush_uwm_ctl_seed42 import base as _base
from config.pandapush_uwm_ctl_seed42 import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
for k in ("p_diag", "adaln_state_weight"):
    base["diffusion"].pop(k, None)
base["diffusion"].update(prefix="prevact/", model="models.AdaLNPINTDenoiser",
                         diffusion="models.PrevActionFlowMatching",
                         loader="datasets.PrevActionGoalDataset", prev_action_sigma=0.1)
