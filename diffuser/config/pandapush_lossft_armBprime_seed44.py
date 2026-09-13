"""Arm B' (denominator control), fine-tuning stochastic seed 44.

Identical to arm B in every respect -- terminal action MASKED, lambda_action=1,
same numerator, same weights, same data, same optimizer -- EXCEPT that the loss
divides by the denominator an UNMASKED run would have used (46560 rather than
46464 at batch 32).

This isolates the registered terminal-mask ablation from the ~0.207%
active-coordinate renormalization it inherits. By construction

    L_B' = (46464 / 46560) * L_B = c * L_B,    c = 0.997938144

so B' differs from B by a constant global scaling of the objective and nothing
else. See mechanism/DENOMINATOR_THEORY.md.
"""
from config.pandapush_lossft_armB import base as _base
from config.pandapush_lossft_armB import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
base["diffusion"].update(
    prefix="lossft_armBprime/",
    dataloader_seed=20260907,
    denominator_ignores_terminal_mask=True,
)
