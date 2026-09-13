"""Arm B at fine-tuning seed 42 -- the ORIGINAL replicate's settings.

The original seed-42 B run used `config.pandapush_lossft_armB` directly (its
dataloader_seed of 20260905 is the base value). This alias exists only so the
B vs B' comparison can name both arms uniformly; it is value-identical to the
base config and introduces no change.
"""
from config.pandapush_lossft_armB import base as _base
from config.pandapush_lossft_armB import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
