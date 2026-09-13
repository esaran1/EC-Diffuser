"""Replication arm B, fine-tuning stochastic seed 44.

Identical to pandapush_lossft_armB except `dataloader_seed`, which must vary
per replicate. It is set HERE rather than on the command line because
`train.py` parses arguments twice -- once for 'diffusion' and once for 'plan' --
and `add_extras` re-applies CLI extras to the plan config, which has no
`dataloader_seed` key and so raises. Declaring it in the config keeps the
pinned training code untouched and puts the seed in a committed file.

The fine-tuning RNG seed (44) is passed as --seed; it is a declared parser
attribute and survives both parses.

Warm start remains the FIXED canonical checkpoint (flow/..._seed42): this varies
the fine-tuning stochastic seed, NOT the pretrained policy.
"""
from config.pandapush_lossft_armB import base as _base
from config.pandapush_lossft_armB import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
base["diffusion"].update(dataloader_seed=20260907)
