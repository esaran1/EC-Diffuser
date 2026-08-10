"""Short single-GPU integration run for the canonical PandaPush flow setup."""

from config.pandapush_flow import base as canonical_base
from config.pandapush_flow import mode_to_args as canonical_mode_to_args


mode_to_args = {
    mode: dict(overrides) for mode, overrides in canonical_mode_to_args.items()
}

base = {
    experiment: dict(settings) for experiment, settings in canonical_base.items()
}

base["diffusion"].update(
    prefix="flow_smoke/",
    n_steps_per_epoch=201,
    n_train_steps=201,
    save_freq=100,
    n_saves=2,
    eval_freq=0,
    device="cuda:0",
)
base["plan"].update(device="cuda:0")
