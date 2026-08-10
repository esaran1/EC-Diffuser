"""One-thousand-step timing run for the canonical single-GPU flow setup."""

from config.pandapush_flow_single_gpu import base as canonical_base
from config.pandapush_flow_single_gpu import mode_to_args as canonical_mode_to_args


mode_to_args = {
    mode: dict(overrides) for mode, overrides in canonical_mode_to_args.items()
}
base = {
    experiment: dict(settings) for experiment, settings in canonical_base.items()
}
base["diffusion"].update(
    prefix="flow_benchmark/",
    n_steps_per_epoch=1000,
    n_train_steps=1000,
    save_freq=1000,
    n_saves=1,
    eval_freq=0,
)
