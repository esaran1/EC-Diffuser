"""Short timing probe: canonical Standard Flow training rate, real train loop."""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    prefix="flow_timing_probe/",
    n_train_steps=600,
    n_steps_per_epoch=200,
    save_freq=0,
    n_saves=1,
    eval_freq=0,
    sample_freq=0,
    require_uncontended_gpu=True,
)
