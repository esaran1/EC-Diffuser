"""Phase-3 loss screen, ARM A — baseline: current objective, unchanged.

Derived from the canonical single-GPU Flow config. The ONLY deviations from
canonical are the two registered loss knobs below plus the 20k screening
budget/checkpoint cadence. Architecture, data, seed, LR, optimizer, batch,
accumulation, EMA, conditioning, horizon and first-action weight (10) are
frozen and identical across arms A/B/C/D.
"""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    prefix="phase3_armA/",
    mask_terminal_action=False,
    lambda_action=1.0,
    n_train_steps=20000,
    n_steps_per_epoch=1000,
    save_freq=5000,
    n_saves=4,
    eval_freq=0,
    sample_freq=0,
    require_uncontended_gpu=True,
    # paired training stream across arms A/B/C/D
    dataloader_seed=20260905,
)
