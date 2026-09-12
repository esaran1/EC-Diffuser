"""LR SAFETY PILOT (A1): canonical warm start, unchanged baseline loss, 300 steps.

Safety only. Never continued into Arm A; Arm A restarts from canonical.
"""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    prefix="lossft_pilot/",
    warm_start_checkpoint=(
        "data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"),
    mask_terminal_action=False,
    lambda_action=1.0,
    learning_rate=1e-5,
    lr_warmup_steps=100,
    step_start_ema=0,
    n_train_steps=300,
    n_steps_per_epoch=100,
    save_freq=300,
    n_saves=1,
    eval_freq=0,
    sample_freq=0,
    require_uncontended_gpu=True,
    dataloader_seed=20260905,
)
