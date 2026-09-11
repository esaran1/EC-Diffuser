"""Loss fine-tune ARM D — terminal mask + 5x action-side loss.

Warm-started from the canonical 500k Standard Flow checkpoint. Deviations from
the canonical training config, all identical across arms except the two
registered loss knobs:
  warm_start_checkpoint  canonical model+ema (Adam NOT restored; restarts at 0)
  step_start_ema = 0     REQUIRED: the canonical default 2000 would hard-copy
                         LIVE->EMA for the whole 2k run and destroy canonical EMA
  learning_rate  = 1e-5  1/8 of canonical 8e-5, bounding the Adam-restart kick
  lr_warmup_steps= 100   ramps the restart transient
  n_train_steps  = 2000  perturbation, not convergence
"""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    prefix="lossft_armD/",
    warm_start_checkpoint=(
        "data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"),
    mask_terminal_action=True,
    lambda_action=5.0,
    learning_rate=1e-5,
    lr_warmup_steps=100,
    step_start_ema=0,
    n_train_steps=2000,
    n_steps_per_epoch=500,
    save_freq=500,
    n_saves=4,
    eval_freq=0,
    sample_freq=0,
    require_uncontended_gpu=True,
    dataloader_seed=20260905,
)
