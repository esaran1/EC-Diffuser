"""Direct-action / query-token arm (tal_followup Stage C, seed 42).

Tal's construction: generative state tokens + a learned action/query token that
maps directly to the action, with NO conventionally noised action trajectory.

Matched to the UWM pilot in every training respect -- same canonical source,
same clean-first-step semantics, same LR/warmup/EMA/schedule, same dataloader
seed -- so the three arms (uwm, ctl, qaction) are mutually comparable.

NOTE: this is a NEW construction, not a reproduction. No prior direct-action
implementation, checkpoint or result exists in this repository (see
experiments/policy_improvement/tal_followup/direct_action/DIRECT_ACTION_RECOVERY.md).
"""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    prefix="qaction/",
    model="models.QueryActionPINTDenoiser",
    diffusion="models.QueryActionFlowMatching",
    warm_start_checkpoint=(
        "data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"),
    mask_terminal_action=False,
    lambda_action=1.0,
    learning_rate=1e-5,
    lr_warmup_steps=100,
    step_start_ema=0,
    n_train_steps=5000,
    n_steps_per_epoch=500,
    save_freq=500,
    n_saves=10,
    eval_freq=0,
    sample_freq=0,
    require_uncontended_gpu=True,
    clean_first_step=True,
    dataloader_seed=20260916,
)
