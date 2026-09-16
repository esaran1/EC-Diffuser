"""UWM independent-time pilot — shared base (tal_followup program).

Warm-started from the canonical 500k Standard Flow checkpoint, with the
corrected CLEAN-FIRST-STEP semantics mandatory for all new training in this
program (directive section 9).

Both arms of the pilot inherit from here and differ in EXACTLY ONE value:

    p_diag = 0.0   faithful UWM   -- t_state, t_action ~ U(0,1) independently
    p_diag = 1.0   matched control -- t_state = t_action (the diagonal)

Everything else -- model class, wrapper class, parameter count, code path,
canonical source, optimizer, LR, schedule, EMA, dataloader stream -- is shared,
so the comparison isolates the training time distribution (directive section 22).

The denoiser is TwoTimePINTDenoiser, which adds ZERO parameters over
AdaLNPINTDenoiser and is bit-exact with it on the diagonal (verified by
experiments/policy_improvement/tal_followup/uwm/test_diagonal_compat.py).

Checkpoints at 500/1000/2000/5000 (section 21); the run may be stopped at the
shortest budget that demonstrably adapts (section 23).
"""
from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}
base["diffusion"].update(
    model="models.TwoTimePINTDenoiser",
    diffusion="models.UWMConditionalFlowMatching",
    warm_start_checkpoint=(
        "data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"),
    # unchanged loss semantics -- terminal supervision preserved (section 1)
    mask_terminal_action=False,
    lambda_action=1.0,
    # zero-parameter two-time construction; any w preserves the diagonal
    adaln_state_weight=0.5,
    # fine-tune schedule, matched to the proven loss-FT pattern
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
