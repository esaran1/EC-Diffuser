"""Phase 2: MeanFlow viability/sanity run on canonical PandaPush.

Boundary ImprovedMeanFlow with the CURRENT joint state/action formulation.
Derived from the canonical single-GPU Flow config; the ONLY deviations are:
  - model      AdaLNPINTDenoiser -> IntervalAdaLNPINTDenoiser (needs (r,t))
  - diffusion  ConditionalFlowMatching -> ImprovedMeanFlow
  - learning_rate 8e-5 -> 4e-5   (viability config; 8e-5 diverged on OGBench)
  - n_train_steps / checkpoint cadence for the staged 20k -> 50k plan

NOT changed: horizon 5, action_weight 10, loss_discount 1, loss_type l1,
loss_weights None, batch 32, grad-accum 2, ema_decay 0.995, architecture, data.
No terminal-action masking, no lambda reweighting, no schedule change, no N2.
"""

from config.pandapush_flow_single_gpu import base as flow_base
from config.pandapush_flow_single_gpu import mode_to_args as flow_modes
from config.pandapush_flow_single_gpu import args_to_watch  # noqa: F401

mode_to_args = {name: dict(values) for name, values in flow_modes.items()}
base = {name: dict(values) for name, values in flow_base.items()}

base["diffusion"].update(
    model="models.IntervalAdaLNPINTDenoiser",
    diffusion="models.ImprovedMeanFlow",
    prefix="imf_viability/",
    learning_rate=4e-5,
    # MeanFlow's JVP holds a second forward graph, so bs=32 OOMs on a 16 GB
    # RTX 4080. Use the audited iMF micro-batch shape; effective batch is
    # 8x8 = 64, matching canonical Flow's 32x2 = 64.
    batch_size=8,
    gradient_accumulate_every=8,
    n_train_steps=20000,        # STAGE A; raised to 50000 for stage B (same run, resumed)
    n_steps_per_epoch=1000,
    save_freq=1000,
    n_saves=60,                 # keep every 1k checkpoint in the staged range
    eval_freq=0,
    sample_freq=0,
    # infrastructure: refuse to start on a shared GPU, and checkpoint+exit
    # cleanly if a foreign GPU process appears mid-run.
    require_uncontended_gpu=True,
)
