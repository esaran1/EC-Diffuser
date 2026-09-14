"""CLEAN-first-step arm B, fine-tuning stochastic seed 44 (clean program).

Identical to the historical arm B at this seed -- same canonical source, same
model/EMA init, fresh Adam, LR 1e-5, warmup 100, step_start_ema=0, dataloader
seed 20260907 (the EXACT historical mapping for seed 44), same minibatch stream,
first-action weight 10, 2000 steps -- EXCEPT that the random-init sanity gradient
left by train.py's pre-training forward/backward is CLEARED before step 1.

LEGACY: step 1 = g_sanity(random-init, batch 1) + g_batch1 + g_batch2
CLEAN:  step 1 =                                  g_batch1 + g_batch2

g_sanity is objective-dependent (rel. to arm A: B 2.25e-2, C 2.50e-1,
D 9.50e-1), so it is a potential confound for the A-vs-B contrast rather than a
shared constant.
"""
from config.pandapush_lossft_armB import base as _base
from config.pandapush_lossft_armB import mode_to_args as _modes

mode_to_args = {name: dict(values) for name, values in _modes.items()}
base = {name: dict(values) for name, values in _base.items()}
base["diffusion"].update(
    prefix="lossft_armBclean/",
    dataloader_seed=20260907,
    clean_first_step=True,
)
