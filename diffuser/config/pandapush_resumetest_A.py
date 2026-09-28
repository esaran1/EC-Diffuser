"""Exact-resume GPU test arm A. Same as the control 10k extension config
but 40 steps with 20-step epochs. A: uninterrupted. B: forced interrupt + resume."""
from config.pandapush_uwm_ctl_seed42_ext import base as _base
from config.pandapush_uwm_ctl_seed42_ext import mode_to_args as _modes
mode_to_args = {n: dict(v) for n, v in _modes.items()}
base = {n: dict(v) for n, v in _base.items()}
base["diffusion"].update(prefix="resumetest_A/", n_train_steps=40,
                         n_steps_per_epoch=20, save_freq=20, n_saves=2)
