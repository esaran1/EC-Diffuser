"""Method-neutral timed copy of the original single-GPU diffusion evaluation."""

from config.plan_pandapush_pint_single_gpu import base as original_base
from config.plan_pandapush_pint_single_gpu import mode_to_args as original_mode_to_args
from config.plan_pandapush_pint_single_gpu import (
    PushT_entity_to_steps, entity_to_steps, kitchen_entity_to_steps,
)

mode_to_args = {mode: dict(values) for mode, values in original_mode_to_args.items()}
base = {experiment: dict(values) for experiment, values in original_base.items()}
base["plan"].update(
    prefix="timed_diffusion_plans/",
    measure_planning_latency=True,
    planning_warmup_calls=10,
    count_denoiser_calls=True,
    verbose=False,
)
