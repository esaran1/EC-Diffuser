from collections import namedtuple
import time

import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np
from diffuser.models import sample_fn_return_attn, default_sample_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GoalConditionedPolicy:
    def __init__(
        self, diffusion_model, normalizer, preprocess_fns,
        measure_planning_latency=False, planning_warmup_calls=10,
        count_denoiser_calls=False, **sample_kwargs
    ):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.measure_planning_latency = bool(measure_planning_latency)
        self.planning_warmup_calls = int(planning_warmup_calls)
        if self.planning_warmup_calls < 0:
            raise ValueError("planning_warmup_calls must be non-negative")
        self.planner_calls = 0
        self.planning_latencies_ms = []
        self.denoiser_calls = 0
        self._denoiser_hook = None
        if count_denoiser_calls:
            denoiser = getattr(diffusion_model, "model", None)
            if denoiser is None:
                raise ValueError("count_denoiser_calls requires diffusion_model.model")
            self._denoiser_hook = denoiser.register_forward_hook(self._count_denoiser_call)

    def _count_denoiser_call(self, _module, _inputs, _output):
        self.denoiser_calls += 1

    def _generate_samples(self, conditions, verbose, return_attention):
        if self.measure_planning_latency and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter() if self.measure_planning_latency else None
        result = self.diffusion_model(
            conditions, verbose=verbose, sort_by_value=False,
            return_attention=return_attention, **self.sample_kwargs
        )
        if self.measure_planning_latency:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if self.planner_calls >= self.planning_warmup_calls:
                self.planning_latencies_ms.append(elapsed_ms)
        self.planner_calls += 1
        return result

    def planning_stats(self):
        latencies = np.asarray(self.planning_latencies_ms, dtype=np.float64)
        stats = {
            "total_planner_calls": self.planner_calls,
            "warmup_calls": min(self.planner_calls, self.planning_warmup_calls),
            "timed_calls": int(latencies.size),
            "denoiser_calls": self.denoiser_calls,
        }
        if latencies.size:
            stats.update(
                mean_ms=float(latencies.mean()), std_ms=float(latencies.std()),
                p50_ms=float(np.percentile(latencies, 50)),
                p90_ms=float(np.percentile(latencies, 90)),
                p95_ms=float(np.percentile(latencies, 95)),
                p99_ms=float(np.percentile(latencies, 99)),
            )
        return stats

    def __call__(self, conditions, batch_size=1, verbose=True, return_attention=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        multi_input = len(conditions[0].shape) != 1
        conditions = self._format_conditions(conditions, batch_size, multi_input=multi_input)

        if return_attention:
            samples, att_dict = self._generate_samples(conditions, verbose, return_attention)
            att_dict = {k: utils.to_np(v) for k, v in att_dict.items()}
        else:
            samples = self._generate_samples(conditions, verbose, return_attention)
        trajectories = utils.to_np(samples.trajectories)

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        action = actions[:, 0] if multi_input else actions[0, 0]

        trajectories = Trajectories(actions, observations, samples.values)
        if return_attention:
            return action, trajectories, att_dict
        return action, trajectories

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _format_conditions(self, conditions, batch_size, multi_input=False):
        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, 'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32)
        if not multi_input:
            conditions = utils.apply_dict(
                einops.repeat, conditions, 'd -> repeat d', repeat=batch_size,
            )
        return conditions
