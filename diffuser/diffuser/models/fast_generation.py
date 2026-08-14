"""Fast trajectory generators implemented from their published objectives."""

import math
import numbers
from collections.abc import Mapping

import torch

from .diffusion import Sample
from .flow_matching import ConditionalFlowMatching


class _IntervalFlowBase(ConditionalFlowMatching):
    """Shared validation and weighted-loss machinery for interval models."""

    def _call_interval_model(self, x, cond, time, interval):
        prediction = self.model(
            x,
            cond,
            time * self.time_scale,
            interval=interval * self.time_scale,
        )
        if not torch.is_tensor(prediction) or prediction.shape != x.shape:
            raise ValueError("interval model output must match trajectory shape")
        if prediction.device != x.device or prediction.dtype != x.dtype:
            raise ValueError("interval model output must match input device and dtype")
        if not torch.isfinite(prediction).all():
            raise ValueError("interval model output contains non-finite values")
        return prediction

    def _weighted_interval_loss(self, prediction, target, reference, cond):
        error = torch.abs(prediction - target)
        if self.loss_type == "l2":
            error = error.square()
        mask = self._make_conditioning_mask(reference, cond)
        weights = self.loss_weight_matrix.to(
            device=reference.device, dtype=reference.dtype
        ).unsqueeze(0)
        active_weights = weights * mask.to(reference.dtype)
        denominator = mask.sum()
        if denominator.item() == 0 or not torch.any(active_weights > 0):
            raise ValueError("conditioning and loss settings leave no active elements")
        loss = (error * active_weights).sum() / denominator.to(reference.dtype)
        return loss, error, mask

    def _prepare_sample(self, cond, horizon, n_steps):
        active_horizon = self.horizon if horizon is None else horizon
        if not isinstance(cond, Mapping) or not cond:
            raise ValueError("conditioning must be a non-empty mapping")
        first = next(iter(cond.values()))
        if not torch.is_tensor(first) or first.ndim != 2:
            raise ValueError("condition values must have shape [batch, observation_dim]")
        self._validate_conditioning(
            cond, first.shape[0], active_horizon, first.device, first.dtype, True
        )
        if first.device != self.device:
            raise ValueError("conditioning device does not match model device")
        if first.dtype != self._model_floating_dtype():
            raise ValueError("conditioning dtype does not match model dtype")
        steps = self._resolve_solver_steps(n_steps)
        x = torch.randn(
            first.shape[0], active_horizon, self.transition_dim,
            device=first.device, dtype=first.dtype,
        )
        self._apply_conditioning(x, cond)
        return x, steps

    @staticmethod
    def _package_sample(x, chain, return_chain, sort_by_value):
        values = x.new_zeros(x.shape[0])
        if sort_by_value and torch.any(values != values[0]):
            indices = torch.argsort(values, descending=True)
            x, values = x[indices], values[indices]
            if return_chain:
                chain = [state[indices] for state in chain]
        chains = torch.stack(chain, dim=1) if return_chain else None
        return Sample(x, values, chains)


class ImprovedMeanFlow(_IntervalFlowBase):
    """Improved MeanFlow (iMF), arXiv:2512.02012, equations 8--12."""

    def __init__(
        self, *args, time_mean=-0.4, time_std=1.0,
        boundary_probability=0.5, adaptive_weighting=False,
        adaptive_power=1.0, adaptive_epsilon=0.01,
        collect_diagnostics=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= boundary_probability <= 1.0:
            raise ValueError("boundary_probability must lie in [0, 1]")
        if time_std <= 0.0:
            raise ValueError("time_std must be positive")
        if not isinstance(adaptive_weighting, bool):
            raise TypeError("adaptive_weighting must be boolean")
        if not isinstance(collect_diagnostics, bool):
            raise TypeError("collect_diagnostics must be boolean")
        if adaptive_weighting and self.loss_type != "l2":
            raise ValueError("adaptive iMF weighting requires loss_type='l2'")
        for name, value in (
            ("adaptive_power", adaptive_power),
            ("adaptive_epsilon", adaptive_epsilon),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError("{} must be a finite positive number".format(name))
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("{} must be a finite positive number".format(name))
        self.time_mean = float(time_mean)
        self.time_std = float(time_std)
        self.boundary_probability = float(boundary_probability)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.adaptive_power = float(adaptive_power)
        self.adaptive_epsilon = float(adaptive_epsilon)
        self.collect_diagnostics = collect_diagnostics

    @staticmethod
    def _subset_mean(values, selection):
        if selection.any():
            return values[selection].mean()
        return values.new_zeros(())

    def _meanflow_diagnostics(self, error, mask, boundary, derivative):
        active = mask.to(error.dtype)
        counts = active.sum(dim=(1, 2)).clamp_min(1.0)
        per_sample_raw_l2 = (error * active).sum(dim=(1, 2)) / counts
        per_sample_jvp_rms = torch.sqrt(
            (derivative.square() * active).sum(dim=(1, 2)) / counts
        )
        interval = ~boundary
        return {
            "boundary_raw_l2": self._subset_mean(
                per_sample_raw_l2, boundary
            ).detach(),
            "interval_raw_l2": self._subset_mean(
                per_sample_raw_l2, interval
            ).detach(),
            "raw_l2_p50": torch.quantile(per_sample_raw_l2, 0.50).detach(),
            "raw_l2_p90": torch.quantile(per_sample_raw_l2, 0.90).detach(),
            "raw_l2_p99": torch.quantile(per_sample_raw_l2, 0.99).detach(),
            "jvp_rms_p50": torch.quantile(per_sample_jvp_rms, 0.50).detach(),
            "jvp_rms_p90": torch.quantile(per_sample_jvp_rms, 0.90).detach(),
            "jvp_rms_p99": torch.quantile(per_sample_jvp_rms, 0.99).detach(),
        }

    def _meanflow_regression_loss(self, prediction, target, reference, cond):
        if not self.adaptive_weighting:
            return self._weighted_interval_loss(
                prediction, target, reference, cond
            )

        squared_error = (prediction - target).square()
        mask = self._make_conditioning_mask(reference, cond)
        weights = self.loss_weight_matrix.to(
            device=reference.device, dtype=reference.dtype
        ).unsqueeze(0)
        active_weights = weights * mask.to(reference.dtype)
        if not torch.any(active_weights > 0):
            raise ValueError("conditioning and loss settings leave no active elements")
        per_sample = (squared_error * active_weights).sum(dim=(1, 2))
        denominator = (
            per_sample + self.adaptive_epsilon
        ).pow(self.adaptive_power).detach()
        loss = (per_sample / denominator).mean()
        return loss, squared_error, mask

    def _sample_times(self, batch_size, device, dtype):
        pair = torch.randn(batch_size, 2, device=device, dtype=dtype)
        pair = torch.sigmoid(pair * self.time_std + self.time_mean)
        t = pair.max(dim=1).values
        r = pair.min(dim=1).values
        boundary = torch.rand(batch_size, device=device) < self.boundary_probability
        r = torch.where(boundary, t, r)
        return r, t, boundary

    def _validate_meanflow_inputs(self, data, cond, noise, r, t):
        self._validate_trajectory(data)
        self._validate_conditioning(
            cond, data.shape[0], self.horizon, data.device, data.dtype, False
        )
        if noise.shape != data.shape or noise.device != data.device:
            raise ValueError("noise must match trajectory shape and device")
        if noise.dtype != data.dtype or not torch.isfinite(noise).all():
            raise ValueError("noise must match trajectory dtype and be finite")
        if r.shape != (data.shape[0],) or t.shape != r.shape:
            raise ValueError("r and t must have shape [batch]")
        if r.device != data.device or t.device != data.device:
            raise ValueError("r and t must match trajectory device")
        if r.dtype != data.dtype or t.dtype != data.dtype:
            raise ValueError("r and t must match trajectory dtype")
        if not torch.isfinite(r).all() or not torch.isfinite(t).all():
            raise ValueError("r and t must be finite")
        if torch.any(r < 0) or torch.any(t > 1) or torch.any(r > t):
            raise ValueError("times must satisfy 0 <= r <= t <= 1")

    def _compute_meanflow_loss(
        self, data, cond, noise=None, r=None, t=None, return_details=False
    ):
        if noise is None:
            noise = torch.randn_like(data)
        else:
            noise = noise.clone()
        if r is None or t is None:
            r, t, boundary = self._sample_times(
                data.shape[0], data.device, data.dtype
            )
        else:
            boundary = r == t
        self._validate_meanflow_inputs(data, cond, noise, r, t)

        data_local = data.clone()
        self._apply_conditioning(data_local, cond)
        self._apply_conditioning(noise, cond)
        zt = (1.0 - t[:, None, None]) * data_local + t[:, None, None] * noise
        self._apply_conditioning(zt, cond)

        def average_velocity(z_value, r_value, t_value):
            return self._call_interval_model(
                z_value, cond, t_value, t_value - r_value
            )

        # iMF equation 12 uses the predicted marginal velocity u(z_t, t, t)
        # as the JVP tangent, not the conditional training target.
        marginal_velocity = average_velocity(zt, t, t)
        average, derivative = torch.func.jvp(
            average_velocity,
            (zt, r, t),
            (marginal_velocity, torch.zeros_like(r), torch.ones_like(t)),
        )
        compound = average + (t - r)[:, None, None] * derivative.detach()
        target = noise - data_local
        loss, error, mask = self._meanflow_regression_loss(
            compound, target, data_local, cond
        )
        info = {
            "meanflow_loss": loss.detach(),
            "unweighted_meanflow_loss": self._masked_mean(error, mask).detach(),
            "boundary_fraction": boundary.to(data.dtype).mean().detach(),
        }
        if self.collect_diagnostics:
            info.update(self._meanflow_diagnostics(
                error, mask, boundary, derivative
            ))
        if return_details:
            details = {
                "data": data_local,
                "noise": noise,
                "zt": zt,
                "r": r,
                "t": t,
                "marginal_velocity": marginal_velocity,
                "average_velocity": average,
                "jvp": derivative,
                "compound_velocity": compound,
                "target_velocity": target,
                "conditioning_mask": mask,
            }
            return loss, info, details
        return loss, info

    def loss(self, x, cond):
        return self._compute_meanflow_loss(x, cond)

    @torch.no_grad()
    def conditional_sample(
        self, cond, horizon=None, sort_by_value=True, return_attention=False,
        n_steps=None, return_chain=False, return_diffusion=None, verbose=True,
        **sample_kwargs
    ):
        if sample_kwargs:
            raise TypeError("unsupported sampling arguments: {}".format(
                sorted(sample_kwargs)
            ))
        if return_attention:
            raise ValueError("ImprovedMeanFlow does not return attention maps")
        if return_diffusion is not None:
            if return_chain and not return_diffusion:
                raise ValueError("return_chain and return_diffusion disagree")
            return_chain = bool(return_diffusion)
        x, steps = self._prepare_sample(cond, horizon, n_steps)
        mask = self._make_conditioning_mask(x, cond).to(x.dtype)
        chain = [x.clone()] if return_chain else None
        grid = torch.linspace(
            1.0, 0.0, steps + 1, device=x.device, dtype=x.dtype
        )
        for index in range(steps):
            t = x.new_full((x.shape[0],), grid[index].item())
            r = x.new_full((x.shape[0],), grid[index + 1].item())
            average = self._call_interval_model(x, cond, t, t - r) * mask
            x = x - (t - r)[:, None, None] * average
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())
        return self._package_sample(x, chain, return_chain, sort_by_value)


class ShortcutModel(_IntervalFlowBase):
    """Shortcut Model, arXiv:2410.12557, equations 3--5 and Algorithm 1."""

    def __init__(
        self, *args, max_base_steps=128, flow_fraction=0.75, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if (
            not isinstance(max_base_steps, int)
            or max_base_steps < 2
            or max_base_steps & (max_base_steps - 1)
        ):
            raise ValueError("max_base_steps must be a power of two >= 2")
        if not 0.0 <= flow_fraction <= 1.0:
            raise ValueError("flow_fraction must lie in [0, 1]")
        self.max_base_steps = max_base_steps
        self.flow_fraction = float(flow_fraction)

    def _sample_shortcut_times(self, batch_size, device, dtype):
        flow_mask = torch.rand(batch_size, device=device) < self.flow_fraction
        # d is the half-step used to bootstrap a target for requested size 2d.
        level_count = self.max_base_steps.bit_length() - 1
        levels = torch.randint(1, level_count + 1, (batch_size,), device=device)
        small_d = torch.pow(
            torch.full((batch_size,), 2.0, device=device, dtype=dtype),
            -levels.to(dtype),
        )
        available = torch.floor((1.0 - 2.0 * small_d) / small_d)
        grid_index = torch.floor(
            torch.rand(batch_size, device=device, dtype=dtype) * (available + 1.0)
        )
        time = grid_index * small_d
        return time, small_d, flow_mask

    def _validate_shortcut_inputs(
        self, data, cond, noise, time, small_d, flow_mask
    ):
        self._validate_trajectory(data)
        self._validate_conditioning(
            cond, data.shape[0], self.horizon, data.device, data.dtype, False
        )
        if noise.shape != data.shape or noise.device != data.device:
            raise ValueError("noise must match trajectory shape and device")
        if noise.dtype != data.dtype or not torch.isfinite(noise).all():
            raise ValueError("noise must match trajectory dtype and be finite")
        for name, value in (("time", time), ("small_d", small_d)):
            if value.shape != (data.shape[0],):
                raise ValueError("{} must have shape [batch]".format(name))
            if value.device != data.device or value.dtype != data.dtype:
                raise ValueError("{} must match trajectory device and dtype".format(name))
            if not torch.isfinite(value).all():
                raise ValueError("{} must be finite".format(name))
        if flow_mask.shape != (data.shape[0],) or flow_mask.dtype != torch.bool:
            raise ValueError("flow_mask must be boolean with shape [batch]")
        if flow_mask.device != data.device:
            raise ValueError("flow_mask must match trajectory device")
        if torch.any(small_d <= 0) or torch.any(time < 0):
            raise ValueError("time and small_d must be positive/in range")
        if torch.any(time + 2.0 * small_d > 1.0 + 1e-6):
            raise ValueError("bootstrap interval exceeds t=1")

    def _compute_shortcut_loss(
        self, data, cond, noise=None, t=None, small_d=None,
        flow_mask=None, return_details=False
    ):
        noise = torch.randn_like(data) if noise is None else noise.clone()
        if t is None or small_d is None or flow_mask is None:
            t, small_d, flow_mask = self._sample_shortcut_times(
                data.shape[0], data.device, data.dtype
            )
        self._validate_shortcut_inputs(
            data, cond, noise, t, small_d, flow_mask
        )

        data_local = data.clone()
        self._apply_conditioning(data_local, cond)
        self._apply_conditioning(noise, cond)
        xt = (1.0 - t[:, None, None]) * noise + t[:, None, None] * data_local
        self._apply_conditioning(xt, cond)
        target = data_local - noise
        requested_d = torch.where(
            flow_mask, torch.zeros_like(small_d), 2.0 * small_d
        )

        bootstrap_mask = ~flow_mask
        if bootstrap_mask.any():
            subset_cond = {
                key: value[bootstrap_mask] for key, value in cond.items()
            }
            subset_x = xt[bootstrap_mask]
            subset_t = t[bootstrap_mask]
            subset_d = small_d[bootstrap_mask]
            with torch.no_grad():
                first = self._call_interval_model(
                    subset_x, subset_cond, subset_t, subset_d
                )
                midpoint = subset_x + subset_d[:, None, None] * first
                self._apply_conditioning(midpoint, subset_cond)
                second = self._call_interval_model(
                    midpoint, subset_cond, subset_t + subset_d, subset_d
                )
                target[bootstrap_mask] = 0.5 * (first + second)

        prediction = self._call_interval_model(
            xt, cond, t, requested_d
        )
        loss, error, mask = self._weighted_interval_loss(
            prediction, target, data_local, cond
        )
        info = {
            "shortcut_loss": loss.detach(),
            "unweighted_shortcut_loss": self._masked_mean(error, mask).detach(),
            "flow_fraction": flow_mask.to(data.dtype).mean().detach(),
        }
        if return_details:
            details = {
                "data": data_local, "noise": noise, "xt": xt, "t": t,
                "small_d": small_d, "requested_d": requested_d,
                "flow_mask": flow_mask, "target": target,
                "prediction": prediction, "conditioning_mask": mask,
            }
            return loss, info, details
        return loss, info

    def loss(self, x, cond):
        return self._compute_shortcut_loss(x, cond)

    @torch.no_grad()
    def conditional_sample(
        self, cond, horizon=None, sort_by_value=True, return_attention=False,
        n_steps=None, return_chain=False, return_diffusion=None, verbose=True,
        **sample_kwargs
    ):
        if sample_kwargs:
            raise TypeError("unsupported sampling arguments: {}".format(
                sorted(sample_kwargs)
            ))
        if return_attention:
            raise ValueError("ShortcutModel does not return attention maps")
        if return_diffusion is not None:
            if return_chain and not return_diffusion:
                raise ValueError("return_chain and return_diffusion disagree")
            return_chain = bool(return_diffusion)
        x, steps = self._prepare_sample(cond, horizon, n_steps)
        if steps > self.max_base_steps or steps & (steps - 1):
            raise ValueError(
                "Shortcut steps must be a power of two no larger than {}".format(
                    self.max_base_steps
                )
            )
        mask = self._make_conditioning_mask(x, cond).to(x.dtype)
        chain = [x.clone()] if return_chain else None
        step_size = 1.0 / float(steps)
        for index in range(steps):
            time = x.new_full((x.shape[0],), index * step_size)
            interval = x.new_full((x.shape[0],), step_size)
            shortcut = self._call_interval_model(
                x, cond, time, interval
            ) * mask
            x = x + step_size * shortcut
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())
        return self._package_sample(x, chain, return_chain, sort_by_value)
