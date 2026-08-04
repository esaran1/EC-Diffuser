"""Conditional straight-path flow matching for trajectory generation."""

import math
import numbers
from collections.abc import Mapping

import torch
from torch import nn

from .diffusion import Sample


class ConditionalFlowMatching(nn.Module):
    """Train and sample a conditional velocity field with forward Euler integration."""

    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=None,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        obs_only=False,
        action_only=False,
        n_solver_steps=None,
        n_diffusion_steps=None,
        time_scale=1000.0,
    ):
        """Initialize a flow wrapper compatible with GaussianDiffusion configs."""
        super().__init__()
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be an integer of at least one")
        if not isinstance(observation_dim, int) or isinstance(observation_dim, bool) or observation_dim < 1:
            raise ValueError("observation_dim must be an integer of at least one")
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 0:
            raise ValueError("action_dim must be a non-negative integer")
        if loss_type not in ("l1", "l2"):
            raise ValueError("loss_type must be 'l1' or 'l2', got {!r}".format(loss_type))
        if obs_only and action_only:
            raise ValueError("obs_only and action_only cannot both be True")
        if isinstance(time_scale, bool) or not isinstance(time_scale, numbers.Real):
            raise TypeError("time_scale must be a finite positive number")
        if not math.isfinite(float(time_scale)) or float(time_scale) <= 0.0:
            raise ValueError("time_scale must be a finite positive number")

        resolved_steps = self._resolve_constructor_steps(
            n_solver_steps, n_diffusion_steps, n_timesteps
        )
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.n_timesteps = resolved_steps
        self.n_solver_steps = resolved_steps
        self.n_diffusion_steps = resolved_steps
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.action_weight = float(action_weight)
        self.loss_discount = float(loss_discount)
        self.loss_weights = None if loss_weights is None else dict(loss_weights)
        self.obs_only = bool(obs_only)
        self.action_only = bool(action_only)
        self.time_scale = float(time_scale)

        # clip_denoised and predict_epsilon are diffusion-only compatibility fields.
        weights = self._make_loss_weights(
            self.action_weight, self.loss_discount, self.loss_weights
        )
        self.register_buffer("loss_weight_matrix", weights)

    @staticmethod
    def _validate_solver_steps(value, name):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("{} must be an integer, not {!r}".format(name, value))
        if value < 1:
            raise ValueError("{} must be at least one, got {}".format(name, value))
        return value

    @classmethod
    def _resolve_constructor_steps(cls, n_solver_steps, n_diffusion_steps, n_timesteps):
        candidates = (
            ("n_solver_steps", n_solver_steps),
            ("n_diffusion_steps", n_diffusion_steps),
            ("n_timesteps", n_timesteps),
        )
        for name, value in candidates:
            if value is not None:
                return cls._validate_solver_steps(value, name)
        return 1000

    def _resolve_solver_steps(self, n_steps):
        if n_steps is None:
            return self.n_solver_steps
        return self._validate_solver_steps(n_steps, "n_steps")

    def _make_loss_weights(self, action_weight, discount, weights_dict):
        if isinstance(action_weight, bool) or not isinstance(action_weight, numbers.Real):
            raise TypeError("action_weight must be a finite non-negative number")
        if not math.isfinite(float(action_weight)) or float(action_weight) < 0.0:
            raise ValueError("action_weight must be a finite non-negative number")
        if isinstance(discount, bool) or not isinstance(discount, numbers.Real):
            raise TypeError("loss_discount must be a finite positive number")
        if not math.isfinite(float(discount)) or float(discount) <= 0.0:
            raise ValueError("loss_discount must be a finite positive number")
        if weights_dict is not None and not isinstance(weights_dict, Mapping):
            raise TypeError("loss_weights must be a mapping of observation indices to weights")

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        if weights_dict is not None:
            for index, weight in weights_dict.items():
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("loss_weights keys must be integer observation indices")
                if index < 0 or index >= self.observation_dim:
                    raise ValueError(
                        "loss_weights index {} is outside observation dimension {}".format(
                            index, self.observation_dim
                        )
                    )
                if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                    raise TypeError("loss weight at observation index {} must be numeric".format(index))
                if not math.isfinite(float(weight)) or float(weight) < 0.0:
                    raise ValueError("loss weight at observation index {} must be finite and non-negative".format(index))
                dim_weights[self.action_dim + index] *= float(weight)

        discounts = float(discount) ** torch.arange(self.horizon, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        weights = torch.einsum("h,t->ht", discounts, dim_weights)
        if self.action_dim:
            weights[0, :self.action_dim] = float(action_weight)
        if self.obs_only:
            weights[:, :self.action_dim] = 0.0
        if self.action_only:
            weights[:, self.action_dim:] = 0.0
        if not torch.any(weights > 0):
            raise ValueError("loss settings leave no active trajectory dimensions")
        return weights

    @property
    def device(self):
        """Return the model parameter/buffer device, then the wrapper buffer device."""
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        model_buffer = next(self.model.buffers(), None)
        if model_buffer is not None:
            return model_buffer.device
        return self.loss_weight_matrix.device

    def _model_floating_dtype(self):
        for tensor in self.model.parameters():
            if tensor.is_floating_point():
                return tensor.dtype
        for tensor in self.model.buffers():
            if tensor.is_floating_point():
                return tensor.dtype
        return self.loss_weight_matrix.dtype

    def _validate_trajectory(self, x):
        if not torch.is_tensor(x):
            raise TypeError("trajectory must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("trajectory must have shape [batch, horizon, transition], got {}".format(tuple(x.shape)))
        if x.shape[0] < 1:
            raise ValueError("trajectory batch dimension must be non-empty")
        if x.shape[1] != self.horizon:
            raise ValueError("trajectory horizon must be {}, got {}".format(self.horizon, x.shape[1]))
        if x.shape[2] != self.transition_dim:
            raise ValueError("trajectory transition dimension must be {}, got {}".format(self.transition_dim, x.shape[2]))
        if not x.is_floating_point():
            raise TypeError("trajectory must have a floating-point dtype")
        if not torch.isfinite(x).all():
            raise ValueError("trajectory contains non-finite values")
        if x.device != self.device:
            raise ValueError("trajectory device {} does not match model device {}".format(x.device, self.device))
        model_dtype = self._model_floating_dtype()
        if x.dtype != model_dtype:
            raise ValueError("trajectory dtype {} does not match model dtype {}".format(x.dtype, model_dtype))

    def _validate_conditioning(self, cond, batch_size, horizon, device, dtype, require_nonempty=True):
        if not isinstance(cond, Mapping):
            raise TypeError("conditioning must be a mapping from timestep to observation tensor")
        if require_nonempty and not cond:
            raise ValueError("conditioning must contain at least one timestep")
        for timestep, value in cond.items():
            if isinstance(timestep, bool) or not isinstance(timestep, int):
                raise TypeError("condition timestep {!r} must be an integer".format(timestep))
            if timestep < 0 or timestep >= horizon:
                raise ValueError("condition timestep {} is outside active horizon {}".format(timestep, horizon))
            if not torch.is_tensor(value):
                raise TypeError("condition at timestep {} must be a torch.Tensor".format(timestep))
            expected = (batch_size, self.observation_dim)
            if tuple(value.shape) != expected:
                raise ValueError("condition at timestep {} must have shape {}, got {}".format(timestep, expected, tuple(value.shape)))
            if not value.is_floating_point():
                raise TypeError("condition at timestep {} must have a floating-point dtype".format(timestep))
            if value.device != device:
                raise ValueError("condition at timestep {} is on {}, expected {}".format(timestep, value.device, device))
            if value.dtype != dtype:
                raise ValueError("condition at timestep {} has dtype {}, expected {}".format(timestep, value.dtype, dtype))
            if not torch.isfinite(value).all():
                raise ValueError("condition at timestep {} contains non-finite values".format(timestep))

    def _apply_conditioning(self, x, cond):
        # Assignment is in-place, so this helper is called only on wrapper-owned tensors.
        for timestep, value in cond.items():
            x[:, timestep, self.action_dim:] = value
        return x

    def _make_conditioning_mask(self, x, cond):
        mask = torch.ones_like(x, dtype=torch.bool)
        for timestep in cond:
            mask[:, timestep, self.action_dim:] = False
        return mask

    def _call_model(self, x, cond, model_time):
        prediction = self.model(x, cond, model_time)
        if not torch.is_tensor(prediction):
            raise TypeError("velocity model must return a tensor")
        if prediction.shape != x.shape:
            raise ValueError("velocity model output shape {} does not match input shape {}".format(tuple(prediction.shape), tuple(x.shape)))
        if prediction.device != x.device or prediction.dtype != x.dtype:
            raise ValueError("velocity model output must match input device and dtype")
        if not torch.isfinite(prediction).all():
            raise ValueError("velocity model output contains non-finite values")
        return prediction

    @staticmethod
    def _masked_mean(values, mask):
        count = mask.sum()
        if count.item() == 0:
            return values.new_zeros(())
        return (values * mask.to(values.dtype)).sum() / count.to(values.dtype)

    def _compute_flow_loss(self, x1, cond, x0=None, t=None, return_details=False):
        """Compute flow loss, optionally with fixed noise/time for deterministic tests."""
        self._validate_trajectory(x1)
        self._validate_conditioning(cond, x1.shape[0], self.horizon, x1.device, x1.dtype, False)
        if x0 is None:
            x0_local = torch.randn_like(x1)
        else:
            if not torch.is_tensor(x0) or x0.shape != x1.shape:
                raise ValueError("x0 must be a tensor with shape {}".format(tuple(x1.shape)))
            if x0.device != x1.device or x0.dtype != x1.dtype:
                raise ValueError("x0 must match trajectory device and dtype")
            if not torch.isfinite(x0).all():
                raise ValueError("x0 contains non-finite values")
            x0_local = x0.clone()

        if t is None:
            time = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        else:
            if not torch.is_tensor(t) or tuple(t.shape) != (x1.shape[0],):
                raise ValueError("t must have shape ({},)".format(x1.shape[0]))
            if t.device != x1.device or t.dtype != x1.dtype:
                raise ValueError("t must match trajectory device and dtype")
            if not torch.isfinite(t).all() or torch.any(t < 0) or torch.any(t > 1):
                raise ValueError("t values must be finite and lie in [0, 1]")
            time = t

        x1_local = x1.clone()
        self._apply_conditioning(x0_local, cond)
        self._apply_conditioning(x1_local, cond)
        broadcast_time = time.view(-1, 1, 1)
        xt = (1.0 - broadcast_time) * x0_local + broadcast_time * x1_local
        self._apply_conditioning(xt, cond)
        target_velocity = x1_local - x0_local
        prediction = self._call_model(xt, cond, time * self.time_scale)
        error = torch.abs(prediction - target_velocity)
        if self.loss_type == "l2":
            error = error.square()

        conditioning_mask = self._make_conditioning_mask(x1_local, cond)
        weights = self.loss_weight_matrix.to(device=x1.device, dtype=x1.dtype).unsqueeze(0)
        active_weights = weights * conditioning_mask.to(x1.dtype)
        denominator = conditioning_mask.sum()
        if denominator.item() == 0 or not torch.any(active_weights > 0):
            raise ValueError("conditioning and loss settings leave no active elements")
        # GaussianDiffusion applies weights once and reduces by element count.
        # Restrict that count to unconditioned elements when masking is present.
        loss = (error * active_weights).sum() / denominator.to(x1.dtype)

        action_mask = conditioning_mask[:, :, :self.action_dim]
        observation_mask = conditioning_mask[:, :, self.action_dim:]
        info = {
            "flow_loss": loss.detach(),
            "unweighted_flow_loss": self._masked_mean(error, conditioning_mask).detach(),
            "action_loss": self._masked_mean(error[:, :, :self.action_dim], action_mask).detach(),
            "observation_loss": self._masked_mean(error[:, :, self.action_dim:], observation_mask).detach(),
        }
        if return_details:
            details = {
                "x0": x0_local,
                "x1": x1_local,
                "xt": xt,
                "time": time,
                "model_time": time * self.time_scale,
                "target_velocity": target_velocity,
                "conditioning_mask": conditioning_mask,
            }
            return loss, info, details
        return loss, info

    def loss(self, x, cond):
        """Return scalar conditional flow-matching loss and detached metrics."""
        return self._compute_flow_loss(x, cond)

    @torch.no_grad()
    def conditional_sample(
        self,
        cond,
        horizon=None,
        sort_by_value=True,
        return_attention=False,
        n_steps=None,
        return_chain=False,
        return_diffusion=None,
        verbose=True,
        **sample_kwargs
    ):
        """Generate trajectories with left-endpoint fixed-step forward Euler."""
        if sample_kwargs:
            raise TypeError("unsupported sampling arguments: {}".format(sorted(sample_kwargs)))
        if return_attention:
            raise ValueError("ConditionalFlowMatching does not return attention maps")
        if return_diffusion is not None:
            if return_chain and not return_diffusion:
                raise ValueError("return_chain and return_diffusion disagree")
            return_chain = bool(return_diffusion)
        active_horizon = self.horizon if horizon is None else horizon
        if not isinstance(active_horizon, int) or isinstance(active_horizon, bool) or active_horizon < 1:
            raise ValueError("sampling horizon must be an integer of at least one")
        if not isinstance(cond, Mapping) or not cond:
            raise ValueError("conditioning must be a non-empty mapping for sampling")
        first_value = next(iter(cond.values()))
        if not torch.is_tensor(first_value) or first_value.ndim != 2:
            raise ValueError("condition values must have shape [batch, observation_dim]")
        batch_size = first_value.shape[0]
        if batch_size < 1:
            raise ValueError("conditioning batch dimension must be non-empty")
        device = first_value.device
        dtype = first_value.dtype
        if device != self.device:
            raise ValueError("conditioning device {} does not match model device {}".format(device, self.device))
        model_dtype = self._model_floating_dtype()
        if dtype != model_dtype:
            raise ValueError("conditioning dtype {} does not match model dtype {}".format(dtype, model_dtype))
        self._validate_conditioning(cond, batch_size, active_horizon, device, dtype, True)
        steps = self._resolve_solver_steps(n_steps)

        x = torch.randn(
            (batch_size, active_horizon, self.transition_dim),
            device=device,
            dtype=dtype,
        )
        self._apply_conditioning(x, cond)
        conditioning_mask = self._make_conditioning_mask(x, cond)
        chain = [x.clone()] if return_chain else None
        dt = 1.0 / float(steps)
        for step in range(steps):
            time = x.new_full((batch_size,), float(step) / float(steps))
            velocity = self._call_model(x, cond, time * self.time_scale)
            velocity = velocity * conditioning_mask.to(dtype)
            x = x + dt * velocity
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())

        values = x.new_zeros(batch_size)
        # Flow matching has no value head, so equal zero values must preserve the
        # caller's batch order. Keep sorting support if values become nonuniform.
        if sort_by_value and torch.any(values != values[0]):
            indices = torch.argsort(values, descending=True)
            x = x[indices]
            values = values[indices]
            if return_chain:
                chain = [state[indices] for state in chain]
        chain_tensor = torch.stack(chain, dim=1) if return_chain else None
        return Sample(x, values, chain_tensor)

    def forward(self, cond, *args, **kwargs):
        """Delegate planner calls to :meth:`conditional_sample`."""
        return self.conditional_sample(cond, *args, **kwargs)
