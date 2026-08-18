"""Deterministic behavior-cloning floor arm.

This exists so that the generative arms in
`experiments/protocols/cube_controlled_v1.md` are measured against a
non-generative control under an otherwise identical setup: same backbone, same
conditioning semantics, same masked loss weighting and reduction, same
trajectory/action representation.

The only intended difference from `ConditionalFlowMatching` is that there is no
probability path and no sampler. The network is asked directly for the
conditioned trajectory, and inference is a single forward pass (1 NFE).

If this arm matches the generative arms on a task, generative modeling is not
earning its inference cost there -- which is a scientific result about the
task, not a bug.
"""

import torch

from .diffusion import Sample
from .flow_matching import ConditionalFlowMatching


class DeterministicBehaviorCloning(ConditionalFlowMatching):
    """Regression policy sharing the flow arm's conditioning and loss contract.

    Subclasses ConditionalFlowMatching purely to inherit the validated
    conditioning, loss-weight, and masking machinery. No flow-matching path,
    time sampling, or ODE integration is used.
    """

    def __init__(self, *args, **kwargs):
        # A deterministic regressor has no solver; pin the inherited solver
        # bookkeeping to a single evaluation so NFE accounting stays honest.
        kwargs.setdefault("n_solver_steps", 1)
        super().__init__(*args, **kwargs)

    def _predict(self, x, cond):
        """One conditioned forward pass at fixed time zero.

        The backbone signature is shared with the flow arm, so a constant time
        input is supplied. Conditioning is imposed on the input exactly as the
        generative arms impose it.
        """
        time = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self._call_model(x, cond, time)

    def _compute_bc_loss(self, x, cond, return_details=False):
        self._validate_trajectory(x)
        self._validate_conditioning(
            cond, x.shape[0], self.horizon, x.device, x.dtype, False
        )

        target = x.clone()
        self._apply_conditioning(target, cond)

        # The network sees the conditioned endpoints and zeros elsewhere, so it
        # never observes the supervised interior it must predict.
        model_input = torch.zeros_like(target)
        self._apply_conditioning(model_input, cond)

        prediction = self._predict(model_input, cond)
        if prediction.shape != target.shape:
            raise ValueError("model output must match trajectory shape")
        if not torch.isfinite(prediction).all():
            raise ValueError("model output contains non-finite values")

        error = torch.abs(prediction - target)
        if self.loss_type == "l2":
            error = error.square()

        conditioning_mask = self._make_conditioning_mask(target, cond)
        weights = self.loss_weight_matrix.to(
            device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        active_weights = weights * conditioning_mask.to(x.dtype)
        denominator = conditioning_mask.sum()
        if denominator.item() == 0 or not torch.any(active_weights > 0):
            raise ValueError("conditioning and loss settings leave no active elements")
        # Identical reduction to ConditionalFlowMatching._compute_flow_loss so
        # the arms' loss magnitudes remain comparable.
        loss = (error * active_weights).sum() / denominator.to(x.dtype)

        action_mask = conditioning_mask[:, :, :self.action_dim]
        observation_mask = conditioning_mask[:, :, self.action_dim:]
        info = {
            "bc_loss": loss.detach(),
            "unweighted_bc_loss": self._masked_mean(error, conditioning_mask).detach(),
            "action_loss": self._masked_mean(
                error[:, :, :self.action_dim], action_mask
            ).detach(),
            "observation_loss": self._masked_mean(
                error[:, :, self.action_dim:], observation_mask
            ).detach(),
        }
        if return_details:
            details = {
                "target": target,
                "model_input": model_input,
                "prediction": prediction,
                "conditioning_mask": conditioning_mask,
            }
            return loss, info, details
        return loss, info

    def loss(self, x, cond):
        return self._compute_bc_loss(x, cond)

    @torch.no_grad()
    def conditional_sample(
        self, cond, horizon=None, sort_by_value=True, return_attention=False,
        n_steps=None, return_chain=False, return_diffusion=None, verbose=True,
        **sample_kwargs
    ):
        """Single deterministic forward pass; exactly one model evaluation."""
        if sample_kwargs:
            raise TypeError(
                "unsupported sampling arguments: {}".format(sorted(sample_kwargs))
            )
        if return_attention:
            raise ValueError(
                "DeterministicBehaviorCloning does not return attention maps"
            )
        if return_diffusion is not None:
            if return_chain and not return_diffusion:
                raise ValueError("return_chain and return_diffusion disagree")
            return_chain = bool(return_diffusion)
        if n_steps is not None and int(n_steps) != 1:
            raise ValueError(
                "DeterministicBehaviorCloning always uses exactly one model call"
            )

        active_horizon = self.horizon if horizon is None else horizon
        if not isinstance(cond, dict) or not cond:
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

        x = torch.zeros(
            first.shape[0], active_horizon, self.transition_dim,
            device=first.device, dtype=first.dtype,
        )
        self._apply_conditioning(x, cond)
        chain = [x.clone()] if return_chain else None

        x = self._predict(x, cond)
        self._apply_conditioning(x, cond)
        if return_chain:
            chain.append(x.clone())

        values = x.new_zeros(x.shape[0])
        chains = torch.stack(chain, dim=1) if return_chain else None
        return Sample(x, values, chains)
