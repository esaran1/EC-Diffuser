"""Action-to-action source (adapted from A2A, arXiv:2602.07322).

The action channel's flow starts from the previously EXECUTED action
(normalized, broadcast over the horizon) plus sigma*N(0,I), instead of pure
Gaussian noise. The state channel is unchanged. Training uses production
_compute_flow_loss with this x0, so weights, conditioning mask and terminal
supervision are identical. Cold start (no previous action) = normalized zero,
i.e. the robot at rest, in both training and closed loop.
"""
import torch

from .diffusion import Sample
from .flow_matching import ConditionalFlowMatching


class PrevActionFlowMatching(ConditionalFlowMatching):
    uses_prev_action = True

    def __init__(self, *args, prev_action_sigma=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_action_sigma = float(prev_action_sigma)

    def _source(self, shape, prev_action, device, dtype):
        x0 = torch.randn(shape, device=device, dtype=dtype)
        ad = self.action_dim
        p = prev_action.to(device=device, dtype=dtype)[:, None, :]
        x0[..., :ad] = p + self.prev_action_sigma * x0[..., :ad]
        return x0

    def loss(self, x, cond, prev_action):
        x0 = self._source(x.shape, prev_action, x.device, x.dtype)
        return self._compute_flow_loss(x, cond, x0=x0)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, prev_action=None, n_steps=None,
                           return_chain=False, verbose=True, **kw):
        if prev_action is None:
            raise ValueError("PrevActionFlowMatching needs prev_action at inference")
        first = next(iter(cond.values()))
        B, dev, dt_ = first.shape[0], first.device, first.dtype
        H = self.horizon if horizon is None else horizon
        steps = self._resolve_solver_steps(n_steps)
        x = self._source((B, H, self.transition_dim), prev_action, dev, dt_)
        self._apply_conditioning(x, cond)
        mask = self._make_conditioning_mask(x, cond).to(dt_)
        chain = [x.clone()] if return_chain else None
        dt = 1.0 / steps
        for k in range(steps):
            t = x.new_full((B,), k / steps)
            x = x + dt * self._call_model(x, cond, t * self.time_scale) * mask
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())
        return Sample(x, x.new_zeros(B), torch.stack(chain, 1) if return_chain else None)
