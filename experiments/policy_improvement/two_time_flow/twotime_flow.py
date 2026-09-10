"""Two-time Conditional Flow Matching: separate flow times for state and action.

Subclasses the repo's ConditionalFlowMatching. Changes ONLY the corruption and
time conditioning. Loss form, weights, terminal-action target, first-action
weight, horizon, data and architecture are untouched.

Convention (VERIFIED against flow_matching.py:_compute_flow_loss):
    x0 = noise, x1 = data
    xt     = (1 - t)*x0 + t*x1        -> t=0 NOISE, t=1 CLEAN
    target = x1 - x0                   (data - noise)
This is exactly Tal's convention: t_state > t_action means STATE IS CLEANER
(inverse-model / plan-to-action regime). No conversion is needed anywhere.
"""
import math

import torch

from diffuser.models.flow_matching import ConditionalFlowMatching


class TwoTimeConditionalFlowMatching(ConditionalFlowMatching):
    """CFM where the state block and action block carry independent flow times.

    p_diag        probability that t_state == t_action (preserves the diagonal)
    remaining mass is split SYMMETRICALLY between t_state<t_action and >.
    """

    def __init__(self, *args, p_diag=0.5, adaln_state_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(p_diag) <= 1.0:
            raise ValueError("p_diag must lie in [0, 1]")
        self.p_diag = float(p_diag)
        self.adaln_state_weight = float(adaln_state_weight)
        if hasattr(self.model, "adaln_state_weight"):
            self.model.adaln_state_weight = float(adaln_state_weight)

    # ---------------- sampling of the two times ----------------
    def sample_two_times(self, batch_size, device, dtype):
        """Mixture B': p_diag on the diagonal, rest split symmetrically.

            u, a1, a2 ~ U(0,1)
            hi = max(a1,a2) ; lo = min(a1,a2)
            k ~ U(0,1)
            k < p_diag              -> (t_s, t_a) = (u, u)
            p_diag <= k < p_diag+q  -> (t_s, t_a) = (hi, lo)     q=(1-p_diag)/2
            otherwise               -> (t_s, t_a) = (lo, hi)

        Marginals of t_s and t_a are each U(0,1) by symmetry of (hi,lo).
        """
        u = torch.rand(batch_size, device=device, dtype=dtype)
        a1 = torch.rand(batch_size, device=device, dtype=dtype)
        a2 = torch.rand(batch_size, device=device, dtype=dtype)
        hi = torch.maximum(a1, a2)
        lo = torch.minimum(a1, a2)
        k = torch.rand(batch_size, device=device, dtype=dtype)
        q = (1.0 - self.p_diag) / 2.0
        on_diag = k < self.p_diag
        first_off = (k >= self.p_diag) & (k < self.p_diag + q)
        t_s = torch.where(on_diag, u, torch.where(first_off, hi, lo))
        t_a = torch.where(on_diag, u, torch.where(first_off, lo, hi))
        return t_s, t_a

    # ---------------- two-time corruption ----------------
    def corrupt(self, x1, x0, t_s, t_a):
        """xt = (1-t)*noise + t*data, blockwise. t=0 noise, t=1 clean."""
        ad = self.action_dim
        ts = t_s.view(-1, 1, 1)
        ta = t_a.view(-1, 1, 1)
        xt = torch.empty_like(x1)
        xt[..., :ad] = (1.0 - ta) * x0[..., :ad] + ta * x1[..., :ad]
        xt[..., ad:] = (1.0 - ts) * x0[..., ad:] + ts * x1[..., ad:]
        return xt

    def _call_two_time_model(self, x, cond, t_s, t_a):
        return self.model(x, cond, t_s * self.time_scale,
                          time_action=t_a * self.time_scale)

    # ---------------- loss ----------------
    def loss(self, x, cond):
        return self._compute_two_time_loss(x, cond)

    def _compute_two_time_loss(self, x1, cond, return_details=False):
        device, dtype = x1.device, x1.dtype
        B = x1.shape[0]
        x1_local = x1.clone()
        x0 = torch.randn_like(x1)
        self._apply_conditioning(x1_local, cond)
        self._apply_conditioning(x0, cond)

        t_s, t_a = self.sample_two_times(B, device, dtype)
        xt = self.corrupt(x1_local, x0, t_s, t_a)
        self._apply_conditioning(xt, cond)

        target = x1_local - x0                      # data - noise (verified)
        prediction = self._call_two_time_model(xt, cond, t_s, t_a)

        error = torch.abs(prediction - target)
        if self.loss_type == "l2":
            error = error.square()

        mask = self._make_conditioning_mask(xt, cond)
        weights = self.loss_weight_matrix.to(device=device, dtype=dtype).unsqueeze(0)
        active = weights * mask.to(dtype)
        denom = mask.sum()
        loss = (error * active).sum() / denom.to(dtype)

        ad = self.action_dim
        am, om = mask[:, :, :ad], mask[:, :, ad:]
        info = {
            "flow_loss": loss.detach(),
            "unweighted_flow_loss": self._masked_mean(error, mask).detach(),
            "action_loss": self._masked_mean(error[:, :, :ad], am).detach(),
            "observation_loss": self._masked_mean(error[:, :, ad:], om).detach(),
            "t_state_mean": t_s.mean().detach(),
            "t_action_mean": t_a.mean().detach(),
            "frac_diag": (t_s == t_a).to(dtype).mean().detach(),
        }
        if return_details:
            return loss, info, {"xt": xt, "x0": x0, "x1": x1_local, "t_s": t_s,
                                "t_a": t_a, "target": target,
                                "prediction": prediction, "mask": mask}
        return loss, info
