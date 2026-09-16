"""Direct-action flow wrapper (tal_followup Stage C).

State remains a generative flow: the state block is noised and the state head
predicts a velocity, exactly as in production. The ACTION block is NOT noised
and the action head predicts the action DIRECTLY.

Losses, weights, conditioning mask, denominator semantics and terminal
supervision follow production as closely as the changed action target permits:

  state coordinates  : |v_pred - (s - eps_s)|  (unchanged)
  action coordinates : |a_pred - a|            (direct, NOT a velocity)

Both are combined with the SAME loss_weight_matrix and the SAME denominator as
production, so the state pathway is numerically comparable to the baseline.

Section 15: this is architecturally distinct from any diffusion endpoint and is
never described as t_action=0 or t_action=1.
"""
import torch

from .flow_matching import ConditionalFlowMatching


class QueryActionFlowMatching(ConditionalFlowMatching):
    """Generative state + directly-predicted action."""

    def loss(self, x, cond):
        return self._compute_query_loss(x, cond)

    def _compute_query_loss(self, x1, cond, x0=None, t=None, return_details=False):
        device, dtype = x1.device, x1.dtype
        B = x1.shape[0]
        ad = self.action_dim

        x1_local = x1.clone()
        x0_local = torch.randn_like(x1) if x0 is None else x0.clone()
        self._apply_conditioning(x1_local, cond)
        self._apply_conditioning(x0_local, cond)

        if t is None:
            t = torch.rand(B, device=device, dtype=dtype)
        bt = t.view(-1, 1, 1)

        # STATE is noised as usual; the ACTION slice is irrelevant to the model
        # (the query token replaces it) but is kept well-defined for shape.
        xt = (1.0 - bt) * x0_local + bt * x1_local
        self._apply_conditioning(xt, cond)

        prediction = self._call_model(xt, cond, t * self.time_scale)

        # blockwise targets: velocity for state, DIRECT action for action
        target = torch.empty_like(x1_local)
        target[..., :ad] = x1_local[..., :ad]                       # direct action
        target[..., ad:] = (x1_local - x0_local)[..., ad:]          # state velocity

        error = torch.abs(prediction - target)
        if self.loss_type == "l2":
            error = error.square()

        mask = self._make_conditioning_mask(x1_local, cond)
        weights = self.loss_weight_matrix.to(device=device, dtype=dtype).unsqueeze(0)
        active = weights * mask.to(dtype)
        denom = mask.sum()
        if denom.item() == 0 or not torch.any(active > 0):
            raise ValueError("conditioning and loss settings leave no active elements")
        loss = (error * active).sum() / denom.to(dtype)

        am, om = mask[:, :, :ad], mask[:, :, ad:]
        info = {
            "flow_loss": loss.detach(),
            "unweighted_flow_loss": self._masked_mean(error, mask).detach(),
            "action_loss": self._masked_mean(error[:, :, :ad], am).detach(),
            "observation_loss": self._masked_mean(error[:, :, ad:], om).detach(),
        }
        if return_details:
            return loss, info, {"xt": xt, "x0": x0_local, "x1": x1_local,
                                "t": t, "target": target, "prediction": prediction,
                                "mask": mask}
        return loss, info

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, sort_by_value=True,
                           return_attention=False, n_steps=None,
                           return_chain=False, return_diffusion=None,
                           verbose=True, **sample_kwargs):
        """Integrate the STATE flow; read the action directly from the final call.

        The action is NOT integrated -- it is produced by the query token from
        the (progressively denoised) state context. The action reported is the
        one predicted at the LAST solver step, i.e. conditioned on the cleanest
        available state.
        """
        from .diffusion import Sample
        if return_attention:
            raise ValueError("QueryActionFlowMatching does not return attention maps")
        if return_diffusion is not None:
            return_chain = bool(return_diffusion)
        active_horizon = self.horizon if horizon is None else horizon
        first_value = next(iter(cond.values()))
        batch_size = first_value.shape[0]
        device, dtype = first_value.device, first_value.dtype
        self._validate_conditioning(cond, batch_size, active_horizon, device, dtype, True)
        steps = self._resolve_solver_steps(n_steps)
        ad = self.action_dim

        x = torch.randn((batch_size, active_horizon, self.transition_dim),
                        device=device, dtype=dtype)
        self._apply_conditioning(x, cond)
        conditioning_mask = self._make_conditioning_mask(x, cond)
        chain = [x.clone()] if return_chain else None
        dt = 1.0 / float(steps)
        action_pred = None
        for step in range(steps):
            time = x.new_full((batch_size,), float(step) / float(steps))
            out = self._call_model(x, cond, time * self.time_scale)
            action_pred = out[:, :, :ad]                    # direct, overwritten
            v = out * conditioning_mask.to(dtype)
            x[..., ad:] = x[..., ad:] + dt * v[..., ad:]    # state only
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())
        x[..., :ad] = action_pred                            # final direct action

        values = x.new_zeros(batch_size)
        chain_tensor = torch.stack(chain, dim=1) if return_chain else None
        return Sample(x, values, chain_tensor)
