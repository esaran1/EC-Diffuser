"""Independent-time (UWM-style) conditional flow matching for EC-Diffuser.

Derived from the closed warm-start branch's TwoTimeConditionalFlowMatching, with
the training distribution corrected to be FAITHFUL to UWM per the literature
audit (../literature/UWM_AUDIT.md section 4):

    UWM samples t_action and t_obs INDEPENDENTLY from the uniform distribution,
    with NO deliberate diagonal mass. The training measure is the plain product
    measure. The diagonal is hit only by chance.

Hence `p_diag` DEFAULTS TO 0.0 here, unlike the prior branch's 0.5. A nonzero
p_diag is the section-20 RESCUE arm and must be explicitly requested and its
trigger documented.

Convention note: UWM is DDPM with t=0 CLEAN. We are flow matching with t=0
NOISE. The axis is inverted; corner semantics invert with it. Our t_state=1
(clean state) corresponds to UWM's t_obs=0 conditioning corner.

Zero added parameters: the shared `time_mlp` is evaluated at both times. This
DIVERGES from UWM, which uses a joint MLP over the concatenated pair -- a
documented, deliberate difference (see UWM_AUDIT.md section 8) chosen so that
the diagonal reduces EXACTLY to the baseline, permitting a strict
diagonal-compatibility test with no parameter-count confound.
"""
import torch

from .flow_matching import ConditionalFlowMatching


class UWMConditionalFlowMatching(ConditionalFlowMatching):
    """CFM with independent state/action flow times.

    p_diag : probability that t_state == t_action.
             0.0 = FAITHFUL UWM (default). Nonzero = section-20 rescue arm.
    """

    def __init__(self, *args, p_diag=0.0, adaln_state_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(p_diag) <= 1.0:
            raise ValueError("p_diag must lie in [0, 1]")
        self.p_diag = float(p_diag)
        self.adaln_state_weight = float(adaln_state_weight)
        if hasattr(self.model, "adaln_state_weight"):
            self.model.adaln_state_weight = float(adaln_state_weight)

    # ------------------------------------------------ time sampling
    def sample_two_times(self, batch_size, device, dtype):
        """Faithful UWM at p_diag=0: t_s, t_a ~ U(0,1) INDEPENDENTLY.

        With p_diag>0, that fraction of the batch is placed exactly on the
        diagonal and the remainder is drawn independently. Marginals stay
        U(0,1) in both cases.
        """
        t_s = torch.rand(batch_size, device=device, dtype=dtype)
        t_a = torch.rand(batch_size, device=device, dtype=dtype)
        if self.p_diag > 0.0:
            u = torch.rand(batch_size, device=device, dtype=dtype)
            on_diag = torch.rand(batch_size, device=device, dtype=dtype) < self.p_diag
            t_s = torch.where(on_diag, u, t_s)
            t_a = torch.where(on_diag, u, t_a)
        return t_s, t_a

    # ------------------------------------------------ corruption
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

    # ------------------------------------------------ loss
    def loss(self, x, cond):
        return self._compute_two_time_loss(x, cond)

    def _compute_two_time_loss(self, x1, cond, x0=None, t_s=None, t_a=None,
                               return_details=False):
        """Mirrors production _compute_flow_loss exactly except for the two
        times. Loss form, weights, mask, denominator and terminal supervision
        are untouched, so the diagonal reduces to the baseline."""
        device, dtype = x1.device, x1.dtype
        B = x1.shape[0]
        x1_local = x1.clone()
        x0_local = torch.randn_like(x1) if x0 is None else x0.clone()
        self._apply_conditioning(x1_local, cond)
        self._apply_conditioning(x0_local, cond)

        if t_s is None or t_a is None:
            t_s, t_a = self.sample_two_times(B, device, dtype)
        xt = self.corrupt(x1_local, x0_local, t_s, t_a)
        self._apply_conditioning(xt, cond)

        target = x1_local - x0_local
        prediction = self._call_two_time_model(xt, cond, t_s, t_a)

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
            "abs_time_gap": (t_s - t_a).abs().mean().detach(),
        }
        if return_details:
            return loss, info, {"xt": xt, "x0": x0_local, "x1": x1_local,
                                "t_s": t_s, "t_a": t_a, "target": target,
                                "prediction": prediction, "mask": mask}
        return loss, info

    # ------------------------------------------------ path-aware sampling
    # The inherited conditional_sample integrates a SINGLE scalar time, which
    # corresponds exactly to the diagonal path P0. That is correct for the
    # matched control and for diagonal evaluation of the UWM arm, but a
    # non-diagonal path needs the chain-rule-correct update (section 25).
    #
    # `inference_path` selects the schedule. It defaults to None, which keeps
    # the inherited (diagonal) behaviour byte-for-byte, so nothing changes
    # unless a path is explicitly requested.
    inference_path = None

    def set_inference_path(self, path):
        """Select a registered path name, or None for the inherited diagonal."""
        self.inference_path = path
        return self

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, sort_by_value=True,
                           return_attention=False, n_steps=None,
                           return_chain=False, return_diffusion=None,
                           verbose=True, **sample_kwargs):
        path = getattr(self, "inference_path", None)
        if path is None or path == "P0_diagonal":
            # exact inherited behaviour; P0 is analytically the same update
            return super().conditional_sample(
                cond, horizon=horizon, sort_by_value=sort_by_value,
                return_attention=return_attention, n_steps=n_steps,
                return_chain=return_chain, return_diffusion=return_diffusion,
                verbose=verbose, **sample_kwargs)

        from .diffusion import Sample
        if return_attention:
            raise ValueError("UWMConditionalFlowMatching does not return attention maps")
        active_horizon = self.horizon if horizon is None else horizon
        first_value = next(iter(cond.values()))
        batch_size = first_value.shape[0]
        device, dtype = first_value.device, first_value.dtype
        self._validate_conditioning(cond, batch_size, active_horizon, device,
                                    dtype, True)
        steps = self._resolve_solver_steps(n_steps)
        ad = self.action_dim

        x = torch.randn((batch_size, active_horizon, self.transition_dim),
                        device=device, dtype=dtype)
        self._apply_conditioning(x, cond)
        mask = self._make_conditioning_mask(x, cond).to(dtype)
        chain = [x.clone()] if return_chain else None

        pts = _schedule_points(path, steps)
        for k in range(steps):
            ts0, ta0 = pts[k]
            ts1, ta1 = pts[k + 1]
            t_s = x.new_full((batch_size,), ts0)
            t_a = x.new_full((batch_size,), ta0)
            v = self._call_two_time_model(x, cond, t_s, t_a) * mask
            # chain rule: each block advances by ITS OWN time increment
            x[..., :ad] = x[..., :ad] + (ta1 - ta0) * v[..., :ad]
            x[..., ad:] = x[..., ad:] + (ts1 - ts0) * v[..., ad:]
            self._apply_conditioning(x, cond)
            if return_chain:
                chain.append(x.clone())

        values = x.new_zeros(batch_size)
        chain_tensor = torch.stack(chain, dim=1) if return_chain else None
        return Sample(x, values, chain_tensor)


# --- registered inference paths (mirrors tal_followup/uwm/two_time_sampler.py) ---
def _p0(u):
    return u, u


def _p1(u):
    return 1.0 - (1.0 - u) ** 2, u ** 2


def _p2(u):
    return u ** 2, 1.0 - (1.0 - u) ** 2


def _p3(u):
    return min(2.0 * u, 1.0), max(2.0 * u - 1.0, 0.0)


def _p4(u):
    return max(2.0 * u - 1.0, 0.0), min(2.0 * u, 1.0)


INFERENCE_PATHS = {"P0_diagonal": _p0, "P1_state_leads": _p1,
                   "P2_action_leads": _p2, "P3_state_first": _p3,
                   "P4_action_first": _p4}


def _lead(alpha):
    lam = 0.5 * float(alpha)

    def f(u):
        if lam == 0.0:
            return u, u
        return (min(max(u / (1.0 - lam), 0.0), 1.0),
                min(max((u - lam) / (1.0 - lam), 0.0), 1.0))
    return f


for _a in (0.0, 0.25, 0.5, 0.75, 1.0):
    INFERENCE_PATHS["lead%.2f" % _a] = _lead(_a)


def _schedule_points(path, steps):
    f = INFERENCE_PATHS[path] if isinstance(path, str) else path
    return [f(k / float(steps)) for k in range(steps + 1)]
