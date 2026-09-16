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
import sys
import torch

sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
from diffuser.models.flow_matching import ConditionalFlowMatching  # noqa: E402


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
