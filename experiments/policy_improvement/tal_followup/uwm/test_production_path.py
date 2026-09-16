"""Section 53 production-path config validation. CPU only, zero GPU.

The historical kwargs defect: diffusion_wrapper_kwargs is a WHITELIST, and the
loss knobs were omitted, so every arm silently trained arm A's objective and
produced byte-identical checkpoints. Validating by directly constructing the
wrapper would NOT have caught it.

This test therefore exercises the real `diffusion_wrapper_kwargs` function that
train.py calls, not a hand-built kwargs dict.
"""
import sys
import types

sys.path.insert(0, "diffuser")
from diffuser.configuration import diffusion_wrapper_kwargs  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print("  {}  {}  {}".format("PASS" if ok else "FAIL", name, detail))


def fake_args(**over):
    a = types.SimpleNamespace(
        horizon=5, n_diffusion_steps=4, loss_type="l2", clip_denoised=False,
        predict_epsilon=True, action_weight=1.0, loss_weights=None,
        loss_discount=1.0, obs_only=False, action_only=False, time_scale=1000.0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def main():
    print("=== 1. TWO-TIME KNOBS SURVIVE THE WHITELIST ===")
    kw = diffusion_wrapper_kwargs(fake_args(p_diag=0.0, adaln_state_weight=0.5),
                                  480, 3)
    check("p_diag forwarded", "p_diag" in kw, "value={}".format(kw.get("p_diag")))
    check("adaln_state_weight forwarded", "adaln_state_weight" in kw,
          "value={}".format(kw.get("adaln_state_weight")))

    print("\n=== 2. VALUES ARE PRESERVED EXACTLY (not defaulted) ===")
    for val in (0.0, 0.25, 0.5, 1.0):
        kw = diffusion_wrapper_kwargs(fake_args(p_diag=val), 480, 3)
        check("p_diag={} preserved".format(val), kw.get("p_diag") == val,
              "got {}".format(kw.get("p_diag")))

    print("\n=== 3. ABSENT KNOB IS OMITTED (control arm unaffected) ===")
    kw = diffusion_wrapper_kwargs(fake_args(), 480, 3)
    check("no p_diag when not configured", "p_diag" not in kw)
    check("no adaln_state_weight when not configured", "adaln_state_weight" not in kw)

    print("\n=== 4. THE TWO ARMS PRODUCE DIFFERENT WRAPPER KWARGS ===")
    # This is the check whose absence allowed the kwargs defect.
    uwm = diffusion_wrapper_kwargs(fake_args(p_diag=0.0), 480, 3)
    ctl = diffusion_wrapper_kwargs(fake_args(p_diag=1.0), 480, 3)
    check("UWM arm != diagonal control arm", uwm != ctl,
          "p_diag {} vs {}".format(uwm.get("p_diag"), ctl.get("p_diag")))

    print("\n=== 5. PRE-EXISTING KNOBS STILL FORWARDED (no regression) ===")
    kw = diffusion_wrapper_kwargs(
        fake_args(mask_terminal_action=True, lambda_action=2.0,
                  denominator_ignores_terminal_mask=True), 480, 3)
    for k, v in (("mask_terminal_action", True), ("lambda_action", 2.0),
                 ("denominator_ignores_terminal_mask", True)):
        check("{} still forwarded".format(k), kw.get(k) == v,
              "got {}".format(kw.get(k)))

    print("\n=== 6. WRAPPER ACCEPTS THE PRODUCED KWARGS ===")
    import torch  # noqa: E402
    sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
    sys.path.insert(0, "experiments/policy_improvement/tal_followup/uwm")
    from diffuser.models.pint import AdaLNPINTDenoiser      # noqa: E402
    from twotime_denoiser import TwoTimePINTDenoiser        # noqa: E402
    from uwm_flow import UWMConditionalFlowMatching         # noqa: E402

    torch.manual_seed(0)
    m = TwoTimePINTDenoiser(features_dim=10, action_dim=3, hidden_dim=32,
                            projection_dim=32, n_head=4, n_layer=1,
                            block_size=5, dropout=0.0, multiview=True,
                            max_particles=None, positional_bias=False)
    kw = diffusion_wrapper_kwargs(fake_args(p_diag=0.0, adaln_state_weight=0.5),
                                  480, 3)
    kw.pop("n_timesteps", None)
    kw["n_solver_steps"] = 4
    try:
        w = UWMConditionalFlowMatching(m, **kw)
        ok, detail = True, "p_diag={} on wrapper".format(w.p_diag)
    except Exception as e:      # noqa: BLE001
        ok, detail = False, "{}: {}".format(type(e).__name__, e)
    check("wrapper constructs from production kwargs", ok, detail)
    if ok:
        check("wrapper.p_diag == configured 0.0", w.p_diag == 0.0,
              "got {}".format(w.p_diag))

    print("\n" + "=" * 60)
    if FAILS:
        print("PRODUCTION-PATH VALIDATION FAILED: {}".format(", ".join(FAILS)))
        return 1
    print("PRODUCTION-PATH VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
