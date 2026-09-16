"""Section 18 diagonal-compatibility gate. CPU only, zero GPU.

Requirement: when t_state == t_action == t, the two-time model must reduce
EXACTLY to the baseline. Tested across multiple random batches, several t
values, and both conditioned and unconditioned coordinates.

FAILS LOUD. This gate must pass before any GPU training.
"""
import sys
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
sys.path.insert(0, "experiments/policy_improvement/tal_followup/uwm")
from diffuser.models.pint import AdaLNPINTDenoiser                 # noqa: E402
from diffuser.models.flow_matching import ConditionalFlowMatching  # noqa: E402
from twotime_denoiser import TwoTimePINTDenoiser                   # noqa: E402
from uwm_flow import UWMConditionalFlowMatching                    # noqa: E402

AD, OD, H = 3, 480, 5
TD = AD + OD
DKW = dict(features_dim=10, hidden_dim=64, projection_dim=64, n_head=4,
           n_layer=2, block_size=H, dropout=0.0, multiview=True,
           max_particles=None, positional_bias=False, action_dim=AD)
FAILS = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print("  {}  {}  {}".format("PASS" if ok else "FAIL", name, detail))


def build():
    torch.manual_seed(20260916)
    base = AdaLNPINTDenoiser(**DKW).eval()
    torch.manual_seed(20260916)
    two = TwoTimePINTDenoiser(**DKW).eval()
    two.load_state_dict(base.state_dict())          # identical weights
    wkw = dict(horizon=H, observation_dim=OD, action_dim=AD, n_solver_steps=4,
               loss_type="l2")
    wb = ConditionalFlowMatching(base, **wkw)
    wt = UWMConditionalFlowMatching(two, p_diag=0.0, **wkw)
    return base, two, wb, wt


def main():
    base, two, wb, wt = build()

    print("=== 1. PARAMETER COUNT IDENTICAL (zero-param construction) ===")
    nb = sum(p.numel() for p in base.parameters())
    nt = sum(p.numel() for p in two.parameters())
    check("param count identical", nb == nt, "{} vs {}".format(nb, nt))

    print("\n=== 2. STATE_DICT COMPATIBILITY ===")
    kb, kt = set(base.state_dict()), set(two.state_dict())
    check("state_dict keys identical", kb == kt,
          "extra={} missing={}".format(len(kt - kb), len(kb - kt)))

    print("\n=== 3. DIAGONAL OUTPUT BIT-EXACT (multiple batches, several t) ===")
    worst = 0.0
    for batch in range(3):
        torch.manual_seed(1000 + batch)
        x = torch.randn(4, H, TD)
        for t_val in (0.0, 0.17, 0.5, 0.83, 1.0):
            t = torch.full((4,), t_val)
            with torch.no_grad():
                ob = base(x, None, t * 1000.0)
                ot = two(x, None, t * 1000.0, time_action=t * 1000.0)
            d = (ob - ot).abs().max().item()
            worst = max(worst, d)
    check("diagonal output bit-exact", worst == 0.0, "worst max|d|={:.3e}".format(worst))

    print("\n=== 4. LOSS IDENTICAL ON THE DIAGONAL (cond + uncond) ===")
    conds = {
        "unconditioned": {},
        "conditioned t=0": {0: torch.randn(4, OD)},
        "conditioned t=0 and goal": {0: torch.randn(4, OD), H - 1: torch.randn(4, OD)},
    }
    for cname, cond in conds.items():
        torch.manual_seed(77)
        x1 = torch.randn(4, H, TD)
        x0 = torch.randn(4, H, TD)
        t = torch.full((4,), 0.4)
        lb, ib = wb._compute_flow_loss(x1, cond, x0=x0, t=t)
        lt, it = wt._compute_two_time_loss(x1, cond, x0=x0, t_s=t, t_a=t)
        dl = abs(lb.item() - lt.item())
        check("{:<24} flow_loss identical".format(cname), dl == 0.0,
              "|d|={:.3e}".format(dl))
        da = abs(ib["action_loss"].item() - it["action_loss"].item())
        do = abs(ib["observation_loss"].item() - it["observation_loss"].item())
        check("{:<24} action+obs loss identical".format(cname),
              da == 0.0 and do == 0.0, "da={:.2e} do={:.2e}".format(da, do))

    print("\n=== 5. CONDITIONING MASK + DENOMINATOR IDENTICAL ===")
    cond = {0: torch.randn(4, OD), H - 1: torch.randn(4, OD)}
    x1 = torch.randn(4, H, TD)
    mb = wb._make_conditioning_mask(x1, cond)
    mt = wt._make_conditioning_mask(x1, cond)
    check("mask identical", torch.equal(mb, mt),
          "sum {} vs {}".format(int(mb.sum()), int(mt.sum())))
    check("denominator identical", int(mb.sum()) == int(mt.sum()),
          "{}".format(int(mb.sum())))

    print("\n=== 6. TERMINAL SUPERVISION UNCHANGED ===")
    check("mask_terminal_action off in both",
          not getattr(wb, "mask_terminal_action", False)
          and not getattr(wt, "mask_terminal_action", False))
    check("terminal action row supervised", bool(mt[:, H - 1, :AD].all()),
          "terminal action mask all True")

    print("\n=== 7. CORRUPTION REDUCES TO THE BASELINE ON THE DIAGONAL ===")
    t = torch.full((4,), 0.63)
    x0 = torch.randn(4, H, TD)
    xt_two = wt.corrupt(x1, x0, t, t)
    xt_base = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
    d = (xt_two - xt_base).abs().max().item()
    check("corrupt() == baseline interpolant", d == 0.0, "max|d|={:.3e}".format(d))

    print("\n=== 8. OFF-DIAGONAL ACTUALLY DIFFERS (gate is not vacuous) ===")
    with torch.no_grad():
        o_diag = two(x1, None, torch.full((4,), 300.0), time_action=torch.full((4,), 300.0))
        o_off = two(x1, None, torch.full((4,), 300.0), time_action=torch.full((4,), 900.0))
    d = (o_diag - o_off).abs().max().item()
    check("off-diagonal output differs", d > 1e-6, "max|d|={:.3e}".format(d))

    print("\n=== 9. FAITHFUL UWM TIME DISTRIBUTION (p_diag=0) ===")
    ts, ta = wt.sample_two_times(20000, torch.device("cpu"), torch.float32)
    frac_diag = (ts == ta).float().mean().item()
    check("p_diag=0 -> no deliberate diagonal mass", frac_diag < 1e-4,
          "frac_diag={:.5f}".format(frac_diag))
    check("t_s marginal ~ U(0,1)", abs(ts.mean().item() - 0.5) < 0.02,
          "mean={:.4f} sd={:.4f}".format(ts.mean().item(), ts.std().item()))
    check("t_a marginal ~ U(0,1)", abs(ta.mean().item() - 0.5) < 0.02,
          "mean={:.4f} sd={:.4f}".format(ta.mean().item(), ta.std().item()))
    corr = torch.corrcoef(torch.stack([ts, ta]))[0, 1].item()
    check("t_s and t_a independent", abs(corr) < 0.03, "corr={:+.4f}".format(corr))

    print("\n=== 10. RESCUE ARM (p_diag=0.5) BEHAVES AS SPECIFIED ===")
    wr = UWMConditionalFlowMatching(two, p_diag=0.5, horizon=H, observation_dim=OD,
                                    action_dim=AD, n_solver_steps=4,
                                    loss_type="l2")
    ts2, ta2 = wr.sample_two_times(20000, torch.device("cpu"), torch.float32)
    fd = (ts2 == ta2).float().mean().item()
    check("p_diag=0.5 -> ~half on diagonal", 0.47 < fd < 0.53, "frac_diag={:.4f}".format(fd))
    check("rescue marginal still ~U(0,1)", abs(ts2.mean().item() - 0.5) < 0.02,
          "mean={:.4f}".format(ts2.mean().item()))

    print("\n" + "=" * 62)
    if FAILS:
        print("DIAGONAL-COMPATIBILITY GATE FAILED: {}".format(", ".join(FAILS)))
        return 1
    print("DIAGONAL-COMPATIBILITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
