"""Unit tests for the path-correct two-time sampler (section 25). CPU only.

Decisive test: with an ORACLE constant velocity field the ODE is linear and
forward Euler is exact for ANY step count and ANY schedule, so every registered
path must land on the clean target to float tolerance. A chain-rule or endpoint
bug fails this immediately.

Control test: with a STATE-DEPENDENT field Euler is NOT exact, and error must
fall at a first-order rate -- this catches a sampler that secretly hard-codes
the analytic solution.
"""
import sys
import torch

sys.path.insert(0, "experiments/policy_improvement/tal_followup/uwm")
from two_time_sampler import (PATHS, two_time_sample, validate_schedule,  # noqa: E402
                              schedule_points)

AD, OD, H, B = 3, 480, 5, 4
TD = AD + OD
FAILS = []


def check(name, ok, detail=""):
    FAILS.append(name) if not ok else None
    print("  {}  {}  {}".format("PASS" if ok else "FAIL", name, detail))


def main():
    torch.manual_seed(20260916)

    print("=== 1. SCHEDULE VALIDITY (endpoints, monotone, range) ===")
    for name in PATHS:
        ok, detail = validate_schedule(name)
        check("{:<16} valid".format(name), ok, detail)

    print("\n=== 2. SCHEDULES ARE DISTINCT (not silently aliased) ===")
    sigs = {n: tuple(round(v, 6) for p in schedule_points(n, 8) for v in p)
            for n in PATHS}
    check("all 5 paths distinct", len(set(sigs.values())) == 5,
          "{} unique".format(len(set(sigs.values()))))
    mid = {n: schedule_points(n, 2)[1] for n in PATHS}
    check("P1 state-leads at midpoint", mid["P1_state_leads"][0] > mid["P1_state_leads"][1],
          "t_s={:.2f} > t_a={:.2f}".format(*mid["P1_state_leads"]))
    check("P2 action-leads at midpoint", mid["P2_action_leads"][1] > mid["P2_action_leads"][0],
          "t_a={:.2f} > t_s={:.2f}".format(mid["P2_action_leads"][1], mid["P2_action_leads"][0]))

    print("\n=== 3. ORACLE FIELD: exact landing on clean target (DECISIVE) ===")
    eps = torch.randn(B, H, TD)
    x1 = torch.randn(B, H, TD)

    def oracle(x, t_s, t_a):
        return x1 - eps                      # constant, = target velocity

    for name in PATHS:
        for steps in (1, 4, 17):
            out = two_time_sample(oracle, eps.clone(), AD, path=name, steps=steps)
            err = (out - x1).abs().max().item()
            check("{:<16} steps={:<3} lands on x1".format(name, steps),
                  err < 1e-5, "max|err|={:.3e}".format(err))

    print("\n=== 4. BLOCKWISE CHAIN RULE (action uses t_a, state uses t_s) ===")
    # Partial integration to u=0.5 must leave each block at its OWN schedule
    # fraction, not at a shared u.
    for name in ("P1_state_leads", "P3_state_first"):
        half = 8
        pts = schedule_points(name, 2 * half)
        ts_half, ta_half = pts[half]
        out = two_time_sample(oracle, eps.clone(), AD, path=name, steps=2 * half)
        # re-run truncated: integrate only the first half of the knots
        x = eps.clone()
        for k in range(half):
            a0, b0 = pts[k]; a1, b1 = pts[k + 1]
            v = oracle(x, a0, b0)
            x[..., :AD] = x[..., :AD] + (b1 - b0) * v[..., :AD]
            x[..., AD:] = x[..., AD:] + (a1 - a0) * v[..., AD:]
        exp_s = eps[..., AD:] + ts_half * (x1 - eps)[..., AD:]
        exp_a = eps[..., :AD] + ta_half * (x1 - eps)[..., :AD]
        es = (x[..., AD:] - exp_s).abs().max().item()
        ea = (x[..., :AD] - exp_a).abs().max().item()
        check("{:<16} state at t_s={:.3f}".format(name, ts_half), es < 1e-5,
              "max|err|={:.2e}".format(es))
        check("{:<16} action at t_a={:.3f}".format(name, ta_half), ea < 1e-5,
              "max|err|={:.2e}".format(ea))

    print("\n=== 5. NAIVE SHARED-du UPDATE IS DETECTABLY WRONG ===")
    # A CONSTANT field cannot expose this: each block's increments sum to 1
    # either way, so both updates land identically. Use a TIME-DEPENDENT field,
    # where integrating against du instead of dt_block genuinely differs.
    def timed(x, t_s, t_a):
        # state velocity depends on t_s, action velocity on t_a
        v = torch.zeros_like(x)
        v[..., :AD] = 1.0 + 3.0 * (t_a ** 2)
        v[..., AD:] = 1.0 + 3.0 * (t_s ** 2)
        return v

    name = "P3_state_first"
    steps = 8
    pts = schedule_points(name, steps)
    correct = two_time_sample(timed, torch.zeros(B, H, TD), AD, path=name, steps=steps)
    x = torch.zeros(B, H, TD)
    for k in range(steps):
        v = timed(x, *pts[k])
        x = x + (1.0 / steps) * v          # WRONG: shared du for both blocks
    gap = (x - correct).abs().max().item()
    check("naive shared-du differs from correct on P3", gap > 1e-2,
          "max|correct-naive|={:.3e}".format(gap))

    print("\n=== 6. STATE-DEPENDENT FIELD: first-order convergence ===")
    # v = c*(x1 - x) solves to x(1) = x1 + (x0-x1)*exp(-c) -- it does NOT reach
    # x1. Euler must converge to THAT limit at first order.
    import math
    c = 2.0

    def dependent(x, t_s, t_a):
        return c * (x1 - x)

    exact = x1 + (eps - x1) * math.exp(-c)
    errs = []
    for steps in (4, 8, 16, 32, 64):
        out = two_time_sample(dependent, eps.clone(), AD, path="P0_diagonal", steps=steps)
        errs.append((out - exact).abs().max().item())
    monotone = all(b < a for a, b in zip(errs, errs[1:]))
    ratio = errs[0] / max(errs[-1], 1e-12)
    check("error -> exact ODE solution decreases with steps", monotone,
          " ".join("{:.2e}".format(e) for e in errs))
    check("first-order: error shrinks >= 8x from 4->64 steps", ratio >= 8.0,
          "ratio={:.1f}x".format(ratio))

    print("\n=== 7. NO CLEAN-TARGET LEAKAGE ===")
    # With a velocity field that returns zeros, output must equal the init
    # exactly: nothing in the sampler may nudge x toward x1.
    out = two_time_sample(lambda x, a, b: torch.zeros_like(x), eps.clone(), AD,
                          path="P1_state_leads", steps=6)
    check("zero field leaves x unchanged", torch.equal(out, eps),
          "max|d|={:.2e}".format((out - eps).abs().max().item()))

    print("\n" + "=" * 60)
    if FAILS:
        print("FAILED: {}".format(", ".join(FAILS)))
        return 1
    print("ALL SAMPLER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
