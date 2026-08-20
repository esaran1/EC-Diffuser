"""Item 17: derive and verify the variance-preserving flow target.

The repository's vanilla flow uses, per
`diffuser/diffuser/models/flow_matching.py:272-274`:

    x_t    = (1-t) * x0 + t * x1        with x0 ~ N(0,I) (noise), x1 = data
    target = x1 - x0

so the convention is t=0 -> noise, t=1 -> data, integrated forward with Euler.
That matches the convention assumed in the directive, so no formula needs
reversing.

For the VP path we set

    n(t) = (1-t) z + t x
    s(t) = sqrt((1-t)^2 + t^2)
    y(t) = n(t) / s(t)

and the velocity target is dy/dt, which is NOT x - z. This script derives it
symbolically, checks it against autodiff and finite differences, and verifies
the endpoint and unit-variance properties.
"""

import numpy as np
import torch


def s(t):
    return np.sqrt((1.0 - t) ** 2 + t**2)


def vp_interpolant(t, z, x):
    return ((1.0 - t) * z + t * x) / s(t)


def vp_velocity(t, z, x):
    """Analytic dy/dt for y(t) = n(t)/s(t).

    n'(t)  = x - z
    s'(t)  = (2t - 1) / s(t)
    dy/dt  = n'/s - n s'/s^2 = (x - z)/s - n (2t-1)/s^3
    """
    n = (1.0 - t) * z + t * x
    st = s(t)
    return (x - z) / st - n * (2.0 * t - 1.0) / st**3


def check_symbolic():
    """Confirm the derivation with sympy rather than trusting the algebra."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed - skipped"

    t, z, x = sp.symbols("t z x", real=True)
    st = sp.sqrt((1 - t) ** 2 + t**2)
    y = ((1 - t) * z + t * x) / st
    derived = sp.simplify(sp.diff(y, t))

    n = (1 - t) * z + t * x
    claimed = sp.simplify((x - z) / st - n * (2 * t - 1) / st**3)

    return "MATCH" if sp.simplify(derived - claimed) == 0 else f"MISMATCH: {derived}"


def check_autodiff():
    """Independent numerical check via torch autograd."""
    torch.manual_seed(0)
    z = torch.randn(64, dtype=torch.float64)
    x = torch.randn(64, dtype=torch.float64)

    worst = 0.0
    for t_val in np.linspace(0.01, 0.99, 25):
        t = torch.tensor(t_val, dtype=torch.float64, requires_grad=True)
        st = torch.sqrt((1 - t) ** 2 + t**2)
        y = ((1 - t) * z + t * x) / st
        grad = torch.autograd.grad(y.sum(), t)[0].item()

        analytic = vp_velocity(t_val, z.numpy(), x.numpy()).sum()
        worst = max(worst, abs(grad - analytic))
    return worst


def check_finite_difference():
    rng = np.random.RandomState(0)
    z, x = rng.randn(256), rng.randn(256)
    eps = 1e-6
    worst = 0.0
    for t in np.linspace(0.02, 0.98, 25):
        numeric = (vp_interpolant(t + eps, z, x) - vp_interpolant(t - eps, z, x)) / (2 * eps)
        worst = max(worst, np.abs(numeric - vp_velocity(t, z, x)).max())
    return worst


def check_endpoints():
    rng = np.random.RandomState(0)
    z, x = rng.randn(1000), rng.randn(1000)
    return (
        float(np.abs(vp_interpolant(0.0, z, x) - z).max()),
        float(np.abs(vp_interpolant(1.0, z, x) - x).max()),
    )


def check_unit_variance():
    """With standardized, independent endpoints the path holds unit variance."""
    rng = np.random.RandomState(0)
    x = rng.randn(200000)  # standardized data stand-in
    z = rng.randn(200000)
    return {float(t): float((vp_interpolant(t, z, x) ** 2).mean()) for t in np.linspace(0, 1, 11)}


def check_conditioned_dimensions():
    """Item 18: conditioned entries must survive the division by s(t) exactly.

    The correct order is: interpolate, divide, then re-impose the condition --
    which is what vanilla flow already does with `_apply_conditioning`. If the
    condition were imposed before the division it would be scaled by 1/s(t) and
    corrupted, so this check contrasts the two orderings.
    """
    rng = np.random.RandomState(0)
    x = rng.randn(500)
    z = rng.randn(500)
    condition_index = np.arange(0, 500, 5)
    condition_value = x[condition_index].copy()

    worst_correct, worst_wrong = 0.0, 0.0
    for t in np.linspace(0.0, 1.0, 21):
        # Correct: divide first, then re-impose exact condition values.
        y = vp_interpolant(t, z, x).copy()
        y[condition_index] = condition_value
        worst_correct = max(worst_correct, np.abs(y[condition_index] - condition_value).max())

        # Wrong: impose before dividing, so the condition is scaled by 1/s(t).
        n = (1 - t) * z + t * x
        n[condition_index] = condition_value
        y_bad = n / s(t)
        worst_wrong = max(worst_wrong, np.abs(y_bad[condition_index] - condition_value).max())

    return worst_correct, worst_wrong


def main():
    print("=== Item 10: convention confirmed from source ===")
    print("  x_t    = (1-t)*x0 + t*x1,  x0 ~ N(0,I) noise, x1 = data")
    print("  target = x1 - x0            (flow_matching.py:272-274)")
    print("  t=0 -> noise, t=1 -> data, forward Euler (flow_matching.py:369-371)")

    print("\n=== Item 17: VP velocity target ===")
    print("  y(t)   = ((1-t)z + tx) / sqrt((1-t)^2 + t^2)")
    print("  dy/dt  = (x-z)/s(t) - n(t)(2t-1)/s(t)^3")
    print("  symbolic (sympy)     :", check_symbolic())
    print(f"  autodiff max err     : {check_autodiff():.3e}")
    print(f"  finite-diff max err  : {check_finite_difference():.3e}")

    t0, t1 = check_endpoints()
    print(f"  endpoint t=0 == noise: max err {t0:.3e}")
    print(f"  endpoint t=1 == data : max err {t1:.3e}")

    print("\n  unit variance across t (standardized independent endpoints):")
    for t, v in check_unit_variance().items():
        print(f"    t={t:.1f}  E[y^2]={v:.4f}")

    correct, wrong = check_conditioned_dimensions()
    print("\n=== Item 18: conditioned dimensions ===")
    print(f"  divide-then-condition (correct) max deviation: {correct:.3e}")
    print(f"  condition-then-divide (wrong)   max deviation: {wrong:.3e}")
    print("  => conditions must be re-imposed AFTER dividing by s(t)")


if __name__ == "__main__":
    main()
