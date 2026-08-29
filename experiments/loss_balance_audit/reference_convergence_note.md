# Flow-ODE reference convergence: measured, not assumed

Phase 2 required a high-accuracy reference and explicit verification. The
verification **failed the naive form of the test**, and that is itself a result.

## RK4 step-refinement ladder (seed 42, 16 episodes, fixed x0, float32)

| refinement | NFE | `|x_n - x_{n/2}|` mean | max |
|---|--:|--:|--:|
| rk4@16 vs rk4@8    |  64 | 0.46582 | 0.96147 |
| rk4@32 vs rk4@16   | 128 | 0.17353 | 0.30440 |
| rk4@64 vs rk4@32   | 256 | 0.12280 | 0.65311 |
| rk4@128 vs rk4@64  | 512 | 0.10427 | 0.89600 |

A convergent RK4 solve shrinks this by ~16x per halving of the step. It does
not: the difference **stalls at ~0.10-0.12** and stops improving.

## Is that float32 nondeterminism?

No. Determinism and independent-scheme checks (seed 42, same x0):

| check | mean | max |
|---|--:|--:|
| rk4@64 run twice (repeat determinism) | **0.00000000** | 0.00000000 |
| rk4@64 vs midpoint@128 (both NFE 256) | 0.17496 | 1.10279 |
| rk4@64 vs euler@512                   | 0.20586 | 1.09942 |

The solve is **bitwise reproducible**, yet three *different* high-order schemes
at NFE 256-512 disagree with each other by ~0.17-0.21. Disagreement between
independent schemes that does not shrink with refinement is not roundoff and is
not a solver bug: the learned velocity field is rough enough that fine-scale
trajectories separate. The Flow ODE is **not integrable to machine precision**
at this model's smoothness.

## Consequence for this study

We therefore do NOT claim a converged reference. We define:

- **reference** = RK4 @ 64 steps = **256 NFE**
- **reference uncertainty floor** = disagreement with an *independent* scheme of
  equal cost (midpoint @ 128 steps, also 256 NFE), reported alongside every
  numerical-error number.

Predefined tolerance: the reference is usable iff its uncertainty floor is at
least 5x smaller than the numerical error of the coarsest arm we interpret.
Measured: floor ~0.17 vs Euler@16 distance ~1.34 (**~8x**), and vs Euler@2
~5.56 (~32x). **Tolerance met** for every arm reported.

Numerical-error differences approaching ~0.17 are NOT resolvable and are not
interpreted.

Endpoint norm for scale: `|x_ref|` ~ 16.95, so the floor is ~1% of the endpoint.
