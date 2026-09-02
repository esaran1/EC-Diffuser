# Benchmark Resolution Surface (N × R)

CPU only. E1 untouched. Derived from frozen calibrated artifacts.

## 1. Exact variance

For a two-arm contrast on a fixed scenario set, with independent arms (Cov = 0
by design) and R physics realizations per scenario:

    Var[Δ̂] = 2w / (N · R)

with **w = 0.08147** the bias-corrected within-scenario outcome variance measured
from the R=8 same-policy bank (seed 42, NFE4, 96 scenarios).

## 2. Predicted 95% half-width (pp)

| N \ R | R=1 | R=2 | R=3 | R=5 | R=8 |
|--:|--:|--:|--:|--:|--:|
| 25 | 15.82 | 11.19 | 9.14 | 7.08 | 5.59 |
| 50 | 11.19 | 7.91 | 6.46 | 5.00 | 3.96 |
| **96** | **8.08** | **5.71** | **4.66** | **3.61** | **2.85** |
| 200 | 5.59 | 3.96 | 3.23 | 2.50 | 1.98 |
| 500 | 3.54 | 2.50 | 2.04 | 1.58 | 1.25 |

## 3. N and R are NOT interchangeable

They enter the variance identically and **differ in meaning**:

- **Increasing R** estimates the **same fixed benchmark** more precisely. The
  estimand is unchanged. This is a measurement decision.
- **Increasing N** changes the benchmark, and therefore **changes the estimand**.
  A 500-scenario benchmark answers a different question than a 96-scenario one.
  This is a benchmark-design decision.

A design table that presents them as a single "just multiply N·R" knob is
misleading, and we do not present one.

## 4. Empirical validation by subsampling (§12)

Scenario subsamples drawn from the 96-bank; two disjoint 4-realization halves of
the R=8 bank form a genuine same-policy null contrast. 4,000 CPU resamples per cell.

| N | R | empirical SD | analytic SD | ratio |
|--:|--:|--:|--:|--:|
| 25 | 1 | 7.87 | 8.07 | 0.98 |
| 25 | 2 | 5.96 | 5.71 | 1.04 |
| 25 | 3 | 5.15 | 4.66 | 1.11 |
| 25 | 4 | 4.74 | 4.04 | 1.17 |
| 50 | 1 | 5.39 | 5.71 | 0.94 |
| 50 | 2 | 4.00 | 4.04 | 0.99 |
| 50 | 3 | 3.53 | 3.30 | 1.07 |
| 50 | 4 | 3.13 | 2.85 | 1.10 |
| 96 | 1 | 3.61 | 4.12 | 0.88 |
| 96 | 2 | 2.57 | 2.91 | 0.88 |
| 96 | 3 | 2.05 | 2.38 | 0.86 |
| 96 | 4 | 1.79 | 2.06 | 0.87 |

**Mean ratio 0.990 (range 0.862–1.173).** The formula is validated.

Two known, stated artifacts:
- **N=96 runs ~13% low** because the "subsample" is the entire bank, so no
  between-scenario sampling variance is induced. This is expected by construction,
  not a model failure.
- **R≥3 runs slightly high** because only 4 distinct realizations exist per half;
  resampling them with replacement reuses realizations.

## 5. Design reading

At the 96-scenario benchmark actually used here, resolving a 5 pp effect needs
**R ≥ 2–3**. Resolving a 2 pp effect is out of reach at any R ≤ 8 without also
enlarging the scenario set — and enlarging it changes the benchmark.
