# Variance decomposition (§10-§11)

Same arm (seed 42, NFE4, 3-cube), R=8 identical repeats, 96 frozen scenarios,
CRN policy noise fixed. Binary outcome.

| component | value | share |
|---|--:|--:|
| between-scenario Var(p_i) | 0.02845 | 28.5% |
| **within-scenario (physics) E[p_i(1−p_i)]** | **0.07129** | **71.5%** |
| total | 0.09974 | |

**ICC = 0.285.** Treating a scenario as a deterministic trial assumes ICC = 1.

| R | design effect 1+(R−1)ICC | effective n / scenario |
|--:|--:|--:|
| 1 | 1.00 | 1.00 |
| 3 | 1.57 | 1.91 |
| 5 | 2.14 | 2.34 |
| 8 | 3.00 | 2.67 |

SE(J) = 3.22 pp treating scenarios as deterministic (R=1) vs 1.97 pp at R=8.

## Concentration

Variance is not uniform: **39/96 scenarios (41%) carry essentially all of it**;
57/96 are robust successes (p_i = 1) and **0/96 are robust failures**. At 4 cubes
the sensitive fraction rises to **~53%** with 5-9 robust failures per arm.

## 3-cube vs 4-cube (§23 — reported as persistence, not a law)

| | 3-cube | 4-cube |
|---|--:|--:|
| physics-sensitive scenarios | 41% | 53% |
| sign disagreement | 25% | 22% |
| within-arm spread | 10.4 pp (R=8, same arm) | 6.6 pp (R=3, mean of 6 arms) |

**No causal claim that compositionality increases evaluator noise.** Success
rate, contact opportunity and difficulty all differ. The defensible statement is
**persistence across a second task regime**, not a monotone compositional law.
The two spread figures are not directly comparable (different R, different
aggregation) and are labelled as such.

## Does physics variance dominate the effects we care about?

Yes, and this is the crux. The policy effects at stake are 0.7-9.0 pp. A single
realization of an arm has SD 4.23 pp, so a single-realization Δ has SD ≈ 5.98 pp.
**The noise is the same size as or larger than most of the treatment effects the
fast-policy literature reports.**
