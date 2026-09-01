# The estimand, and what a "trial" is

## The problem this fixes

Prior reports were statistically fuzzy about what "benchmark performance" means.
That vagueness is the root of the confusion between bias, variance, and coverage.

## Definition (frozen benchmark, §14-A)

Our benchmark is a **fixed set** of N=96 scenarios, not a sample from a scenario
distribution. So for policy π:

```
J(π) = (1/N) Σ_i  E_physics[ Y | scenario_i, π ]
```

and the policy contrast is

```
Δ = (1/N) Σ_i ( E[Y | scenario_i, π_B] − E[Y | scenario_i, π_A] )
```

The inner expectation is over **simulator realizations and policy sampling**,
conditional on a fixed scenario. `p_i = E[Y | scenario_i, π]` is the per-scenario
success *probability* — not a deterministic label.

## What is a trial? (§13)

This is the conceptual core. Conventional practice takes the trial unit to be
**one scenario**, implicitly assuming

```
Y | scenario_i, π   is deterministic   ⟺   p_i ∈ {0,1}
```

Our measurements refute that assumption: with the policy, scenarios, and
policy-noise all fixed, **39/96 scenarios (3-cube) and ~53% (4-cube) have
0 < p_i < 1**. The nominal trial contains a hidden realization level.

### Can we just call each realization an i.i.d. trial?

**No.** Realizations within a scenario are exchangeable *given the scenario* but
not across scenarios, because scenarios have heterogeneous difficulty. Pooling
N×R realizations as i.i.d. and applying a standard binomial/sequential test:

1. uses the wrong variance (ignores between-scenario heterogeneity), and
2. targets the wrong estimand if the benchmark is a *fixed* scenario set — the
   sampling variability of interest is over physics, not over scenarios.

The correct treatment is a **nested/clustered** one: scenarios at the top level,
realizations nested within scenario. That is the estimator all our calibrated
results use.

## Measured variance components (same arm, R=8, 3-cube, seed 42 NFE4)

For a binary outcome, total variance decomposes as between-scenario variance of
`p_i` plus the mean within-scenario Bernoulli variance:

| component | value | share |
|---|--:|--:|
| between-scenario, Var(p_i) | 0.02845 | **28.5%** |
| within-scenario (physics), E[p_i(1−p_i)] | 0.07129 | **71.5%** |
| total | 0.09974 | 100% |

**Intra-scenario correlation ICC = 0.285.**

> Treating a scenario as one deterministic trial implicitly assumes **ICC = 1**.
> We measure **0.285** — about **71% of outcome variance lives *below* the
> nominal trial unit.**

Design effect `1 + (R−1)·ICC` and effective sample size per scenario:

| R | design effect | effective n / scenario |
|--:|--:|--:|
| 1 | 1.00 | 1.00 |
| 3 | 1.57 | 1.91 |
| 5 | 2.14 | 2.34 |
| 8 | 3.00 | 2.67 |

Replication buys real but strongly sublinear information — which is exactly why
the resolution curve flattens (7.55 → 2.67 pp from R=1 to R=8).

SE of J(π): **3.22 pp** treating scenarios as deterministic at R=1, versus
**1.97 pp** with R=8 nested realizations.

## Bias vs variance vs coverage (§15) — three distinct things, only one is ours

| candidate failure | verdict |
|---|---|
| Is R=1 **biased**? | **NO.** Mean of the 9 reconstructed single-realization views equals the calibrated estimate to 2 dp on every comparison (e.g. +9.03 vs +9.03; −1.04 vs −1.04). |
| Do episode-only CIs **under-cover**? | **NO.** Pseudo-coverage vs the R=8 reference: 96.8% (R=1), 98.7% (R=3), 99.8% (R=5) — nominal 95%, i.e. *conservative*. |
| Is R=1 **high-variance / low-resolution**? | **YES.** SD of a single-realization Δ is 2.15-4.91 pp per comparison; the paired resolution is 7.55 pp at R=1. |

**The claim must be resolution, never bias, never undercoverage.** An R=1
episode-only interval is ~12.1 pp wide: it covers the truth *because* it is too
wide to resolve the 3-6 pp effects the fast-policy literature reports.
