# Resolution audit — theory (CPU only; E1 untouched)

## 1. Exact fixed-benchmark variance

Scenarios i = 1…N fixed. For arm A, `μ_Ai = E_r[Y_Air]`, `σ²_Ai = Var_r(Y_Air)`.
Estimand and estimator:

```
Δ   = (1/N) Σ_i (μ_Bi − μ_Ai)
Δ̂  = (1/N) Σ_i (Ȳ_Bi − Ȳ_Ai)
```

**Exact conditional variance** (conditioning on the fixed scenario set):

```
Var[Δ̂] = (1/N²) Σ_i [ σ²_Ai/R_Ai + σ²_Bi/R_Bi − 2·Cov_i(Ȳ_Ai, Ȳ_Bi) ]
```

Δ̂ is **unbiased for the fixed-benchmark Δ** with no distributional assumption;
between-scenario heterogeneity `Var(μ_i)` **does not appear** — it cancels because
the scenario set is fixed, not sampled.

For binary outcomes `σ²_Ai = μ_Ai(1−μ_Ai)`, so no separate variance model is
needed.

### When the simplified form is exact

With `R_Ai = R_Bi = R` and `Cov_i = 0`:

```
Var[Δ̂] = (1/(N²R)) Σ_i (σ²_Ai + σ²_Bi) = 2·w̄/(N·R),
   w̄ ≡ (1/2N) Σ_i (σ²_Ai + σ²_Bi)
```

This is an **identity, not an approximation**, once `w̄` is defined as the
across-arm mean. Verified numerically on all six 3-cube contrasts: exact and
simplified SE agree to **0.0000 pp**. Homoskedasticity is **not** assumed — the
per-scenario variances are averaged, not equated.

The two real assumptions are: **(a) equal R across arms; (b) zero cross-arm
covariance.**

## 2. Covariance / CRN — set to zero, deliberately

Policy-noise CRN is bit-exact, so it removes policy sampling as a source of
*arm-specific* variation. But the physics realization is **not** controllable:
two arms executed at different wall-clock times receive independent GPU
execution orderings. There is no mechanism producing a positive `Cov_i`, and the
cache contains no design that could estimate one (arms were never executed in a
paired-realization structure).

**Therefore `Cov_i = 0` in the primary formula.** We do **not** claim CRN
variance reduction — consistent with the earlier finding that CRN produced no
measurable variance reduction (CRN-off SD sat at the 35th percentile of CRN-on).
Any true positive covariance would make our resolution estimate *conservative*.

## 3. The design rule

Requiring a two-sided (1−α) half-width ≤ δ:

```
z_{1−α/2} · sqrt( 2w̄ / (N·R) ) ≤ δ
  ⇒  R_required(δ, α) = ceil( 2 w̄ z²_{1−α/2} / (N δ²) )
```

Heterogeneous form (retain when R differs by arm):
`R ≥ z² Σ_i(σ²_Ai+σ²_Bi) / (N²δ²)`.

This is **standard stratified/repeated-measures design**, claimed as a design
rule, **not** as new statistics.

## 4. ICC = 0.285 does **NOT** survive scrutiny — WITHDRAWN

The earlier figure used a plug-in `Var(p̂_i)` that is **inflated by sampling
noise**: `E[Var(p̂_i)] = Var(p_i) + E[p_i(1−p_i)]/R`. Correcting this, and
comparing principled alternatives on the same R=8 bank:

| estimator | value |
|---|--:|
| naive moment ratio (**what we reported**) | **0.285** |
| bias-corrected moment | 0.215 |
| beta-binomial (Fleiss–Cuzick) | **0.181** |
| one-way ANOVA on 0/1 data | **0.183** |

The three principled estimators cluster at **0.18–0.22**; 0.285 was an artifact.
**"71.5% of variance lies below the trial unit" is withdrawn** (the corrected
share is 78.5%/21.5%, and even that is estimator-dependent).

### 5. Replacement wording — estimator-free

More importantly, **the variance share is not the operative quantity for a fixed
benchmark**, since `Var(μ_i)` cancels from `Var[Δ̂]` entirely. The honest,
directly interpretable statements are:

- **39 of 96 scenarios are non-degenerate** (0 < p_i < 1) for a fixed policy,
  mean p_i = 0.724 among them; **0/96 fail in all realizations**
- a single realization of the whole benchmark has **SE = 2.73 pp**
- a two-arm contrast at R=1 has a **95% half-width of 7.55 pp**

These require no ICC definition and no Gaussian-on-Bernoulli assumption.

## 6. Analytical vs bootstrap — they disagree, and the tool uses the safer one

| contrast (R=3) | analytical | hierarchical bootstrap | ratio |
|---|--:|--:|--:|
| 3c s42 NFE4−NFE1 | 5.32 pp | 6.94 pp | 0.77 |
| 3c s43 NFE4−NFE1 | 5.57 pp | 8.16 pp | 0.68 |
| 3c s44 NFE4−NFE1 | 4.51 pp | 6.94 pp | 0.65 |
| 3c s42 NFE4−NFE2 | 4.72 pp | 7.47 pp | 0.63 |

The analytical form is **~35% narrower** because it conditions on the fixed
scenario set, whereas the bootstrap **also resamples scenarios** — i.e. they
target different estimands (fixed-benchmark vs population). Both are correct for
their own estimand.

**Consequence:** the paper presents the closed form for intuition and design, but
any reported interval must state which estimand it targets. For a frozen
benchmark the analytical (conditional) interval is the right one; a reviewer
generalising beyond the scenario set needs the bootstrap.
