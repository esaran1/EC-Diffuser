# Simulator resolution audit — feasibility (CPU only, E1 untouched)

Verdict: **KEEP the calibration/resolution result. DROP adaptive allocation.**

---

## 1. My earlier STEP/N-SCORE claim was WRONG — corrected

I previously wrote that ICC < 1 "violates the i.i.d. assumption" of STEP and
N-SCORE. **That is incorrect.** From STEP (arXiv 2503.10966), Assumption 1
verbatim:

> *"In each evaluation trial, the initial state s0 and the observation o0 are
> drawn in an independent and identically distributed (i.i.d.) fashion from the
> underlying distribution D_{s0,o0}. We assume access to samples from D_{s0,o0},
> but do not assume D_{s0,o0} itself to be known."*
>
> *"the nth evaluation trial involves making an i.i.d. draw of an environment
> from D_{s0,o0} and running the policy π1 … yields a binary success/failure
> outcome z1,n corresponding to an i.i.d. draw from a Bernoulli random variable
> with mean p1."*

The i.i.d. requirement is on the **scenario draw**, and `z ~ Ber(p)` marginalises
over *both* the scenario and any execution randomness. **Simulator
nondeterminism is simply absorbed into the outcome distribution.** Their
guarantee does **not** fail on our data.

### The correct conceptual relationship

The mismatch is one of **estimand**, not violated assumptions:

| | STEP / N-SCORE | ours |
|---|---|---|
| estimand | population success rate `p` over `D_{s0,o0}` | fixed-benchmark `J(π) = (1/N) Σ_i E_r[Y_ir]` |
| trial | fresh i.i.d. scenario draw | one realization **within** a fixed scenario |
| variance | `p(1−p)/n` | `(1/N²)Σ_i Var(p̂_i)` |
| between-scenario variance | **enters** | **cancels** |

Both are legitimate. But **benchmark practice fixes the scenario set** (frozen
episode lists, published seeds, leaderboards), which is the fixed-benchmark
estimand — and under it, the *only* remaining variance is the within-scenario
simulator term that single-realization evaluation sets to R=1 by default and
never reports. STEP/N-SCORE optimise *when to stop drawing scenarios*; they do
not address *how many realizations a fixed scenario needs*. Those are
complementary, not competing.

Corrected wording for the paper: **not** "their assumption is violated", but
"their formulation targets a population estimand and does not optimise over the
inner realization distribution that a fixed benchmark induces."

## 2. Fixed-benchmark estimand and its variance (derived)

`Var[Ĵ] = (1/N²) Σ_i p_i(1−p_i)/R = within/(N·R)`, and for a two-arm contrast
with independent arms `Var[Δ̂] ≈ 2·within/(N·R)`.

Measured (`within` = 0.07129, N = 96): SE(Δ) = **3.85 / 2.23 / 1.72 / 1.36 pp**
at R = 1/3/5/8 → 95% half-widths **7.55 / 4.36 / 3.38 / 2.67 pp**.

**These match the empirically measured resolution curve exactly**, so the theory
and the measurements corroborate each other.

## 3. The positive result: same-arm calibration predicts held-out reliability

**Strict holdout**: for each comparison, σ is predicted *only* from same-policy
repeats of **other checkpoints** in the same task regime. The comparison being
tested never informs its own noise estimate.

| held-out predicted \|Δ\|/σ | comparisons | sign correct |
|---|--:|--:|
| [0.0, 0.5) | 4 | **63.9%** |
| [0.5, 1.0) | 4 | 75.0% |
| [1.0, 1.5) | 2 | **94.4%** |
| ≥ 1.5 | 2 | **100%** |

Monotone, and it **transfers across checkpoints and across task regimes**
(3-cube ↔ 4-cube). This is the actionable finding: *a short same-policy pilot
tells you, in advance, which effect sizes your evaluator can resolve.*

## 4. The negative result: adaptive allocation FAILS — dropped

### Honest-holdout feasibility (§H)

The cache **cannot** support an honest adaptive-contrast evaluation: only one arm
(seed 42, NFE4) has a deep R=8 bank; contrast arms have R=3, which splits into a
degenerate pilot (a Bernoulli variance from 1-2 draws takes only values {0, 0.25}).

### Measured, in the most favourable single-arm case

| method | rollouts | RMSE(Ĵ) |
|---|--:|--:|
| uniform R=1 | 96 | 2.759 pp |
| uniform R=2 | 192 | 1.786 pp |
| **uniform R=3** | **288** | **1.308 pp** |
| two-stage R0=2 +1 | 277 | **3.211 pp** |
| two-stage R0=2 +2 | 286 | 3.264 pp |
| two-stage R0=3 +1 | 374 | 2.343 pp |

**Adaptive is ~2.5× worse than uniform at matched cost.** Against the predeclared
thresholds (STRONG ≥25% savings / PROMISING 10-25% / WEAK <10%) this is not merely
weak — it is **negative**.

### Why, mechanistically

`Var[Ĵ] = (1/N²)Σ_i v_i/R_i` with Neyman optimum `R_i ∝ √v_i`. But `v_i = 0` for
**57/96 scenarios** (p_i ∈ {0,1}) and 0.25 for the rest — a near-binary variance
profile. Even the **oracle** Neyman allocation, which knows `v_i` exactly and is
not deployable, saves only 44-54% of variance (≈25-30% RMSE), because zero-variance
scenarios still need ≥1 draw to be observed and they are the majority. A
realizable scheme must pay a pilot on *every* scenario to discover `v_i`, and
that pilot cost exceeds the gain.

Pilot variance does carry signal (corr with held-out variance 0.38 / 0.47 / 0.50
at R0 = 2/3/4) — it simply cannot pay for itself.

**Per §N this direction is killed. No contact-aware variant was pursued**, since
the ceiling argument applies regardless of how scenarios are selected.

## 5. Consequence for the paper

D2 stays an **empirical study with a calibration procedure**, not an empirical
study with an allocation algorithm. The defensible contribution:

> A short same-policy repeatability pilot measures the evaluator's resolution
> **in advance**, and that measurement predicts, on held-out comparisons and
> across task regimes, which policy differences the evaluator can reliably sign.
> Uniform replication is then the right way to spend the budget — we show that a
> plausible adaptive alternative is worse, and give the variance argument for why.

The negative result is worth keeping: it pre-empts the obvious reviewer
suggestion ("why not allocate adaptively?") with a measurement and a derivation.
