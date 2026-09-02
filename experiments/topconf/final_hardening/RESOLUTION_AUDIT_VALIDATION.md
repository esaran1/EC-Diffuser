# Resolution audit — empirical validation (CPU only; E1 untouched)

All calibration is **held out**: σ is estimated from same-policy repeats that
exclude the comparison being predicted.

## 1. Pilot estimation uncertainty (§4)

Reference within-scenario variance (R=8, bias-corrected): **w = 0.08147**.

| R0 | mean ŵ | bias | sd | 2.5% | 97.5% |
|--:|--:|--:|--:|--:|--:|
| 2 | 0.08157 | +0.00009 | 0.01808 | 0.0469 | 0.1146 |
| 3 | 0.08158 | +0.00011 | 0.01118 | 0.0556 | 0.0972 |
| 4 | 0.08148 | +0.00000 | 0.00793 | 0.0642 | 0.0946 |
| 5 | 0.08130 | −0.00017 | 0.00607 | 0.0677 | 0.0917 |

**Essentially unbiased at every R0** (|bias| ≤ 2e-4), but noisy at small R0.

## 2. Under-recommendation — the failure mode that matters (§5, §11)

"Under-recommendation" = the pilot says R is sufficient when the reference says
it is not.

| target | R0 | true R | point estimate | **UCB (conservative)** |
|--:|--:|--:|--:|--:|
| 3 pp | 2 | 8 | **57.2%** | 13.9% |
| 3 pp | 3 | 8 | 39.0% | 10.4% |
| 3 pp | 5 | 8 | 27.7% | 6.2% |
| **5 pp** | 2 | 3 | 13.9% | **0.0%** |
| **5 pp** | 3 | 3 | 3.8% | **0.0%** |
| 10 pp | 2 | 1 | 0.0% | 0.0% |

**The point-estimate rule is unsafe for tight targets** (57% under-recommendation
at 3 pp with R0=2). The conservative variant — plug the **upper one-sided 95%
bound** on ŵ into the design rule — eliminates under-recommendation at the
practical 5 pp target and cuts it to ~10-14% at 3 pp.

**Decision: the tool defaults to the conservative (UCB) rule.**

## 3. Leave-one-checkpoint-out (§7) — reported per seed, not pooled

σ from the *other two* checkpoints in the same task regime; compared to the
observed SD of the 9 single-realization deltas.

| task | held-out seed | predicted SE(R=1) | observed SD | ratio |
|---|--:|--:|--:|--:|
| 3-cube | 42 | 4.58 | 3.17 | 1.44 |
| 3-cube | 43 | 4.31 | 3.52 | 1.22 |
| 3-cube | 44 | 4.76 | 2.80 | 1.70 |
| 4-cube | 42 | 6.04 | 4.83 | 1.25 |
| 4-cube | 43 | 5.83 | 3.76 | 1.55 |
| 4-cube | 44 | 6.25 | 4.89 | 1.28 |

**Every ratio > 1: the predictor is systematically conservative** — mean relative
error **+47.7%**, median +29.2%, worst +121.7%. It never under-predicted the
observed spread on any of the 12 comparisons. Safe, but imprecise; the paper must
say "conservative", not "accurate".

## 4. Cross-task transfer (§8, §9) — **asymmetric**

| direction | calibrated w | predicted SE | observed SD | ratio |
|---|--:|--:|--:|--:|
| **3-cube → 4-cube** | 0.09954 | 4.55 pp | 4.49 pp | **1.01** |
| 4-cube → 3-cube | 0.17535 | 6.04 pp | 3.17 pp | **1.91** |

3→4 transfer is near-exact; 4→3 over-predicts ~2×, because 4-cube is intrinsically
noisier (its scenarios sit further from the success ceiling). **Do not claim
symmetric cross-task calibration.** The safe statement: *calibration transfers
from an easier to a harder regime conservatively, and from harder to easier very
conservatively.*

## 5. Sign reliability vs held-out signal-to-resolution ratio (§9)

x = |Δ_reference| / predicted SE(Δ) at R=1, **SE from other checkpoints only**.

| ratio bin | comparisons | sign correct | practical-category correct |
|---|--:|--:|--:|
| [0.00, 0.25) | 1 | **44.4%** (4/9) | 77.8% |
| [0.25, 0.50) | 3 | 70.4% (19/27) | 77.8% |
| [0.50, 1.00) | 4 | 75.0% (27/36) | 72.2% |
| [1.00, ∞) | 4 | **97.2%** (35/36) | 75.0% |

**Sign reliability is monotone in the held-out ratio** — 44% (coin-flip) to 97%.

**Hierarchy:** 12 calibrated comparisons, 9 nested views each. Bins contain 1-6
comparisons. These are **descriptive rates, not inferential estimates**, and no
interval is computed over the 108 nested views.

## 6. Practical-category reliability — a NEGATIVE result (§10)

Category accuracy is **flat at 72-78% across every ratio bin** and does **not**
improve with signal-to-resolution. Mechanism: the ±5 pp categories create
boundaries that a *large* true effect can still cross under noise, so a bigger
effect does not protect the category the way it protects the sign.

**Consequence:** the calibration predicts *sign* reliability, not *practical
conclusion* reliability. This must be stated — it limits the procedure's claim.

## 7. Frozen Resolution Audit algorithm (§14)

```
INPUT: frozen scenarios (N), one representative policy, pilot repeats R0,
       target effect δ (pp), confidence 1−α
1. run the policy R0 times on all N scenarios (no policy comparison involved)
2. ŵ = (1/N) Σ_i p̂_i(1−p̂_i) · R0/(R0−1)          [bias-corrected]
3. bootstrap scenarios → upper one-sided (1−α) bound ŵ_UCB
4. half-width(R) = z_{1−α/2} · sqrt(2·ŵ_UCB/(N·R))
5. R* = min{ R : half-width(R) ≤ δ }               [conservative by construction]
6. run the policy comparison at R = R*; report the fixed-benchmark Δ̂ with its
   conditional interval, and state the estimand
WARN if R0 < 3, or if the ŵ bootstrap CI spans > 2× its point estimate.
NO adaptive allocation — measured worse than uniform.
```

## 8. Strongest defensible claim (§15)

> A same-policy calibration, performed without using the treatment comparison,
> predicts on held-out checkpoints and across task regimes which effect sizes the
> evaluator can reliably **sign**: reliability rises from ~44% at a
> signal-to-resolution ratio below 0.25 to ~97% above 1.0. The predictor is
> systematically **conservative** (mean +48% on the half-width), which is the
> safe direction for a design rule.

**Not** claimed: accurate half-width prediction; symmetric cross-task transfer;
prediction of practical-category reliability; novel statistics.

## 9. Biggest remaining limitation

The deep repeated-realization bank exists for **one arm** (seed 42, NFE4, R=8).
Every other arm has R=3, so pilot-uncertainty quantification rests on a single
policy, and the reliability curve's bins hold 1-6 comparisons. A second deep bank
on a different checkpoint would materially strengthen this — but it is a GPU
experiment and is **not** proposed now.

---

## ADDENDUM (figure package) — tie-handling correction

Building the paper figures exposed a defect in the sign-agreement counts in §5
above. `np.sign(0) == 0`, so a nested view with **exactly** Δ = 0 never matched
the reference sign and was silently counted as a *disagreement*. With 96 binary
scenarios, exact ties are common: **7 of 108 nested views**.

A tie is not a sign reversal. Corrected counts:

| quantity | as published above | corrected |
|---|--:|--:|
| sign disagreement, pooled | 23/108 = 21.3% | **16/108 = 14.8%** (strict reversals) |
| ...ties excluded from denominator | — | 15.8% |
| §5 bin [0, 0.25) sign correct | 44.4% (4/9) | **66.7% (6/9)** |
| §5 bin [0.25, 0.5) | 70.4% | **77.8%** |
| §5 bin [0.5, 1) | 75.0% | **83.3%** |
| §5 bin [1, ∞) | 97.2% | 97.2% (unchanged) |

**What survives:** sign reliability is still monotone in the held-out
signal-to-resolution ratio, and the [1, ∞) bin is unchanged. The qualitative
conclusion of §5 stands.

**What changes:** the low bin is no longer at chance. The claim "single-realization
sign is a coin flip for weak effects" is **withdrawn** — 66.7%, not 44.4%.
The practical-category negative result (§6) is unaffected: still flat, 72–78%.

Superseded headline numbers, in order: 24% → 21.3% → **14.8%**.
