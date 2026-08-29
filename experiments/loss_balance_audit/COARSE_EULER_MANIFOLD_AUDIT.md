# Why does coarse Euler predict better than the converged Flow ODE?

**No training. No loss change. No solver change. No new policy.**
Compute: ~0.07 GPU-h (223 s endpoint extraction + 80 s spectrum), rest CPU (58 s).

Scripts: `extract_endpoints.py`, `manifold_audit.py`, `plot_manifold_audit.py`
Data: `cached_endpoints.npz`, `manifold_audit.json`
Figure: `experiments/figures/manifold_audit.png`

**Answer: H1 is NOT supported. Outcome B — task-specific solver bias.**

---

## 0. A reproducibility problem found first, and how it was handled

Only per-sample **chamfer scalars** were cached by the previous studies, never the
latent endpoints, so the endpoints had to be regenerated (permitted by §3 for
genuinely absent artifacts). Replaying the identical protocol — same frozen
episode set, noise seed 777, same env advance — did **not** reproduce the earlier
per-sample values: max |Δchamfer| ≈ 0.04–0.056, present already at rollout step 0.

**Isaac Gym's DLP observations are not bit-reproducible across processes.** Sample
*means* agree closely (e.g. s42 euler16: 0.04143 now vs 0.04059 before), so no
previous conclusion is affected, but **cross-run per-sample pairing is invalid**.

Consequence: every paired quantity below is computed **within this single run**,
where all arms provably share one x0 and one environment trajectory. The
within-run replication of the headline effect is itself a useful check:

| seed | E16 | E512 | paired Δ |
|---|--:|--:|--:|
| 42 | 0.04143 | 0.04326 | +0.00183 |
| 43 | 0.03922 | 0.04071 | +0.00149 |
| 44 | 0.04239 | 0.04391 | +0.00152 |
| **pooled** | **0.04101** | **0.04263** | **+0.00161 [+0.00107, +0.00215]** |

**The E512 > E16 degradation replicates in a fresh run, all three seeds** (prior
estimate +0.00210; both CIs exclude zero).

## 1. Real-state reference dataset

| property | value |
|---|---|
| source | `ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl` |
| full shape | (2000, 100, 48, 10) |
| **reference split** | episodes **0–1499**, 4,000 states sampled |
| **held-out split** | episodes **1500–1999**, 288 states sampled (**disjoint**) |
| sampling | uniform without replacement over (episode, timestep) |
| rng seed | 20260829 |
| sha256(observations) | recorded in `manifold_audit.json` |

Held-out real queries are **never** in the reference database, as required.

**View correction:** the cached endpoints are **view 0 only (24 particles)**,
because the arms extract `[:, horizon=1, view=0]`. My first run built a
48-particle (two-view) reference; matching a 24-particle query against it would
bias every distance. Fixed — reference, held-out set and Mahalanobis all use
view 0.

## 2. Manifold metric definition (primary)

**Permutation-invariant symmetric Chamfer on the particle POSITION block**
(`latent_metric.POS = dims 0:2`), the already-validated metric:

```
d(q, x) = 0.5 * [ mean_i min_j ||q_i - x_j||  +  mean_j min_i ||q_i - x_j|| ]
manifold_distance_k(q) = mean of the k smallest d(q, x) over the 4000 reference states
```

**k = 1 and k = 5, both predeclared before looking at results** and both reported
throughout. No k was chosen post hoc.

## 3. Flattened Mahalanobis: **REJECTED**

Tested as instructed before any use:

| test | value |
|---|--:|
| condition number | 2.13e5 |
| matrix rank | 240 / 240 |
| **effective rank (entropy)** | **112.2 of 240** |
| **mean relative score change under particle permutation** | **8.44%** |
| mean absolute change | 1.271 (on a base score of 15.28) |

Permuting particles **within a view** — a semantically null operation — changes
the Mahalanobis score by ~8.4%. The covariance is also ill-conditioned with
effective rank less than half its dimension.

**Verdict: INVALID. Flattened Mahalanobis is not used for any scientific claim.**

## 4-7. Calibration table (all values recomputed, within-run)

| Source | Ground-truth error | Manifold dist (k=1) | Manifold dist (k=5) |
|---|--:|--:|--:|
| **Held-out real state** | **0.0** | **0.05779** | **0.06425** |
| Copy-current | 0.07563 | **0.05946** | **0.06622** |
| Gaussian@100 | 0.04109 | 0.06215 | 0.06785 |
| Flow Euler@8 | 0.04428 | 0.06352 | 0.06939 |
| **Flow Euler@16** | **0.04101** | 0.06347 | 0.06945 |
| **Flow Euler@512** | **0.04263** | 0.06368 | 0.07002 |

Per-seed (k=5): E16 = 0.06933 / 0.07032 / 0.06869; E512 = 0.06978 / 0.07084 /
0.06944 (seeds 42/43/44).

**Held-out real states sit at 0.06425, not zero** — exactly the calibration the
protocol demanded. All generated arms lie 0.004–0.006 above that baseline, i.e.
in a narrow band ~7–9% worse than genuine unseen data.

**The decisive observation is copy-current.** It has the **best manifold distance
of any non-real source** (0.06622, closest to the real baseline) and by far the
**worst prediction** (0.07563). Manifold proximity and prediction quality are
therefore **not** monotonically related — a source can be maximally realistic and
maximally wrong.

## 8-9. Paired E512 − E16 (positive = converged endpoint farther from support)

| k | pooled Δ manifold | 95% CI | Δ gt error | Spearman ρ | p |
|--:|--:|---|--:|--:|--:|
| 1 | **+0.00022** | [−0.00028, +0.00070] **spans 0** | +0.00161 | +0.148 | 0.012 |
| 5 | **+0.00057** | [+0.00025, +0.00090] | +0.00161 | +0.192 | 0.0011 |

Per seed (k=5): +0.00046 / +0.00051 / +0.00075 — consistent in sign.
Per seed (k=1): −0.00015 / −0.00006 / +0.00085 — **two of three negative**.

So the manifold effect is **k-dependent and an order of magnitude smaller than
the prediction effect**: Δ manifold = +0.00057 against Δ gt = +0.00161, and at
k=1 it is not distinguishable from zero.

## 10. Does manifold drift explain the degradation?

**Spearman ρ = +0.148 (k=1) and +0.192 (k=5).** Both positive and nominally
significant, but ρ ≈ 0.19 corresponds to **~3.7% of variance explained**. Samples
that move away from data support are only very weakly more likely to become worse
predictions.

**A correlation this weak cannot carry the implicit-regularization story.** No
causal claim is made in either direction.

## 11. Robust range violations (0.5th–99.5th percentile, per block)

| source | pos | scale | depth | vis | transp |
|---|--:|--:|--:|--:|--:|
| **held-out real** | **0.0104** | **0.0100** | **0.0116** | **0.0094** | **0.0097** |
| Gaussian@100 | 0.0115 | 0.0126 | 0.0126 | 0.0066 | 0.0048 |
| Euler@16 (3-seed) | 0.0084 | 0.0064 | 0.0054 | 0.0026 | 0.0020 |
| Euler@512 (3-seed) | 0.0103 | 0.0127 | 0.0084 | 0.0033 | 0.0058 |

E512 violates more often than E16 on **all five blocks** (clearest in `scale`:
0.0127 vs 0.0064), directionally agreeing with the k=5 result.

But the calibration inverts the reading: **held-out real data violates these
ranges ~1% of the time**, and **both** Flow arms are at or *below* that rate on
every block. Neither endpoint is out-of-distribution in absolute terms; E512 is
simply *closer to how often real data does it*. Reported as the weak secondary
diagnostic it is.

## 12. Temporal-spectrum diagnostic (optional, performed — one batch, ~80 s)

Dense `v(x,t)` at **fixed x**, 512 samples over t ∈ [0,1], seed 42.
‖v‖ ranges 40.1 → 178.9. Power of the fluctuation by temporal frequency:

| band (cycles over [0,1]) | share of power |
|---|--:|
| 0–2 | **66.6%** |
| 2–8 | 19.8% |
| 8–32 | 9.4% |
| 32–128 | 3.3% |
| 128–256 | 0.9% |

95% of power below **26 cycles** (21 cycles across all velocity components).
A dt = 1/16 step resolves ~8 cycles (Nyquist).

**Useful refinement of the previous finding: the model uses only a small fraction
of the embedding's representable bandwidth.** `time_scale = 1000` could express
~1000 cycles; the trained field concentrates **86% of its temporal power below 8
cycles**, which Euler@16 already resolves. The unresolved content is the last
~14%. Consistent with §14's mandated wording — practical integrators
under-resolve *some* real rapid time dependence — while showing the model is far
from using the full high-frequency capacity. **No claim that time_scale=1000 is
wrong.**

## 13. Classification: **B — TASK-SPECIFIC SOLVER BIAS**

Evidence against A (manifold regularization):

1. The manifold gap between E16 and E512 is **+0.00057 (k=5)**, and **not
   distinguishable from zero at k=1** (CI spans 0; 2 of 3 seeds negative) — while
   the prediction gap is +0.00161 with every CI excluding zero. The candidate
   explanation is smaller and less robust than the thing it must explain.
2. **Correlation is weak: ρ ≈ 0.15–0.19, ~4% of variance.**
3. **Copy-current decisively breaks the premise**: it is the most manifold-realistic
   non-real source (0.06622 vs held-out real 0.06425) and the worst predictor
   (0.07563). Being near the data manifold does not make a prediction good.
4. All arms sit in a narrow band 0.004–0.006 above the held-out-real baseline;
   E16 and E512 differ by ~10% of that already-small offset.
5. Range violations of both Flow arms are *below* the held-out-real rate, so
   neither is meaningfully off-distribution.

Evidence for B: E16 is reliably closer to the **specific conditional future**
(+0.00161, 3/3 seeds) without being correspondingly closer to the **general data
distribution**. That is precisely the B pattern — a useful bias toward the
conditional target, not generic realism.

Not C (E512 is not *more* on-manifold — it is slightly less, at k=5).
Not D (the prediction effect is robust and replicated; it is the *manifold*
explanation that fails, not the effect).

## 14. Best mechanistic interpretation

Coarse Euler's truncation error is not generic smoothing toward the data
manifold. If it were, we would see a manifold gap at least as large as the
prediction gap and a strong correlation between them; we see neither, and
copy-current falsifies the premise outright.

The evidence instead favours: **under-resolved Euler lands nearer the particular
conditional future the task is scored against, while being no more "realistic" in
distribution than the converged endpoint.** Where the two endpoints differ
(§11, §12), the converged one is if anything *more* statistically real.

Stated conservatively: the mechanism is a **conditional-target bias of the
discrete solver**, and it is **not explained by data-manifold proximity**. Why
that bias points in a helpful direction is not established here, and I am not
speculating further on this evidence.

## 15. Previous claims that remain valid

All nine accepted conclusions stand. Specifically re-verified in this audit:

- **Euler@16 predicts the real future better than converged Euler@512** —
  independently replicated in a fresh run, +0.00161 [+0.00107, +0.00215], 3/3
  seeds (accepted conclusion 7).
- The rapid-time-variation finding (conclusion 4), now **quantified**: 86% of
  temporal power below 8 cycles, 95% below 26.
- Conclusions 1, 2, 3, 5, 6, 8, 9 are untouched by this analysis.

**New limitation to record:** Isaac Gym DLP observations are not bit-reproducible
across processes (max |Δchamfer| ≈ 0.05). Sample means are stable; per-sample
cross-run pairing is not. Future paired analyses must be computed within one run.

## 16. Loss-hypothesis status: **LOWER PRIORITY** (unchanged)

Nothing here motivates a loss change, and outcome B does not implicate the
objective. Combined with the small converged residual whose CI overlaps Gaussian,
there is no evidence that would make loss modification the right next move.
Per §17, no loss change is made regardless of result.

## 17. Exactly ONE next experiment

**Test whether the E16 advantage is a genuine predictive bias or an artifact of
scoring a stochastic generator by a single sample, by comparing per-sample
variance and multi-sample aggregates from the already-cached machinery.**

Concretely: for one Flow seed and the same frozen 96 examples, draw **8 different
initial noises x0 per example**, generate at Euler@16 and Euler@512, and compare
(a) the mean chamfer per example, (b) the **best-of-8**, and (c) the spread across
noises. No training, no new policy, ~0.1 GPU-h.

Rationale: outcome B says coarse Euler is biased toward the conditional target.
A concrete and cheap alternative is that E512 endpoints are more *diverse* — a
sharper conditional distribution sampled more faithfully — so any single draw
lands farther from one particular future while the distribution is better. That
would look exactly like the observed effect while meaning the opposite. Comparing
mean-vs-best-of-8 and the noise-induced spread separates "biased toward the
target" from "more diverse around the target", and it must be resolved before any
claim about what coarse integration is doing.

---

## HARD STOP OBSERVED

No training. No loss change. No MeanFlow. No VP. No solver change. No new policy.
