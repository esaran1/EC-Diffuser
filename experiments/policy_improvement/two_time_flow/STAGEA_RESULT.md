# Stage A Falsification — **the phenomenon does not survive. NO-GO.**

GPU: ~0.20 GPU-h (Stage-1 rerun 0.07 + Stage A 0.13) of the 1.40 approved.
**Stages B and C were not run — Stage A killed the phenomenon, so I stopped.**
320 samples/cell (vs 48 at the gate).

---

## Verdict against the §15 NO-GO criteria

| criterion | status |
|---|---|
| **A. U-shape already present pre-FT AND caused by conditioning interpolation** | **BOTH HALVES MET** |
| **B. degradation only at exact `t_s=1`** | **not met** — degradation is smooth and starts by `t_s≈0.7` |
| **C. effect disappears with larger sample** | **effectively met** — the post-FT interior advantage collapses to +0.0%/+1.7% |
| D. only in vector error, not reconstruction | not separately decisive |
| E. path choice has no consequence | not tested (Stage B not run) |

**A and C together are disqualifying.**

## 1. The interior minimum is NOT learned — pre-FT it is exactly the diagonal

**PRE-FT** (retrofit at init, no off-diagonal training), action velocity error:

| t_s \ t_a | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|--:|--:|--:|--:|--:|
| **0.00** | **0.0257** | 0.1293 | 0.2795 | 0.4799 | 0.9083 |
| **0.25** | 0.1250 | **0.0317** | 0.1733 | 0.3939 | 0.9133 |
| **0.50** | 0.2775 | 0.1696 | **0.0443** | 0.2728 | 0.9206 |
| **0.75** | 0.5250 | 0.4249 | 0.2737 | **0.0809** | 0.9079 |
| **1.00** | 1.4345 | 1.0105 | 0.6613 | 0.5647 | 0.7949 |

**The minimum of every row lies exactly on the diagonal `t_s = t_a`.** That is the
bit-exact-equivalence corridor — the only region where the retrofit inherits
canonical behavior. Everything off it is out-of-distribution and degrades
sharply. There is no semantic "partially clean state helps" structure at init;
there is only "stay on the corridor the model was trained on."

## 2. After fine-tuning, the "interior optimum" is essentially the diagonal

| t_a | diagonal cell | best cell | best beats diagonal by |
|--:|--:|--:|--:|
| 0.00 | 0.0469 | 0.0414 (t_s=0.25) | **+11.8%** |
| 0.25 | 0.0459 | 0.0451 (t_s=0.50) | **+1.7%** |
| 0.50 | 0.0584 | 0.0584 (t_s=0.50) | **+0.0%** |
| 0.75 | 0.0834 | 0.0834 (t_s=0.75) | **+0.0%** |

In **2 of 4** non-degenerate rows the best cell **is** the diagonal. In the other
two the margin is +1.7% and +11.8%. The gate's headline — *"at `t_a=0.50` the
best state context is `t_s=0.50` (0.0964) vs `t_s=1.00` (0.4490), 4.7× better"* —
survives only as a comparison **against the endpoint**, and §7 below shows the
endpoint is an embedding artifact. Against the **diagonal**, the advantage is
**zero**.

**What fine-tuning actually did** was flatten the off-diagonal penalty (7.39× at
`t_a=0`, 4.72× at 0.25) while **degrading every diagonal cell** (+82.7%, +44.8%,
+31.7%, +3.1%). It bought off-diagonal robustness by giving up diagonal accuracy —
not by discovering an interior optimum.

## 3. Boundary strip — rules out the pure-endpoint artifact, but not the effect

Degradation begins **well before** `t_s = 1` (320 samples, 95% bootstrap CI):

| t_state | 0.50 | 0.70 | 0.90 | 0.95 | 0.99 | 1.00 |
|---|--:|--:|--:|--:|--:|--:|
| err (t_a=0.25) | 0.0451 | 0.0547 | 0.0813 | 0.1017 | 0.1569 | 0.2524 |
| vs min | 0.0% | +21.4% | +80.3% | +125.7% | +248.0% | **+460.1%** |

So §15-B is **not** met: this is not an exact-endpoint pathology. But that does
not rescue the claim, because §7 shows the *smooth* degradation also tracks the
embedding geometry.

## 4. §7 conditioning geometry — the actual explanation

`|e_state| = |time_mlp(t_s · 1000)|` is **non-monotonic in `t_s`**:

| t_s | 0.00 | 0.25 | 0.50 | 0.75 | 0.90 | 0.95 | 0.99 | 1.00 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| \|e_state\| | 5.088 | 3.931 | 2.928 | **2.152** | 2.216 | 2.629 | 3.957 | **8.808** |

It falls to a minimum at `t_s ≈ 0.75`, then rises, jumping **~4×** at the exact
endpoint. Cosine of the combined AdaLN vector to the diagonal conditioning
collapses from ~0.96 to **0.16–0.25** at `t_s = 1`.

**Action error tracks this almost perfectly:**

| t_action | corr(err, cos-to-diagonal) | corr(err, \|e_state\|) |
|--:|--:|--:|
| 0.25 | **−0.995** | +0.922 |
| 0.50 | **−0.999** | +0.949 |
| 0.75 | **−1.000** | +0.973 |

Correlations of −0.995 to −1.000 mean the error surface is explained by **how far
the conditioning vector has drifted from the pretrained diagonal manifold**, not
by the semantic cleanliness of the state plan. `time_mlp` was pretrained only on
diagonal inputs; feeding it `t_s ≠ t_a` produces conditioning it has never seen,
and the damage scales with the geometric distance.

**This is §15-A: the U-shape is a property of the retrofit's conditioning
interpolation.**

## 5. Corrected claims

**Withdrawn:**
- "Partially denoised state context is optimal for action inference."
- "Interior optimum in 5/5 rows" → the count was right but the fifth row is
  degenerate; the honest figure was 4/4, and now **even that is withdrawn**
  because the post-FT optimum coincides with the diagonal.
- "At `t_a=0.5`, fully clean state is 4.7× worse" → true as measured, but the
  `t_s=1` cell is an embedding artifact, so it does not support a semantic claim.
- The matched-cleanliness asymmetry (already retracted at the gate for
  own-block confounding) stays retracted.

**What survives, and it is modest:**

> A zero-parameter two-time retrofit of a pretrained flow policy is bit-exact on
> the diagonal but sharply out-of-distribution off it, because the pretrained
> time embedding was never exposed to `t_state ≠ t_action`. 1,000 steps of
> off-diagonal fine-tuning flattens that penalty by 1.5–7.4× at the cost of
> 3–83% diagonal degradation. Off-diagonal action error is almost perfectly
> predicted (|r| = 0.995–1.000) by the cosine distance of the combined time
> conditioning from its diagonal value.

That is a **methodological warning about warm-starting two-time models**, not a
finding about inverse/world-model structure.

## 6. Why this does not reach a paper

UWM already owns the endpoint semantics. Our differentiator was the interior, and
the interior structure we measured is an artifact of our own retrofit's
conditioning geometry. Against the diagonal — the correct baseline — the interior
advantage is **+0.0%** in half the rows.

A model **trained from scratch** on the 2D domain (rather than warm-started)
would not inherit this artifact and could still show genuine interior structure.
But that is exactly the from-scratch training Phase 3 showed we cannot afford to
do informatively at this budget: 20k steps produced a policy scoring 0/96.

## 7. Recommendation: **NO-GO for the remaining program**

Stop the two-time branch. Do not spend the remaining ~1.20 GPU-h on Stages B/C.

The path test (Stage B) would inherit the same artifact: paths that stray further
from the diagonal pass through worse-conditioned regions, so any path ordering
would likely measure embedding drift rather than semantics. Closed-loop control
(Stage C) would compound it.

**If this direction is ever revisited**, the prerequisite is a time-conditioning
scheme whose geometry is well-behaved off-diagonal — e.g. a separately
parameterized two-time embedding trained from the start — plus a training budget
at which the baseline is functional. Both were out of scope here.
