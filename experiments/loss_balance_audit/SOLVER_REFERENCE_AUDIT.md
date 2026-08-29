# Solver / reference audit: why RK4 "failed to converge"

**No training. No loss change. No model change.** Compute: **~0.09 GPU-h total**
(39 s audit + 253 s converged-reference + 75 s metric floor), inside the 0.5 h target.

Scripts: `solver_reference_audit.py`, `converged_reference_gt.py`,
`perfect_future_floor.py`, `plot_reference_audit.py`
Data: `solver_reference_audit.json`, `converged_reference_gt.json`,
`perfect_future_floor.json`
Figures: `experiments/figures/audit_raw_convergence.png`, `audit_rk_ood.png`

**Headline: the previous "does not converge" finding was a measurement artifact
of my own convergence norm. In raw free-coordinate space the solvers DO
converge.** The corrected picture changes two of the three flagged conclusions.

---

## 1. Raw-space convergence curves (Phase 2)

Measured in the **raw generated transition tensor**, on **unconditioned
coordinates only** — no DLP decode, no Hungarian, no Chamfer. Subset of 8
samples, seed 42, fixed x0. Mask sizes: 11,640 free coords (120 action /
11,520 observation), 7,680 conditioned and excluded.

Successive-refinement difference, mean |Δ| (free coords):

| NFE | euler | order | midpoint | order | rk4 | order |
|--:|--:|--:|--:|--:|--:|--:|
| 32 | 0.007925 | — | 0.008470 | — | 0.018208 | — |
| 64 | 0.004413 | 0.84 | 0.003017 | 1.49 | 0.007576 | 1.27 |
| 128 | 0.002966 | 0.57 | 0.001601 | 0.91 | 0.003103 | 1.29 |
| 256 | 0.001480 | 1.00 | 0.000792 | 1.02 | 0.001324 | 1.23 |
| 512 | 0.000722 | 1.04 | 0.000327 | 1.27 | 0.000595 | 1.15 |

**All three schemes converge, monotonically, toward zero — but all at order ~1,
including RK4 (expected 4) and midpoint (expected 2).**

Cross-scheme agreement at 512 NFE (the key test the earlier study got wrong):

| pair | mean_abs | rms | max_abs |
|---|--:|--:|--:|
| euler vs midpoint | 0.000676 | 0.001960 | 0.031501 |
| euler vs rk4 | 0.000893 | 0.002157 | 0.031550 |
| midpoint vs rk4 | 0.000441 | 0.000770 | 0.008903 |

Independent schemes agree to ~5e-4, and the agreement tightens with refinement.
**A converged endpoint exists.**

### Why the earlier study concluded otherwise

The previous audit measured `(a-b).flatten(1).norm(dim=1)` — a **full-tensor L2
norm including the 7,680 conditioned coordinates and summing 19,320 raw
elements**. That inflates the number, does not normalize by element count, and
mixes coordinate scales. Its "floor of ~0.17" was an artifact of that norm, not
a property of the ODE. **This is an error in the prior report's reference
analysis and it is withdrawn.**

## 2. Action vs observation coordinates

They converge together, with no separate pathology (right panel of
`audit_raw_convergence.png`). Euler@512: action 0.000710, observation 0.000722.
At every rung the two agree within ~10%. The 120 action coordinates are not a
special source of irregularity.

## 3. Exact conditioning behavior (Phase 3)

Canonical sampler (`flow_matching.py:358-373`):

```
x = randn(...);              apply_conditioning(x, cond)
cmask = make_conditioning_mask(x, cond)     # False on cond'd obs channels
for step in range(steps):
    t = step/steps
    v = model(x, cond, t * time_scale)
    v = v * cmask                            # velocity ZEROED on cond'd coords
    x = x + dt*v
    apply_conditioning(x, cond)              # cond'd coords rewritten
```

`_apply_conditioning` writes a **constant** (`x[:, ts, action_dim:] = value`) and
`_make_conditioning_mask` zeroes the velocity on exactly those coordinates.

My diagnostic solvers re-impose conditioning **after every stage state is formed
and before every network call**, matching the canonical discipline.

**This is NOT a projected dynamical system in any consequential sense.** Because
the velocity is masked to zero on conditioned coordinates, those coordinates
never move; re-imposing them is a no-op after the first application. Projection
is therefore idempotent and cannot introduce order reduction.

## 4. Two conditioning variants (Phase 4) — proven equivalent

- **A** = canonical projected behavior (write-back each stage)
- **B** = integrate only free coordinates, hold conditioned ones analytically fixed

Measured: **max |A − B| = 0.0 (exactly)**, and conditioned coordinates move by
**0.0** under A. The two formulations are bit-identical. **Option B of the
central question is ruled out.**

## 5. Time-variable audit (Phase 5) — this is the cause

- integration interval: exactly [0, 1]; `dt = 1/steps`
- t values are exact floats, e.g. euler@4 → `[0.0, 0.25, 0.5, 0.75]`;
  rk4@1 → `[0.0, 0.5, 0.5, 1.0]` (correct Butcher nodes)
- the model receives **scaled** time `t * time_scale`, `time_scale = 1000.0`
- `SinusoidalPosEmb` consumes the float directly: **no rounding, no
  quantization, no clipping** anywhere in the time path

Embedding sensitivity at t=0.5 confirms full continuity at tiny scales:

| δt | max |Δembedding| |
|---|--:|
| 1e-7 | 1.22e-4 |
| 1e-6 | 9.76e-4 |
| 1e-5 | 9.64e-3 |

Smooth and non-quantized. **But the frequency content is the problem:**

`projection_dim = 512` → `half_dim = 256`, frequencies from 1 down to 1e-4 **in
scaled time**. The fastest mode is `sin(1000·t)`, of period **Δt = 0.00628 in t**.

| setting | dt | phase advance of fastest mode |
|---|--:|--:|
| NFE 2 | 0.5 | 500 rad = **79.6 periods** |
| NFE 4 | 0.25 | 250 rad = 39.8 periods |
| NFE 16 | 0.0625 | 62.5 rad = **9.9 periods** |
| NFE 128 | 0.00781 | 7.81 rad = 1.24 periods |
| NFE 512 | 0.00195 | 1.95 rad = 0.31 periods |

**A high-order method's Taylor expansion is only valid when dt resolves the
field's variation. Here that needs NFE ≳ 512.** At every practical budget, RK4's
intermediate stages sample an oscillating field at effectively unrelated phases,
so the extra evaluations buy no order.

## 6. Network smoothness audit (Phase 6)

`AdaLNPINTDenoiser` / `pint.py` contains **5× GELU and nothing else nonsmooth**:

| construct | count | smooth? |
|---|--:|---|
| `nn.GELU` | 5 | yes, C^∞ |
| `LayerNorm` (AdaLN) | — | yes (away from zero variance) |
| `Dropout` | 1 | inactive under `.eval()` |
| ReLU / clamp / clip / threshold / discrete ops | **0** | — |

**The network is analytically smooth. "Rough field" is NOT supported by the
architecture**, and my earlier wording was wrong to imply it.

## 7. Local regularity (Phase 7)

Mid-trajectory, seed 42:

| perturbation | ‖Δv‖/‖δ‖ (state) | ‖Δv‖/|δt| (time) |
|---|--:|--:|
| 1e-5 | 2.125 | 93.74 |
| 1e-4 | 1.948 | 93.89 |
| 1e-3 | 1.943 | 99.30 |
| 1e-2 | 1.943 | 90.32 |

**Both are flat across four orders of magnitude — no discontinuity, no
quantization, at any scale probed.** The field is locally well-behaved.

The informative part is the **ratio**: sensitivity to time is ~**48×**
sensitivity to state. The dynamics are dominated by explicit time dependence,
exactly as the `time_scale = 1000` embedding predicts.

## 8. RK intermediate-state OOD excursion (Phase 8)

Distance of each network-evaluation state from the canonical Euler@16 trajectory:

| arm | stage | t | ‖x‖ | dist to path | ‖v‖ |
|---|---|--:|--:|--:|--:|
| euler@4 | k1 | 0.750 | 17.18 | 1.441 | 37.18 |
| midpoint@2 | k2 | 0.750 | 17.29 | 1.238 | 37.35 |
| heun@2 | k2 | 1.000 | 16.04 | **4.475** | **15.72** |
| rk4@1 | k3 | 0.500 | 24.35 | **3.561** | 39.82 |
| rk4@1 | k4 | 1.000 | 15.85 | **5.994** | **15.90** |

All methods share an identical k1 (‖x‖ = 39.790, ‖v‖ = 40.452), confirming a
common starting point.

**Yes — high-order methods obtain their "better" local estimates by evaluating
the network far off the trajectory.** RK4's k3/k4 are the most distant states of
any scheme, and at t=1 the velocity collapses to ~15.9 versus ~38-40 on-path.
Combined with §5, this is why higher order does not pay here.

## 9. Frozen-field solver-interface control (Phase 9) — the decisive test

Same solver code, a small **smooth GELU** field with the **identical
tensor/conditioning interface**, varying only `time_scale`:

| method | time_scale = 1 | time_scale = 1000 |
|---|--:|--:|
| euler | order **1.00, 1.00, 1.00** | order 1.73, −0.17, 3.01 (erratic) |
| midpoint | order **1.88** | order 1.46, 0.70, 0.98 |
| rk4 | error at float32 floor (~5e-8) | order 1.12, 1.65, 0.17 |

At `time_scale = 1` the solvers recover their **textbook orders** (Euler exactly
1.00; midpoint 1.88; RK4 already at float32 roundoff, which is why its apparent
"order" goes negative there). Switching **only** the time frequency to 1000
collapses every method to order ≈1 with errors 4-5 orders of magnitude larger —
**reproducing the exact pathology seen on the trained model**.

Plus the analytic ODE tests from the prior study (`dx/dt = x`): orders 0.96 /
1.97 / 1.97 / 3.81 for euler / midpoint / heun / rk4.

**The solver implementation is correct. The order loss is caused by the
high-frequency time embedding, not by the solver, the conditioning, or a rough
network.**

## 10. Classification: **C — trained field is numerically irregular *in time***

...but with an important refinement that separates it from the naive reading of C,
and it borders on **D/E** for the *previous* report specifically:

- **not E** (implementation bug): the frozen-field control recovers full order
  with the same code; conditioning variants are bit-identical; analytic ODE tests
  pass. The solvers are correct.
- **not B** (projected dynamics): projection is provably idempotent (max
  |A − B| = 0.0), so it cannot reduce order.
- **not the naive C**: the field is **not** non-smooth. Activations are C^∞ and
  local regularity is flat to 1e-5 in both state and time.
- **D applies to the earlier report**: the prior "non-convergence" was a
  **metric artifact** of an unnormalized full-tensor norm over conditioned
  coordinates. In raw free-coordinate space endpoints converge.

The accurate statement: **the learned field is smooth but varies on a time scale
(~0.0063) far shorter than any practical step size, so no method achieves its
formal order below ~512 NFE.** This is a genuine and important property of the
trained model — a consequence of `time_scale = 1000` in a flow-matching model on
t ∈ [0,1] — and it is why high-order solvers cannot help at deployable budgets.

## 11. Does a valid high-accuracy Flow reference exist?

**Yes.** Raw free-coordinate endpoints from three independent schemes agree to
mean_abs ~5e-4 at 512 NFE, with monotone refinement. **Euler@512 is a usable
converged reference**, and I now use it as such. (The earlier RK4@64 "reference"
at 256 NFE was not converged, and the numerical-error column built on it is
withdrawn.)

## 12. What remains valid from MATCHED_NFE_SOLVER_STUDY

**VALID (no solver bug was found; explicitly preserved per Phase 11):**

- **At matched practical NFE, canonical Euler has lower ground-truth imagination
  error than midpoint and Heun, on all three Flow seeds, at NFE 2/4/8.** Every
  paired CI excludes zero except one. This is the study's central empirical claim
  and it stands.
- The mechanistic account of Heun's collapse (full-step predictor at dt=1 →
  ‖v2−v1‖/‖v1‖ = 0.96), now *reinforced* by the OOD analysis in §8.
- Solver NFE accounting, Euler bit-identity (0.000e+00), determinism.
- The vector-field diagnostics table.
- **Practical recommendation unchanged: use Euler; do not adopt a higher-order
  solver.** The audit strengthens the reason — it is not merely empirical but
  follows from the time-frequency analysis in §5.

**WITHDRAWN:**

- "The Flow ODE is not integrable to machine precision." **False**, and caused by
  my convergence norm. Withdrawn entirely.
- "Independent high-order schemes at NFE 256-512 disagree by ~0.17 and refinement
  does not close it." Artifact of the same norm; the corrected figure is ~5e-4
  and shrinking.
- The "reference uncertainty floor" construction and the numerical-error column
  of the results table (they were computed against a non-converged reference).
- "Euler@16 has reached the Flow ODE's exact limit." Not established as stated —
  see §13, where the corrected measurement gives a different and more interesting
  result.

**SOFTENED:**

- "The residual Flow-vs-Gaussian gap is definitively model-level." Now measured
  against a converged reference and **substantially smaller** — see §13.
- "Rough field" wording → replaced by "smooth field that varies rapidly in time."

## 13. The residual, re-measured against a CONVERGED reference

Full 96-sample set, 3 seeds, environment advanced with the Euler@8 action:

| arm | s42 | s43 | s44 | 3-seed mean | sd | residual vs Gaussian |
|---|--:|--:|--:|--:|--:|--:|
| euler@8 | 0.04382 | 0.04257 | 0.04351 | 0.04330 | 0.00065 | +0.00326 |
| **euler@16** | 0.04059 | 0.03981 | 0.04068 | **0.04036** | 0.00047 | **+0.00032** |
| euler@128 | 0.04211 | 0.04124 | 0.04226 | 0.04187 | 0.00055 | +0.00183 |
| **euler@512 (converged)** | 0.04267 | 0.04190 | 0.04280 | **0.04246** | 0.00048 | **+0.00242** |
| Gaussian@100 | | | | 0.04004 | | — |

Pooled bootstrap (n=288): euler@512 = 0.04246 [0.04092, 0.04404];
euler@16 = 0.04036 [0.03884, 0.04188].

Two results, both replicating across all three seeds:

1. **The residual at the converged endpoint is +0.00242, not +0.00712.** The
   earlier figure was inflated by using a non-converged 256-NFE RK4 endpoint.
   The unpaired 95% CI for the converged arm **spans Gaussian's 0.04004**, so at
   this sample size the converged Flow endpoint is **not significantly worse than
   Gaussian**.
2. **Euler@16 remains better than the converged solution**: paired difference
   euler@512 − euler@16 = **+0.00210 [+0.00158, +0.00262]**, CI excludes zero.
   Converging the ODE more accurately measurably *worsens* the prediction.

(These absolute values differ slightly from the previous study — e.g. euler@16
0.04036 here vs 0.04455 there — because the shared environment trajectory is now
advanced with the Euler@8 action rather than Euler@2. Within-study comparisons
are paired and unaffected.)

## 14. Perfect-future metric floor (Phase 13, secondary)

| input | chamfer |
|---|--:|
| **perfect future latent (identity)** | **0.0** |
| perfect future through normalize→unnormalize | 4.58e-8 |
| copy-current | 0.07353 |
| Gaussian@100 | 0.04004 |
| Flow best practical (euler@16) | 0.04036 |

**The metric floor is exactly zero.** There is no representation or metric
overhead inflating the comparison. Per instruction, this is not used to explain
away solver behavior — it simply removes one candidate explanation, and in doing
so makes the (now smaller) residual fully attributable to the models.

## 15. Revised interpretation of the Flow-vs-Gaussian residual

Using the mandated wording: **a residual Flow-vs-Gaussian gap remains at the best
tested practical Euler setting — but it is small (+0.00032 at Euler@16, +0.00242
at the converged endpoint) and its unpaired CI includes zero.**

What is now solidly established is a different and sharper fact: **the converged
Flow ODE solution is significantly worse at predicting the future than the
deliberately under-resolved Euler@16 solution** (+0.00210, CI excludes zero, 3/3
seeds). Euler's truncation error is not noise being cleaned up by more compute —
it is systematically landing closer to the data than the exact ODE endpoint does.

I am **not** attributing this to any single cause. It is consistent with the
learned field's exact flow terminating slightly off the data manifold while
coarse integration happens to compensate, but that is a hypothesis, not a result.

## 16. Is loss modification still scientifically motivated?

**Status: B — LOWER PRIORITY** (revised down from the previous report's "C —
plausible again").

Reasoning: the previous upgrade to C rested on "a substantial residual survives
accurate integration." With a converged reference the residual is **~3× smaller
(+0.00242) and not significantly different from Gaussian**. There is no longer a
substantial unexplained model-level deficit for a loss change to target, and the
one robust effect (converged < Euler@16) is not an obvious loss phenomenon.

**No loss change is implemented or recommended.**

## 17. Exactly ONE next experiment

**Measure whether the converged Flow endpoint is off-manifold relative to Euler@16,
using the training-data distribution — no training, no generation beyond what
exists.**

Concretely: for the same 96 paired samples and 3 seeds, take the already-computed
Euler@16 and Euler@512 endpoints and compare each to the empirical training-latent
distribution (per-feature normalized ranges, Mahalanobis distance under the
training mean/covariance already computed by `training_stats`, and fraction of
coordinates outside observed training ranges).

Rationale: §13 established that more accurate integration reliably produces worse
predictions. There are two competing explanations — (a) the exact flow terminates
slightly off the data manifold and coarse steps truncate toward it, or (b) the
difference is unrelated to manifold proximity. This measurement separates them
directly, costs a few minutes on cached endpoints, and determines whether the
next real question is about the learned field's terminal distribution or about
something else. It must precede any objective-level work.

---

## HARD STOP OBSERVED

No training. No loss change. No new policy method. No new seeds.
