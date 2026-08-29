# Matched-NFE solver study: separating numerical integration error from model error

**No training. No model, loss, normalization, conditioning, data or checkpoint
changes.** Only the numerical integration of `dx/dt = v_theta(x,t)` varies.

Scripts: `solvers.py`, `solver_study.py`, `analyze_solver_study.py`
Data: `solver_study.json` · Figures: `experiments/figures/solver_*.png`
Reference note: `reference_convergence_note.md`
Compute: **160 s wall (~0.045 GPU-h)**, well inside the 1 GPU-hour budget.

Protocol: frozen episode set `35144910`, 16 episodes x 6 rollout steps = **96
paired samples per arm**, EMA checkpoints at internal step 499000, noise seed
777. Within each (episode, rollout step, seed) **every solver and NFE receives
the identical x0, current observation, goal and conditioning**. The environment
is advanced with one shared arm's action so all arms see one common trajectory.

---

## 1. Exact high-accuracy reference definition

**Reference = classical RK4, 64 uniform steps on t in [0,1] = 256 NFE**, from
the same x0, with conditioning re-imposed after every stage and every step, and
the velocity masked by the canonical conditioning mask.

## 2. Reference convergence validation — the test FAILED in its naive form

This is a real finding and it constrains every downstream claim.

RK4 step-refinement ladder (seed 42, fixed x0):

| refinement | NFE | mean `|x_n - x_{n/2}|` | max |
|---|--:|--:|--:|
| rk4@16 vs rk4@8   |  64 | 0.46582 | 0.96147 |
| rk4@32 vs rk4@16  | 128 | 0.17353 | 0.30440 |
| rk4@64 vs rk4@32  | 256 | 0.12280 | 0.65311 |
| rk4@128 vs rk4@64 | 512 | 0.10427 | 0.89600 |

RK4 should shrink this ~16x per halving of the step. It **stalls at ~0.10-0.12**.

Is that float32 noise? No:

| check | mean | max |
|---|--:|--:|
| rk4@64 run twice (repeat determinism) | **0.00000000** | 0.00000000 |
| rk4@64 vs midpoint@128 (both NFE 256) | 0.17496 | 1.10279 |
| rk4@64 vs euler@512                   | 0.20586 | 1.09942 |

The solve is **bitwise reproducible**, yet three *independent* high-order schemes
at NFE 256-512 disagree by ~0.17-0.21. Disagreement between independent schemes
that does not shrink under refinement is neither roundoff nor a solver bug: the
learned velocity field is rough enough that fine-scale trajectories separate.
**The Flow ODE is not integrable to machine precision at this model's smoothness.**

So we do **not** claim a converged reference. We report a **reference
uncertainty floor** = disagreement with an independent scheme of equal cost.

**Predefined tolerance** (fixed before interpreting results): the reference is
usable iff its uncertainty floor is >=5x smaller than the numerical error of the
coarsest arm interpreted.

Measured over the full 96-sample set: floor **mean 0.14072, median 0.06677,
max 0.82656**, versus Euler@16 at 1.380 (**9.8x**) and Euler@2 at 5.753 (41x).
**Tolerance met for every arm reported.** Endpoint norm ~16.95, so the floor is
~1% of the endpoint. Numerical-error *differences* approaching ~0.14 are not
resolvable and are not interpreted.

## 3. Implementation validation (Phase 3)

CPU unit test on `dx/dt = x`, x(0)=1, measured convergence order:

| method | order measured (steps 4->32) | expected |
|---|--:|---|
| euler | 0.86 -> 0.96 | 1 |
| midpoint | 1.86 -> 1.97 | 2 |
| heun | 1.86 -> 1.97 | 2 |
| rk4 | 3.81 (floors at ~1e-7 on float32 roundoff) | 4 |

On the real model, per seed:

| seed | Euler diagnostic vs canonical sampler | determinism (same x0 twice) | NFE(euler@4) |
|---|--:|--:|--:|
| 42 | **0.000e+00** | 0.000e+00 | 4 |
| 43 | **0.000e+00** | 0.000e+00 | 4 |
| 44 | **0.000e+00** | 0.000e+00 | 4 |

Euler is bit-identical to `conditional_sample`'s loop; time interval is exactly
[0,1]; NFE counters are asserted from an internal counter, not assumed.

## 4. NFE accounting

`NFE = velocity-network forward calls per generated trajectory`, counted
programmatically: euler = steps, midpoint = 2x steps, heun = 2x steps,
rk4 = 4x steps. All comparisons below are at **equal NFE**, never equal steps.

## 5-7. Results, all three seeds

### A. Ground-truth imagination error (chamfer position; lower is better)

| arm | NFE | s42 | s43 | s44 | mean | sd |
|---|--:|--:|--:|--:|--:|--:|
| euler@2 | 2 | 0.12373 | 0.12751 | 0.12423 | **0.12516** | 0.00205 |
| midpoint@1 | 2 | 0.12854 | 0.13758 | 0.13342 | 0.13318 | 0.00452 |
| heun@1 | 2 | 0.27141 | 0.26427 | 0.27011 | 0.26859 | 0.00380 |
| euler@4 | 4 | 0.06298 | 0.06518 | 0.06323 | **0.06379** | 0.00120 |
| midpoint@2 | 4 | 0.11918 | 0.11958 | 0.12004 | 0.11960 | 0.00043 |
| heun@2 | 4 | 0.17311 | 0.16906 | 0.17156 | 0.17124 | 0.00204 |
| euler@8 | 8 | 0.04653 | 0.04688 | 0.04657 | **0.04666** | 0.00019 |
| midpoint@4 | 8 | 0.06441 | 0.06418 | 0.06271 | 0.06376 | 0.00092 |
| heun@4 | 8 | 0.11499 | 0.11377 | 0.11408 | 0.11428 | 0.00063 |
| euler@16 | 16 | 0.04413 | 0.04500 | 0.04451 | **0.04455** | 0.00044 |
| **RK4@64 reference** | **256** | 0.04611 | 0.04837 | 0.04699 | **0.04716** | 0.00114 |
| Gaussian@100 | 100 | | | | 0.04004 | |
| copy-current | 0 | | | | 0.07484 | |

**Euler wins at every matched NFE budget, on every seed, without exception.**

### B. Numerical ODE error (latent distance to the RK4@64 reference)

| arm | NFE | s42 | s43 | s44 | mean | sd |
|---|--:|--:|--:|--:|--:|--:|
| euler@2 | 2 | 5.71346 | 5.78602 | 5.75871 | 5.75273 | 0.03665 |
| midpoint@1 | 2 | 5.38948 | 5.43596 | 5.41324 | **5.41289** | 0.02324 |
| heun@1 | 2 | 16.72893 | 16.76861 | 16.77410 | 16.75722 | 0.02465 |
| euler@4 | 4 | 3.27170 | 3.33849 | 3.29749 | **3.30256** | 0.03368 |
| midpoint@2 | 4 | 3.79778 | 3.85863 | 3.80624 | 3.82088 | 0.03296 |
| heun@2 | 4 | 8.66549 | 8.66453 | 8.63153 | 8.65385 | 0.01933 |
| euler@8 | 8 | 2.10068 | 2.10757 | 2.07424 | 2.09416 | 0.01759 |
| midpoint@4 | 8 | 2.01560 | 2.01789 | 2.02936 | **2.02095** | 0.00737 |
| heun@4 | 8 | 4.33805 | 4.34892 | 4.32713 | 4.33804 | 0.01089 |
| euler@16 | 16 | 1.37855 | 1.41550 | 1.34668 | 1.38024 | 0.03444 |

Seed sd is tiny (<=0.04) and every ordering replicates across all three
independently trained vector fields.

## 8. Paired matched-NFE effects (95% paired-bootstrap CI; negative beats Euler)

| comparison | NFE | ground-truth error | numerical error |
|---|--:|---|---|
| midpoint@1 - euler@2 | 2 | +0.0048 / +0.0101 / +0.0092 (worse; s42 CI spans 0) | **-0.324 / -0.350 / -0.345 (better)** |
| heun@1 - euler@2 | 2 | +0.148 / +0.137 / +0.146 (far worse) | +11.02 / +10.98 / +11.02 (far worse) |
| midpoint@2 - euler@4 | 4 | **+0.0562 / +0.0544 / +0.0568 (worse)** | +0.526 / +0.520 / +0.509 (worse) |
| heun@2 - euler@4 | 4 | +0.110 / +0.104 / +0.108 (far worse) | +5.39 / +5.33 / +5.33 (far worse) |
| midpoint@4 - euler@8 | 8 | +0.0179 / +0.0173 / +0.0161 (worse) | **-0.085 / -0.090 / -0.045 (slightly better)** |
| heun@4 - euler@8 | 8 | +0.068 / +0.067 / +0.068 (far worse) | +2.24 / +2.24 / +2.25 (far worse) |

Values are per seed 42/43/44. Every CI excludes zero except `midpoint@1-euler@2`
ground-truth on seed 42.

**The decisive row is midpoint at NFE 2 and NFE 8: it integrates the ODE
*better* than Euler (numerical error lower, CI excludes zero) while producing
*worse* imagination (ground-truth error higher, CI excludes zero).** More
faithful integration of the learned field yields a worse prediction of reality.

## 9. Vector-field diagnostics (seed 42)

| arm | mean \|update\| | cos(v_k, v_{k-1}) | rel \|v2-v1\|/\|v1\| within a step |
|---|--:|--:|--:|
| euler@2 | 19.19 | 0.9728 | -- |
| euler@4 | 9.48 | 0.9691 | -- |
| euler@8 | 4.73 | 0.9847 | -- |
| euler@16 | 2.35 | 0.9944 | -- |
| midpoint@1 | 36.95 | -- | 0.2339 |
| midpoint@2 | 18.73 | 0.9790 | 0.2186 |
| midpoint@4 | 9.41 | 0.9783 | 0.1453 |
| heun@1 | 22.05 | -- | **0.9622** |
| heun@2 | 14.69 | 0.9855 | **0.6179** |
| heun@4 | 8.35 | 0.9821 | **0.4014** |

The new relative-change diagnostic explains Heun cleanly. Heun's predictor
`x + dt*v1` takes a **full** step; at NFE 2 that is dt=1 with `|v|~40`, landing
far off-manifold where the velocity is entirely different -- `|v2-v1|/|v1| =
0.96`, i.e. the second evaluation disagrees with the first by ~100%. Averaging
those two velocities is worse than using the first alone. Midpoint takes a
**half** step and stays much closer (0.23), which is why the two nominally
equal-order methods separate so sharply here.

Supporting probe (fixed x, sweeping t): `|v|` rises monotonically 39.8 -> 203.6
as t: 0 -> 1. The field is strongly state- and time-dependent, so large
explicit predictor steps are unreliable. Training samples t ~ U[0,1), so this is
a step-size effect, not a coverage gap.

## 10. High-accuracy Flow vs Gaussian residual (Phase 12)

Pooled over 288 samples (3 seeds x 96):

- **Flow at 256 NFE (RK4@64): 0.04716**  (seed sd 0.00114)
- Gaussian@100: **0.04004**
- **Residual: +0.00712**, i.e. the accurately integrated Flow ODE is still
  ~18% worse than Gaussian.

And the sharper observation:

| comparison | paired difference [95% CI] |
|---|---|
| RK4@256NFE - Euler@8 | +0.00050 [-0.00055, +0.00155] (indistinguishable) |
| RK4@256NFE - Euler@16 | **+0.00261 [+0.00202, +0.00320] (reference is WORSE)** |

**Euler@16 has already reached, and slightly passed, the Flow ODE's own
accuracy limit.** Spending 16x more compute to integrate the ODE more faithfully
does not improve the prediction — it very slightly degrades it. The remaining
+0.00712 gap to Gaussian survives essentially exact integration and therefore
**cannot be numerical integration error**.

## 11. Classification: **C — BETTER NUMERICS DO NOT IMPROVE IMAGINATION MUCH**

Evidence:

1. Higher-order methods do reduce **numerical** error where the theory predicts
   (midpoint beats Euler at NFE 2 and NFE 8 on reference distance, CIs exclude
   zero) — so the solvers are correct and the numerical axis is real.
2. That improvement **does not transfer** to ground-truth error. At NFE 2 and 8
   midpoint is numerically *closer* to the true ODE solution and simultaneously
   *worse* at predicting the actual future, on all three seeds.
3. The 256-NFE reference — 16x the compute of Euler@16 — is statistically
   **indistinguishable from Euler@8** and significantly **worse than Euler@16**.
4. The Flow-vs-Gaussian residual (+0.00712) is unchanged by accurate integration.

This also revises the previous sweep's mechanism. The earlier report said the
NFE curve was "consistent with first-order integration error" and flagged that
as unproven. It is now **falsified as the explanation for the state deficit**:
integrating the learned field more accurately does not produce better states.
The gains from NFE 1->8 are real, but they are not gains from *converging to the
ODE solution* — Euler's truncation error happens to land nearer the data than
the exact ODE endpoint does.

## 12. Revised interpretation of the imagination deficit

The learned velocity field's **exact** trajectory terminates ~0.047 from the
true future, versus Gaussian's 0.040. Since:

- the Euler NFE curve plateaus exactly where the ODE solution itself sits,
- and no more accurate solver goes below that plateau,

the deficit is a property of **where the learned Flow ODE's exact solution
lands**, not of how we get there. That is a model-level property. Candidates —
none tested here, and deliberately not narrowed to one:

- the learned vector field / its smoothness (we measured directly that it is not
  integrable to machine precision, which is itself a statement about the field),
- the training objective and its weighting,
- the conditional generative formulation (linear-path CFM with endpoint
  conditioning),
- the DLP representation and the decoder.

**We do not attribute the residual to the loss.** Phase 6's instruction applies
directly: this is the "residual is not numerical integration error" branch, and
it licenses only "model-level", not "loss".

One caveat, stated plainly: the residual is small (+0.00712, ~18%) and is
**not iso-compute** — Gaussian uses 100 NFE against Flow's 16 (or 256). A fair
practical statement is that Euler@8-16 is the efficient operating point for this
Flow model and gets within ~0.006-0.007 of a 100-NFE Gaussian.

## 13. Revised loss-hypothesis status: **C — PLAUSIBLE AGAIN**

A substantial fraction of the residual survives accurate integration: accurate
integration removes **none** of the Flow-Gaussian gap (0.04716 vs Euler@16's
0.04455 — accurate integration is if anything slightly worse). Under the
Phase 13 definitions this is C, not B: integration does **not** explain the
residual.

This raises the loss hypothesis from LOWER PRIORITY back to PLAUSIBLE — but
strictly to *plausible*, and no higher. Nothing here implicates the objective
specifically over the representation, the field's smoothness, or the conditional
formulation. Reaching D would require evidence that singles out the objective.
**No loss change is implemented, and none is recommended on this evidence alone.**

## 14. Is a small Isaac Gym solver control experiment justified?

**No.** The premise of Phase 11 was "if a second-order solver materially
improves imagination at matched NFE." It does not — it is worse at every matched
budget on every seed. There is nothing to carry into a control evaluation.
Euler remains the correct sampler and the existing control results stand.

## 15. Exactly ONE next action

**Test whether the residual is in the generative model at all, by decoding the
ground-truth future latent through the same pipeline.**

Encode the true future observation to its DLP latent and push it through the
identical unnormalize -> chamfer path used for every arm above, giving the
metric's own noise floor on perfect input. Cost: no generation, no training,
minutes.

Rationale: every number in this study is measured in DLP latent space through
one fixed decode/metric path. If a *perfect* future scores meaningfully above
zero on this metric, part of the +0.00712 residual is representation/metric
overhead shared by Flow and Gaussian alike, and the model-level gap is smaller
than it looks. If a perfect future scores ~0, the residual is entirely
attributable to the generative models and the model-level question is sharp.
This is the cheapest measurement that can shrink the space of explanations, and
it must precede any objective-level work.

---

## HARD STOP OBSERVED

No training. No loss modification. No MeanFlow. No VP. No new seeds. No full
control evaluation.
