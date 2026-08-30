# Isaac Gym evaluator noise floor: calibration study

**No training. No policy change. No physics-parameter tuning. No NFE4-vs-NFE32.**
Compute: **0.51 GPU-h total** (open-loop 156 s + 8 CRN repeats 1127 s +
3 no-CRN repeats 425 s + CPU/subscene diagnostics). The §8 calibration budget of
0.30 GPU-h applied to the same-arm repeat study, which came in at **0.313 GPU-h**
(R=8, estimated 0.296).

Scripts: `openloop_physics.py`, `crn_control.py`, `analyze_noise_floor.py`
Data: `openloop_physics.json`, `noise_floor.json`, `results/`

**HEADLINE: the evaluator noise floor is now measured. A single realization per
episode gives a 95% CI of ±6.3 pp and cannot resolve 5 pp. But your correction
was right — replication works: R=3 brings the paired-difference resolution to
~4.4 pp and R=5 to ~3.4 pp. The variance is concentrated in 41% of episodes,
and it is NOT threshold flipping — failed realizations miss by 0.304 versus
0.018 when they succeed.**

---

## 1. Canonical physics configuration (recorded, unchanged)

From `env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml`:

| parameter | value |
|---|---|
| physics engine | PhysX |
| **`use_gpu`** | **True** |
| **`use_gpu_pipeline`** | **True** |
| solver_type | 1 (TGS) |
| num_position_iterations | 8 |
| num_velocity_iterations | 1 |
| substeps | 2 (effective 120 Hz) |
| dt | 0.01667 (60 Hz) |
| num_threads | 4 |
| **num_subscenes** | **4** (runtime log: "useGpu is set, forcing single scene (0 subscenes)") |
| numEnvs | **16 batched** |
| aggregateMode | 3 |
| contact_offset / rest_offset | 0.005 / 0.0 |
| contact_collection | 0 (CC_NEVER) |
| control | OSC @ 3 Hz |
| episodeLength (3C) | 100 |
| success threshold | 0.04 (cube radius) |

**Nothing was modified.** Environments are batched 16-at-a-time on GPU, so
physics ordering/numerics are shared across the batch — a plausible route for
cross-environment coupling, though not isolated here.

## 2. Stochasticity ledger

| # | source | location | status |
|---|---|---|---|
| 1 | Flow initial noise `torch.randn` | `flow_matching.py:358` | **controllable** (CRN); one draw per plan, no per-step noise |
| 2 | Franka DOF reset noise `torch.rand` | `isaac_panda_push_env.py:698` | **inert** (`frankaDofNoise: 0.0`) but consumes the RNG stream |
| 3 | start pos/rot noise | `:448-454` | inactive, all config values 0.0 |
| 4 | DLP encoding | `dlp_utils.py:286,342` | **deterministic** (`deterministic=True`) |
| 5 | policy path | `policies.py` | no additional RNG |
| 6 | **Isaac Gym GPU physics** | `env.step` | **NOT controllable — dominant source** |

## 3. Open-loop repeated-trajectory divergence

10 repeats, 16 episodes, **identical serialized initial state, identical fixed
action sequence, no policy inference**, measured on **raw PhysX tensors**
(q, qd, root, eef) — not DLP.

| step | cube_max | cube_rms | eef_max | dof_max |
|--:|--:|--:|--:|--:|
| **0** | **5.877e-04** | 1.463e-05 | 8.583e-07 | 1.907e-06 |
| 1 | 5.739e-04 | 1.438e-05 | 2.861e-06 | 4.548e-05 |
| 2 | 4.593e-03 | 2.377e-04 | 1.681e-03 | 2.243e-02 |
| 5 | 4.593e-03 | 2.453e-04 | 2.845e-03 | 7.391e-02 |
| 8 | 2.259e-01 | 5.373e-03 | 1.026e-02 | — |
| 10 | 9.216e-01 | 2.014e-02 | 1.064e-02 | 8.861e-02 |
| 50 | 7.175e-01 | 3.241e-02 | 5.180e-02 | 1.133e+00 |
| **100** | **9.225e-01** | 5.490e-02 | 1.565e-01 | 2.364e+00 |

**Divergence is nonzero at t=0, before any action is applied** (5.9e-4 in cube
position) — the reset itself is not bit-reproducible.

## 4. Contact-related divergence — **the amplification is contact-triggered**

Growth by segment: steps 0→2 **×7.8**, steps **2→10 ×200.7**, then 10→20 ×0.8,
20→50 ×1.0, 50→100 ×1.3.

Divergence is small and roughly flat until ~step 7, jumps by two orders of
magnitude between steps 7–10, then **plateaus**. First excursion above 1e-2 is
step 7; above 1e-1 is step 8.

This is **not** smooth exponential chaos. It is a contact-triggered
discontinuity: once a near-tie contact resolves differently, a cube takes a
macroscopically different path and the difference persists rather than
compounding. Terminal cube divergence is **23× the 0.04 success threshold**.

## 5. Closed-loop same-arm protocol

Flow **seed 42, NFE 4** (cheap canonical operating point), frozen
`replicate0_n96` (sha `35144910…`), identical CRN policy-noise bank
(base seed 20260905), identical initial states and goals, **R = 8 independent
physics realizations**. NFE verified at exactly 4.0 calls/plan on every repeat.
No NFE comparison was run.

## 6. Run-level success across repeats

| repeat | success |
|--:|--:|
| 1 | 0.8229 (79/96) |
| 2 | 0.8958 (86/96) |
| 3 | 0.9167 (88/96) |
| 4 | 0.8229 (79/96) |
| 5 | 0.9167 (88/96) |
| 6 | 0.9167 (88/96) |
| 7 | 0.9271 (89/96) |
| 8 | 0.8854 (85/96) |

**mean 0.8880, SD 0.0423, range [0.8229, 0.9271] — a 10.4 pp spread with the
checkpoint, episodes, and policy noise all held fixed.**

This also explains the earlier "8.3 pp anomaly" (canonical 0.8854 vs rerun
0.8021): both values fall inside this distribution. **It was never evidence of a
protocol error — it is the evaluator's ordinary behaviour.**

## 7-8. Episode success-frequency distribution

| class | count | share |
|---|--:|--:|
| robust success (p_i = 1) | 57/96 | 59.4% |
| robust failure (p_i = 0) | **0/96** | 0.0% |
| **physics-sensitive (0 < p_i < 1)** | **39/96** | **40.6%** |

Histogram of successes-out-of-8: `[0,0,0,3,3,8,10,15,57]` for bins 0…8.

**No episode fails in all 8 realizations.** 41% of episodes carry *all* the
run-to-run variance, with mean p_i = 0.724 among them.

## 9. Relationship to the 0.04 threshold — **NOT threshold flipping**

| class | n | mean max_obj_dist |
|---|--:|--:|
| robust success | 57 | 0.0173 |
| physics-sensitive | 39 | 0.0970 |

Splitting the sensitive episodes by outcome:

| realization outcome | n | mean max_obj_dist |
|---|--:|--:|
| succeeded | 226 | **0.0184** |
| **failed** | 86 | **0.3036** |

**Failures miss by 0.304 — 16× the distance seen when the same episode succeeds,
and 7.6× the threshold.** `cubes_placed` on sensitive episodes ranges 0–3 with
per-episode SD 0.567 (vs exactly 0.000 on robust episodes).

So physics divergence does not nudge a cube across a scoring line; it causes
**qualitatively different task outcomes** — a cube gets pushed away or missed
entirely. This matters: variance-reduction tricks aimed at threshold margins
would not help.

## 10-11. Hierarchical uncertainty estimate and CI width vs R

**Estimator:** episodes are the primary sampling unit and physics repeats are
nested within episode. Bootstrap resamples 96 episodes with replacement, then
resamples R physics realizations within each drawn episode. Repeats are **not**
treated as independent samples.

| R | 95% CI for overall success | **width** |
|--:|---|--:|
| 1 | [0.8229, 0.9479] | **12.50 pp** |
| 2 | [0.8333, 0.9375] | 10.42 pp |
| 3 | [0.8403, 0.9306] | 9.03 pp |
| 5 | [0.8438, 0.9271] | 8.33 pp |
| 8 | [0.8477, 0.9258] | **7.81 pp** |

Width falls with R but **asymptotes near ~7.5 pp**, because episode-to-episode
difficulty variation (component A) does not shrink with physics replication —
only the within-episode term does. This is an absolute-rate CI; the paired
design below is far more efficient.

## 12. Resolution for a paired two-arm comparison

Within-episode physics variance `Var[p̂_i] = p_i(1−p_i)/R`; measured mean
`p_i(1−p_i)` = 0.1073. For a **paired** design on the same frozen episodes, the
episode-difficulty term cancels, leaving the noise-only floor:

| R | SE of Δ | 95% half-width | smallest resolvable \|Δ\| |
|--:|--:|--:|--:|
| 1 | 3.85 pp | 7.55 pp | **~7.6 pp** |
| 3 | 2.23 pp | 4.36 pp | **~4.4 pp** |
| 5 | 1.72 pp | 3.38 pp | **~3.4 pp** |
| 8 | 1.36 pp | 2.67 pp | **~2.7 pp** |

**Your correction is confirmed empirically.** The claim that ±5 pp is
unresolvable "at any episode count" was wrong: at **R ≥ 3** a 5 pp effect is
resolvable, and R = 8 approaches 2.7 pp. A **2 pp** effect needs R ≈ 15 by
extrapolation — which I flag as beyond the observed range (§13's caution) and do
not treat as established.

## 13-14. Does policy-noise CRN still help? **No measurable benefit**

Three additional repeats with **CRN disabled**, everything else identical:

| condition | rates | SD | spread |
|---|---|--:|--:|
| CRN ON (R=8) | 0.823, 0.896, 0.917, 0.823, 0.917, 0.917, 0.927, 0.885 | 0.0423 | 10.4 pp |
| CRN OFF (R=3) | 0.823, 0.854, 0.885 | 0.0312 | 6.2 pp |
| CRN ON subsampled to R=3 | — | mean 0.0383, 95% range [0.006, 0.057] | — |

**CRN-off SD (0.0312) sits at the 35th percentile of the CRN-on R=3
distribution** — statistically indistinguishable, and nominally *lower*.

**Physics variance completely swamps policy sampling noise.** CRN is correct and
harmless (validated bit-exact, marginals unchanged), but it buys nothing
measurable here. Recommendation: **keep it enabled** — it is free, it removes one
genuine source by construction, and it may matter for arms with larger sampling
spread (e.g. Gaussian, which draws noise at every one of 100 steps) — but do not
rely on it for variance reduction.

## 15-16. CPU / deterministic-pipeline diagnostic — **BLOCKED, not refuted**

I built an isolated CPU config copy in scratch (`use_gpu: False`,
`use_gpu_pipeline: False`); **the canonical config was not modified**.

- **CPU physics initializes** (4 subscenes, 4 articulations each) but the run
  fails in `isaac_env_wrappers.py:492` with a device mismatch: the wrapper
  assumes CUDA tensors and does `torch.cat` across cuda:0 and cpu. Fixing this
  means editing the canonical wrapper — out of scope for this audit.
- `num_subscenes: 1` / `num_threads: 1` on GPU **crashes** (core dump).

So **CPU reproducibility is untested**, not disproven. And per §18, even if it
were deterministic it would not automatically become the evaluator: GPU-vs-CPU
dynamic equivalence would need separate validation before any benchmark use.

**No physics parameters were tuned** (§19): solver iterations, contact offsets,
substeps, timestep, friction, damping and thresholds are all untouched.

## 17. Corrected interpretation of prior NFE control results

**SUPPORTED**
- Isaac Gym GPU physics is nondeterministic under identical initial states and
  identical actions.
- Fixed-action trajectories measurably diverge, contact-triggered, to 23× the
  success threshold.
- Policy-noise CRN alone cannot make the current evaluator repeatable, and adds
  no measurable variance reduction.
- Prior fine NFE ordering is unresolved.

**PROVISIONAL**
- NFE1 appears materially worse than the NFE2+ regime. Its gap (0.8056 vs
  0.868–0.899, i.e. 6–9 pp) is comparable to the single-realization resolution
  of ~7.6 pp, so even this is **not fully noise-controlled** despite its original
  McNemar p = 0.0356.

**NOT ESTABLISHED**
- "behavioural equivalence region"; "control saturates exactly at NFE2";
  "NFE8 is optimal"; "NFE16 declines"; any sub-~10 pp control difference from a
  single realization per episode — which includes the entire prior NFE ordering
  and the +3.5 pp NFE4-vs-NFE32 result.

Raw measurements are preserved and unaltered throughout.

## 18. Recommended protocol for ALL future closed-loop comparisons

1. **Frozen episode set** (`replicate0_n96`), identical across arms and seeds.
2. **R = 5 physics realizations per episode per arm** (predeclared before running).
3. **CRN enabled** — free, removes one source, no measurable cost.
4. Primary statistic: per-episode success **frequency** `p̂_i = (successes)/R`,
   never a single binary label.
5. Paired analysis on `Δ_i = p̂_i^B − p̂_i^A` across the same episodes.
6. Hierarchical bootstrap: resample episodes first, then realizations within
   episode.
7. **Predeclare the effect size**; anything below ~3.4 pp at R=5 is not resolvable.
8. Report the same-arm noise floor alongside every result.
9. **Retire single-realization McNemar** for effects under ~8 pp.

## 19. Cost of the recommended protocol

Measured: one NFE4 96-episode realization = 133 s.

| design | realizations | GPU-h |
|---|--:|--:|
| 2 arms × 1 seed × R=5, NFE4-class | 10 | **0.37** |
| 2 arms × 3 seeds × R=5, NFE4-class | 30 | **1.11** |
| NFE4 vs NFE32, 3 seeds, R=5 | 15 × (133 s) + 15 × (462 s) | **2.48** |

The previously "approved 0.508 GPU-h" NFE4-vs-NFE32 design was, in hindsight,
**~5× underpowered** for the effect it sought.

## 20. ONE next action

**Re-run NFE4 vs NFE32 under the §18 protocol at R=5, three seeds — but only
after deciding whether the 2.48 GPU-h is worth spending on this particular
question.**

My recommendation is that it is **not the best use of that budget**. The
offline evidence predicts no large effect, the measured resolution at R=5 is
~3.4 pp, and the practical decision (deploy NFE4) is already settled by the 8×
cost difference regardless of a sub-5 pp outcome. The calibration itself is the
durable result: **it retroactively bounds every control claim in this project
and specifies what future comparisons must do.**

If a closed-loop question is worth 2.48 GPU-h, the higher-value target is
**NFE1 vs NFE4** — the one contrast whose effect size (6–9 pp) is near the
resolvable range, whose current status is only *provisional*, and which
underpins the project's headline low-NFE claim.

**Nothing launched.** Both options need explicit approval.

---

## HARD STOP OBSERVED

No NFE4-vs-NFE32 rerun. No training. No policy modification. No physics
modification.
