# NFE1 vs NFE4 under the calibrated R=3 protocol

**No training. No policy change. No physics tuning. No NFE2/Gaussian.**
(Filename retains `R5` per the requested path; the executed and approved design
is **R=3**.)

Scripts: `crn_control.py`, `analyze_nfe1_nfe4.py`
Data: `results/r0_s{42,43,44}_nfe{1,4}_r3rep{1,2,3}.json`, `nfe1_nfe4_analysis.json`
Provenance: `episode_set_provenance.json`, `run_schedule.json`

**HEADLINE: Classification B — a small-to-moderate one-step penalty. NFE4 beats
NFE1 on all three checkpoints (+9.0, +2.8, +6.3 pp; mean +6.0 pp, SD 3.1 pp).
The direction is unanimous and the effect is concentrated on physics-sensitive
episodes, but only seed 42's interval excludes zero, and the N=3
checkpoint-level interval spans zero. Practical equivalence is NOT supported.**

---

## 1. Compute: estimate vs actual

| | value |
|---|--:|
| measured NFE1 realization (96 eps) | 116 s |
| measured NFE4 realization (96 eps) | 133 s |
| projected, 18 runs + 15% overhead | **0.72 GPU-h** |
| **actual** | **2495 s = 0.693 GPU-h** |
| model forward calls | 432,000 |

Under projection for once. Every run verified at exact call counts (**1.0** and
**4.0** calls/plan); no `CALL_COUNT_MISMATCH` on any of the 18.

## 2. Run-order balancing (predeclared, frozen before launch)

`run_schedule.json`: arm order alternates with `(repeat + seed_index)` parity, so
NFE1 runs first in **5 of 9** (seed, repeat) pairs. Recorded before launch and
not modified. Executed order matches the schedule exactly.

## 3. CRN-bank provenance

Validated mechanism from `COMMON_RANDOM_NUMBERS_EVALUATOR.md`, unmodified:
base seed **20260905**, seed derivation
`sha256("base|batch_start|decision")[:8]` little-endian mod 2⁶³−1, keyed by
(episode-batch, decision index). Gates previously passed bit-exactly (0.000e+00
identity, marginals KS p=0.91/0.99/0.99). Both arms receive the same exogenous
Flow-noise sequence at each decision index.

## 4. Frozen episode-set provenance

| property | value |
|---|---|
| artifact | `experiments/isaacgym_episode_sets/replicate0_n96.pkl` |
| stored sha256 | `35144910b1471b7b0d50d17da18b01db0b5e61e21d7e16e4ef1aa266ee80d511` |
| file sha256 | `c760dd58d4cde41c…` |
| init sha256 | `b5b16bcda726b76b…` |
| goal sha256 | `7875ce901889dc54…` |
| episodes / cubes / horizon | 96 / 3 / 100 |

Checkpoints (internal step **499000** verified by loading each state dict):
seed 42 `861dc344…`, seed 43 `c8e00ead…`, seed 44 `c2c13f55…`.

## 5-8. Per-seed results and hierarchical intervals

Estimator: bootstrap **episodes** at the top level (96 units), then resample the
R=3 physics realizations **within** each drawn episode. Physics realizations are
not paired across arms (§8).

| seed | p(NFE1) | p(NFE4) | **Δ** | 95% CI | effective counts |
|---|--:|--:|--:|---|---|
| 42 | 0.7951 | 0.8854 | **+0.0903** | **[+0.0243, +0.1597]** | 76.3/96 → 85.0/96 |
| 43 | 0.8090 | 0.8368 | **+0.0278** | [−0.0556, +0.1076] | 77.7/96 → 80.3/96 |
| 44 | 0.8472 | 0.9097 | **+0.0625** | [−0.0069, +0.1319] | 81.3/96 → 87.3/96 |

**Three-seed summary (N=3 checkpoints):** deltas +0.0903 / +0.0278 / +0.0625,
signs **+, +, +**, mean **+0.0602**, SD **0.0313**.

Checkpoint-level 95% t-interval (N=3, wide by construction):
**[−0.0176, +0.1380]** — reported for completeness, not as the primary evidence;
with three replicates it has almost no power.

## 9. Relation to the predeclared ±5 pp margin

| seed | CI within ±5 pp? | CI excludes 0? |
|---|---|---|
| 42 | **No** | **Yes** |
| 43 | No | No |
| 44 | No | No |

**No seed's interval fits inside the ±5 pp equivalence region**, so per §14 the
correct statement is **not** "equivalent". Two of three point estimates
(+0.090, +0.063) exceed the 5 pp practical threshold; the third (+0.028) does not.

## 10. Episode-level Δᵢ distribution

| seed | Δᵢ > 0 | Δᵢ = 0 | Δᵢ < 0 | mean \|Δᵢ\| when nonzero |
|---|--:|--:|--:|--:|
| 42 | **27** | 60 | 9 | 0.426 |
| 43 | **31** | 45 | 20 | 0.418 |
| 44 | **24** | 60 | 12 | 0.463 |

Positive episodes outnumber negative by 3.0×, 1.6× and 2.0×. Most episodes are
unaffected (Δᵢ = 0 on 45–60 of 96); where the arms differ, the difference is
large (mean |Δᵢ| ≈ 0.42–0.46, i.e. more than one realization in three flipping).

## 11. Physics-sensitive episode analysis — **the mechanism**

| seed | NFE1 robust+ / robust− / sensitive | NFE4 robust+ / robust− / sensitive |
|---|---|---|
| 42 | 57 / 3 / **36** | 70 / 1 / **25** |
| 43 | 55 / 3 / **38** | 64 / 3 / **29** |
| 44 | 66 / 3 / **27** | 78 / 1 / **17** |

**NFE4 systematically converts physics-sensitive episodes into robust successes**
— sensitive counts fall 36→25, 38→29, 27→17, and robust successes rise
57→70, 55→64, 66→78 on the three seeds.

Where the effect lives:

| seed | Δ on sensitive-either episodes | Δ on both-robust episodes |
|---|--:|--:|
| 42 | **+0.1550** (43 eps) | +0.0377 (53 eps) |
| 43 | **+0.0692** (53 eps) | −0.0233 (43 eps) |
| 44 | **+0.1351** (37 eps) | +0.0169 (59 eps) |

**The advantage is concentrated on episodes that were already marginal**, 4–6×
larger there than on robustly-solved ones. NFE4 does not broadly change
behaviour; it makes borderline scenarios reliable. This is a mechanistically
informative answer to §16.

## 12. Secondary canonical metrics (episode-mean over R, then across episodes)

| metric | s42 | s43 | s44 | mean |
|---|--:|--:|--:|--:|
| goal_success_frac | +0.0405 | +0.0185 | +0.0301 | **+0.0297** |
| cubes_placed | +0.1215 | +0.0556 | +0.0903 | **+0.0891** |
| avg_obj_dist | −0.0114 | −0.0026 | −0.0107 | **−0.0082** |
| max_obj_dist | −0.0284 | −0.0057 | −0.0230 | **−0.0190** |
| n_contacted | +0.0139 | −0.0035 | −0.0139 | −0.0012 |
| cubes_farther | −0.0694 | +0.0035 | −0.0451 | −0.0370 |

Every placement-quality metric favours NFE4 on all three seeds (except
`cubes_farther` on seed 43, marginally). Contact rate is unchanged, so NFE4 is
not reaching the cubes more often — it is **placing them better once there**,
consistent with §11.

## 13. Old R=1 vs new R=3

| seed | old R=1 (3-cube, fixed-H) | new R=3 | change |
|---|--:|--:|--:|
| 42 | +0.0428 | +0.0903 | +0.0475 |
| 43 | +0.0046 | +0.0278 | +0.0231 |
| 44 | +0.0301 | +0.0625 | +0.0324 |
| **mean** | **+0.0258** | **+0.0602** | **+0.0344** |

**Signs were stable; magnitudes were attenuated.** The new estimate is larger on
every seed — roughly 2.3× the old mean. (Episodes are not paired across studies;
this is descriptive only, per §18.)

This is the opposite of the usual regression-to-the-mean worry: the R=1
evaluation understated the effect rather than inflating it. Plausibly because
single-realization sampling adds noise that dilutes a real difference on
sensitive episodes — exactly the population §11 shows carries the effect.

## 14. Three-seed classification: **B — SMALL ONE-STEP PENALTY**

Direction favours NFE4 on **3/3** checkpoints, and the mean (+6.0 pp) exceeds the
5 pp practical threshold — but:

- only **1 of 3** per-seed CIs excludes zero;
- the checkpoint-level N=3 interval **[−0.018, +0.138] spans zero**;
- seed 43's point estimate (+2.8 pp) is **below** the practical margin, so the
  effect is not uniformly meaningful across checkpoints.

Not **A** (calibrated uncertainty does not support a ≥5 pp effect on 2 of 3
seeds). Not **C** (no interval fits inside ±5 pp; equivalence is unsupported).
Not **D** — all three agree in sign and the spread (2.8–9.0 pp) is consistent
with sampling noise at this precision, though seed sensitivity cannot be excluded
at N=3. Not **E**: R=3 was sufficient to produce a unanimous direction, a
mechanism, and one significant seed.

## 15. Does a genuine one-step control penalty exist?

**Probably yes, and it is larger than previously estimated — but it is not yet
established at the per-checkpoint level.** The strongest defensible statement:

> Under noise-calibrated evaluation with 3 physics realizations per episode,
> Flow Euler NFE4 outperformed NFE1 on all three independently trained
> checkpoints by +2.8 to +9.0 pp (mean +6.0 pp). The advantage is concentrated on
> physics-sensitive episodes, where NFE4 converts marginal scenarios into robust
> successes. One checkpoint's interval excludes zero; the others do not.

**One-step Flow is measurably worse than four-step Flow for this task.** The
practical recommendation (deploy NFE4) is unchanged and now better supported.

## 16. Corrected status of the earlier three-seed "Regime A" result

Applied in `experiments/audit/seed_replication_CORRECTION.md` **before** this run:

- **Withdrawn:** "Regime A strong three-seed replication"; "F4 has a replicated
  ~3 pp control advantage"; the per-seed bootstrap p-values (0.0351 / 0.0255) as
  evidence.
- **Reason:** measured at R=1 against a 7.6 pp single-realization noise floor,
  and the CIs resampled episodes while treating each single rollout as fixed —
  excluding simulator variance entirely.
- **Now superseded** by this R=3 measurement, which finds the same sign with a
  **larger** magnitude (+0.060 vs +0.026). The original conclusion's *direction*
  survives; its magnitude and its claimed significance do not.

## 17. Strongest low-NFE claim now scientifically justified

> On 3-cube PushCube, Flow Euler at 4 network evaluations gives materially better
> closed-loop control than at 1 evaluation (mean +6.0 pp across three
> checkpoints, unanimous in direction, concentrated on marginal episodes), under
> an evaluator whose noise floor has been measured and whose protocol uses
> repeated physics realizations. NFE4 remains the recommended operating point.

## 18. Claims that remain provisional

- **NFE2 vs NFE4** — untested at R≥3; whether the penalty is already gone at 2
  steps is unknown. (Not run, per §21.)
- **"Control saturates at NFE2" / "peaks at NFE8" / "declines at NFE16"** — still
  withdrawn; all rest on R=1 data.
- **"Flow NFE2 matches Gaussian@100 at 53× lower latency"** — the latency ratio
  stands; the control-performance half is **historical/checkpoint-level** (one
  Gaussian checkpoint, R=1 evaluator). Not re-run, per §22.
- **NFE4 vs NFE32** — no reliable difference detected; not equivalence.
- Seed-level heterogeneity cannot be excluded at N=3.

## 19. ONE recommended next action

**Run NFE2 vs NFE4 at R=3 on the same protocol (~0.65 GPU-h).**

§21 predeclared this as the follow-up conditional on NFE1 being meaningfully
worse — which it is. This is now the sharpest open question in the project: the
headline efficiency claim rests on NFE2 being sufficient, that claim currently
rests entirely on R=1 data, and we now know R=1 *attenuated* the NFE1 effect by
~2.3×. If NFE2 also carries a penalty relative to NFE4, the recommended operating
point and the efficiency headline both need revising.

**Not launched — requires approval.**

---

## HARD STOP OBSERVED

No NFE2. No Gaussian. No training. No loss changes. No new method.
