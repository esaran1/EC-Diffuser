# 4-cube fixed-horizon evaluation (R=3) — Run 1

**Cost: 2557 s = 0.710 GPU-h + 0.010 probe = 0.720 total** (cap 0.78).
All 18 runs fresh; no call-count mismatches (2.0 / 4.0 verified on every run).

---

## 1. Old-R=1 compatibility audit — **NOT REUSABLE**

| requirement | historical `4cube_H100_flow_*` |
|---|---|
| arms NFE2 & NFE4 | **NFE1 and NFE4 only — no NFE2 arm** |
| `training_seed` | **NOT RECORDED** |
| `checkpoint` | **NOT RECORDED** |
| CRN policy-noise bank | **absent** (predates the validated evaluator) |
| scenario set | ✓ `5962c3ab…` — the one field that matched |
| H=100, 96 episodes | ✓ |

Four load-bearing fields fail. Per protocol §5 the artifacts were **not** reused
and all 18 realizations were run fresh. The frozen scenario set was reused.

## 2-3. Compute

Measured preflight (16-episode probe): NFE4 = 26 s/16 ep → 156 s per 96-episode
realization. Projected 0.748 + 0.010 probe = 0.758 GPU-h → exceeded the original
0.75 cap → **stopped and reported** rather than trimming R/seeds/episodes.
Cap raised to 0.78. **Actual: 0.710 + 0.010 = 0.720 GPU-h**, inside the cap.

## 4. Scenario-set provenance

`experiments/isaacgym_control/fourcube/episode_set_4cube.pkl`
stored sha256 `5962c3abb4367eaa78b44949c0340389203621a153f28fdcec0efa83f10d6b9d`,
init sha256 `fb0da9662fd7aeded87a4fc3ec67b02b`, goal sha256
`4a29109a362308f1d866cb725117cde2`, 96 episodes, **4 cubes**, generation seed
40404. `env.num_objects == 4` and `env.horizon == 100` asserted at runtime.

## 5. Run schedule

Predeclared balanced order: arm order alternates with `(repeat + seed_index)`
parity, so NFE2 leads in 5/9 (seed, repeat) pairs. Frozen before results and
followed exactly (see execution log). CRN base seed 20260905, identical to the
3-cube campaigns.

## 6-8. Per-seed results

| seed | p(NFE2) | p(NFE4) | Δ (4−2) | 95% CI (hierarchical) |
|--:|--:|--:|--:|---|
| 42 | 0.6389 | 0.7118 | **+7.30 pp** | [−1.7, +16.3] |
| 43 | 0.6424 | 0.6771 | **+3.45 pp** | [−6.2, +13.2] |
| 44 | 0.6910 | 0.7188 | **+2.85 pp** | [−6.3, +12.2] |

## 9. Three-seed summary

**Δ = +7.30 / +3.45 / +2.85 pp — signs +, +, + — mean +4.53 pp, SD 2.41.**

No individual CI excludes zero (consistent with a ~4.5 pp effect against this
evaluator's resolution), and none fits inside ±5 pp, so **neither a difference
nor equivalence is formally established at R=3**. The claim is directional across
three independently trained checkpoints.

## 10. Within-arm physics variability

| seed / arm | per-realization success | spread |
|---|---|--:|
| s42 NFE2 | 61.5 / 61.5 / 68.8 | 7.3 pp |
| s42 NFE4 | 75.0 / 70.8 / 67.7 | 7.3 pp |
| s43 NFE2 | 65.6 / 66.7 / 60.4 | 6.2 pp |
| s43 NFE4 | 69.8 / 68.8 / 64.6 | 5.2 pp |
| s44 NFE2 | 64.6 / 67.7 / 75.0 | **10.4 pp** |
| s44 NFE4 | 70.8 / 74.0 / 70.8 | 3.1 pp |

Mean within-arm spread **6.6 pp** — the same policy on the same scenarios.

## 11. Physics-sensitive episodes

Mean **50.5 / 96 = 52.6%** of scenarios have 0 < p_i < 1 (vs **41%** on 3-cube).
Robust failures are now non-zero (5-9 per arm), unlike 3-cube where 0/96 failed
in all realizations.

## 12-14. Reconstructed single-realization views

| seed | R=3 Δ | single-realization Δ (9 views) | sign flips | \|diff\|>5 pp | range |
|--:|--:|---|--:|--:|--:|
| 42 | +7.3 | 13.5, 9.4, 6.2, 13.5, 9.4, 6.2, 6.2, 2.1, **−1.0** | 1/9 | 4/9 | 14.6 pp |
| 43 | +3.5 | 4.2, 3.1, **−1.0**, 3.1, 2.1, **−2.1**, 9.4, 8.3, 4.2 | 2/9 | 2/9 | 11.5 pp |
| 44 | +2.8 | 6.2, 9.4, 6.2, 3.1, 6.2, 3.1, **−4.2**, **−1.0**, **−4.2** | 3/9 | 3/9 | 13.5 pp |

**Sign disagreement 6/27 = 22%. Magnitude disagreement >5 pp: 9/27 = 33%.**

These are reconstructed single-realization *views nested inside* the calibrated
experiment — **not 27 independent experiments**. Combined with 3-cube:
**26/108 (24%) of reconstructed single-realization views disagree in sign** with
the corresponding multi-realization estimate.

## 15. 3-cube vs 4-cube evaluator noise

| | 3-cube | 4-cube |
|---|--:|--:|
| within-arm spread | 10.4 pp (R=8, same arm) | 6.6 pp (R=3, mean over 6 arms) |
| physics-sensitive episodes | 41% | **53%** |
| sign disagreement | 25% (81 views) | 22% (27 views) |
| robust failures | 0/96 | 5-9/96 |

**Evaluator instability persists at higher compositional load and does not
diminish.** The sensitive-episode fraction rises (41% → 53%); the sign-flip rate
is essentially unchanged (25% → 22%). The spread figures are not directly
comparable (R=8 same-arm vs R=3 across arms) and are reported as such.

## 16. Contact localization

Physics-sensitive episodes have **mean max_obj_dist 0.1041 vs 0.0214** for
robust-success episodes — a 4.9× separation reproducing the 3-cube bifurcation
(0.097 vs 0.017). Sensitive episodes also contact more cubes (3.94 vs 3.69) and
make first contact later (0.551 vs 0.167). **The same contact-triggered
qualitative bifurcation is present at 4 cubes.**

## 17. Does the NFE2 operating point transfer? **NO — it does not**

| task | Δ (NFE4 − NFE2) | signs |
|---|--:|---|
| 3-cube | −0.69 / −1.04 / +1.74, **mean ≈ 0.00** | −, −, + |
| **4-cube (H=100)** | +7.30 / +3.45 / +2.85, **mean +4.53** | **+, +, +** |

This is protocol outcome **B**: *increased compositional load may increase the
behavioral value of inference compute.* On 3 cubes NFE2 matched NFE4; on the
fixed-horizon 4-cube stress test NFE4 leads on all three checkpoints by ~4.5 pp.

### The confound I must state

4-cube success is **65.7% / 70.3%** versus ~85-89% on 3-cube. The task sits
15-20 pp further from ceiling, so there is simply more room for arms to differ.
**A compression of differences near ceiling is a live alternative explanation for
the 3-cube null**, and this experiment cannot separate it from a genuine
compositional-compute interaction. Any paper claim must carry this caveat.

Wording: this is a **fixed-horizon zero-shot policy-compositional stress test**
(policy trained on 3-cube data; DLP saw more objects; H=100 vs native 150). Its
raw success rate is **not** canonical native-horizon 4-cube performance.

## 18. Does compositional load change evaluator instability?

Sensitive episodes **41% → 53%**; sign disagreement **25% → 22%** (unchanged
within noise). Instability **persists and its scenario footprint grows**; the
rate at which it corrupts conclusions does not fall. For the D2 paper this is the
important result — the problem is not confined to the canonical task.

## 19. Is 5-cube justified? **NO**

§14 permits 5-cube only if a specific question demands it. Neither trigger fires
cleanly:

- *Evaluator instability trend*: sign-flip rate is flat (25% → 22%), so a third
  point would not establish a trend.
- *Compositional-compute interaction*: the 4-cube effect (+4.5 pp) has no CI
  excluding zero, and the **ceiling confound** above means a 5-cube point would
  inherit the same ambiguity at 1.27 GPU-h. It cannot resolve the confound —
  only a ceiling-matched design could, which is a different experiment.

**Recommendation: do not run 5-cube.**

## 20. Revised paper verdict

**RSS: PLAUSIBLE-to-STRONG (unchanged, slightly strengthened).** The evaluation
contribution now spans two task regimes with a consistent ~24% sign-disagreement
rate and a reproduced contact bifurcation. The compositional result adds a
concrete consequence: *the inference budget you would deploy depends on a
comparison the conventional evaluator cannot reliably make.*

**CoRL: PLAUSIBLE (unchanged).**
**NeurIPS / ICLR: WEAK (unchanged)** — still no method or new statistics.

## 21. ONE next action

**Write the paper. Run no further experiments.**

Every must-have result is in hand: evaluator calibration (3-cube R=8), resolution
curves, 24% sign disagreement across two task regimes, contact-triggered
bifurcation reproduced at 4 cubes, the deterministic PushT contrast, the external
Diffusion Policy Pareto result, and the corrected low-NFE conclusions. The
remaining risk is expository, not experimental — and the strongest next
investment is drafting §2 (thesis) and Figure 3 (sign instability), not more GPU.

**HARD STOP.**
