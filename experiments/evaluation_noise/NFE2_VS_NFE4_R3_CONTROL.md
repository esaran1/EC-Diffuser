# NFE2 vs NFE4 under the calibrated R=3 protocol

**No training. No policy change. No physics tuning. No Gaussian. No loss work.**
Final Flow-NFE control experiment for this phase.

Scripts: `crn_control.py`, `analyze_nfe2_nfe4.py`
Data: `results/r0_s{42,43,44}_nfe{2,4}_n24rep{1,2,3}.json`, `nfe2_nfe4_analysis.json`
Schedule: `run_schedule_nfe2.json`

**HEADLINE: Classification B — a small NFE4 advantage that is statistically
indistinguishable from zero and, at the three-seed level, exactly zero
(mean −0.00003, SD 0.0151, signs −/−/+). Combined with the NFE1 result, 100% of
the measured NFE1→NFE4 gain is already realized by NFE2. The deployment
recommendation is NFE2, at 2.11× lower planner latency. Formal ±5 pp equivalence
is NOT claimed — no interval fits inside the band.**

---

## 1-2. Compute: measured projection vs actual

| | value |
|---|--:|
| measured NFE2 realization (96 eps) | **122 s** |
| measured NFE4 realization (96 eps) | **133 s** |
| raw projection (18 runs) | 0.638 GPU-h |
| with 15% overhead | **0.73 GPU-h** (gate: 0.80) |
| **actual** | **2739 s = 0.761 GPU-h** |

4% over the overhead-adjusted projection, still under the 0.80 gate. Two seed-44
NFE4 runs ran long (203 s, 235 s) — the only anomaly, and it affects wall time
only, not results.

**Calls/plan verified exactly on all 18 runs: 2.0 and 4.0.** No
`CALL_COUNT_MISMATCH`.

## 3. Frozen schedule

`run_schedule_nfe2.json`, **sha256
`49800b1f00d59d00f793e7c3f0bffc1b94c60c97401c8b3453f88c75fa360634`**, written
before launch and unmodified. Arm order alternates with `(repeat + seed_index)`
parity; **NFE2 runs first in 5 of 9** (seed, repeat) pairs. Executed order
matches exactly.

**GPU contention note:** launch was deferred until an unrelated
`evetac-streaming` RVT validation job (PID 3652358, 1496 MiB, 96% util) finished.
The GPU was clean (208 MiB, 0%) before the throughput probe and the main run, so
neither timing nor either arm was contended.

## 4. Provenance

| checkpoint | sha256 (first 40) | internal step |
|---|---|--:|
| seed 42 | `861dc34434474455a25dc3a15ea4e1754066202d` | 499000 |
| seed 43 | `c8e00eadfed9b8a0b54c5423864457e5330434b7` | 499000 |
| seed 44 | `c2c13f557aca7cf0eeb29b7baa572cf62a002702` | 499000 |

Episode set `replicate0_n96`: stored sha `35144910b1471b7b…`, file sha
`c760dd58d4cde41c…`, init `b5b16bcda726b76b…`, goal `7875ce901889dc54…`,
96 episodes / 3 cubes / horizon 100. CRN base seed 20260905, evaluator unmodified.

## 5-8. Per-seed results

| seed | p(NFE2) | p(NFE4) | **Δ = p4 − p2** | 95% CI | effective counts |
|---|--:|--:|--:|---|---|
| 42 | 0.8681 | 0.8611 | **−0.0069** | [−0.0764, +0.0625] | 83.3/96 → 82.7/96 |
| 43 | 0.8299 | 0.8194 | **−0.0104** | [−0.0903, +0.0694] | 79.7/96 → 78.7/96 |
| 44 | 0.8819 | 0.8993 | **+0.0174** | [−0.0486, +0.0833] | 84.7/96 → 86.3/96 |

**Three-seed summary:** deltas −0.0069 / −0.0104 / +0.0174, signs **−, −, +**,
**mean −0.00003**, SD 0.0151.

Two of three checkpoints nominally favour **NFE2**. The mean is within 0.003 pp
of exactly zero.

## 9. Hierarchical uncertainty

Identical estimator to the NFE1 study, unchanged: bootstrap **episodes** at the
top level (96 units), then resample the R=3 realizations **within** each drawn
episode. Physics realizations not paired across arms.

## 10. ±5 pp equivalence assessment — **NOT satisfied**

| seed | CI | width | share of CI inside ±5 pp | fully inside? |
|---|---|--:|--:|---|
| 42 | [−0.0764, +0.0625] | 13.9 pp | 72% | **No** |
| 43 | [−0.0903, +0.0694] | 16.0 pp | 63% | **No** |
| 44 | [−0.0486, +0.0833] | 13.2 pp | 75% | **No** |

Per §11, "no significant difference" is **not** equivalence. No interval is
contained in [−0.05, +0.05], so the correct statement is **"no reliable
difference detected"**, not "equivalent". I am explicitly not forcing an
equivalence result because NFE2 is computationally attractive.

That said, the point estimates cluster tightly at zero (|Δ| ≤ 1.7 pp on every
seed) and the interval width is set by the evaluator's R=3 resolution (~4.4 pp
half-width), not by any observed effect.

## 11. Episode-level Δᵢ distribution

| seed | Δᵢ > 0 | Δᵢ = 0 | Δᵢ < 0 | mean \|Δᵢ\| when nonzero |
|---|--:|--:|--:|--:|
| 42 | 18 | 57 | **21** | 0.410 |
| 43 | 22 | 51 | **23** | 0.452 |
| 44 | **18** | 65 | 13 | 0.441 |

Nearly balanced in both directions — a sharp contrast with NFE1-vs-NFE4, where
positives outnumbered negatives 3.0×, 1.6× and 2.0×.

## 12. Physics-sensitive episode analysis

| seed | NFE2 robust+/robust−/sensitive | NFE4 robust+/robust−/sensitive |
|---|---|---|
| 42 | 70 / 3 / **23** | 63 / 0 / **33** |
| 43 | 59 / 0 / **37** | 65 / 6 / **25** |
| 44 | 69 / 1 / **26** | 75 / 1 / **20** |

| seed | Δ on sensitive-either | Δ on both-robust |
|---|--:|--:|
| 42 | −0.0152 | +0.0000 |
| 43 | +0.0000 | −0.0217 |
| 44 | +0.0721 | −0.0169 |

**This answers the §13 question directly: NFE2 already stabilises the marginal
episodes.** In the NFE1→NFE4 comparison, NFE4 converted sensitive episodes into
robust successes on all three seeds (36→25, 38→29, 27→17) with Δ of +0.155,
+0.069, +0.135 there. Going 2→4 shows no such consistent conversion — seed 42
actually has *more* sensitive episodes at NFE4, and the sensitive-episode Δ is
inconsistent in sign.

## 13. Secondary placement/contact metrics

| metric | s42 | s43 | s44 | mean |
|---|--:|--:|--:|--:|
| goal_success_frac | +0.0012 | −0.0035 | +0.0012 | −0.0004 |
| cubes_placed | +0.0035 | −0.0104 | +0.0035 | −0.0012 |
| avg_obj_dist | −0.0005 | +0.0006 | −0.0022 | −0.0007 |
| max_obj_dist | +0.0013 | +0.0035 | −0.0096 | −0.0016 |
| n_contacted | +0.0174 | +0.0035 | −0.0069 | +0.0046 |
| cubes_farther | +0.0278 | +0.0313 | −0.0035 | +0.0185 |

**Every metric is ~zero and inconsistent in sign.** Compare NFE1→NFE4, where
`cubes_placed` improved +0.089 and `max_obj_dist` fell −0.019 consistently. The
§12 question — "does the effect again come from better placement after contact?"
— has no effect to explain here.

## 14. Calibrated boundary table (R=3 only; no R=1 values mixed in)

| seed | **NFE4 − NFE1** | **NFE4 − NFE2** | derived 1→2 |
|---|---|---|--:|
| 42 | **+0.0903** [+0.024, +0.160] | −0.0069 [−0.076, +0.062] | +0.0972 |
| 43 | +0.0278 [−0.056, +0.108] | −0.0104 [−0.090, +0.069] | +0.0382 |
| 44 | +0.0625 [−0.007, +0.132] | +0.0174 [−0.049, +0.083] | +0.0451 |
| **mean** | **+0.0602** (SD 0.0313) | **−0.0000** (SD 0.0151) | **+0.0602** |

**§15 caveat, applied:** the 1→2 column is an **indirect derived contrast**
(Δ(1→4) − Δ(2→4)) computed from **two separate experiments with independent
physics realizations**. NFE1 and NFE2 were never evaluated against each other in
a paired design. **No paired CI is attached, and none is valid.**

## 15. Where the major low-NFE improvement occurs

**Entirely in the 1 → 2 transition.** The derived 1→2 contrast (+0.060) accounts
for **100%** of the measured 1→4 gain (+0.060), and the directly measured 2→4
step contributes **−0.00003**.

The second network evaluation carries the benefit; the third and fourth add
nothing measurable.

## 16. Deployment recommendation: **NFE2**

Per the §16 decision rule: NFE4−NFE2 is far below 5 pp (|Δ| ≤ 1.7 pp per seed,
mean ≈ 0) and nominally favours NFE2 on 2 of 3 checkpoints, so **NFE2 is the
recommended operating point**, halving denoiser evaluations versus NFE4 at no
detectable behavioural cost.

Qualifier: this rests on "no reliable difference detected" rather than
demonstrated equivalence, and the evaluator cannot exclude an effect up to
~±5-9 pp per seed. A deployment that cannot tolerate a possible few-point
regression should stay at NFE4, whose cost is still modest.

## 17. Exact inference-call and latency advantage

**Planner-only latency** (measured by the evaluator's internal timer, excluding
simulator time, warmup-corrected):

| arm | ms/plan | denoiser calls/plan |
|---|--:|--:|
| NFE2 | **38.47** | **2.0** |
| NFE4 | 81.07 | 4.0 |

**2.0× fewer model evaluations, 2.11× lower planner latency.** (Whole-run wall
time — 122 s vs 133 s — is simulator-dominated and is deliberately *not* used to
characterise inference speed, per §17.)

## 18. Classification: **B — SMALL NFE4 ADVANTAGE**

Strictly, "small" here means *indistinguishable from zero and nominally negative*.

Not **A**: no meaningful NFE4 advantage; mean is −0.00003.
Not **C**: no CI is contained in ±5 pp, so practical equivalence is unsupported
by the calibrated uncertainty (§11), despite point estimates at zero.
Not **D**: the spread (−1.0 to +1.7 pp) is far smaller than the CI width, so
there is no evidence of checkpoint heterogeneity.
Not **E**: R=3 was sufficient to bound the effect near zero and to produce a
clear contrast against the NFE1 boundary. **No R increase is warranted or taken.**

**B** is chosen over C only on the formal equivalence standard; the practical
reading is that 2 and 4 steps behave the same on this task.

## 19. Strongest scientifically justified low-NFE claim

> On the canonical 3-cube PushCube task, under an evaluator whose noise floor has
> been measured and with 3 physics realizations per episode, **two Flow network
> evaluations retain four-step closed-loop performance** (three-seed mean
> difference −0.00003, |Δ| ≤ 1.7 pp per checkpoint) **while halving denoiser
> calls and reducing planner latency 2.11×**. Reducing further to a single
> evaluation costs a directionally replicated ~6 pp (all three checkpoints).
> The useful minimum inference budget for this task is **2**.

## 20. Claims replaced

- **"Control saturates at NFE2"** — was withdrawn as R=1-based; now **restored
  on calibrated evidence**, in the specific and narrower form above.
- **"Minimum sufficient NFE is 2"** — the original R=1 claim; now supported by
  calibrated measurement at both boundaries (1→2 carries the gain, 2→4 adds
  nothing).
- **"NFE4 is the recommended operating point"** (from the NFE1-vs-NFE4 report) —
  **superseded**: NFE4 is not measurably better than NFE2, so NFE2 is preferred
  on cost.
- **Still withdrawn / not restored:** "peaks at NFE8", "declines at NFE16" — both
  rest on R=1 data and were not re-measured.
- **Still historical:** "Flow NFE2 matches Gaussian@100 at 53× lower latency" —
  the latency ratio stands; the control half remains one-checkpoint, R=1, and was
  not re-run (§19).
- **Unchanged:** NFE1-vs-NFE4 stays Classification B (directional, ~6 pp, only
  seed 42's CI excludes zero). No "Regime A" language.

## 21. Is any further Flow-NFE control experiment necessary?

**No.** Both boundaries around the operating point are now measured under the
calibrated protocol: 1→2 carries the entire benefit, 2→4 adds nothing. NFE8 and
NFE16 sit above a boundary already shown to be flat, and the earlier NFE4-vs-NFE32
study found no reliable difference either — so more steps are not a live question.

The one genuinely open comparison is **Gaussian vs Flow under the calibrated
evaluator**, which is a different experiment with its own limitations (single
Gaussian checkpoint), and is explicitly out of scope here.

## 22. Loss branch

**NOT MOTIVATED.** Unchanged and closed. Nothing in this result reopens it.

---

## HARD STOP OBSERVED

No R increase. No NFE8. No Gaussian. No training. No loss work. No new method.
