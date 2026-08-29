# Enriched solver-bias audit: goal, current-state and action geometry

**No training. No loss change. No new solver. No control benchmark.**

Scripts: `enriched_multinoise.py`, `analyze_enriched.py`
Data: `enriched_endpoints.npz` (53 MB), `enriched_analysis.json`
Schema: `enriched_cache_SCHEMA.md`

**Answer: Classification D — ACTION/STATE DECOUPLING, but in the OPPOSITE
direction to the §18 hypothesis. The action prediction degrades ~5-6× MORE than
the state prediction, not less. Goal-directedness (A) and conservative bias (C)
are both effectively ruled out.**

## COMPUTE — budget overrun, reported

| run | outcome | cost |
|---|---|--:|
| attempt 1 | crashed in the save block after generating seed 42 | 0.14 GPU-h |
| attempt 2 | same crash (my first fix targeted the wrong key) | 0.14 GPU-h |
| attempt 3 | completed, 1533 s | 0.43 GPU-h |
| **total** | | **~0.71 GPU-h vs ~0.43 approved** |

The cause was a cache key (`noise_idx`) initialised but never written, so
`np.concatenate` received an empty list — *after* all GPU work. I have added a
pre-save guard that raises on any unpopulated key, and a static check that every
initialised key is written. **The overrun is mine and I am flagging it rather
than burying it.** Storage came in at 53 MB against a 5 GB limit.

---

## 1. Aggregate replication (§5 gate — PASSED)

| seed | E16 | E512 | paired Δ [95% CI] | best-of-8 E16/E512 | dispersion E16/E512 | medoid E16/E512 | E16 wins |
|---|--:|--:|---|--:|--:|--:|--:|
| 42 | 0.04499 | 0.04649 | **+0.00150** [+0.00110, +0.00188] | 0.03247 / 0.03358 | 0.03930 / 0.04294 | 0.03881 / 0.03993 | 65.5% |
| 43 | 0.04322 | 0.04426 | **+0.00104** [+0.00068, +0.00139] | 0.03090 / 0.03189 | 0.03805 / 0.04104 | 0.03768 / 0.03933 | 59.4% |
| 44 | 0.04361 | 0.04499 | **+0.00138** [+0.00102, +0.00173] | 0.03173 / 0.03237 | 0.03870 / 0.04207 | 0.03817 / 0.03878 | 66.0% |

Prior run: Δ = +0.00164 / +0.00122 / +0.00177, E16 winning 66/62/68%.
**Every sign, magnitude and ordering reproduces** (E512 worse on mean, best-of-8
and medoid; more dispersed; E16 favoured on ~60-66% of noises). Gate passed —
proceeding to the new geometry.

## 2. Cache schema and reproducibility

45 keys, full transition tensors at **all** timesteps in both model and data
space, plus conditioning, targets, mask, and per-condition noise hashes. See
`enriched_cache_SCHEMA.md`; artifact sha256
`82ad69c229a762aa3323adff222bf2b4b968834c8cd9bdc4da511f48ee89ef3c`.
All pairing below is **within this single run**.

## 3. Temporal semantics (verified from code and data)

- generated **t=0** conditioned = current; **t=4** conditioned = goal
- **scored imagination = generated t=1**; supervised target = **observed t=1**
- **action[t] drives t → t+1**, so `action[0]` produces the scored t=1 state.
  That is the action compared against the executed `real_action`.

## 4-6, 11. Distance table (permutation-invariant position chamfer)

Seed 42 (43 and 44 are within ~2% on every entry):

| source | → current | → observed t1 | → goal |
|---|--:|--:|--:|
| current state | 0.00000 | 0.07452 | 0.13590 |
| observed t1 state | 0.07452 | 0.00000 | 0.13573 |
| **Euler@16** | 0.07189 | **0.04499** | 0.14405 |
| **Euler@512** | 0.07271 | 0.04649 | **0.14343** |

Paired E512 − E16, all three seeds:

| reference | s42 | s43 | s44 | E512 closer |
|---|--:|--:|--:|--:|
| → current | **+0.00082** [+.00041,+.00124] | +0.00052 [+.00011,+.00094] | +0.00039 [−.00004,+.00082] | 42-45% |
| → observed t1 | **+0.00150** [+.00111,+.00189] | +0.00104 [+.00069,+.00140] | +0.00138 [+.00102,+.00173] | 34-41% |
| → goal | **−0.00061** [−.00113,−.00010] | −0.00048 [−.00104,+.00009] | −0.00029 [−.00081,+.00023] | 49-53% |

**E512 is farther from the observed future AND farther from the current state,
while being very slightly closer to the goal** — but the goal effect is ~2.5×
smaller than the future effect and its CI **includes zero on 2 of 3 seeds**.

## 7, 12-13. Normalized goal progress and the overshoot test

`progress = 1 − d(x, goal) / d(current, goal)`:

| seed | observed t1 | E16 | E512 | E512 − observed | E16 − observed |
|---|--:|--:|--:|--:|--:|
| 42 | −0.0113 | −0.0762 | −0.0704 | **−0.0591** | −0.0649 |
| 43 | −0.0176 | −0.0671 | −0.0620 | −0.0444 | −0.0495 |
| 44 | −0.0222 | −0.0733 | −0.0687 | −0.0465 | −0.0511 |

**No temporal overshoot.** Both arms show *negative* progress — generated t=1
states are **farther** from the goal than the current state is — and **both are
well below the observed t=1 state's progress**. E512 is marginally less negative
than E16 (+0.005), far too small to matter and in a regime where neither arm is
advancing toward the goal at all.

The demonstrated t=1 state itself barely progresses (−0.011 to −0.022), which is
expected for a one-step-ahead target: one env step covers little of a
100-step episode.

## 8. Evidence for/against temporal overshoot: **AGAINST**

E512 is not "racing ahead" to the goal. It does not pass the observed t=1 state
in goal progress; it does not even reach it. §13's proposed mechanism is not
supported.

## 9. Current-state / conservative-bias analysis: **AGAINST**

The §14 hypothesis required E16 to be closer to current **and** closer to the
observed future. E16 *is* closer to both — but this cannot be conservatism,
because **the observed t=1 state is 0.0745 from current while E16 is only 0.0719
from current**: E16 sits *nearer to current than the real future does*. Yet both
arms are essentially equidistant from current (difference ≤ 0.0008, ~1%), so
neither is meaningfully more conservative than the other. The +0.0008 gap is 5×
smaller than the future-error gap it would have to explain.

## 10. Goal-direction alignment

Direction cosines, E16-anchored (primary):

| seed | cos_future | cos_goal | cos_current | \|para\|/‖δ‖ (fut / goal / cur) |
|---|--:|--:|--:|---|
| 42 | +0.0283 | **+0.0016** | −0.0017 | 0.161 / 0.138 / 0.161 |
| 43 | +0.0268 | **+0.0116** | −0.0056 | 0.165 / 0.145 / 0.164 |
| 44 | +0.0451 | **+0.0073** | +0.0013 | 0.159 / 0.147 / 0.159 |

**cos_goal ≈ +0.007 — essentially zero.** The shift is not goal-directed. All
three references carry only ~14-17% of the displacement on-axis, confirming the
previous audit's finding that **~85% of the movement is orthogonal to every
semantic direction we can test.**

## 11. Assignment-anchor sensitivity

E512-anchored gives cos_future ≈ +0.234, cos_goal ≈ +0.139, cos_current ≈ +0.181
— all inflated, and inflated *uniformly*, because re-matching the target to E512
absorbs E512's own movement into the assignment. **The cosines remain
anchor-unstable and I again decline to interpret their magnitudes.** The
*ordering* (future > current > goal) is stable across anchors, and it is the
ordering — not the magnitude — that supports the conclusions above.

## 12. Demonstrated-action error, E16 vs E512

| seed | ‖a − demo‖ E16 | E512 | paired Δ [95% CI] | relative increase |
|---|--:|--:|---|--:|
| 42 | 0.12915 | 0.15167 | **+0.02252** [+0.02039, +0.02479] | **+17.4%** |
| 43 | 0.13244 | 0.15320 | **+0.02077** [+0.01871, +0.02283] | **+15.7%** |
| 44 | 0.12959 | 0.15104 | **+0.02145** [+0.01972, +0.02313] | **+16.6%** |

Action magnitudes are nearly identical (E16 0.8345 vs E512 0.8382 vs demo
0.8376), so this is a **direction/placement error, not a scale error**.

## 13. Normalized state vs action solver displacement

Per-coordinate RMS displacement divided by each space's own data scale:

| seed | state | action | ratio |
|---|--:|--:|--:|
| 42 | 0.1359 | 0.0424 | 3.20× |
| 43 | 0.1313 | 0.0404 | 3.25× |
| 44 | 0.1313 | 0.0384 | 3.42× |

The **state** representation moves ~3.3× more than the action under accurate
integration. But displacement is not error — and the error tells the opposite
story:

| seed | state error increase | action error increase | action / state |
|---|--:|--:|--:|
| 42 | +3.33% | **+17.43%** | **5.2×** |
| 43 | +2.40% | **+15.68%** | **6.5×** |
| 44 | +3.16% | **+16.55%** | **5.2×** |

## 14. Does action/state NFE sensitivity decouple? **YES — inverted**

§18 hypothesised that the action endpoint would change *very little* while the
state changed substantially, which would explain control saturating at low NFE.

**The data refute that.** On a like-for-like fractional basis the action
prediction degrades **5.2-6.5× more** than the state prediction when moving from
E16 to E512. The action channel is *more* sensitive to integration accuracy, not
less.

**This does NOT explain control-vs-imagination saturation — it deepens the
puzzle**, because control is known to saturate by NFE 2-4 while these action
predictions keep changing out to NFE 512. Two readings remain open, and this
experiment cannot separate them: (a) the closed-loop controller is robust to
per-step action error of this size, so a 16% degradation is simply absorbed; or
(b) the *demonstrated* action is not the right target for control quality. I am
not claiming either.

Per §16, only E16 and E512 were generated; no NFE 1/2/4/8 sweep was run.

## 15-16. Per-noise and per-seed

All effects hold on **3/3 seeds** with non-overlapping-of-zero CIs for the state
and action results. The goal-distance effect is the sole exception: CI includes
zero on 2 of 3 seeds, which is why it is reported as not supported.

## 17. Primary classification: **D — ACTION/STATE DECOUPLING (inverted)**

Chosen over the alternatives because:

- **A (goal-directed) — rejected.** cos_goal ≈ +0.007; goal-distance advantage is
  −0.0005 with CI spanning zero on 2/3 seeds, and 2.5× smaller than the
  future-distance penalty.
- **B (temporal overshoot) — rejected.** Both arms show *negative* goal progress
  and neither reaches the observed t=1 state's progress.
- **C (conservative E16) — rejected.** The two arms are within ~1% on distance to
  current, and E16 is already nearer to current than the true future is.
- **E (still orthogonal) — partially true and reported as the secondary finding:**
  ~85% of the displacement remains orthogonal to all three semantic axes.
- **D — supported and quantitatively the largest effect in the audit:** a clean,
  3/3-seed, 5-6× asymmetry between action and state degradation.

## 18. Strongest mechanism supported

**Accurate integration moves the generated state substantially (≈3.3× the action
displacement, ~85% of it orthogonal to any semantic target) yet costs only ~3% in
state accuracy, while costing ~16% in action accuracy against the demonstrated
action.** The solver-induced shift is therefore *disproportionately harmful in
the action subspace*, even though that subspace moves least.

The action channel is 3 of 483 coordinates and is conditioned at no timestep, so
it is the part of the transition tensor with the least constraint from the
conditioning — and it is where accurate integration diverges most from the
demonstration.

## 19. Strongest alternative still alive

**The demonstrated action may not be the correct target.** `real_action` is the
action the E16-noise-0 arm actually executed, so E16 has a structural advantage
in matching it: the environment was advanced by an E16 action, making E16's
action channel closer to the executed one *by construction*. **This is a genuine
confound in §12 and I am flagging it as such** — the +16% action gap should be
read as "E512 differs more from the E16-executed action", not necessarily as
"E512's action is worse". The state comparison does not share this confound
(both arms are scored against the same realised next state), which is why the
state result is the more trustworthy of the two.

## 20. Is "E16 imagines better" still defensible?

**Yes, but only in the narrow, literal sense** — and the enriched geometry now
supports the wording more strongly than before, because the two alternative
explanations that would have undermined it are ruled out:

> Euler@16 produces t=1 states closer to the single observed t=1 future than
> converged Euler@512 does, consistently across three seeds and eight noises.
> This is **not** because E512 pursues the conditioned goal instead (cos_goal ≈ 0,
> no goal-progress advantage), and **not** because E16 is more conservative
> (both arms are equidistant from the current state).

Still off-limits, unchanged: we have **one** observed future per condition, so
this cannot establish that E16's futures are objectively better, only that they
are closer to the single realised one.

## 21. Loss-hypothesis status: **LOWER PRIORITY** (unchanged)

Nothing here implicates the objective. The effect remains a sampler property.
The action/state asymmetry is a property of the discrete map, not of the loss
weighting — and note it runs *opposite* to the old dimensional-imbalance
intuition, which predicted the 3 action dims were under-served.

## 22. Exactly ONE next experiment

**Resolve the §19 confound by re-scoring actions against an arm-independent
target: advance the environment with a fixed, arm-neutral action and compare
both arms' action channels against that same executed action.**

Concretely: rerun the identical protocol but advance the env with the *observed
dataset action* for that state (or a held-out reference policy action) rather
than the E16 noise-0 action, so neither arm authored the trajectory. Everything
else — noise bank, seeds, conditions — unchanged. ~0.43 GPU-h, no training.

Rationale: §12 produced the largest effect in this audit (+16%, 3/3 seeds), and
§19 identifies a structural reason it may be inflated. That single confound is
now the main obstacle to interpreting the action result, and it is the difference
between "accurate integration hurts action prediction" — which would matter for
control — and a measurement artifact. It must be settled before the action
finding is used for anything.

---

## HARD STOP OBSERVED

No training. No loss change. No MeanFlow. No VP. No solver modification.
No new architecture.
