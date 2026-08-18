# Protocol: hard-task causal diagnosis (PREDECLARED, NOT RUN)

Status: **predeclared, awaiting GPU availability and review.** No new method
is proposed here — this protocol only separates causes.

## 1. Starting point: a verified statistic, an UNTESTED hypothesis

`experiments/goal_horizon_verification.md` established, and
`experiments/scripts/verify_goal_horizon.py` reproduces:

**Verified:**
- The 99.2% figure is numerically correct, reproduced to every digit.
- It is the **arithmetic consequence** of uniform within-episode goal sampling
  over 1001-step episodes with H=5 (analytic 0.99250 vs empirical 0.99233) --
  not a corrupted adapter, not a leak.
- **87.97%** of training windows contain **zero button-state changes**; goals
  differ from the current state by **6.64 of 16 buttons** on average.

**Withdrawn:** the earlier "OGBench 2.6x vs 3-cube 1.1x" displacement contrast.
Those metrics were not comparable (OGBench mixes 32 binary dims with
continuous state across a ~100x std range; the 3-cube figure used unnormalized
DLP latents). On comparable metrics the ordering reverses: OGBench is 1.68x
standardized, 3-cube is 3.66x in metres.

**Not established:** that any of this is a bottleneck. A goal beyond the H=5
window is **normal and expected** in goal-conditioned control -- hindsight
relabeling trains on distant goals by design, and the policy replans every
step. Distance-beyond-horizon is not itself evidence of a flaw.

The open question this protocol tests is therefore narrow:

> Does prediction/control quality actually **degrade with goal offset** (D1),
> and does changing the relabeling rule (D3) or the horizon (D4) actually
> **improve performance**? Only affirmative answers justify the word
> "bottleneck".

## 2. Competing hypotheses

| # | Hypothesis | Claim |
|---|---|---|
| H-ALG | Generative objective | The low-NFE objective cannot fit the conditional action distribution |
| H-TEMP | Temporal / horizon | H=5 is too short relative to goal distance; the target is unreachable in-window |
| H-DATA | Data / goal stitching | Uniform relabeling produces mostly-unreachable goals; coverage of *near* goals is too sparse to learn from |
| H-REP | Representation | Flat 83-D state lacks the entity structure needed for combinatorial composition |
| H-CTRL | Control | Contact/action scaling prevents execution regardless of the plan |

H-CTRL is already **largely excluded** by the v3 diagnostic: raising
`action_weight` to 10 cut action clipping from 0.829 to 0.183 without
improving task completion, and every corrected episode produced button
transitions. Retained only as a control.

## 3. Ordered ablations — cheapest discriminating first

Each changes **one** factor. All use the existing 5,000-step checkpoints
unless stated. Ordering is by information per GPU-hour.

### D1 — Goal-distance-conditioned evaluation (NO TRAINING, ~0.5 h)

**Cheapest and most discriminating.** Partition *evaluation* episodes by goal
distance and re-evaluate the existing checkpoint. Also compute offline
one-step prediction error on held-out windows bucketed by goal offset
(4–10, 10–50, 50–200, 200+ steps).

- *If true (H-TEMP/H-DATA):* error and task progress are near-normal for
  offset ≤ 10 and degrade monotonically with offset. The model is fine; the
  conditioning is unreachable.
- *If false:* error is flat across offset buckets ⇒ the horizon mismatch is
  not the operative cause, and H-ALG/H-REP move up.

This single experiment separates the goal-offset hypothesis from the
generative objective and requires no retraining.

### D2 — NFE sweep on the existing checkpoint (NO TRAINING, ~0.5 h)

Evaluate the same checkpoint at NFE 1/2/4/8/16.

- *If progress improves materially with NFE:* integration error is implicated
  (H-ALG).
- *If flat:* the bottleneck is not the number of solver steps. Given D1, this
  is the expected outcome, and a flat curve is strong evidence against H-ALG.

### D3 — Capped-offset goal relabeling (TRAINING, ~1.3 h)

Retrain Flow with goals sampled from `[t+H-1, t+K]`, `K = 20`, instead of the
full episode. One seed, 50k steps. Everything else identical.

- *If true (H-DATA/H-TEMP):* validation error drops sharply and task progress
  improves, because targets become reachable.
- *If false:* no improvement ⇒ unreachable goals were not the binding
  constraint.

This is the direct experimental test of the §1 mechanism.

### D4 — Horizon extension (TRAINING, ~2–4 h)

Retrain Flow at H=20 (and optionally H=50) with the original goal policy, at
matched examples-seen.

- *If true (H-TEMP):* longer horizon closes the reachability gap and improves
  progress, even with unchanged goal sampling.
- *If false:* horizon alone is insufficient; the issue is goal *selection*
  (D3), not chunk length.

D3 and D4 are deliberately paired: D3 changes *which goal*, D4 changes *how
far the model can travel*. Both address H-TEMP/H-DATA from opposite sides,
and their combination identifies which is binding.

### D5 — Representation contrast (TRAINING, ~2 h; only if D1–D4 leave H-REP open)

Compare the flat sequence backbone against the entity-structured PINT
backbone on the *same* task/data/budget. This is only meaningful on a task
with entity structure exposed; on OGBench puzzle the 16 button states are
already an entity-like factorization worth exploiting.

- *If true (H-REP):* entity-structured conditioning improves compositional
  progress at matched parameters and budget.
- *If false:* representation is not the bottleneck here.

D5 is the bridge to the surviving novelty gap in the literature audit and
should be run **last**, only if the cheaper ablations have not explained the
failure.

## 4. Preregistered decision rule

- D1 degrading with offset **and** D3 improving ⇒ **H-DATA/H-TEMP confirmed**;
  the OGBench puzzle failure is a goal-relabeling/horizon artifact, and the
  task must be re-specified before it can test generative objectives at all.
- D1 flat **and** D2 improving with NFE ⇒ **H-ALG**.
- D1 flat, D2 flat, D3/D4 flat, D5 improving ⇒ **H-REP**.
- Everything flat ⇒ report that the 5,000-step budget is the binding
  constraint and that no causal claim is supported at this budget.

No root cause will be asserted without the corresponding ablation.

## 5. Cost

| Step | Training? | GPU-h |
|---|---|--:|
| D1 goal-distance buckets | no | ~0.5 |
| D2 NFE sweep | no | ~0.5 |
| D3 capped relabeling | yes, 50k × 1 seed | ~1.3 |
| D4 horizon extension | yes, 50k × 1–2 configs | ~2–4 |
| D5 representation contrast | yes, 50k × 1 seed | ~2 |
| **Total** | | **~6.3–8.3** |

D1+D2 together cost **~1 GPU-h and require no training**. However, see §7:
they diagnose **puzzle-4x4 specifically** and are no longer recommended as the
first GPU spend. If cube-triple is adopted, run **D1-cube** instead, on
cube-triple checkpoints.

## 6. Explicitly excluded

No new method. No hyperparameter tuning. No iMF arm (cancelled). No
architecture search. No claim of a root cause without its ablation.

## 7. Transfer analysis: do D1/D2 inform cube-triple? — PARTIALLY

This section exists because D1/D2 were previously proposed as the first GPU
experiments. That recommendation is **withdrawn**; the analysis follows.

### What is shared

`OGBenchPuzzleWindowDataset._goal_index` implements uniform within-episode
relabeling over `[t+H-1, episode_end]`. A cube-triple adapter does not exist
yet (`benchmark_sequence.py` has only Puzzle, MimicGen, DexJoCo classes), but
if it is written by analogy — the natural and likely choice — it will reuse
`_BaseBenchmarkDataset._batch` and the same `_goal_index` rule, over episodes
of the same 1001-step length. In that case:

- the **goal-relabeling mechanism** is shared;
- the ~99% beyond-endpoint arithmetic is shared, because it depends only on
  episode length and H;
- **D1** (does quality degrade with goal offset?) therefore asks a question
  that transfers, and its methodology transfers directly.

### What does NOT transfer

The specific hypothesis motivating D1 does **not** carry over:

| | puzzle-4x4 | cube-triple |
|---|---|---|
| Goal structure | 16 discrete button toggles | continuous cube poses |
| "88% of windows have no button change" | the core observation | **no analogue** |
| Sparse discrete subgoal supervision | plausible bottleneck | not applicable |
| Combinatorial state space | 2^16 button configs | continuous rearrangement |

The 87.97%-zero-button-change finding is *specific to discrete toggles*. Cube
positions change continuously in essentially every window, so the
"supervision present in only 12% of windows" mechanism has no cube analogue.

**D2** (NFE sweep) transfers as a method but not as evidence: it would be run
on a puzzle checkpoint, and NFE sensitivity is task- and checkpoint-specific.

### Measured on the real cube data (2026-08-18, CPU only)

The cube datasets are now downloaded and their manifests/normalizers frozen,
so the transfer question was answered empirically rather than by assumption:

| Task | obs dim | standardized in-window | to-goal | ratio | median offset | beyond endpoint |
|---|--:|--:|--:|--:|--:|--:|
| puzzle-4x4 | 83 | 6.910 | 11.641 | 1.68x | 190 | 0.9923 |
| **cube-triple** | 46 | 2.489 | 6.661 | **2.68x** | 192 | 0.9925 |
| cube-double | 37 | 2.529 | 6.372 | 2.52x | 187 | 0.9930 |

Two consequences:

1. The goal-offset structure **does transfer**: cube episodes are also 1001
   steps, so the ~99.2% figure and the offset distribution are nearly
   identical. D1's question is well-posed on cube-triple.
2. The reachability gap is **larger** on cube-triple (2.68x) than on puzzle
   (1.68x) on the same standardized metric. If the goal-offset hypothesis has
   force anywhere, cube-triple is the stronger place to test it -- the opposite
   of what the puzzle-first ordering assumed.

What still does not transfer is the *discrete-subgoal* mechanism (the 88%
zero-button-change statistic); cube poses move continuously. So D1 should be
run on cube-triple with a continuous progress metric, not by porting the
button-Hamming analysis.

### Verdict

D1/D2 **as originally written** diagnose **puzzle-4x4 specifically**. They would tell us why our puzzle
pilots underperformed — a task we have now largely set aside in favour of
cube-triple, and one where a published one-step method already reaches 40.

Running them first would spend GPU-hours characterizing a task that is not
the intended next benchmark. **They are therefore demoted from first place.**

They become worthwhile in two cases only:
1. cube-triple is adopted *and* its adapter reuses the same `_goal_index`
   rule — then D1 should be run **on cube-triple**, not on puzzle; or
2. puzzle-4x4 is reinstated as a Tier-A task.

### Replacement: D1-cube

If cube-triple proceeds, the transferable experiment is **D1 restated on
cube-triple**: bucket held-out windows by goal offset and measure offline
prediction error and task progress per bucket. Same question, right task, and
it can run immediately after the cube arms train — reusing those checkpoints
at no extra training cost.
