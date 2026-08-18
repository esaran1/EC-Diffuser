# Protocol: hard-task causal diagnosis (PREDECLARED, NOT RUN)

Status: **predeclared, awaiting GPU availability and review.** No new method
is proposed here — this protocol only separates causes.

## 1. Starting point: a verified mechanism

`experiments/goal_horizon_verification.md` established, and
`experiments/scripts/verify_goal_horizon.py` reproduces:

- The 99.2% figure is **correct** (reproduced to every digit) but is the
  **arithmetic consequence** of uniform within-episode goal sampling over
  1001-step episodes with H=5 (analytic 0.99250 vs empirical 0.99233). It is
  not a corrupted adapter.
- The consequence that matters is measured: the conditioning slot demands a
  displacement **2.6×** larger than what the 5-step window can traverse
  (‖o[t]−goal‖ = 4.64 vs ‖o[t]−o[t+4]‖ = 1.82). Median goal distance is
  **187–190 steps**.
- 3-cube PushCube has ratio **1.1×** (fixed task goal, 100-step episodes), so
  the two tasks are structurally different at the same H=5.

So on OGBench puzzle, the model is trained to hit a target it **cannot reach
within the horizon**, and that same unreachable state is imposed as a hard
conditioning constraint at sampling time. Any claim about the generative
objective on this task is confounded by that fact.

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

This single experiment separates the *verified mechanism* from the objective
and requires no retraining.

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

D1+D2 together cost **~1 GPU-h and require no training**, and may resolve the
question outright. They are the correct first spend.

## 6. Explicitly excluded

No new method. No hyperparameter tuning. No iMF arm (cancelled). No
architecture search. No claim of a root cause without its ablation.
