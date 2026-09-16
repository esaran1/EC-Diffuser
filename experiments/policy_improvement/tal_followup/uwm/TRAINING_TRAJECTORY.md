# Training-loss trajectories — with the caveat that makes them readable

**These two numbers are NOT directly comparable as quality.** Each arm computes
its loss over its **own** time distribution: the UWM arm over independent
`(t_s, t_a)`, the control over the diagonal. A lower number in one column does
not mean a better model. What they show is each arm optimizing its own
objective.

| step | UWM flow | UWM action | CTL flow | CTL action |
|--:|--:|--:|--:|--:|
| 0 | **0.4595** | 0.3905 | **0.2339** | 0.0978 |
| 500 | 0.2491 | 0.2608 | 0.2236 | 0.0434 |
| 1000 | 0.2389 | 0.1371 | 0.2301 | 0.0503 |
| 1500 | 0.2349 | 0.1238 | 0.2341 | 0.0563 |
| 1900 | 0.2434 | 0.1510 | 0.2367 | 0.0937 |

## What the step-0 gap means

At step 0 both arms hold **byte-identical canonical weights**. The UWM arm's
loss is nearly **2× higher** (0.4595 vs 0.2339) purely because it is immediately
being evaluated on off-diagonal times the canonical model has never seen.

That gap is a direct, independent measurement of the phenomenon the closed
warm-start branch identified: **a diagonal-trained model is genuinely
out-of-distribution off the diagonal.** Here it is quantified at the very first
step, before any optimization, with no confound — the weights are identical and
only the time distribution differs.

## What the convergence means

Over 1900 steps the UWM arm's flow loss falls from 0.4595 to ~0.2434, converging
toward the control's ~0.2367, and its action loss falls 0.39 → ~0.13–0.15 while
the control's stays roughly flat around 0.05–0.09.

The natural reading is that the UWM arm is **learning the off-diagonal regime** —
it is closing most of the gap that the off-diagonal distribution opened.

## Why this is NOT yet the adaptation verdict

Three reasons, all of which the field diagnostic addresses and this does not:

1. **Different distributions.** The UWM arm's remaining loss is an average over
   the whole 2D square; the control's is an average over the diagonal alone.
   Convergence of two differently-defined averages is suggestive, not decisive.
2. **No matched inputs.** These are training losses on different sampled times,
   not the same states and noise evaluated at the same grid cells.
3. **Single seed, no resolution model.** Training loss has no established
   relationship to closed-loop control here.

The registered verdict comes from `field_diagnostics.py` on the **same** 5×5
grid with **identical** replay states, noise and conditioning for both arms
(criterion in `ADAPTATION_CRITERION.md`), and any control claim requires
closed-loop R=3 (§48).
