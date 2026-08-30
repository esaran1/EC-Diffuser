# Mechanism diagnostics: why does NFE2 beat NFE1?

**No training. No simulator.** Arm-neutral recorded replay (recorded current,
goal, next-state and demonstration action; nothing model-authored). 96
conditions x 4 noises x 3 checkpoints. Runtime 13 s.

Artifacts: `mechanism_diagnostics.py`, `mechanism_diagnostics.json`
Evidence level: **LEVEL 1** (offline, in-distribution replay).

---

## 1. Second-step decomposition (§14)

Variants of the second Euler half-step, all sharing step 1:

| variant | action err s42 | s43 | s44 |
|---|--:|--:|--:|
| NFE1 (h=1) | 0.01668 | 0.01742 | 0.01944 |
| **A full 2-step** | 0.01785 | 0.01605 | 0.01761 |
| **B action-coords only** | 0.01785 | 0.01605 | 0.01761 |
| C state-coords only | 0.65265 | 0.65210 | 0.65292 |
| D refine state, then recompute action | 0.02356 | 0.02093 | 0.02028 |

### CORRECTION — B ≡ A is arithmetic, not a mechanism

`|B − A| = 0.00e+00` on all three seeds. I initially read this as "the action
improvement comes purely from action self-refinement." **That inference was
wrong and is withdrawn.** B and A consume the *same* velocity `v1` from a single
model call and differ only in which coordinates they write; the action
coordinates must therefore agree identically. This is a tautology of the probe
design, not evidence about the model.

C is likewise uninformative for the action: it never updates action coordinates,
so its "action error" is just the un-updated half-step action.

D is the only informative variant, and it is **worse** than A on all three seeds
(+0.0057 / +0.0049 / +0.0027) — recomputing the action from a refined state costs
accuracy relative to simply completing the Euler step.

## 2. The informative probe: does the action velocity read the state?

Perturb **only** the state coordinates at t=0 and measure the induced change in
the **action** velocity component (seed 42):

| state perturbation | Δ action-velocity | as % of ‖v_a‖ |
|--:|--:|--:|
| 0.01 | 0.0001 | **0.00%** |
| 0.10 | 0.0010 | **0.02%** |
| 0.50 | 0.0099 | **0.24%** |
| 1.00 | 0.0316 | **0.78%** |

**The action velocity is nearly invariant to the (noisy, partially-integrated)
state coordinates.** Even a unit-scale perturbation — comparable to the noise
level at t=0 — moves the action velocity by under 1%.

Note what this does and does not say. The action token attends to the *particle
tokens*, and those are conditioned on the **recorded current observation and
goal** (which are clamped by conditioning at t=0 and t=4 and never perturbed
here). So the action prediction is driven by the *conditioning*, not by the
in-flight state sample. That is consistent with §1.7's finding that state
prediction quality is not proven causally necessary for control.

## 3. Projected 1-step vs 2-step residual

Model-space displacement between the NFE1 and NFE2 endpoints:

| seed | action block | state block | ratio |
|---|--:|--:|--:|
| 42 | 0.0627 | 4.5738 | 72.9× |
| 43 | 0.0638 | 4.5813 | 71.8× |
| 44 | 0.0508 | 4.5831 | 90.2× |

The second call moves the **state** enormously (state error −48.8% / −49.0% /
−48.7%) and the **action** barely at all. The two blocks are on completely
different scales of change per unit of integration.

## 4. Implication for method design

The mechanistically supported reading is:

> The action projection is already close to converged after one Euler step, is
> driven primarily by the clamped conditioning rather than by the in-flight state
> sample, and moves ~70-90× less than the state block during the second step.

**M1/M2/M3 verdict (§14): not resolved by this probe.** The decomposition cannot
separate them because of the shared-`v1` tautology above. What *is* established
is the near-invariance of action velocity to state coordinates, plus the
residual-scale asymmetry.

**Caution for the candidate method.** These results cut *against* the naive
motivation for an action-only shortcut objective: if the action is already nearly
converged at NFE1 offline (0.0167 vs 0.0179 — NFE1 is actually *better* on seed
42), then the ~6 pp closed-loop NFE1 penalty (§1.4) is **not** explained by
action-endpoint error on recorded replay. Something else accounts for it.

That is a genuine tension in the evidence and must be resolved before committing
to a method. Candidate explanations to test:
1. the NFE1 control penalty is driven by the *state* branch after all (via
   closed-loop compounding), not by single-step action accuracy;
2. it is driven by action *distribution* properties (multimodality/variance) that
   L2-to-demonstration cannot see;
3. it is an artifact within the evaluator's resolution (the +6 pp had only 1 of 3
   CIs excluding zero).

**Explicitly: this diagnostic does not yet support GO on an action-projected
shortcut method.**
