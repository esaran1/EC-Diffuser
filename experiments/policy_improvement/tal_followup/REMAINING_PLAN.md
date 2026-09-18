# Remaining registered work, and the order it runs in

State as of the session resume. **Spent: 1.0011 of 10.0 GPU-h.**

## Done

| stage | outcome |
|---|---|
| A — literature audit (UWM, registers, 2602.07322, MeanFlow) | complete, 4 documents |
| B — architecture audit | complete, traced from production code |
| C — direct-action *recovery* | **no prior artifact exists**; built instead |
| D — two-time implementation + gates | complete, all gates pass |
| D — pilot training, 2 arms × 5k | complete, divergence sentinel passed |
| E — 2D field + path diagnostics | complete, **ADAPTED** verdict |
| E — §23 extension, UWM arm 5k→10k | complete (`bc2027e4d112ac7f`) |
| F — harder task | **BLOCKED** structurally (~33 GPU-h) |
| G — register diagnostic | **NO ARTIFACT**; implementation pilot not justified |
| H — action dataset audit | bottleneck **NOT AUTHORIZED** (75.7% vs ≥99%) |

## In flight

| item | cost | note |
|---|--:|---|
| extension, control arm 5k→10k (rerun after session death) | ~0.33 | required so both arms keep identical optimizer budgets |

## Remaining, in priority order (§6)

| # | item | cost | rationale |
|--:|---|--:|---|
| 1 | re-run field + replay diagnostics on both **extended** arms | ~0.02 | does the adaptation continue past 5k? decides whether 10k changes the verdict |
| 2 | **closed-loop R=3**: CTL-diagonal vs UWM-diagonal | 1.327 | the decisive test (§48). Registered in `uwm/CLOSED_LOOP_PLAN.md` |
| 3 | Stage C: direct-action training (5k) | 0.33 | answers Tal's H1 directly |
| 4 | **closed-loop R=3**: direct-action | 0.664 | H1's control endpoint |
| | **projected total** | **~3.35 of 10.0** | |

## Conditional, only if warranted

| item | cost | trigger |
|---|--:|---|
| 3-seed confirmation of the UWM candidate | 3.96 | only if the closed-loop effect **exceeds ~3 pp** (§30, §49). A one-seed pilot is not evidence of generality |

If the closed-loop difference is **below ~3 pp**, §30 is explicit: do not burn
large compute proving a tiny effect. The correct call then is **UNRESOLVED at
routine resolution**, and the recommended next action is the one that changes
what we can measure — not another seed.

## Why no register or latent-action pilot

Both were closed on evidence, not convenience:

- **Registers**: the diagnostic found no artifact to fix (max/mean ≤ 2.53 vs
  Darcet's ~10×). Adding register tokens would add capacity with no diagnosed
  pathology, and per the audit any improvement would be uninterpretable — the
  absorption test has no outlier to absorb.
- **Action latent**: the 2-D bottleneck failed its own **predeclared** criterion
  (75.7% of variance vs the required ≥99%). A same-dim 3→3 VAE remains
  permissible but has a weak prior, and the A2A audit independently argues the
  idea does not transfer at our action dimensionality.
