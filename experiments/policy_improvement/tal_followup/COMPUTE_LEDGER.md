# Compute ledger — Tal-followup program

**Fresh hard cap: 10.0 GPU-h.** Historical programs are separate and are not
charged here. CPU work and idle waiting do not count.

| stage | training valid | training invalid | eval valid | eval invalid | diagnostics | total |
|---|--:|--:|--:|--:|--:|--:|
| A — literature audit (zero GPU) | 0 | 0 | 0 | 0 | 0 | **0.000** |
| B — architecture audit (zero GPU) | 0 | 0 | 0 | 0 | 0 | **0.000** |
| C — direct-action recovery (zero GPU) | 0 | 0 | 0 | 0 | 0 | **0.000** |
| D — two-time impl + diagonal gate (CPU) | 0 | 0 | 0 | 0 | 0 | **0.000** |
| D — UWM pilot training (2 arms, 5k steps, 1 seed) | in progress | 0 | 0 | 0 | 0 | **live** |
| **SPENT TO DATE** | | | | | | **see pilot_runner.log** |

Live accounting is computed from lease acquire/release pairs in
`uwm/pilot_runner.log`, `uwm/closed_loop.log` and `uwm/qaction_runner.log` by
`run_closed_loop.py:program_spent()`, which sums TRAINING and EVALUATION
together against the single 10.0 cap.

## Measured unit costs (from the completed evaluator-resolution program)

| unit | cost |
|---|--:|
| one five-set R=1 evaluation (480 episodes, NFE=4) | **0.2212 GPU-h** |
| one R=3 evaluation of one checkpoint | **0.664 GPU-h** |
| one R=3 A/B pair (two checkpoints) | **1.327 GPU-h** |

These are measured, not estimated, and are the basis of every projection below.

## Live projection (updated before each GPU stage)

Per directive §6, before any substantial GPU stage:
`spent + projected stage + minimum cost of higher-priority remaining work ≤ 10.0`.

Priority under budget pressure (§6):
UWM impl/training > direct-action reconstruction > harder-task eval >
register pilot > latent-action pilot > 2602.07322 implementation.
