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

## Measured unit costs

| unit | cost | source |
|---|--:|---|
| **5000-step training arm** | **0.326 GPU-h** | **measured, this program (arm 1: 19.5 min)** |
| one five-set R=1 evaluation | 0.2212 GPU-h | measured, evaluator-resolution program |
| one R=3 evaluation | 0.664 GPU-h | derived |
| one R=3 A/B pair | 1.327 GPU-h | derived |

Arm 1 came in at 0.326 GPU-h against a 0.42 estimate, so the projection below
uses the measured figure.

## Actual spend to date (measured from lease acquire/release pairs)

| item | GPU-h | class |
|---|--:|---|
| pilot arm uwm (5k) | 0.3258 | valid training |
| pilot arm ctl (5k) | 0.3258 | valid training |
| Stage-E diagnostics (all arms, one lease) | 0.0144 | valid diagnostics |
| extension arm uwm (5k→10k) | 0.3317 | valid training |
| **extension arm ctl — killed by session death** | **0.0033** | **WASTED / INVALID** |
| extension arm ctl (rerun) | *in progress* | valid training |
| **TOTAL SPENT** | **1.0011** | of 10.0 |
| **REMAINING** | **8.9989** | |

The 0.0033 GPU-h is recorded as waste, not absorbed silently. Its partial
artifacts were deleted and the arm rerun from scratch rather than resumed
mid-stream.

## Stages closed WITHOUT spending their budget

| stage | status | budget freed |
|---|---|--:|
| F — harder multi-object task | **BLOCKED** (~33 GPU-h needed, 3.3× the cap) | — |
| G — register implementation pilot | **NOT JUSTIFIED** (no artifact found) | ~0.99 |
| H — action-latent bottleneck | **NOT AUTHORIZED** by its own criterion (75.7% vs ≥99%) | ~0.99 |

## Projection with Stage F blocked

| stage | GPU-h | cumulative |
|---|--:|--:|
| pilot, 2 arms | 0.652 | 0.652 |
| Stage E offline diagnostics (one lease) | 0.150 | 0.802 |
| closed-loop R=3: control + best UWM path | 1.327 | 2.129 |
| Stage C direct-action training | 0.326 | 2.455 |
| closed-loop R=3: direct-action | 0.664 | **3.119** |
| Stage F harder task | **BLOCKED** (~33 GPU-h needed) | — |

**Projected total 3.119 of 10.0; headroom 6.881.** Affordable with that
headroom, in priority order: 3-seed confirmation (3.958), register pilot
(0.990), latent-action pilot (0.990).

## Superseded estimate

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
