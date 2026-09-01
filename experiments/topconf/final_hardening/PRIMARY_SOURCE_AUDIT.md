# Primary-source audit (§1) — what each closest work owns

Verified by reading actual paper content. Where a number is load-bearing I
fetched and quoted the source PDF myself rather than trusting a summary.

## The five closest works

| # | work | owns | evaluates policy ranking under sim nondeterminism? | slot/index? | num_envs? | sign reversal? | verdict |
|--:|---|---|:--:|:--:|:--:|:--:|---|
| 1 | **GPUSimBench** 2607.13059, IROS 2026 | four GPU stochasticity regimes; EMD divergence in cm; simulator-choice guidance | **no — runs zero learned policies** | observes index divergence, **never permutes** | scales for throughput only | no | **WOUNDS** |
| 2 | **RoboDojo** 2607.04434 (v3) | eval infrastructure, 42 sim + 18 real tasks, leaderboard | cross-**GPU-device** only | **no** | **no** | no | **WOUNDS — must confront** |
| 3 | **N-SCORE** 2603.13616, RSS 2026 | SAVI+KDE anytime-valid sequential testing, generalized progress metrics | no | no | no | no | ADJACENT |
| 4 | **STEP** 2503.10966, RSS 2025 | near-optimal-stopping two-policy comparison, ≤32% fewer trials | no | no | no | no | ADJACENT |
| 5 | **SIMPLER** 2405.05941, CoRL 2024 | real-to-sim eval; **MMRV** rank-violation metric | sim-vs-**real** rank disagreement | no | no | **different axis** | ADJACENT |

Also audited: Active Experiment Selection (2502.09829, CoRL 2025) — repeats each
experiment 3× but **pools repeats as extra i.i.d. Bernoulli draws rather than
decomposing within- vs between-scenario variance**; AutoEval (2503.24278) and
RoboArena (2506.18123) — real-hardware only; Discounted-Liveness OPE
(2605.11479) — fully offline.

## The two that matter, read carefully

### GPUSimBench — pre-empts the physics claim, cannot reach ours

Design: inclined-plane rolling-ball / cube-array collision, **16 parallel envs,
10 runs**, pairwise EMD of cube positions at t=5 s. *"we fix random seeds and
disable all task-level randomization"* — scripted physics, **no policy, no
success rate, no comparison**.

It **owns**: "GPU simulators are nondeterministic and env-index-dependent" as a
*physics* observation, including the four regimes.

It **cannot own**: any statement about learned-policy comparison decisions. It
also **observes** inter-env divergence without ever **permuting** which slot a
scenario occupies — so the causal question is untouched.

### RoboDojo — publicly asserts the opposite conclusion (verified verbatim)

§6.4.1, quoted from the PDF:

> *"we conduct stability experiments on three RTX 4090 GPUs. Specifically, we use
> **layout 0 of each task** and evaluate three representative policies … with
> **three random seeds on each GPU**."*
>
> *"the largest success-rate standard deviation is only **1.1 percentage
> points** … RoboDojo yields stable simulation scores across GPU devices, making
> it **suitable for fair leaderboard comparison** despite minor nondeterminism."*

**Why ≤1.1 pp and our ~10 pp are not in contradiction — they measure different
things.** Three structural reasons, all verifiable from the quoted design:

1. **It varies GPU *device*, not slot assignment or `num_envs`.** A between-device
   contrast cannot detect within-device slot effects.
2. **Its statistic is a cross-GPU SD of values already averaged over three
   seeds.** Averaging is precisely the operation that suppresses within-scenario
   realization variance; our 10.4 pp is the *un-averaged* spread of single
   evaluations.
3. **It uses layout 0 of each task — one scenario per task.** Per-scenario
   bifurcation, which is where 71.5% of our variance lives, cannot appear in that
   design.

This is the strongest available motivation: **a public benchmark is asserting
leaderboard-grade stability on the basis of a design that is structurally blind
to the effect we measure.** Cite early and confront directly; do not treat as a
threat to be buried.

## What remains exclusively ours

1. **Slot-permutation as a policy-evaluation variable** — nobody has run it.
2. **`num_envs` as a policy-evaluation variable** — unclaimed.
3. **Sign-reversal rate of a calibrated policy comparison under simulator
   nondeterminism** (~24%) — SIMPLER's MMRV is sim-vs-real, a different axis.
4. **The two-level estimand with ICC / design effect / R-resolution** — STEP and
   N-SCORE both assume flat i.i.d. trials; Active Selection pools rather than
   decomposes.
5. **Contact bifurcation tied to evaluation validity** — GPUSimBench measures
   contact divergence with no policy; the contact literature treats bifurcation
   as a control problem, never an evaluation-validity one.

## Positioning

Move off "GPU simulators are nondeterministic" — GPUSimBench owns that at the
physics layer as of July 2026. The defensible frame:

> The statistical evaluation line (STEP, N-SCORE) assumes i.i.d. trials; the
> GPU-simulator line (GPUSimBench) documents execution nondeterminism without
> policies; the benchmark line (RoboDojo) infers leaderboard stability from a
> design that averages over seeds and uses one layout per task. We connect the
> three and show the i.i.d.-trial assumption fails at contact: 71.5% of outcome
> variance lies below the nominal trial, and ~24% of comparison signs flip.
