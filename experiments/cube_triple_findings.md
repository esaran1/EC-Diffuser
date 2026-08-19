# Cube-triple under pure behavior cloning: findings

Date: 2026-08-18. Branch `fast-generative-policies`. All numbers below are
traceable to a protocol, a checkpoint SHA256, an episode-set SHA256, and a raw
result file under `data/phase9_evaluations/`.

## 1. What was asked

> Under pure behavior cloning at 1–100 NFE, with no critic at training or test
> time, how far can a policy get on OGBench cube-triple?

### Correction to an earlier framing in this repository

Earlier notes compared our results against MVP (0.32–0.52) and FMQ+QGBS (0.88)
on cube-triple. **Those numbers are from `cube-triple-play-singletask-task4-v0`,
a different benchmark setting**, and the comparison was invalid. Per the
official OGBench repository, singletask variants are for standard
reward-maximizing offline RL while `-play-v0` is goal-conditioned, and the two
"target fundamentally different problem formulations, so direct success rate
comparisons would not be meaningful." Those numbers are dropped here rather
than merely relabeled.

### The correct external reference

We ran the goal-conditioned `cube-triple-play-v0` benchmark. The OGBench paper
(arXiv:2410.20092, Table 2) reports for that exact setting, over 8 seeds and 5
goals:

| Method | cube-single | cube-double | **cube-triple** |
|---|--:|--:|--:|
| **GCBC** (our closest analogue) | 6 ±2 | 1 ±1 | **1 ±1** |
| GCIVL | 53 ±4 | 36 ±3 | 1 ±0 |
| GCIQL | 68 ±6 | 40 ±5 | 3 ±1 |
| QRL | 5 ±1 | 1 ±0 | 0 ±0 |
| CRL | 19 ±2 | 10 ±2 | 4 ±1 |
| HIQL | 15 ±3 | 6 ±2 | 3 ±1 |

**cube-triple-play is near-zero for every published method**, value-based ones
included — GCIQL collapses 68 → 40 → 3 as cubes are added. The task is
effectively unsolved in the goal-conditioned setting.

## 2. Setup, held fixed across every arm

| Item | Value |
|---|---|
| Dataset | official `cube-triple-play-v0`, 3000 train episodes, 46-D state, 5-D actions |
| Windows | 2,991,000 horizon-5 windows |
| Goal relabeling | uniform within-episode future state, identical code path for all arms |
| Backbone | `IntervalTemporalUnet`, dim 128, mults (1,2,4,8) |
| **Parameters** | **63,249,715 — identical for all three arms** |
| **Peak train VRAM** | **1482.57 MiB — identical for all three arms** |
| Optimizer steps | 50,000; effective batch 64; 3.2M examples seen |
| Seed | 42 (single training seed — see §7) |
| Checkpoint | final EMA only, no selection on evaluation episodes |
| Success | official OGBench `info["success"]` (all three cubes within 0.04 m) |

Training cost 0.40 GPU-h per arm, 1.20 GPU-h total. Every arm's held-out
validation fell monotonically with no divergence, so no arm is degenerate:

| Arm | Validation (EMA) |
|---|---|
| behavior_cloning | 0.7223 → 0.0627 |
| conditional_flow_matching | 1.0844 → 0.1445 |
| gaussian_diffusion | 0.7976 → 0.1171 |

## 3. Headline result

All arms evaluated on **identical episode seeds**, tasks 2/3/4.

| Method | Paradigm | NFE | Calls verified | Success | 95% UB | Cubes placed | Latency |
|---|---|--:|--:|--:|--:|--:|--:|
| behavior_cloning | BC | 1 | 1 | 0/60 | 4.87% | 0 | 2.77 ms |
| conditional_flow_matching | BC | 1 | 1 | 0/60 | 4.87% | 0 | 2.85 ms |
| conditional_flow_matching | BC | 2 | 2 | 0/60 | 4.87% | 0 | 5.49 ms |
| conditional_flow_matching | BC | 4 | 4 | 0/60 | 4.87% | 0 | 10.99 ms |
| conditional_flow_matching | BC | 8 | 8 | 0/60 | 4.87% | 0 | 20.96 ms |
| gaussian_diffusion | BC | 100 | 100 | 0/30 | 9.50% | 0 | (contended) |

Latency scales exactly with NFE and model calls are verified exact, so the
compute was genuinely spent — the flat result is not an instrumentation
artifact.

**The NFE axis is completely flat at zero across a 100x compute range, and the
non-generative BC floor is identical to the generative arms.** Nothing here
indicts the generative objective: the regression policy fails in exactly the
same way as flow matching and as diffusion at 100 NFE.

The diffusion arm was evaluated on 30 episodes that are an exact verified
subset of the 60, so the comparison is paired. On those identical 30 episodes
every arm scores 0/30:

| Arm | NFE | Success | Cubes placed | Δdist (m) | Clip |
|---|--:|--:|--:|--:|--:|
| behavior_cloning | 1 | 0/30 | 0 | — | 3.6% |
| conditional_flow_matching | 1 | 0/30 | 0 | — | 2.6% |
| conditional_flow_matching | 2 | 0/30 | 0 | −0.00025 | 3.6% |
| conditional_flow_matching | 4 | 0/30 | 0 | +0.00153 | 5.4% |
| conditional_flow_matching | 8 | 0/30 | 0 | +0.00132 | 6.7% |
| **gaussian_diffusion** | **100** | **0/30** | **0** | **−0.00611** | **20.1%** |

Diffusion clips 20.1% of steps — three to eight times more than the low-NFE
arms — and still moves cubes slightly *away* from their goals. More aggressive
actions are therefore not sufficient on their own, which tempers the
saturation-deficit lead in §5 without refuting it.

## 4. Four explanations excluded with evidence

| Explanation | Test | Result |
|---|---|---|
| **Fitting failure** | offline action MAE on held-out windows | 0.116 (BC), 0.145 (Flow) against action scale 0.297 mean / 0.418 std — models fit the data |
| **Action saturation** | clip fraction at execution | 2.4–6.8%, versus 82.9% in the earlier puzzle diagnostic |
| **Near-misses** | per-cube goal distance | 0.243 → 0.241 m against a 0.04 m threshold |
| **Goal difficulty** | D1-cube, predeclared easy/hard split | both 0/40; task 4 (goals 40% closer) also zero |
| **A defect in our pipeline** | comparison with published GCBC on the same benchmark | GCBC scores 1 ±1; our 0/60 has a 4.87% upper bound — statistically indistinguishable, so the zero reproduces known behavior |

The cube-distance metric is read from the same MuJoCo state the environment
uses in `CubeEnv._compute_successes` and was verified to reproduce
`_compute_successes()` exactly.

## 5. Where the failure actually is

Executed action magnitude was measured at rollout and compared with the
demonstrations:

| | abs mean | p50 | p90 | steps with any component at limit |
|---|--:|--:|--:|--:|
| Demonstrations | 0.3163 | 0.1880 | 0.8912 | **0.3207** |
| behavior_cloning @1 | 0.2695 | 0.1047 | 0.8540 | **0.0944** |
| conditional_flow_matching @4 | 0.2859 | 0.1389 | 0.8107 | **0.0583** |

**There is no mean collapse** — executed magnitudes are close to the data,
ruling out the classic BC-on-play regression-to-the-mean story. The deficit is
specific: the policies reach the action limit on 5.8–9.4% of steps versus
32.1% in the demonstrations, a **3.4×–5.5× shortfall in exactly the saturated
actions that produce gripper contact and cube motion**.

This is a measured association, not a demonstrated cause. It says *where*
executed behavior departs from the data; it does not prove that restoring
saturation would produce success.

**The diffusion arm bears directly on this and weakens it.** At 100 NFE
diffusion clips 20.1% of steps — much closer to the demonstrations' 32.1% than
any low-NFE arm — yet it still places no cube and its mean cube distance
*increases* by 0.0061 m. So closing the saturation gap is demonstrably not
sufficient by itself. The deficit remains a real measured difference from the
data, but it is not established as the operative cause.

## 6. What is and is not claimable

**Supported:** *No arm exceeds ~5% success (~9.5% for the 30-episode diffusion
arm) on cube-triple under this training budget and goal distribution, and
success is invariant to NFE across a 100x range (1 to 100) and to the presence
of a generative objective at all.*

**Also supported, and this is the strongest form of the claim:** *our pipeline
reproduces the published GCBC baseline on this benchmark.* GCBC scores 1 ±1 and
our BC-class arms score 0/60 (upper bound 4.87%) — indistinguishable. The zero
is a faithful measurement, not a defect.

**Not supported:** *"Behavior cloning cannot solve cube-triple."* That needs
multiple training seeds, a converged budget, and a critic baseline measured
here rather than cited. Note also that value-based methods score only 0–4 on
this task, so the BC-versus-critic gap that exists on cube-single and
cube-double has largely vanished by cube-triple.

**Explicitly not supported:** *"Low-NFE generative modeling fails on
compositional manipulation."* The non-generative floor fails identically, so
this experiment cannot indict low-NFE generation.

## 7. Limitations, stated plainly

- **One training seed (42).** All arms share it, so pooling across arms gives
  evaluation-episode confidence only, never training-seed confidence.
- **50,000 steps is not a converged budget** (~1/10 of the EC-Diffuser 500k
  reference).
- **No critic baseline was run.** The BC-versus-critic contrast is inferred
  from published work, not measured here.
- **The diffusion arm used 30 episodes, not 60**, because measured 100-NFE
  latency (789 ms/plan) put the full protocol at ~6.6 h, over the compute
  gate. Its bound is 9.50% rather than 4.87%. The episode set is a verified
  subset of the 60, so the comparison remains paired.
- **The D1 easy/hard split was flawed**: it separated standardized
  *observation* distance but not physical cube distance, because the
  separating dimensions were arm proprioception. The stronger difficulty
  control is the intrinsic task-4 contrast, reported alongside it.
- **The benchmark offers no within-task difficulty gradient** — physical goal
  distance is nearly constant inside each task — so a fully matched easy/hard
  comparison cannot be built from the official goals.

## 8. Consequence for the project

cube-triple under pure BC has **no dynamic range** for the low-NFE question:
every arm sits at the floor, so no NFE or objective comparison can be
resolved. Crucially, this is not specific to BC — **no published method
exceeds 4% on this task**, so adding a critic would not create dynamic range
either. GCIQL manages 3 ±1.

The productive move is therefore **not** to add a critic but to change task.
The published numbers name the right target directly: **cube-double-play**,
where GCIQL reaches 40 ±5 and GCBC only 1 ±1. That is a task with real
headroom AND a large, documented BC-versus-value gap — exactly the dynamic
range cube-triple lacks. It is also already downloaded, converted, normalized,
and adapter-supported in this repository.

The measured saturation deficit (§5) remains the most specific quantified
departure from the demonstration distribution, but the diffusion arm shows it
is not sufficient on its own: diffusion nearly closes the saturation gap
(20.1% versus 32.1%) and still fails. Any follow-up should treat it as one
component of a closed-loop failure rather than the cause.
