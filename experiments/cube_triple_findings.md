# Cube-triple under pure behavior cloning: findings

Date: 2026-08-18. Branch `fast-generative-policies`. All numbers below are
traceable to a protocol, a checkpoint SHA256, an episode-set SHA256, and a raw
result file under `data/phase9_evaluations/`.

## 1. What was asked

Published results on OGBench `cube-triple` all come from **critic-based**
methods: FMQ+QGBS reaches 0.88 on task 4 but needs a learned critic *at
inference* with NFE 20–32; MVP uses Q-guided generate-and-select; QC is
Q-learning throughout. **No published number exists for a pure
behavior-cloning policy at low NFE.** That is the regime this project studies,
so we measured it.

> Under pure behavior cloning at 1–8 NFE, with no critic at training or test
> time, how far can a policy get on cube-triple?

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

**Not supported:** *"Behavior cloning cannot solve cube-triple."* That needs
multiple training seeds, a converged budget, and a critic baseline measured
here rather than cited.

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
resolved. Pursuing the low-NFE question here requires either adopting a critic
(which changes the paradigm under study) or moving to a task where BC reaches
non-trivial performance.

The measured saturation deficit (§5) remains the most specific quantified
departure from the demonstration distribution, but the diffusion arm shows it
is not sufficient on its own: diffusion nearly closes the saturation gap
(20.1% versus 32.1%) and still fails. Any follow-up should treat it as one
component of a closed-loop failure rather than the cause.
