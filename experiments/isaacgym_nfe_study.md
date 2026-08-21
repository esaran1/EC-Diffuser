# Paired Isaac Gym NFE study

Date: 2026-08-20. Branch `fast-generative-policies`.

**Question.** What is the minimum Flow NFE that reliably retains or exceeds the
canonical Gaussian EC-Diffuser performance on Isaac Gym 3-cube PushCube?

**Context.** The diagnosis in `experiments/isaacgym_flow_diagnosis.md` is closed:
no vanilla-Flow implementation failure, no action-normalization failure, no
undertraining problem, and no contact-control failure. This study asks the
original low-NFE question in that now-validated environment. **No training was
run.** NFE is an inference-time solver override.

## 1. Design

| Element | Value |
|---|---|
| Arms | Flow at 1, 2, 4, 8, 16 solver steps; Gaussian at 100 |
| Flow checkpoint | `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42`, EMA, step 499k |
| Gaussian checkpoint | `.../3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt`, EMA |
| Episode sets | **3 independently generated sets of 96 episodes** |
| Total | 6 arms x 3 sets x 96 = **1,728 episodes** |
| Task | 3-cube PushCube, random colours, 100-step episodes |

### 1.1 These are evaluation replicates, not training seeds

**One** trained Flow checkpoint and **one** Gaussian checkpoint are used
throughout. Between replicates only the *episodes* change; between Flow arms
only the *solver-step count* changes. Nothing about training varies.

So the between-replicate spread reported in §4 measures **evaluation-sampling
variance only**. It says nothing about training-seed variance, and no claim in
this document should be read as covering it. Establishing training-seed
variance would require retraining, which was explicitly out of scope.

### 1.2 Pairing is enforced, not assumed

Each replicate's initial and goal cube states are recorded once, frozen to disk,
and hashed with SHA256. Every arm loads that same file, and each result records
the hash. The aggregator **refuses to report** if the arms within a replicate do
not share one hash.

Episode-set hashes:

| Replicate | SHA256 (prefix) |
|---|---|
| 0 | `35144910b1471b7b` |
| 1 | *(recorded at run time)* |
| 2 | *(recorded at run time)* |

### 1.3 Model calls are verified, not requested

A forward hook on the denoiser counts actual calls. Every arm is checked against
its requested NFE and flagged on mismatch. A CPU unit test
(`tests/test_nfe_study_integrity.py`) independently pins that the solver honours
1/2/4/8/16 — including 16, which exceeds the checkpoint's trained default of 4 —
and that the override reaches flow wrappers while leaving `GaussianDiffusion`
untouched, so the reference arm cannot silently drop below 100 steps.

## 2. Statistical power, stated up front

At 288 paired episodes per arm (3 x 96), with a baseline near 0.85-0.88, the
approximate power to detect a true difference is:

| True difference | Power |
|---|--:|
| 3 points | ~0.20 |
| 5 points | ~0.44-0.54 |
| 7 points | ~0.75-0.86 |
| 10 points | ~0.98 |

**This study can detect differences of roughly 7 points or larger. It cannot
resolve 3-point differences.** Any "no significant difference" below must be
read as "not larger than about 5-7 points", not as "identical".


## 3. Results

All 18 runs complete: 6 arms x 3 replicates x 96 episodes = **1,728 episodes**.

### 3.1 Episode-set integrity

| Replicate | SHA256 (prefix) | Arms | Distinct hashes |
|---|---|--:|--:|
| 0 | `35144910b1471b7b` | 6 | 1 (OK) |
| 1 | `0047468fa69c00b8` | 6 | 1 (OK) |
| 2 | `586e5b8d2f7c44f8` | 6 | 1 (OK) |

Three distinct hashes across replicates, so the sets are independently
generated; one hash within each, so all six arms saw byte-identical episodes.

### 3.2 Verified model calls and latency

| Arm | Requested | **Measured** | ms / batch-16 plan | **ms / episode-step** |
|---|--:|--:|--:|--:|
| Flow 1 | 1 | **1.00** | 19.22 | **1.20** |
| Flow 2 | 2 | **2.00** | 38.21 | **2.39** |
| Flow 4 | 4 | **4.00** | 77.14 | **4.82** |
| Flow 8 | 8 | **8.00** | 156.50 | **9.78** |
| Flow 16 | 16 | **16.00** | 315.59 | **19.72** |
| Gaussian 100 | 100 | **100.00** | 2044.37 | **127.77** |

Counted by a forward hook on the denoiser: every arm honoured its request
exactly. Latency scales linearly with calls, so the compute was genuinely spent.
One planner call covers all 16 parallel envs, hence the per-episode column.

### 3.3 Per-set and pooled success

| Arm | r0 | r1 | r2 | **Pooled (n=288)** | 95% CI |
|---|--:|--:|--:|--:|--:|
| Flow 1 | 82/96 | 73/96 | 77/96 | **0.8056** | [0.755, 0.850] |
| Flow 2 | 85/96 | 83/96 | 82/96 | **0.8681** | [0.823, 0.905] |
| Flow 4 | 85/96 | 85/96 | 86/96 | **0.8889** | [0.847, 0.923] |
| **Flow 8** | 81/96 | 86/96 | 92/96 | **0.8993** | [0.859, 0.932] |
| Flow 16 | 82/96 | 88/96 | 85/96 | **0.8854** | [0.843, 0.920] |
| Gaussian 100 | 79/96 | 84/96 | 87/96 | **0.8681** | [0.823, 0.905] |

### 3.4 Aggregate metrics

| Arm | Success | Goal-succ frac | Obj-goal dist | Cubes placed | Contact rate | Wrong-direction pushes |
|---|--:|--:|--:|--:|--:|--:|
| Flow 1 | 0.8056 | 0.9051 | 0.0369 | 2.715 | **1.0000** | 0.170 |
| Flow 2 | 0.8681 | 0.9375 | 0.0281 | 2.813 | **1.0000** | 0.122 |
| Flow 4 | 0.8889 | 0.9479 | **0.0243** | 2.844 | **1.0000** | **0.076** |
| **Flow 8** | **0.8993** | **0.9595** | 0.0257 | **2.879** | **1.0000** | 0.090 |
| Flow 16 | 0.8854 | 0.9317 | 0.0297 | 2.795 | **1.0000** | 0.115 |
| Gaussian 100 | 0.8681 | 0.9282 | 0.0289 | 2.785 | **1.0000** | 0.101 |

**Contact rate is 1.0000 for every arm at every NFE from 1 to 100.**

### 3.5 Paired differences versus Gaussian @100

Episode-level pairing, pooled over replicates (n = 288 paired episodes):

| Arm | Delta | b (Flow wins) | c (Gaussian wins) | McNemar p |
|---|--:|--:|--:|--:|
| **Flow 1** | **−0.0625** | 24 | 42 | **0.0356** |
| Flow 2 | +0.0000 | 32 | 32 | 1.0000 |
| Flow 4 | +0.0208 | 31 | 25 | 0.5044 |
| Flow 8 | +0.0312 | 32 | 23 | 0.2806 |
| Flow 16 | +0.0174 | 27 | 22 | 0.5682 |

**Flow at 1 NFE is significantly worse than Gaussian (p = 0.0356).** No other
arm differs significantly, and Flow 2 ties the reference exactly (32 vs 32).

### 3.6 Failure taxonomy

| Arm | Failures | 1. Never approaches | 2. No contact | 3. Wrong direction | 4. Insufficient |
|---|--:|--:|--:|--:|--:|
| Flow 1 | 56 | **0** | **0** | 41 | 15 |
| Flow 2 | 38 | **0** | **0** | 26 | 12 |
| Flow 4 | 32 | **0** | **0** | 19 | 13 |
| Flow 8 | 29 | **0** | **0** | 18 | 11 |
| Flow 16 | 33 | **0** | **0** | 24 | 9 |
| Gaussian 100 | 38 | **0** | **0** | 23 | 15 |

**Across 1,728 episodes there is not a single approach failure or contact
failure at any NFE.** Every failure is post-contact, and the dominant mode is
pushing a cube the wrong way. Reducing NFE from 100 to 1 does not degrade
approach or contact — it degrades push accuracy: wrong-direction failures rise
from 18-24 at higher NFE to 41 at 1 NFE.

## 4. Between-replicate variance

Variability of **one fixed checkpoint** across independently drawn episode sets:

| Arm | min | max | range | std |
|---|--:|--:|--:|--:|
| Flow 1 | 0.7604 | 0.8542 | 0.0938 | 0.0384 |
| Flow 2 | 0.8542 | 0.8854 | 0.0312 | 0.0130 |
| Flow 4 | 0.8854 | 0.8958 | **0.0104** | 0.0049 |
| Flow 8 | 0.8438 | 0.9583 | **0.1146** | 0.0468 |
| Flow 16 | 0.8542 | 0.9167 | 0.0625 | 0.0255 |
| Gaussian 100 | 0.8229 | 0.9062 | 0.0833 | 0.0344 |

**The evaluation noise floor is 3-11 points on 96-episode sets**, with no clean
relationship to NFE — Flow 8 has the widest spread and Flow 4 the narrowest,
which is itself sampling noise on three draws. This is the single most important
number for designing any future comparison on this task: **a 96-episode
evaluation cannot resolve differences below roughly 10 points, and even n=288
resolves only about 6-7.**

## 5. Does the NFE trend replicate?

This is the section that most affects how the study should be read.

The three replicates produced **three different curve shapes**:

| Replicate | Spearman rho | p | success at NFE 1/2/4/8/16 | shape |
|---|--:|--:|---|---|
| 0 | −0.369 | 0.541 | 0.854 0.885 0.885 0.844 0.854 | flat |
| 1 | **+1.000** | **<0.001** | 0.760 0.865 0.885 0.896 0.917 | monotone rising |
| 2 | +0.700 | 0.188 | 0.802 0.854 0.896 **0.958** 0.885 | rises then falls |

Same checkpoint, same six arms, same protocol — only the 96 episodes differ.

Read alone, replicate 0 supports "the NFE axis is flat"; replicate 1 supports
"success rises monotonically with NFE"; replicate 2 supports "success peaks at
8 NFE and declines". **All three would have been reported confidently from a
single 96-episode run, and two of the three would have been wrong.**

This is the concrete justification for the three-replicate design. It means **no
per-set trend may be quoted on its own** — only the pooled curve is defensible.

What the replicates *do* agree on: **1 NFE is the worst arm in all three sets**
(0.854 / 0.760 / 0.802, and lowest within its own set every time). That is the
finding that survives replication, and it is the one the pooled paired test
confirms at p = 0.0356.

## 6. Answer to the study question

> **What is the minimum Flow NFE that reliably retains or exceeds the canonical
> Gaussian EC-Diffuser performance?**

**Two.**

- **Flow at 2 NFE ties Gaussian at 100 NFE exactly** — 0.8681 versus 0.8681,
  paired b = 32, c = 32, p = 1.0000 — at **1/50th the network calls** and
  **2.39 ms versus 127.77 ms per episode-step, a 53x latency reduction**.
- **Flow at 1 NFE is significantly worse** (0.8056, p = 0.0356). This is the
  only significant contrast in the study, and it is the floor of the curve.
- **Flow at 4-16 NFE is nominally better than Gaussian** (+1.7 to +3.1 points)
  but not significantly so.

The practical recommendation is **4 NFE**: it is the cheapest arm that is
nominally above the reference on every aggregate metric (obj-goal distance
0.0243 versus 0.0289, wrong-direction pushes 0.076 versus 0.101), it had the
narrowest between-replicate spread, and it costs 4.82 ms per episode-step —
still a **26x** saving over Gaussian.

**Stated precisely.** "Retains" here means "not measurably worse at n = 288".
Per the power table in §2 this study can exclude a deficit larger than about 6-7
points; it cannot certify equivalence to within 2-3 points. The honest claim is:
**the low-NFE question is answered in the direction of low NFE — one call is
too few, two calls suffice, and beyond four there is no measurable return.**

## 7. Is 3-cube saturated?

**Yes, effectively.**

- The best arm (Flow 8) reaches **0.8993 with 29 failures in 288 episodes**.
- The spread between the best and worst *non-degenerate* arm (Flow 2 through 16)
  is **3.1 points**, well inside the 3-11 point evaluation noise floor.
- All arms have **100% contact rate** and zero approach failures; the task no
  longer discriminates on the behaviours it was chosen to test.
- Only the deliberately-crippled 1-NFE arm separates from the pack.

A benchmark where five of six arms fall within noise of each other cannot
resolve the questions that remain open — the probability path, state-prediction
fidelity, or finer NFE distinctions. **Headroom is required before any of those
are testable.**

See `experiments/isaacgym_harder_task_feasibility.md` for what 4 cubes would
cost. The short version: the DLP encoder emits a fixed 24 particles per view
regardless of cube count, `num_entity` never reaches the backbone, and cubes are
created procedurally, so a zero-shot 4-cube evaluation needs **no retraining and
no new data** — only a `--num-entity` flag. Minimum useful probe ~0.7 GPU-h.

## 8. Compute

| Item | Cost |
|---|--:|
| 18 evaluation runs, 1,728 episodes | **1.75 GPU-h** |
| Training | **none** |

Flow arms cost 89-275 s per 96 episodes; the Gaussian arm cost ~2,100 s each,
i.e. **the single Gaussian reference consumed more wall time than all fifteen
Flow runs combined.**
