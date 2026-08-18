# Few-step generative policy literature and novelty audit

Date: 2026-08-17. Search performed through the current date using primary
sources (arXiv abstracts/HTML, official project pages, benchmark papers).

This audit exists to answer one question before any further method work:

> Which of the contributions this project could claim are already published?

## 1. Directly competing published work

| Work | Venue/ID | Objective | NFE | Benchmarks | Status for us |
|---|---|---|---|---|---|
| MP1 | arXiv:2507.10543, AAAI | MeanFlow identity, distillation-free | 1 | Adroit, Meta-World, real Franka | **Blocks "MeanFlow for robot policies"** |
| OMP | arXiv:2512.19347 | MeanFlow + directional alignment; DDE approximates JVP | 1 | Adroit, Meta-World | Blocks "fix MeanFlow pathologies" framing |
| DMPO | arXiv:2601.20701 | MeanFlow + dispersive regularization + RL fine-tune | 1 | RoboMimic, Kitchen, Gym, real Franka | Blocks "MeanFlow representation collapse" framing |
| One-Step Gen. Policies w/ Q-Learning | arXiv:2511.13035 | MeanFlow reformulation + Q-learning | 1 | OGBench (73 tasks incl. puzzle-4x4) | **Closest to our task setting** |
| Shortcut Models | arXiv:2410.12557 | self-consistency bootstrap | 1–4 | images; robot adaptations exist | Baseline only |
| Shortcut for policies (CVPRW 2025) | CVPR-W MEIS | shortcut + SO(3) | few | task-specific manipulation | Baseline only |
| BFQ | arXiv:2606.10613 | bootstrapped flow Q-learning, single-step | 1 | OGBench, avg score 68 | Strong current SOTA reference |

### Consequence

The following are **not novel** and must not be claimed:

- applying MeanFlow to robot policies (MP1);
- one-step manipulation in itself (MP1, OMP, DMPO);
- applying Shortcut Models to control (CVPRW 2025);
- replacing diffusion with flow matching in a policy (widely done);
- identifying MeanFlow instability/collapse as a phenomenon (OMP, DMPO).

Notably, OMP and DMPO independently report MeanFlow pathologies. Our own
`imf_source_to_code_audit.md` observed live-weight interval-loss divergence
with EMA masking. That observation is **real but not novel**; it corroborates
published findings rather than establishing a new one.

## 2. What remains open

The published one-step policy literature is concentrated on **short-horizon,
single-object, non-combinatorial** tasks: Adroit, Meta-World, RoboMimic
Lift/Can/Square/Transport, Kitchen. Reported failures are precision and
representation collapse, not compositional sequencing.

Two gaps survive this audit:

**Gap A (as first drafted; see §3, now substantially narrowed) — low-NFE
policies on combinatorially hard, long-horizon, multi-object tasks.**
OGBench puzzle-4x4-play requires combinatorial generalization over 2^16
button states plus continuous control. Published offline GCRL baselines
(OGBench, arXiv:2410.20092, Table 2):

| Method | puzzle-3x3 | puzzle-4x4 |
|---|---:|---:|
| GCBC | — | 0% |
| GCIVL | — | 13% |
| GCIQL | 95% | **26%** |
| QRL | — | 0% |
| CRL | — | 0% |
| HIQL | — | 7% |

puzzle-3x3 is saturated (95%); puzzle-4x4 is the headroom task. This
satisfies the "meaningful performance headroom, not saturated" requirement
from primary published numbers rather than our own pilots.

**Gap B — object-centric/entity-structured inductive bias for few-step
generation.** Every method above uses a flat sequence or image backbone.
EC-Diffuser's PINT entity-factored representation is unusual, and no audited
work combines entity-structured conditioning with a low-NFE objective. This
is the project's genuine structural asset.

## 3. Resolved: a one-step policy already reports strong puzzle-4x4

**This verification item is now closed, and it narrows Gap A substantially.**

arXiv:2511.13035 (one-step MeanFlow reformulation + Q-learning) was read
directly from the PDF. Its `singletask` results (5 tasks per row, mean ± s.e.)
include, against the strongest flow baseline FQL:

| OGBench row | FQL | Theirs |
|---|---:|---:|
| puzzle-3x3-singletask | 30 ±1 | **66 ±8** |
| puzzle-4x4-singletask | 17 ±2 | **40 ±6** |
| cube-double-singletask | 29 ±2 | 3 ±2 |
| humanoidmaze-large | 4 ±2 | 20 ±3 |

So a **one-step generative policy already reaches 40 on puzzle-4x4**, well
above the 26% GCIQL number from the OGBench paper. Two consequences:

1. "Low-NFE policies cannot do combinatorial puzzle tasks" is **false** as a
   general claim and must not be asserted. Our own puzzle-4x4 failure at
   5,000 training steps is an undertraining/behavior-cloning artifact, not
   evidence of a structural limit of few-step generation.
2. Their gain comes from **Q-learning**, not from the generative objective —
   they state naive MeanFlow degrades via out-of-range actions and bad critic
   estimates. Our setup is pure conditional behavior cloning with no critic.
   This is the single largest uncontrolled difference between our pipeline and
   the published state of the art on this exact task.

Note also `cube-double`: their one-step method collapses to 3 ±2 where FQL
gets 29 ±2. Multi-object composition is where the published one-step method
is *weakest*, and cube-double is the closest OGBench analogue to our
entity-structured 3-cube task.

### Revised gap statement

Gap A as originally written (combinatorial long-horizon) is **largely closed**
for state-based puzzle tasks by arXiv:2511.13035. What survives is narrower
and better targeted:

> **Gap A′ — multi-object compositional manipulation is where published
> one-step policies visibly break** (cube-double: 3 ±2 vs FQL 29 ±2), and no
> audited work applies an entity-structured/object-centric inductive bias to
> a low-NFE objective there.

Gap A′ and Gap B are the same gap approached from two directions, which is a
stronger position than either alone.

## 3b. Second-pass audit (2026-08-17): MVP and the Q-guided family

A second search pass was run specifically for work missed in the first pass.
It found the strongest directly-competing method and materially changes the
cube-double plan.

| Work | ID / venue | Paradigm | OGBench coverage | Relevance |
|---|---|---|---|---|
| **MVP** | arXiv:2602.13810, **ICLR 2026** | MeanFlow + instantaneous velocity constraint (IVC); offline→online with Q-guided generate-and-select | **cube-double t2/3/4, cube-triple t2/3/4** | **Closest competitor** |
| FMQ / QGBS | arXiv:2605.12416 | flow-map policy + Q-guided beam search | OGBench + RoboMimic, 12 tasks | Beats MVP by 21.3% relative IQM |
| Q-Flow | arXiv:2605.13435 | flow policy + Bellman critic | incl. cube-double, puzzle-4x4 | Reports the failures we care about |
| QGF | github zhouzypaul/qgf | BC flow + TD critic, test-time guidance | OGBench | Reference implementation |
| One-Step Flow Policy (OFP) | arXiv:2603.12480 | self-distillation, self-consistency | visuomotor | Few-step baseline |
| ElasticFlow | arXiv:2605.08799 | one-step, elastic horizons | language-guided | Horizon-adaptive precedent |

### MVP reported results (state-based OGBench, vs QC baseline)

| Task | QC | MVP |
|---|---:|---:|
| cube-double-task2 / task3 | 1.00 | 1.00 |
| cube-double-task4 | 0.93 | 0.95 |
| cube-triple-task2 | 0.82 | 0.88 |
| cube-triple-task3 | 0.69 | 0.71 |
| **cube-triple-task4** | 0.46 | **0.52** |

### Three consequences, one of which overturns the plan

**(i) cube-double is saturated, not hard.** MVP reports 1.00 / 1.00 / 0.95.
The "3 ±2 vs FQL 29 ±2" figure from arXiv:2511.13035 is one particular
method's weakness, **not a property of the task**. Selecting cube-double as
the flagship hard task on the strength of that single number would have been
a mistake. **cube-triple-task4 (0.52) is the task with real headroom.**

**(ii) The BC-vs-Q-learning confound is now confirmed as decisive and
universal.** MVP, FMQ, Q-Flow and QGF are all critic-based; MVP's gains come
from Q-guided generate-and-select at inference. No audited method reaches
these numbers from pure behavior cloning. Any comparison of our BC pipeline
against these published numbers measures *the critic*, not the generative
objective. This confirms the caution in the current instructions and is now
a hard constraint on protocol design.

**(iii) MeanFlow-family + OGBench cubes is occupied.** MVP is MeanFlow-family,
ICLR 2026, on exactly the cube tasks. This further closes the iMF direction
and independently supports cancelling the 50k iMF run.

### What still survives — narrowed statement of the gap

A third search pass specifically targeted structured few-step policies. The
closest work:

| Work | Structure | NFE | Multi-object relational? |
|---|---|---|---|
| **EfficientFlow** (AAAI, arXiv:2512.02020) | **global SO(2)** scene equivariance | 1/3/5 | No — global symmetry, no per-object decomposition |
| **ActionFlow** (arXiv:2409.04576) | SE(3)-invariant transformer over **poses** | few-step FM | Relational over poses, but not entity-factored multi-object composition |
| DM1 (arXiv:2510.07865) | dispersive regularization | 1 | No structure |
| MP1 / OMP / DMPO / MVP | none (flat/visual) | 1 | No |
| EC-Diffuser (ICLR 2025) | **entity-centric DLP + PINT** | ~100 (diffusion) | **Yes — but multi-step** |
| Hier. Entity-centric RL (arXiv:2602.02722) | entity-factored subgoals | multi-step diffusion; inference cost not a goal | Yes, but not low-NFE |

The pattern is consistent: work with **entity-level relational structure**
uses **multi-step** generation, and work at **1–4 NFE** uses **flat or
globally-equivariant** architectures. Note that "structured low-NFE policy"
as a gap statement does **not** survive EfficientFlow — it is equivariant and
runs at 1 NFE.

The gap must therefore be stated narrowly:

> **Entity-level *relational* structure — per-object tokens with inter-object
> attention — for low-NFE (1–4) multi-object action generation, and whether
> that structure preserves compositional generalization as object count grows
> when the sampler is reduced to one or a few steps.**

Three qualifiers do the work, and each is needed:

1. **Entity-level relational**, not global equivariance. EfficientFlow's
   SO(2) symmetry is a property of the whole scene; it does not decompose the
   scene into objects or model inter-object relations. ActionFlow reasons over
   relative SE(3) poses but is not an entity-factored multi-object policy.
2. **Low-NFE (1–4)**, not multi-step. EC-Diffuser has exactly the structure we
   mean but runs full diffusion; its compositional-generalization result is
   unmeasured under a one/few-step sampler.
3. **Compositional generalization vs object count**, not just success. This is
   the specific property EC-Diffuser claims (zero-shot to more objects than
   trained on) and the one most likely to be *fragile* under aggressive NFE
   reduction — which is a testable, falsifiable prediction rather than a
   general aspiration.

**Falsifiable prediction:** entity-structured low-NFE policies degrade more
slowly than flat low-NFE policies as object count grows (3-cube → cube-triple,
and cube-double → cube-triple), even at matched parameters and training
budget. If flat and entity-structured degrade identically, the gap is empty
and we report that.

**Honest caveat:** this is a *representation* contribution, not a new
generative objective. Its value depends entirely on the empirical result. It
is also only testable under pure BC — every critic-based competitor confounds
representation with policy improvement.

## 4. Method-selection implications

- **Baselines to keep:** Gaussian diffusion (canonical), vanilla Flow
  Matching (done), Shortcut (implemented). These are cheap and already built.
- **iMF:** keep as an implemented comparison, but do **not** invest further
  in stabilizing it as a contribution — OMP/DMPO already own that narrative.
- **Do not implement:** OMP, DMPO, MP1 re-implementations. They are
  published, need their own architectures/RL machinery, and would consume the
  entire compute budget to reproduce someone else's result.
- **Direction with residual novelty:** entity-structured / object-centric
  low-NFE generation evaluated where composition actually breaks.

## 5. Sources

- MP1: https://arxiv.org/abs/2507.10543
- OMP: https://arxiv.org/abs/2512.19347
- DMPO: https://arxiv.org/html/2601.20701
- One-Step Generative Policies with Q-Learning: https://arxiv.org/pdf/2511.13035
- BFQ: https://arxiv.org/pdf/2606.10613
- Shortcut Models: https://arxiv.org/abs/2410.12557
- OGBench: https://arxiv.org/html/2410.20092v1
- Improved Mean Flows: https://arxiv.org/abs/2512.02012
- MVP (ICLR 2026): https://arxiv.org/abs/2602.13810
- FMQ / QGBS: https://arxiv.org/html/2605.12416
- Q-Flow: https://arxiv.org/html/2605.13435v2
- One-Step Flow Policy: https://arxiv.org/pdf/2603.12480
- ElasticFlow: https://arxiv.org/pdf/2605.08799
- EC-Diffuser (ICLR 2025): https://arxiv.org/abs/2412.18907
- DexJoCo: https://arxiv.org/html/2605.16257v1
- Isaac Gym deprecation: https://developer.nvidia.com/isaac-gym
- Isaac Lab: https://developer.nvidia.com/isaac/lab
