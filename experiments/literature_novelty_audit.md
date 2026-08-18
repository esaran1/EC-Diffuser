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
- DexJoCo: https://arxiv.org/html/2605.16257v1
- Isaac Gym deprecation: https://developer.nvidia.com/isaac-gym
- Isaac Lab: https://developer.nvidia.com/isaac/lab
