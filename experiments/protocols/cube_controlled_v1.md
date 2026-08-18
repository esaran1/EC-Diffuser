# Protocol: controlled multi-object cube experiment (PREDECLARED, NOT RUN)

Status: **predeclared, awaiting GPU availability, a data download decision,
and review.**

## 1. Target correction: cube-triple, not cube-double

The instruction was to keep cube-double as the main hard-task target. The
second-pass literature audit (see `experiments/literature_novelty_audit.md`
§3b) found evidence that changes this, and it is reported here rather than
silently followed or silently ignored.

**MVP (arXiv:2602.13810, ICLR 2026) reports cube-double at 1.00 / 1.00 / 0.95**
across tasks 2/3/4. cube-double is **saturated**, which fails the project's
own "prefer tasks with meaningful headroom" rule.

The `3 ±2 vs FQL 29 ±2` figure that motivated cube-double comes from
arXiv:2511.13035 and is **one method's weakness on one task**, not a property
of the task. Two independent methods (MVP 0.95+; QAM-E 65±5 in Q-Flow's
table) do well on cube-double.

Meanwhile **cube-triple-task4 sits at 0.52 (MVP) / 0.46 (QC)** — genuine
headroom, and it is explicitly the *cyclic permutation of three cubes*, i.e.
the compositional multi-object structure this project cares about, at the
same object count as our 3-cube PushCube task.

**Recommendation: make cube-triple (tasks 2/3/4) the target and retain
cube-double-task4 (0.95) only as an easier control.** This preserves the
intent of the instruction — multi-object composition as the hard target —
while pointing it at a task that is not already solved.

If you prefer to keep cube-double as primary regardless, the protocol below
runs unchanged with the task id swapped; only the headroom argument weakens.

## 2. The question

> Does low-NFE generative modeling fail on multi-object compositional
> manipulation **under a controlled policy-learning setup**?

"Controlled" is load-bearing: every published number on these tasks comes
from a **critic-based** method (MVP: Q-guided generate-and-select; FMQ:
Q-guided beam search; Q-Flow: Bellman critic; QC/QAM-E: Q-learning). Our
pipeline is pure conditional behavior cloning.

**Therefore: no arm in this experiment may be compared against a published
number as evidence about the generative objective.** Published values appear
in write-ups only as context, explicitly labeled as a different training
paradigm. Internal comparisons between our own arms are the only valid
evidence here.

## 3. Arms

All arms share dataset, observation/goal representation, backbone capacity,
training budget, horizon, evaluation protocol, and seeds. Only the objective
(and NFE at inference) varies.

| Arm | Method | Training | Inference NFE |
|---|---|---|---|
| 1 | **BC (deterministic MLP/regression head)** | BC | 1 (no sampling) |
| 2 | **GaussianDiffusion** | BC-style denoising | 100 |
| 3 | **ConditionalFlowMatching** | BC-style flow | 1, 2, 4, 8 |
| 4 | **ShortcutModel** | BC-style self-consistency | 1, 2, 4 |

Arm 1 is the floor: if a plain regression policy matches the generative arms,
generative modeling is not earning its cost on this task, which is itself a
publishable negative result.

Arm 4 is chosen over iMF because iMF is **cancelled** per instruction and,
independently, because MVP now occupies the MeanFlow-family/cube-task
intersection.

### Separately labeled, not an apples-to-apples arm

| Arm | Method | Why separate |
|---|---|---|
| 5 (optional, Tier B) | any critic-based baseline (e.g. FQL/QC-style) | Different training paradigm. Reported in its own table with a paradigm column. Never averaged with arms 1–4. |

Arm 5 is **not required** to answer the question in §2 and should be run only
if arms 1–4 all fail, to establish whether the failure is BC-specific.

## 4. Held fixed

- Dataset: one OGBench cube dataset, one split manifest, train-only normalizer
  (same machinery as `experiments/datasets/converted/*`).
- Observation/goal: official state observations; identical goal-relabeling
  policy across arms.
- Backbone: shared `IntervalTemporalUnet`-class flat sequence backbone at
  matched parameter count (the OGBench pilots used 63,282,904 params).
  Report exact counts; ShortcutModel and Flow must match within ~1%.
- Training budget: identical optimizer steps, effective batch, LR schedule,
  EMA settings, and **examples seen**.
- Horizon / action chunk: identical across arms (see §6 — this is a variable
  in the diagnostic protocol, fixed here).
- Evaluation: identical episode seeds, episodes per seed, native horizon,
  success predicate (official OGBench).
- Seeds: 3 training seeds (42/43/44); paired evaluation seed set.

## 5. Preregistered outcomes

- **If arms 3/4 at 1–2 NFE approach arm 2 (Diffusion@100) and arm 1:**
  low-NFE generative modeling does *not* fail here; the bottleneck is the
  policy-learning paradigm (BC without a critic), not the NFE budget.
- **If arms 3/4 collapse at 1–2 NFE but recover by 4–8:** the failure is
  integration/objective-related, and NFE is the operative variable.
- **If all of arms 1–4 fail:** the bottleneck is BC itself on this task, and
  the honest conclusion is that this task cannot discriminate generative
  objectives without a critic. Arm 5 then becomes justified.

All three outcomes are informative and will be reported.

## 6. Data prerequisite — measured, NOT blocking

**Only `puzzle-4x4-play-v0` is present locally** (train 262,414,247 B +
validation 26,216,338 B). No cube dataset is downloaded.

Download sizes were measured directly by HTTP HEAD against the official
Berkeley RAIL mirror (the URL pattern is flat: `<dataset>-v0.npz` and
`<dataset>-v0-val.npz`; the pattern was validated by confirming
`puzzle-4x4-play-v0.npz` returns exactly 262,414,247 bytes, byte-identical to
the local copy):

| File | Size |
|---|--:|
| `cube-triple-play-v0.npz` | 1.004 GB |
| `cube-triple-play-v0-val.npz` | 0.101 GB |
| `cube-double-play-v0.npz` | 0.297 GB |
| `cube-double-play-v0-val.npz` | 0.030 GB |
| **Total (both tasks)** | **1.431 GB** |

**1.43 GB is well under the 20 GB download gate**, so this is not a blocker.
Note this is the *standard* play dataset, not `cube-triple-play-100m-v0`
(100M transitions), which must **not** be downloaded.

Pre-launch checklist:
1. `wget` the four files above into `data/benchmarks/source/ogbench/`
   (git-ignored). ~1.4 GB.
2. Build conversion manifests with the existing
   `diffuser/scripts/build_phase6_conversion_manifests.py` machinery.
3. Freeze a train-only normalizer; record its SHA256 alongside the existing
   entries in `experiments/datasets/benchmark_normalizers_v1.json`.
4. Run the existing `audit_phase6_benchmarks.py` integrity checks
   (finiteness, episode boundaries, train/val leakage) before training.

## 7. Cost estimate (indicative, pending the §6 measurement)

Training, from the measured OGBench pilot rates (sec/step at 63.3M params,
effective batch 64, on this 4080):

| Method | sec/step | 50k steps | × 3 seeds |
|---|--:|--:|--:|
| Flow | 0.03001 | 0.42 h | 1.25 h |
| Shortcut | 0.04272 | 0.59 h | 1.78 h |
| Diffusion | 0.02887 | 0.40 h | 1.20 h |
| BC | ~0.02 (est.) | ~0.28 h | ~0.83 h |
| **Training total** | | | **~5.1 h** |

Evaluation: 4 arms × NFE settings × 3 seeds × episodes, on 500-step native
episodes — dominated by Diffusion@100. Budget **~4 h** with Diffusion
restricted to fewer episodes, mirroring the Stage-1 reduction.

**Indicative total: ~9–10 GPU-h**, plus download. Exact figures to be fixed
after §6 and before launch.

## 8. Not included

No new method. No architecture search. No hyperparameter tuning beyond
matching the arms. No iMF arm.
