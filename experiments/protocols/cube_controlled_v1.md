# Protocol: controlled multi-object cube experiment (PREDECLARED, NOT RUN)

Status: **predeclared, awaiting GPU availability, a data download decision,
and review.**

## 1. Target framing: cube-triple is hard *for our setting*, not unsolved

Earlier drafts of this protocol described cube-triple-task4 as having broad
SOTA headroom on the strength of MVP's 0.52. **That framing is withdrawn.**

Newer work reports much higher numbers on the same task:

| Method | cube-triple-task4 | Paradigm | Test-time cost |
|---|---:|---|---|
| QC | 0.37 ±0.26 | offline RL + critic | multi-step |
| MVP (ICLR 2026) | 0.32 ±0.07 (FMQ's table) / 0.52 (own table) | offline→online, Q-guided generate-and-select | 1 NFE + critic selection |
| **FMQ + QGBS** | **0.88 ±0.07** | offline RL + **online** critic fine-tuning | **NFE=20–32, critic required at test time** |

So cube-triple-task4 is **not unsolved**: FMQ reaches 0.88. It must not be
described as a task where the field is stuck.

### Why it is still the right target for this project

The three results above share a property our setting does not have:

- **FMQ** needs a learned critic *at inference*, using Q-guided beam search
  over 4 particles × 4 branches (NFE=20–32). That is neither one-step nor
  BC — it is closer to a search procedure than a fast policy.
- **MVP** needs Q-guided generate-and-select, i.e. a critic to pick among
  sampled actions.
- **QC** is Q-learning throughout.

**No published number on cube-triple comes from a pure-BC, low-NFE policy.**
The open question is therefore not "can this task be solved" (yes, at 0.88,
with a critic and 20–32 NFE) but:

> Under pure behavior cloning at 1–4 NFE, with no critic at training or test
> time, how far can a policy get on cube-triple — and does entity-level
> relational structure change that?

That is the regime this project studies, and it is empirically uncharted.

### Mandatory reporting rule

Every results table must carry a **paradigm column** with at least:
`BC` / `offline-RL` / `offline→online` and a **test-time NFE incl. critic
calls** column. Published critic-based numbers may appear only as clearly
labeled context rows, never averaged or ranked against our BC arms. A
comparison that omits these columns is invalid.

### Consequence for task selection

cube-triple remains the target because (a) it is genuinely multi-object
compositional, (b) it matches our 3-cube object count, and (c) the pure-BC
low-NFE regime on it is unmeasured. But the **headroom claim is now
conditional**: it must be established by our own arm 1/2 results, not
assumed. If plain BC and Diffusion@100 both do well at 1–4 NFE, the task is
not a discriminator for us either, and we report that and move on.

cube-double (MVP 1.00/1.00/0.95) is retained only as an easier control.

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
