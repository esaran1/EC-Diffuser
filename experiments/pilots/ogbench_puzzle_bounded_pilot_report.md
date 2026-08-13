# OGBench Puzzle bounded pilot report

## Scope

Single seed (`42`), 5,000 optimizer steps per method, effective batch 64, identical 63,282,904-parameter `IntervalTemporalUnet`, official OGBench train/validation split, frozen train-only normalizer, and four deterministic held-out batches. These are optimization/integration pilots, not task-performance results.

## Training results

| Method | Runtime | sec/step | Projected 500k | Peak allocated VRAM | Fixed validation, live | Fixed validation, EMA | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| GaussianDiffusion | 144.34 s | 0.02887 | 4.01 h | 1,483 MiB | 0.93884 → 0.66682 | 0.70316 | stable |
| ConditionalFlowMatching | 150.03 s | 0.03001 | 4.17 h | 1,483 MiB | 1.08520 → 0.57745 | 0.64561 | stable |
| ShortcutModel | 213.59 s | 0.04272 | 5.93 h | 1,481 MiB | 0.89128 → 0.52667 | 0.59872 | stable |
| ImprovedMeanFlow (boundary iMF) | 422.12 s | 0.08442 | 11.73 h | 1,483 MiB | raw L2 16.49371 → 37.03447 | raw L2 4.47104 | finite, strong EMA dependence; live instability |

All runs exited `0`, completed exactly 5,000 optimizer steps, used 320,000 examples, produced finite gradients, and wrote Trainer-compatible live+EMA checkpoints. Checkpoint hashes are in the machine-readable result.

The paper-default iMF adaptive scalar remains close to one when `p=1`; raw unweighted L2 is therefore the informative diagnostic. A post-run fixed-batch gradient audit found finite parameters and gradients but a larger live gradient norm (`1.66` initially, `3.96` live step 5,000, `2.69` EMA). The selected EMA weights recover strongly. This is optimizer/EMA dynamics, not NaN divergence.

## Native OGBench integration

Official OGBench 1.2.1 `puzzle-4x4-play-v0` initialized and executed real actions from EMA checkpoints. Exact model calls matched requested NFEs:

- Flow: 1/1, 2/2, 4/4, 8/8.
- iMF: 1/1 and 4/4.
- Shortcut: 4/4.
- Gaussian: 100/100.

The five-step Flow integration check reported 11.37 ms mean planner-only latency at 4 NFE and 380 MiB allocated inference memory. Its `0/1` success and 60% action-clipping rate are explicitly non-scientific because the model saw only 5,000 steps and the episode was truncated to five of the native 500 steps.

## Decision

Do not launch any 500k run. Gaussian, vanilla Flow, and Shortcut pass the bounded optimization screen. iMF passes checkpoint/native integration but requires a focused EMA/optimizer-dynamics study before longer training. Full native evaluation is also premature at this training budget.
