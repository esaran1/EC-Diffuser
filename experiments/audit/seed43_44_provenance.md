# Provenance check: do canonical Flow replicas already exist at seeds 43/44?

Date: 2026-08-22. **The seed-43 automatic launch was disarmed before this check
began** (gate PID 3851988 and its child 3851989 killed; verified no training or
launcher process remains).

**Verdict: NO. The existing seed-43/44 checkpoints are a different experiment
entirely. Train clean seed 43.**

## 1. Seed-43 checkpoint inventory

Exhaustive search (`find -name "*.pt" -path "*seed4[34]*"`) returns **exactly
four** files, all under `data/phase7_runs/ogbench_puzzle_state/`:

| Path | SHA256 | Bytes | mtime | Internal step | Keys | #params |
|---|---|--:|---|--:|---|--:|
| `imf_aux_multiseed_boundary_seed43_5000/state_5000.pt` | `37891f8a3ef8caf25d5a4a6e54dc3dc3572f5fa42dfea51e34fd623ad4858c9d` | 506,437,584 | 2026-08-13T22:35:34 | **5000** | step, model, ema | 63,283,344 |
| `imf_aux_multiseed_auxiliary_seed43_5000/state_5000.pt` | `41c5ce6f6deb96775a475f07c80559642500f5c23d788b71fc289760a5ff5d3d` | 627,561,836 | 2026-08-13T22:46:31 | **5000** | step, model, ema | 78,415,464 |

## 2. Seed-44 checkpoint inventory

| Path | SHA256 | Bytes | mtime | Internal step | #params |
|---|---|--:|---|--:|--:|
| `imf_aux_multiseed_boundary_seed44_5000/state_5000.pt` | `ce6bb54c3b96dc02b31b8e98e8a5303f8a921a218b7577ebd715987570f47b30` | 506,437,584 | 2026-08-13T22:54:33 | **5000** | 63,283,344 |
| `imf_aux_multiseed_auxiliary_seed44_5000/state_5000.pt` | `c003f18f97cd1c00a5454ddae965220221dee1413184d91a635d9c1cf393a409` | 627,561,836 | 2026-08-13T23:04:49 | **5000** | 78,415,464 |

The other `seed43`/`seed44` directories found (`improved_meanflow_smoke_plans/`,
`shortcut_smoke_plans/`) are **planning-output directories containing no
checkpoints**; there `seed43`/`seed44` denotes an *evaluation* seed, not a
training seed.

## 3. What the "IMF auxiliary" runs actually were

Each run directory carries a `summary.json` with full recorded metadata (the
`*_config.pkl` files were not written for these runs, so provenance comes from
that summary, not filenames).

| Field | boundary seed43/44 | auxiliary seed43/44 |
|---|---|---|
| `method` | `improved_meanflow` | `auxiliary_improved_meanflow` |
| `wrapper` | **`ImprovedMeanFlow`** | **`AuxiliaryImprovedMeanFlow`** |
| `backbone` | **`IntervalTemporalUnet`** | **`AuxiliaryIntervalTemporalUnet`** |
| `parameter_count` | 63,282,904 | 78,415,024 |
| `optimizer_steps` | **5,000** | **5,000** |
| `learning_rate` | 4e-05 | 4e-05 |
| `adam_betas` | (0.9, **0.95**) | (0.9, **0.95**) |
| `lr_warmup_steps` | **78** | **78** |
| `dataset_manifest` | **`ogbench_puzzle_4x4_play_v0`** | **`ogbench_puzzle_4x4_play_v0`** |
| `git_commit` | `49af9be3f185b7b43d596c217518910f00996a59` | same |
| `runtime_seconds` | 435 | 586 |

**Classification: B — separately trained auxiliary/diagnostic models.** They are
Improved-MeanFlow variants trained on the **OGBench puzzle-4x4** task for a
5,000-step diagnostic, from a different commit, with a different objective,
backbone, optimizer settings and dataset. The MeanFlow/iMF line was later
cancelled.

## 4. Field-by-field comparison against canonical seed 42

| Field | Canonical seed 42 (Flow) | Existing seed 43 (iMF) | Match |
|---|---|---|---|
| Task / dataset | `push_cubes/3C_randcolor` (Isaac Gym) | `ogbench_puzzle_4x4_play_v0` | **NO** |
| Dataset SHA256 | `7abf83b8…` | different corpus | **NO** |
| DLP checkpoint | `dlp_push_6C` (`a8a11130…`) | none — state-based puzzle task | **NO** |
| Wrapper | `ConditionalFlowMatching` | `ImprovedMeanFlow` / `AuxiliaryImprovedMeanFlow` | **NO** |
| Probability path | linear Flow | MeanFlow (interval-conditioned) | **NO** |
| Backbone | `AdaLNPINTDenoiser` 512/12/512 | `IntervalTemporalUnet` / auxiliary variant | **NO** |
| Parameter count | **60,646,925** | 63,282,904 / 78,415,024 | **NO** |
| Horizon | 5 | puzzle adapter horizon 5 (different task) | n/a |
| obs / act dims | 480 / 3 | 83-D puzzle state / 5-D | **NO** |
| Loss | l1 | MeanFlow objective | **NO** |
| action_weight | 10 | not applicable | **NO** |
| Optimizer | Adam, betas (0.9, 0.999) | Adam, betas (0.9, **0.95**), 78-step warmup | **NO** |
| Learning rate | 8e-05 | **4e-05** | **NO** |
| Batch / grad-accum | 32 / 2 | 32 / 2 | yes |
| Training budget | **500,000 updates** | **5,000 updates** | **NO** |
| Normalization | SafeLimits / ParticleLimits | puzzle benchmark normalizer (`2f5ec416…`) | **NO** |
| `--rand_color` | yes | not applicable | **NO** |
| Save semantics | `save_freq` 1000, `label_freq` 100000 | single terminal `state_5000.pt` | **NO** |

Agreement on 1 of 20 fields (batch/grad-accum). **These are not canonical Flow
replicas by any reading.**

## 5. Has seed 43 ever been evaluated on the cube tasks?

**No.** Enumerating the `checkpoint` field of every Isaac Gym result file yields
exactly two distinct checkpoints across the entire project:

```
data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42
ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100
```

- Evaluated at Flow@1/@4? **No** — no seed-43 model has been run on PushCube.
- Object counts? **None.**
- Native or fixed horizon? **Neither.**
- Frozen episode sets? **Never used with seed 43.**
- Per-object success inspected? **No.**

(Textual `grep` hits for "43" inside NFE-study result files are episode indices
and numeric values; those files carry no `checkpoint` or `seed` field at all.)

## 6. Is the replication endpoint genuinely prospective?

**Yes, with respect to seed 43.**

The endpoint — equal-weight mean of per-object `Flow@4 − Flow@1` across 3/4/5
cubes at H=100 — was frozen in commit `afbd02e` before any seed-43 Flow model
existed, and **no seed-43 cube-task outcome has ever been observed**, because no
such model has ever been trained.

The standing caveat is unchanged and must be repeated: the metric is
**prospective for seed 43 but derived from exploratory seed-42 evidence**. It
was not preregistered before seed 42.

## 7. Decision

**Case B — the existing seed-43 artifacts differ scientifically.** Different
task, dataset, objective, backbone, parameter count, optimizer, learning rate,
and a 100x smaller training budget.

- **Do not reuse.** They cannot serve as a Flow replication.
- **Preserve.** They are valid records of the cancelled iMF diagnostic line;
  nothing about them is modified by this check.
- **Seed 44 likewise**: identical provenance (iMF/auxiliary, puzzle task, 5,000
  steps). There is **no** canonical seed-44 Flow checkpoint to reuse later, so a
  future seed 44 would also require a full ~32.5 GPU-h run. Not evaluated on the
  new endpoint, per instruction.
- **Seed number remains 43.** The `[42, 43, 44]` sequence is a project-wide
  training-seed convention; its prior use on a different experiment does not
  consume the number for this one. Both are seed 43 of their own line.

**TRAIN CLEAN SEED 43** from the exact-commit worktree `7506ce48`, as staged.
