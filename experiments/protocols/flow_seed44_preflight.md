# Preflight: Flow training seed 44 (PREDECLARED, NOT LAUNCHED)

Date: 2026-08-24. **Status: awaiting approval. Nothing launched.**

Seed 44 is the **third and final** predeclared Flow training seed. Its purpose is
to test whether the aggregate inference-budget effect is stable enough across
independent trainings to justify mechanism work. The object-count interaction
hypothesis is **not** a target and will not be pursued.

## 1. Numerical seed: 44

The project-wide sequence `[42, 43, 44]` was prospectively committed in `d5d84e5`
(`cube_controlled_v1.md:123`, `imf_auxiliary_multiseed_replication_results_v1.json:8`,
`imf_auxiliary_offline_failure_diagnostic_v1.json:20`). Flow occupies positions
1 and 2 with seeds 42 and 43, so the third is **44**.

No canonical Flow run at seed 44 exists (`data/panda_push/flow*/*seed44*` is
empty). The iMF seed-44 checkpoints under `phase7_runs/ogbench_puzzle_state/` are
puzzle-task diagnostics from a different objective, backbone, dataset and budget,
established in audit commit `217b2d6`; they are unrelated and untouched.

## 2. Worktree at the exact seed-42 commit

```
git worktree add --detach /home/jren313/ecdiff-seed44-7506ce48 7506ce48cc5e0ccbaf8ae41be7f3b8acf4944ba7
```

- HEAD verified `7506ce48cc5e0ccbaf8ae41be7f3b8acf4944ba7`; `git status --porcelain` empty.
- The `fast-generative-policies` workspace was **not** reset or modified; three
  worktrees now coexist (main, seed43, seed44).
- `ecdiffuser-data` → **symlink** to the main tree, so dataset and DLP are the
  *same bytes*.
- `data/` → **new empty directory** (0 entries), so seed-44 output cannot reach
  seed-42 (main tree) or seed-43 (its own worktree) files.

## 3. Canonical invariants — all verified in the historical code

| Invariant | Expected | Resolved | |
|---|---|---|---|
| diffusion wrapper | `ConditionalFlowMatching` | same | OK |
| probability path | linear (`x_t=(1-t)x0+tx1`, target `x1-x0`) | unchanged in `flow_matching.py` | OK |
| model | `AdaLNPINTDenoiser` | same | OK |
| hidden / projection dim | 512 / 512 | 512 / 512 | OK |
| layers / heads | 12 / 8 | 12 / 8 | OK |
| **parameter count** | **60,646,925** | **60,646,925** | **PASS** |
| horizon | 5 | 5 | OK |
| observation / action dim | 480 / 3 | features_dim 10 x 48 particles / 3 | OK |
| loss | l1 | l1 | OK |
| action_weight / loss_discount | 10 / 1 | 10 / 1 | OK |
| optimizer / lr | Adam / 8e-05 | Adam / 8e-05 | OK |
| batch / grad-accum | 32 / 2 (effective 64) | 32 / 2 | OK |
| optimizer updates | 500,000 | 500,000 | OK |
| save schedule | `save_freq` 1000, `n_saves` 5, `label_freq` 100000 | same | OK |
| normalizers | SafeLimits / ParticleLimits | same | OK |
| padding / conditioning | `use_padding` True, GoalDataset | same | OK |
| EMA | 0.995 | 0.995 | OK |
| n_diffusion_steps / time_scale | 4 / 1000.0 | 4 / 1000.0 | OK |
| max_particles / positional_bias / multiview | None / False / True | same | OK |

**`--rand_color` is mandatory**: it resolves the mode key to `3C_dlp_randcolor`,
which supplies 512/12/512. Without it the mode resolves to `3C_dlp` → 256/6/256,
a different architecture.

## 4. Programmatic config diff

```
keys compared: 48   differing: 2
  _results_folder: ...seed42 -> ...seed44
  seed: 42 -> 44
=> ONLY seed + output metadata
```

Effective config SHA256: `b33683244bd6f8d63d967268343f66346a47b943adcc3b6c2f4f82d0b4941a85`

## 5. Data identity (hashed from inside the worktree)

| Artifact | SHA256 | Matches |
|---|---|---|
| Dataset | `7abf83b82fcf2bae801ddae6fa6138d505f4957570d638e37da1e5a5290baf12` | yes |
| DLP encoder | `a8a1113048df79c0fd00cdd4539779a7b3c588cefc4e5f26eb50c0482f756236` | yes |

## 6. Environment

Python 3.8.20, torch 2.1.2+cu121, CUDA 12.1, conda `ecdiffuser-linux`, one
RTX 4080, `WANDB_MODE=offline`. Historical imports of
`ConditionalFlowMatching`, `AdaLNPINTDenoiser` and `Trainer` all succeed with
CUDA available. The `Trainer` signature has **17 kwargs**, confirming the
pre-drift code path (none of the five later additions).

**UNKNOWN, not assumed** (identical to seed 43): `CUDA_VISIBLE_DEVICES` and
`PYTHONPATH` at the original seed-42 launch were never recorded, and peak
training VRAM was never instrumented for seed 42 or 43. It will be reported as
**UNVERIFIED** unless actually measured.

## 7. Checkpoint rule — frozen, unchanged

> Evaluate the **EMA** weights of the terminal checkpoint whose internal `step`
> field is the maximum written by the unchanged save schedule — expected
> **499,000**, in the file labelled `state_400000.pt`.

The label is `self.step // 100000 * 100000`, not the step count. Verified across
seeds 42 and 43, both of which wrote internal steps 99k/199k/299k/399k/499k.
**No terminal save will be added, no intermediate checkpoint evaluated, and no
checkpoint selected on evaluation performance.**

## 8. GPU gate (already validated)

No unrelated process at or above 1000 MiB; total memory below 2000 MiB;
utilization at or below 5%; sustained 8 consecutive 60s checks; final query
immediately before exec. Any violation resets the counter. Small idle CUDA
contexts are permitted and will be recorded, with the wording **"no material
competing GPU workload detected at launch"** rather than any claim of
exclusivity.

Current GPU: 222 MiB, 0% util, no compute apps — the gate would pass now.

## 9. Launch command

```bash
cd /home/jren313/ecdiff-seed44-7506ce48
export PYTHONPATH="$PWD:$PWD/diffuser"
WANDB_MODE=offline python diffuser/scripts/train.py \
    --config config.pandapush_flow_single_gpu \
    --num_entity 3 \
    --rand_color \
    --seed 44
```

## 10. Cost

Measured 0.2343 s/step (seed 42) and 0.2339 s/step (seed 43) →
**~32.5 GPU-h** uncontended; seed 43 took 36.4 h with partial contention.
Storage ~2.4 GB.

## 11. Frozen evaluation endpoint (unchanged)

After training, evaluate the same checkpoint at Flow@1 and Flow@4 on the frozen
H=100 sets — 3-cube `35144910/0047468f/586e5b8d` (n=288 pooled), 4-cube
`5962c3ab` (n=96), 5-cube `f8dff00d` (n=96), all verified present.

Primary: `D_s = mean over {3,4,5} of [per-object(F4) - per-object(F1)]`.
Recomputed references: **D_42 = +0.0318**, **D_43 = +0.0387**.

No reweighting, no task dropped, no metric substitution, no redefinition.
