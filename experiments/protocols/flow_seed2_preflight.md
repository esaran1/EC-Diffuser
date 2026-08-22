# Pre-flight: Flow training seed 2 (PREDECLARED, NOT LAUNCHED)

Date: 2026-08-22. HEAD at time of writing: see `git rev-parse HEAD`.
**Status: awaiting approval. Nothing has been launched.**

## 1. Exact seed-42 training configuration

Extracted from the committed artifacts of the seed-42 run itself
(`data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/*.pkl`), not from
prose. Launch provenance from the offline W&B run `l1vkhnp9`:

```
program: diffuser/scripts/train.py
args   : ['--config', 'config.pandapush_flow_single_gpu', '--num_entity', '3', '--rand_color']
start  : 2026-08-10T20:30:48
host   : MEL05962D.me.gatech.edu, NVIDIA GeForce RTX 4080 x1
```

| Item | Value | Source |
|---|---|---|
| **Seed** | **42** | `ArgsParser.seed` default (`diffuser/utils/args.py:16`) — *not* passed on the command line |
| Dataset | `ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl` | `dataset_config.pkl` |
| Dataset SHA256 | `7abf83b82fcf2bae801ddae6fa6138d505f4957570d638e37da1e5a5290baf12` | recomputed |
| DLP encoder | `ecdiffuser-data/latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth` | `env_config/generalization_num_cubes/Config.yaml:20` |
| DLP SHA256 | `a8a1113048df79c0fd00cdd4539779a7b3c588cefc4e5f26eb50c0482f756236` | recomputed |
| Model | `AdaLNPINTDenoiser` | `model_config.pkl` |
| hidden_dim / projection_dim | 512 / 512 | `model_config.pkl` |
| n_layer / n_head | 12 / 8 | `model_config.pkl` |
| features_dim | 10 | `model_config.pkl` |
| max_particles / positional_bias | None / False | `model_config.pkl` |
| multiview | True | `model_config.pkl` |
| **Parameter count** | **60,646,925** | instantiated from `model_config.pkl` |
| Horizon | 5 | `diffusion_config.pkl` |
| observation_dim / action_dim | 480 / 3 | `diffusion_config.pkl` |
| Flow path | `ConditionalFlowMatching`, `n_timesteps=4`, `time_scale=1000.0` | `diffusion_config.pkl` |
| Loss | `l1`, `action_weight=10`, `loss_discount=1`, `loss_weights=None` | `diffusion_config.pkl` |
| Optimizer | Adam (`Trainer` default) | `diffuser/utils/training.py` |
| Learning rate | 8e-05 | `trainer_config.pkl` |
| Batch size | 32 | `trainer_config.pkl` |
| Gradient accumulation | 2 (effective batch 64) | `trainer_config.pkl` |
| EMA decay | 0.995 | `trainer_config.pkl` |
| **Total optimizer updates** | **500,000** (`n_train_steps`) | config, mode `3C_dlp_randcolor` |
| Checkpoint schedule | `save_freq=1000`, `n_saves=5`, `label_freq=100000` | `trainer_config.pkl` |
| Retained checkpoints (seed 42) | `state_{0,100000,200000,300000,400000}.pt` | directory listing |
| Normalization | `SafeLimitsNormalizer` (actions), `ParticleLimitsNormalizer` (particles) | `dataset_config.pkl` |
| Conditioning | `GoalDataset`, current obs at t=0 and goal at t=H-1, `use_padding=True` | `dataset_config.pkl` |
| Hardware | 1x RTX 4080, `cuda:0` | config + W&B metadata |

### Config-resolution note worth recording

`--rand_color` changes the resolved mode key to **`3C_dlp_randcolor`**, which is
what supplies `hidden_dim 512 / n_layers 12 / projection_dim 512`. Without that
flag the mode resolves to `3C_dlp`, which yields **256 / 6 / 256** and would be a
*different architecture*. The flag is therefore not cosmetic and must be passed.

## 2. Proposed seed-2 configuration

Identical in every field above, with:

| Item | Value |
|---|---|
| **Seed** | **2** (passed explicitly as `--seed 2`) |
| Output folder | `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed2` |

The seed reaches `random.seed`, `np.random.seed` and `torch.manual_seed` via
`diffuser/utils/setup.py:15-18`, called from `Parser.set_seed` (line 176-180).

**No hyperparameter is changed on the basis of anything learned from seed 42.**

## 3. Programmatic config diff

Computed by resolving `base["diffusion"]` updated with
`mode_to_args["3C_dlp_randcolor"]` for each seed:

```
total keys compared: 48   differing: 2
  [RUN-OUTPUT METADATA] _results_folder:
      seed42: .../3C_dlp_adalnpint_randcolor_H5_T4_seed42
      seed2 : .../3C_dlp_adalnpint_randcolor_H5_T4_seed2
  [SEED] seed:  42 -> 2

VERDICT: ONLY seed + run-output metadata differ
```

## 4. Checkpoint-selection rule — PREDECLARED

**Use the final training checkpoint, exactly as for seed 42.**

Seed 42's central checkpoint is `state_400000.pt`, whose *internal* step field is
**499000** (filenames are epoch labels offset from step counts). The seed-2
equivalent is therefore **the highest-numbered `state_*.pt` written by the run**,
which will likewise correspond to ~499k steps.

**The seed-2 checkpoint will NOT be chosen by evaluation performance.** No
seed-2 checkpoint sweep will be run, and if one ever is, it is exploratory and
may not be used for checkpoint selection.

Evaluation weights: **EMA**, as in every prior evaluation.

## 5. Expected cost, from measured throughput

Measured on the seed-42 run itself (`l1vkhnp9`), not estimated:

| Quantity | Value |
|---|--:|
| Steady-state median | **0.2343 s/step** |
| **Projected wall time, 500k steps** | **32.5 GPU-h** |
| Observed seed-42 wall time | 39.0 h (included contention spikes) |
| Storage | **~2.4 GB** (5 checkpoints x 485 MB, matching seed 42) |
| Peak training VRAM | **UNVERIFIED** — not recorded for this run. Seed 42 ran on the same 16 GB card alongside other work, so headroom is ample, but no measured number exists and none is invented here. |

If the GPU is contended the run will take longer; contention should be recorded,
not silently absorbed.

## 6. Exact launch command

```bash
source /home/jren313/miniconda3/etc/profile.d/conda.sh
conda activate ecdiffuser-linux
export PYTHONPATH="$PWD:$PWD/diffuser"
python diffuser/scripts/train.py \
    --config config.pandapush_flow_single_gpu \
    --num_entity 3 \
    --rand_color \
    --seed 2
```

Differs from the seed-42 invocation only by the explicit `--seed 2`.

## 7. Integrity requirements at launch

Launch only when `nvidia-smi --query-compute-apps` returns **no rows** and GPU
memory/utilization are effectively idle. To record: launch command, commit SHA,
start/end timestamps, exit code, full training log, GPU telemetry at launch,
final checkpoint SHA256, actual wall time, and peak VRAM.

## 8. Evaluation plan after seed 2 (predeclared)

Same trained checkpoint evaluated at **Flow @1** and **Flow @4** — no retraining
per NFE.

**Primary matrix (fixed horizon, removes the object-count/horizon confound):**

| Cubes | Horizon | Frozen episode set |
|---|--:|---|
| 3 | 100 | `replicate{0,1,2}_n96.pkl` (`35144910`, `0047468f`, `586e5b8d`) |
| 4 | 100 | `episode_set_4cube.pkl` (`5962c3ab`) |
| 5 | 100 | `episode_set_5cube.pkl` (`f8dff00d`) |

Secondary (native horizon) only if cheap and already frozen; it must not delay
the primary result.

Metrics per object count: full success, per-object success, cubes placed,
mean/max object-goal distance, completion distribution, contact rate,
wrong-direction pushes, exact NFE.

**Flow@4 − Flow@1 is computed separately for seed 42 and seed 2. Episodes are
never pooled across training seeds as though they came from one model.**

## 9. Predeclared interpretation

| Class | Criterion |
|---|---|
| **A — Replicated strongly** | Same direction and similar magnitude across the important tasks |
| **B — Replicated directionally** | Flow@4 > Flow@1 again, magnitude unstable |
| **C — Seed-sensitive** | Effect very small or inconsistent |
| **D — Contradicted** | Seed 2 reverses the central effect |

Only **A or B** justify spending another ~32.5 GPU-h on Flow seed 3. On **C or
D**: stop, do not brute-force more seeds, diagnose why the result is
training-seed sensitive.

Note the seed-42 fixed-horizon gaps this replicates against: **+0.043 at 3
cubes, −0.029 at 4 cubes, +0.081 at 5 cubes** — already non-monotonic and
already zero-crossing at 4 cubes.
