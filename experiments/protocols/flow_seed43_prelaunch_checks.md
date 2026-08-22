# Flow replication seed: three pre-launch checks

Date: 2026-08-22. **Result: all three checks pass. One correction to the
previously proposed seed.**

## Check 1 — numerical seed against existing predeclarations

**A training-seed sequence already exists and was executed.**

Evidence:

- `experiments/protocols/cube_controlled_v1.md:123` — *"Seeds: 3 training seeds
  (42/43/44); paired evaluation seed set."*
- `experiments/pilots/imf_auxiliary_multiseed_replication_results_v1.json:8` —
  `"training_seeds": [42, 43, 44]`, with `per_seed` entries for 42, 43 and 44.
- `experiments/pilots/imf_auxiliary_offline_failure_diagnostic_v1.json:20` —
  `"training_seeds": [42, 43, 44]`.
- First committed in `d5d84e5`.
- Seeds 43 and 44 were actually run: checkpoints exist under
  `data/phase7_runs/ogbench_puzzle_state/imf_aux_multiseed_*_seed4{3,4}_5000/`.

**Final numerical seed: 43.** Not 2. The earlier proposal of `--seed 2` was
"replication run #2" reasoning, which the instruction explicitly prohibits when
a sequence exists. The Flow run at seed 42 is position 1 of `[42, 43, 44]`, so
the next independent Flow model is **seed 43**.

Recorded before launch, in this commit.

## Check 2 — execution-environment equivalence

Seed 42's environment, recovered from the offline W&B run `l1vkhnp9`
(`files/wandb-metadata.json`) — not reconstructed from memory:

| Item | Seed 42 (recorded) | Planned seed 43 | Match |
|---|---|---|---|
| **Git commit** | `7506ce48cc5e0ccbaf8ae41be7f3b8acf4944ba7` | current HEAD | **see drift analysis** |
| Python | 3.8.20 | 3.8.20 | yes |
| PyTorch / CUDA | 2.1.2+cu121 / 12.1 | 2.1.2+cu121 / 12.1 | yes |
| Conda env | `/home/jren313/miniconda3/envs/ecdiffuser-linux` | same | yes |
| Host / OS | MEL05962D, Linux-7.0.0-28-generic, glibc2.17 | same | yes |
| GPU | 1x RTX 4080 (17171480576 B) | same physical card | yes |
| Repo root | `/home/jren313/EC-Diffuser-1` | same | yes |
| Entrypoint | `diffuser/scripts/train.py` | same | yes |
| Dataset SHA256 | `7abf83b8...0baf7` | same file | yes |
| DLP SHA256 | `a8a11130...756236` | same file | yes |

### Code drift since the seed-42 commit — analysed, not assumed

Eight training-path files compared with
`git diff 7506ce48 HEAD -- <file>`:

| File | Status |
|---|---|
| `diffuser/scripts/train.py` | UNCHANGED |
| `diffuser/diffuser/models/flow_matching.py` | UNCHANGED |
| `diffuser/diffuser/datasets/sequence.py` | UNCHANGED |
| `diffuser/diffuser/datasets/normalization.py` | UNCHANGED |
| `diffuser/config/pandapush_flow_single_gpu.py` | UNCHANGED |
| `diffuser/diffuser/utils/setup.py` | UNCHANGED |
| `diffuser/diffuser/models/pint.py` | **CHANGED** |
| `diffuser/diffuser/utils/training.py` | **CHANGED** |

Both changes are **additive with backward-compatible defaults**:

- `pint.py`: refactors `self.time_mlp(time)` into a `_time_embedding` hook and
  adds a new subclass `IntervalAdaLNPINTDenoiser`. The base class's behaviour is
  unchanged when `interval=None` (the default), and the config instantiates
  `AdaLNPINTDenoiser`, not the subclass.
- `training.py`: adds `max_grad_norm`, `collect_step_diagnostics`, `adam_betas`,
  `lr_warmup_steps`, `dataloader_seed`. Verified defaults: `None`, `False`,
  `(0.9, 0.999)`, `0`, `None` — i.e. no clipping, no warmup, torch's own Adam
  betas, and `generator=None` so the DataLoader draws from the global RNG
  exactly as before. Seed 42's `trainer_config.pkl` does not contain any of
  these keys, so they take defaults on replay.

**Conclusion: the executed training computation is equivalent.** The code is not
byte-identical to `7506ce48`, and that is stated rather than glossed.

### What is UNKNOWN and not assumed

- `CUDA_VISIBLE_DEVICES` at seed-42 launch: **not recorded** by W&B. The config
  pins `device: cuda:0` and the machine has one GPU, so the effective device is
  the same, but the variable's value is unknown.
- `PYTHONPATH` at seed-42 launch: **not recorded**.
- W&B mode: the run directory is `offline-run-*`, so seed 42 ran with W&B in
  **offline** mode. Seed 43 will match.
- `cuda` field in W&B metadata is `None`; the CUDA version above comes from the
  current interpreter, not from the seed-42 record.
- Peak training VRAM for seed 42: **never recorded**.

## Check 3 — checkpoint selection by actual training state

The filename is **not** the step count. From
`diffuser/diffuser/utils/training.py:267-269`:

```python
if self.save_freq and self.step % self.save_freq == 0:
    label = self.step // self.label_freq * self.label_freq   # label_freq = 100000
    self.save(label)
```

The label is `self.step` floored to the nearest 100,000, so **every save within a
100k window overwrites the same file**. Verified across all five seed-42
checkpoints:

| File | Internal `step` | `step // 100000 * 100000` | Consistent |
|---|--:|--:|---|
| `state_0.pt` | 99,000 | 0 | yes |
| `state_100000.pt` | 199,000 | 100,000 | yes |
| `state_200000.pt` | 299,000 | 200,000 | yes |
| `state_300000.pt` | 399,000 | 300,000 | yes |
| **`state_400000.pt`** | **499,000** | 400,000 | yes |

**Did training execute 500,000 updates or steps 0-499,999?** Both descriptions
are the same thing: `train.py` runs `for step in range(n_train_steps)` with
`n_train_steps = 500000`, i.e. indices 0..499,999 = **500,000 optimizer
updates**. `self.step` increments once per iteration, and the final save fires
at the last `self.step` divisible by `save_freq=1000` inside that range, which
is **499,000**. The last 1,000 updates are therefore not represented in any
saved file — for seed 42 or for seed 43.

Checkpoint contents: `['step', 'model', 'ema']`. Evaluation has always used the
**`ema`** weights (EMA decay 0.995, updated every 10 steps after step 2,000).

### Predeclared replication rule

> **Evaluate the EMA weights from the terminal checkpoint produced by the
> unchanged save schedule — the file whose internal `step` field is the maximum
> written by the run, expected to be 499,000, matching seed 42's actual
> optimizer state.**

The rule is stated in terms of optimizer state, not filename. The seed-43 run
must be verified to have reached internal step 499,000 before evaluation; if it
terminates elsewhere, the discrepancy is reported rather than absorbed.

**No seed-43 checkpoint may be selected using evaluation performance.**

## Frozen replication endpoint

**Primary seed-level metric (prospectively defined, before seed-43 training
begins):**

> Equal-weight mean of `per_object_success(Flow@4) − per_object_success(Flow@1)`
> across 3-cube H=100, 4-cube H=100, and 5-cube H=100.

Seed 42's values on that metric: **+0.043, −0.029, +0.081**, equal-weight mean
**+0.0316**.

This metric is **prospectively defined for seed 43 but derived from an
exploratory seed-42 finding.** It was *not* preregistered before seed 42, and
must never be described as such.

Secondary diagnostics, retained but not primary: full-success difference, cubes
placed, object-goal distance difference, per-task effects.

Within-seed p-values do not constitute algorithm-level replication and will not
be presented as such.
