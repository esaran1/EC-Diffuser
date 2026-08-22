# Is `--num-entity 4` a valid compositional generalization test?

Date: 2026-08-21. **CPU-only verification, run before any GPU work.**
Verdict: **all five conditions hold. The probe is valid.**

## 1. Model architecture supports 4 entities without shape hacks — **YES**

The canonical checkpoint's `model_config.pkl` declares:

```
AdaLNPINTDenoiser
  features_dim: 10        <- PER PARTICLE, not per scene
  max_particles: None     <- no particle-count cap
  positional_bias: False  <- no particle-count-dependent bias table
  hidden_dim: 512, n_layer: 12, n_head: 8, block_size: 5
```

Every projection in `pint.py:96-127` maps `features_dim -> hidden -> projection`,
i.e. it acts on each particle independently; the transformer then attends over
the particle set. Nothing is sized by particle count.

**Verified empirically** by instantiating the canonical `model_config.pkl` and
running forward passes at several particle counts:

| Particles | Input | Output | Finite |
|--:|---|---|---|
| 48 | (2, 5, 483) | (2, 5, 483) | yes |
| 60 | (2, 5, 603) | (2, 5, 603) | yes |
| 96 | (2, 5, 963) | (2, 5, 963) | yes |

No modification, no padding, no reshape. 60,646,925 parameters throughout.

Note `positional_bias: False` matters: had it been True, the relative positional
bias table would be indexed by particle count and a change could have silently
mis-indexed. It is off in this checkpoint.

## 2. DLP representation supports the extra object — **YES, and the count does not change**

`dlp_push_6C/hparams.json`: `n_kp_enc = 24`.

The encoder emits **exactly 24 particles per view regardless of how many cubes
are in the scene**. At 2 views that is 48 particles x 10 features = 480 dims,
identical at 3 and 4 cubes. The model's observation dimension does not move.

The extra cube is represented by *reassigning* particles within the fixed budget,
which is precisely the entity-centric mechanism under test.

**Caveat, stated explicitly:** this DLP encoder was trained on scenes with up to
**six** cubes (`dlp_push_6C`). So the *representation* has seen 4-cube scenes even
though the *policy* has not. This is the same encoder EC-Diffuser used for its
own zero-shot generalization results, so the protocol matches the paper — but the
claim must be "zero-shot for the policy", not "zero-shot for the whole stack".

## 3. Environment genuinely creates a 4-cube task — **YES**

- `eval_utils.py:28` sets `isaac_env_cfg['env']['numObjects'] = args.num_entity`.
- `_obj_id_dict` / `_init_obj_state_dict` are built as
  `{f"cube{i+1}": ... for i in range(self.num_objects)}`, so they size themselves.
- Cube assets are **procedural**: `object_assets = [gym.create_box(...) for _ in
  range(self.num_objects)]` (`isaac_panda_push_env.py:331`), with **nine** colours
  defined (line 313). No asset file is needed.
- Reset uses the general branch (`isaac_panda_push_env.py:809-811`), looping over
  every cube with `check_valid=True` collision checking. All special branches
  (`AdjacentGoals`, `OrderedPush`, `SortPush`, `RandNumObj`) are absent or False
  in `env_config/generalization_num_cubes`, so none interferes.
- Goals are produced by the same `get_set_goal` path, which resets the env to the
  goal configuration and re-images it — so a 4-cube goal is a genuine 4-cube scene.
- `entity_to_steps` already maps `4: 150`, so episode length is defined.

## 4. No 4-cube information was used during training — **YES for the policy**

Training data (`push_cubes/3C_randcolor`, the dataset both canonical checkpoints
were trained on, per their `args.json`):

```
state_observations  (2000, 100, 4, 6)   <- 4 entity slots = 1 eef + 3 cubes
observations        (2000, 100, 48, 10)
```

and the training env config carries `numObjects: 3`, `RandNumObj: False`. There
is no 4-cube episode anywhere in the policy's training data, and no mixed-count
curriculum.

The DLP encoder is the one exception, documented in §2.

## 5. Success criteria scale correctly from 3 to 4 cubes — **YES, automatically**

`isaac_env_wrappers.py:405-438`:

```python
dist = np.linalg.norm(a_goal[..., :2] - d_goal[..., :2], axis=-1)
obj_goal_reached = dist < self.dist_threshold        # 0.04 m, per object
goal_frac_reached = np.mean(obj_goal_reached, -1)    # k / N
avg_obj_dist      = np.mean(dist, -1)
max_obj_dist      = np.max(dist, -1)
```

Per-object threshold is absolute (0.04 m, the cube's effective radius) and the
reductions are over the object axis, so:

- `goal_success_frac` becomes k/4 instead of k/3 — finer-grained, correct.
- Full success requires **all four** cubes placed, so the bar is genuinely
  stricter. This is the intended difficulty increase, not a metric artifact.
- No constant needs rescaling anywhere.

One consequence worth stating for interpretation: because full success needs all
4 rather than all 3, some success drop is expected **even from an unchanged
per-cube competence**. If per-cube success were an independent p, overall success
would fall from p^3 to p^4. At p = 0.96 that alone predicts 0.885 -> 0.849. Any
drop beyond that arithmetic is the interesting part, so **per-object success is
recorded as a first-class metric** and not just the all-or-nothing rate.

## Verdict

All five conditions hold. `--num-entity 4` is a valid zero-shot compositional
generalization test for the policy, with the single documented caveat that the
DLP encoder saw up to 6 cubes during its own training.

The probe is authorized to run once the GPU is free.

---

# Addendum: is `--num-entity 5` also valid? (2026-08-22)

Re-verified for **five** cubes. **All five conditions hold; the 5-cube probe is
valid.** Only the items that could differ from the 4-cube case are re-argued.

| # | Condition | Verdict | Evidence |
|---|---|---|---|
| 1 | Env genuinely creates 5 cubes | **YES** | `_obj_state_dict` and `object_assets` are both built by looping `range(num_objects)`; cubes are procedural `create_box`; **9** colours defined and `numColors: 6` >= 5; reset uses the general collision-checked branch; `entity_to_steps[5] = 200` so episode length is defined upstream. A runtime assert in the probe fails loudly if `env.num_objects` differs from the request. |
| 2 | Architecture accepts the representation | **YES** | `max_particles: None`, per-particle `features_dim: 10`. Re-ran forward passes through the canonical `model_config.pkl` at 48/72/120 particles — all clean. In practice the input is **unchanged at 480 dims** (see 3), so 5 cubes is not even a new shape for the model. |
| 3 | DLP uses the same fixed particle budget | **YES** | `n_kp_enc = 24` per view. Observation is 2 x 24 x 10 = 480 at 3, 4 **and** 5 cubes. The fifth cube is absorbed by reassigning particles inside a fixed budget — the entity-centric mechanism under test. |
| 4 | No 5-cube policy data in training | **YES** | Policy training tensors are `(2000, 100, 4, 6)` = eef + 3 cubes, with `numObjects: 3` and `RandNumObj: False`. No 4- or 5-cube episode exists. |
| 5 | Criterion requires 5/5, per-object is k/5 | **YES** | `check_success` is unchanged: per-object absolute 0.04 m threshold, then `mean` over the object axis. Verified arithmetically: N=3 -> 3/3, N=4 -> 4/4, N=5 -> **5/5**, and `goal_success_frac` becomes k/5. |

## The caveat, restated

**This is zero-shot _policy_ generalization, not zero-shot representation
learning.** The DLP encoder `dlp_push_6C` was trained on scenes containing up to
**six** cubes, so a 5-cube scene is comfortably inside the representation's
training range while being entirely outside the policy's. Every claim from this
probe must be phrased accordingly.

## One difficulty note specific to 5 cubes

Episode length rises 150 -> 200 steps (`entity_to_steps[5]`), so the policy gets
proportionally more time per cube than at 4 cubes did relative to 3. That
partially offsets the added difficulty and should be kept in mind when reading
the 3 -> 4 -> 5 scaling curve: the x-axis is not a pure difficulty axis, because
the time budget moves with it.
