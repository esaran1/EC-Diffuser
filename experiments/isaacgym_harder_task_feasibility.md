# What a 4-cube PushCube task would require

Status: **analysis only. Nothing implemented, nothing trained, nothing run.**
This documents what the change would cost, so the decision can be made on
evidence. It is contingent on the NFE study showing 3-cube is saturated.

## 1. Why a harder task is under consideration

From `experiments/isaacgym_flow_diagnosis.md`:

- Flow at 4 NFE reaches **95.8%** on 3-cube (92 of 96 episodes).
- Only **4 failures remain** in 96 episodes.
- The same checkpoint measured 84.4% on 32 episodes and 95.8% on 96 — an
  **11-point swing from episode sampling alone**.

A benchmark with 4 failures left and a ~10-point noise floor cannot resolve the
differences that remain interesting (probability path, state-prediction
fidelity, further NFE reduction). Headroom is needed before those questions are
answerable.

## 2. The finding that makes 4 cubes cheap

**The model input dimension does not depend on the number of cubes.**

| Component | Shape | Depends on cube count? |
|---|---|---|
| DLP observation | 2 views x **24 particles** x 10 features = 480 | **No** |
| Actions | 3 (EE dx, dy, dz) | **No** |
| Model input | 483 | **No** |
| `state_observations` (bookkeeping only) | (eef + N cubes) x 6 | Yes, but **not a model input** |

`n_kp_enc = 24` is fixed in the DLP encoder's `hparams.json`, so the encoder
emits exactly 24 particles per view whether the scene holds 3 cubes or 6.
Grepping the model package confirms `num_entity` never reaches the backbone —
it appears only in `eval_utils.py` (env construction), `rendering.py`
(visualization) and `setup.py` (naming). The PINT backbone is set-based over
particles.

**Consequence: a 4-cube evaluation requires no retraining of the policy and no
retraining of the DLP encoder.** This is the same zero-shot generalization
protocol the EC-Diffuser paper itself used, reporting rollouts on 4, 5 and 6
cubes from a 3-cube-trained policy.

## 3. Exactly what changes

### 3.1 Environment — one argument

`diffuser/diffuser/eval_utils.py:28` already does the work:

```python
isaac_env_cfg['env']['numObjects'] = args.num_entity
```

so `num_entity = 4` is sufficient. Supporting facts:

- The config directory in use is literally `env_config/generalization_num_cubes`,
  built for varying cube counts, with `numColors: 6` and `RandColor: True`.
- The DLP checkpoint it points at is `dlp_push_6C` — **trained on up to six
  cubes**, not three.
- `entity_to_steps` in the plan config already maps `4: 150`, so the episode
  length for 4 cubes is defined (150 steps, up from 100).
- Cube placement uses the general branch of `_reset_init_cube_states`, which
  loops over `self._init_obj_state_dict` with validity checking and has no
  3-cube assumption.
- `_obj_id_dict` / `_init_obj_state_dict` are built as
  `{f"cube{i+1}": ... for i in range(self.num_objects)}`, so they size
  themselves.

### 3.2 Assets — **verified, no work required**

Cubes are **generated procedurally**, not loaded from files
(`isaac_panda_push_env.py:331`):

```python
object_assets = [self.gym.create_box(self.sim, *([self.cube_size] * 3), asset_options)
                 for _ in range(self.num_objects)]
```

The loop runs `num_objects` times, so a fourth cube is created automatically.
`self.object_colors` (line 313) lists **nine** colours — red, green, blue,
yellow, purple, cyan, pink, brown, orange — well beyond the four needed.

**There is no asset dependency and no asset work.**

### 3.3 Our evaluation harness — small, mechanical

`experiments/scripts/isaacgym_control.py` currently hardcodes `num_entity = 3`
in the `Args` stand-in. It must become a flag. The diagnostics generalize
without change: `entity_positions` slices `[1:]` for cubes, and
`summarize_episode` operates on whatever cube count it is handed.

One real consideration: `episodeLength` rises 100 -> 150, so wall time per
episode grows ~50%.

### 3.4 Data — **no change at all for evaluation**

No new demonstrations are needed. The existing 3-cube checkpoint is evaluated
zero-shot. This is the crucial cost saving.

*If* a 4-cube-**trained** policy were later wanted, that is a different and much
larger project: it needs a 4-cube demonstration dataset, which this repository
does not have and cannot generate — the original data came from a trained ECRL
expert policy that is not in this tree (see the parked Isaac Lab work). That
option is **not** proposed here.

## 4. Estimated compute

Measured: 96 episodes at 100 steps takes ~89 s for Flow at 1 NFE and ~1280 s for
Gaussian at 100 NFE. Scaling by the 1.5x episode length:

| Item | Estimate |
|---|--:|
| Flow 4-cube, 96 episodes, one NFE setting | ~2-8 min depending on NFE |
| Gaussian 4-cube, 96 episodes | ~32 min |
| One full 6-arm NFE sweep at 4 cubes, 1 replicate | **~55 min** |
| Three replicates | **~2.7 GPU-h** |
| **Minimum useful probe** (Flow at best NFE + Gaussian, 96 episodes, 1 set) | **~40 min** |

**Recommended first step is the minimum probe, ~0.7 GPU-h**: two arms, one
replicate, purely to establish whether 4 cubes actually opens headroom. If
success drops into a range with room to move (say 40-80%), the full sweep is
justified. If it stays near ceiling, escalate to 5 or 6 cubes, which cost the
same to test.

## 5. Risks and honest caveats

1. **Zero-shot means the comparison changes character.** At 4 cubes both arms
   are out of distribution. That is a legitimate generalization benchmark and
   the paper's own framing, but it is no longer "the task the model was trained
   for", and results must be labelled as generalization rather than in-domain.
2. **It may overshoot.** EC-Diffuser's published 3-cube number is 0.894 and
   OGBench-style tasks collapse quickly with object count. 4 cubes might land
   near the floor rather than in a useful middle, which would make it as
   uninformative as a ceiling. This is exactly why the cheap probe comes first.
3. **Episode length is a confound.** 150 steps versus 100 gives more time to
   recover, partially offsetting the added difficulty. Worth reporting both at
   the default 150 and at a matched 100 if the result is marginal.

## 6. Recommendation

Do not implement yet. Contingent on the NFE study confirming saturation:

1. Add a `--num-entity` flag to the evaluation harness (small, mechanical) —
   this is the *only* code change required.
2. Run the **minimum probe** at ~0.7 GPU-h before committing to a full sweep.

Asset availability is verified, so no other prerequisite remains.
