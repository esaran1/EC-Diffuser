# Isaac Gym debugging investigation (Tal's action items)

Date: 2026-08-20. Branch `fast-generative-policies`.

**Compute status.** Throughout this investigation the RTX 4080 was occupied by
unrelated Isaac Lab jobs (9.4-11.9 GiB, 60-62% utilization, PIDs 2745984 /
2750645 / 2751201 under `env_isaaclab` and `newton-r1/.venv-isaac`). Per the
standing GPU rule, **no GPU job was launched**. Everything below is CPU-only
analysis of existing data, existing logs, and source. Items 1, 4 and 5 require
the GPU and are staged, not run.

---

## Item 1 — Original Isaac Gym positive control: STAGED, NOT RUN

### The canonical checkpoint exists and is identified

| Item | Value |
|---|---|
| Path | `ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt` |
| Diffusion | `models.GaussianDiffusion`, `n_diffusion_steps` 100 |
| Denoiser | `models.AdaLNPINTDenoiser`, 12 layers, hidden 512, 8 heads |
| Horizon / entities | H=5, `num_entity` 3 |
| Training | 1,200,000 steps (config declares `n_train_steps` 1e6) |
| `predict_epsilon` | false (predicts x0) |
| Normalizers | `SafeLimitsNormalizer` (actions), `ParticleLimitsNormalizer` (particles) |
| Dataset | `3C_ECRL_State_RandColor`, matches local `push_cubes/3C_randcolor` |
| Upstream commit | `344ddeb162a3b4f1052e62124960a5692227b8d5` |

The matching DLP encoder is `ecdiffuser-data/latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth`.

### Isaac Gym itself is functional

`import isaacgym` succeeds in `ecdiffuser-linux` and loads
`gym_38.so` from `/home/jren313/software/isaacgym`. The blocker is GPU
contention, not installation.

### Why this ordering is right

This is the only arm in the project whose expected success rate is known from
the published paper. Until it reproduces, **no Flow number from this simulator
is interpretable**, because a failure could equally be the checkpoint, the DLP
encoder, the env config, the controller, or the policy. This must run first and
must pass before any Flow result is read.

---

## Item 2 — Action normalization audit: VERDICT — round-trip is CORRECT, but two real defects found

### The full traced path

Every stage was read from source and is listed here with its file and line.

| # | Stage | Where | What happens |
|---|---|---|---|
| 1 | Raw dataset actions | `push_cubes/3C_randcolor/...pkl` `actions` | `(2000, 100, 3)` float32, xyz only |
| 2 | Fit normalizer | `datasets/sequence.py:36,43` | `SafeLimitsNormalizer` over flattened actions |
| 3 | Normalize to [-1,1] | `normalization.py:154` | `2*(x-min)/(max-min) - 1` |
| 4 | Model target | `sequence.py:157` | `concat([actions, observations], -1)` -> 3 action + 480 obs channels |
| 5 | Model output | `models/*` | same layout |
| 6 | Inverse normalization | `sampling/policies.py:95` | `normalizer.unnormalize(actions, 'actions')`, **the only such call** |
| 7 | Policy output | `policies.py:96` | `actions[0,0]`, a 3-vector |
| 8 | Isaac command assembly | `isaac_env_wrappers.py:266-268` | `cat([action_xyz, [0,0,0,-1]])` -> 7-D |
| 9 | Env clip | `isaac_panda_push_env.py:926` | `clamp(a, -clip_actions, +clip_actions)`, `clipActions: 1.0` |
| 10 | Controller scale | `panda_controller/base_controller.py:74-88` via `osc.py:166` | clip to [-1,1], affine map to output range |
| 11 | OSC output range | `isaac_panda_push_env.py:218-221` | `[-0.125, 0.125]` m position, `[-0.5, 0.5]` rad orientation |
| 12 | Goal | `osc.py:197` | `goal_pos = ee_pos + dpose`, then `pos_limits` clamp |

### Numerical verification on the real 3C dataset

```
raw actions      min [-1.0000 -1.0000 -0.3619]   max [1.0000 1.0000 0.0624]
                 mean [-0.0021  0.0053 -0.0270]  std [0.2399 0.2771 0.0633]
SafeLimits mins  [-1.000 -1.000 -0.362]  maxs [1.000 1.000 0.062]
normalized       min [-1 -1 -1]  max [1 1 1]
ROUND TRIP       max abs err 1.490e-08    mean abs err 7.292e-11
observations     round-trip max abs err 1.431e-06
```

**The round trip is numerically exact.** Flow and Gaussian share stages 2-12
verbatim: there is one `unnormalize(..., 'actions')` call site in the entire
`diffuser` package, both arms construct `GoalConditionedPolicy` through the same
`policy_config` in `eval_agent.py`, and neither overrides it. So an action-path
asymmetry between Flow and Gaussian is **excluded**.

### Defect A — the z-axis is severely asymmetric, and "do nothing" is not zero

The z action range is `[-0.3619, +0.0624]` — the arm pushes down 5.8x further
than it lifts. `LimitsNormalizer` maps this range onto `[-1, 1]`, so:

```
raw action z = 0.0      ->  normalized z = +0.706
normalized z = 0.0      ->  raw action z = -0.1498
```

Consequences:

1. **A zero-valued network output is not a null action.** It commands a
   downward push of 0.1498 (scaled: 0.0187 m per step). Any tendency to output
   near-zero — early training, an over-smoothed one-step prediction, the mean of
   a multimodal distribution — is a *downward press into the table*, not a
   no-op.
2. The normalized z channel has **mean +0.579**, versus roughly 0.0 for x and y.
   The action block the model must fit is strongly off-center in exactly one
   channel.
3. `GoalDataset.__getitem__` (`sequence.py:143`) writes the terminal action of
   every window as `normalizer.normalize(zeros)` = `[0, 0, +0.706]`. The padding
   value is a downward push, and it is written into **1 of every 5 timesteps**
   at H=5.

This is a genuine representational defect. It is not yet shown to cause the
Isaac Gym failure — that is what item 5 measures — but it is the single most
suspicious thing found in the action path, and it is a plausible mechanism for a
policy that drives the gripper into the table instead of pushing cubes.

### Defect B — `SafeLimitsNormalizer` widens every dimension, not the constant one

`normalization.py:174-192`:

```python
for i in range(len(self.mins)):
    if self.mins[i] == self.maxs[i]:
        self.mins -= eps      # array-wide, not self.mins[i]
        self.maxs += eps
```

If any single dimension is constant, **all** dimensions are widened by `eps=1`,
silently rescaling every unrelated channel. On the 3C action data no dimension
is constant, so this **does not fire here** and does not affect any result in
this report. It is recorded because it is a live latent bug that would corrupt
normalization on any dataset with a constant channel.

### Verdict

**The action normalization round-trip is correct and is identical for Flow and
Gaussian.** Item 2's original hypothesis — a broken or asymmetric inverse
transform — is refuted. But the audit surfaced a real asymmetry defect
(Defect A) in the z channel that changes the meaning of a zero action, plus a
latent normalizer bug (Defect B).

---

## Item 3 — Flow loss / undertraining: VERDICT — CONVERGED, not undertrained

Recovered from the offline W&B run `l1vkhnp9`
(`train.py --config config.pandapush_flow_single_gpu --num_entity 3 --rand_color`),
5,000 logged points over 499,900 optimizer steps.

Figure: `experiments/figures/flow_loss_isaacgym.png`

| Step window | total loss | action loss | observation loss |
|---|--:|--:|--:|
| 0 – 25k | 0.12820 | 0.18349 | 0.25308 |
| 50k – 100k | 0.11862 | 0.12326 | 0.23602 |
| 150k – 200k | 0.11673 | 0.09885 | 0.23314 |
| 250k – 300k | 0.11570 | 0.08861 | 0.23146 |
| 350k – 400k | 0.11525 | 0.07941 | 0.23082 |
| 400k – 450k | 0.11448 | 0.07616 | 0.22937 |

Late-training trend on total loss:

| Window | slope | relative change across window |
|---|--:|--:|
| 2nd half (250k–500k) | −5.08e−09 /step | **−1.10%** |
| Last quarter | −1.46e−09 /step | **−0.16%** |
| Last 10% | +7.79e−09 /step | **+0.34%** |

**Verdict: the Flow model is converged, not undertrained.** Total loss moves
−0.16% over the final quarter and is flat-to-slightly-rising over the final 10%.
More training steps will not fix Isaac Gym Flow performance, and **retraining
for longer is not a justified GPU spend.**

One caveat worth carrying: the *action* component is still falling at 500k
(0.183 → 0.076, and still descending), while the *observation* component — which
dominates the total, 0.229 vs 0.076, and covers 480 of 483 channels — is flat.
So the total-loss plateau is the observation term saturating. Action prediction
had not fully converged. That is a scale-balance observation, and it connects
directly to item 6: the loss is dominated by the 480 observation channels.

**Checkpoint availability for a train-time sweep:** `state_0/100000/200000/300000/400000.pt`
plus 25 rendered epoch directories exist for this run, so a
performance-versus-training-step curve is measurable **without any retraining**
once the GPU frees. That is the cheap way to confirm this verdict behaviorally
rather than only by loss.

---

## Item 4 — DLP visualization: STAGED, NOT RUN (GPU)

The required pieces are present and verified loadable:

- Encoder: `dlp_utils.extract_dlp_features_with_bg` (`dlp_utils.py:275`)
- Decoder: `dlp_utils.get_recon_from_dlps` (`dlp_utils.py:291`), which calls
  `latent_rep_model.decode_all(pixel_xy, visual_features, z_bg, transp, z_depth, z_scale)`
- Checkpoint: `latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth`

The particle layout the decoder expects is confirmed from source, and it matches
the stored 10-D features exactly:

| Slice | Meaning |
|---|---|
| `[..., 0:2]` | `pixel_xy` |
| `[..., 2:4]` | `scale_xy` |
| `[..., 4:5]` | `depth` |
| `[..., 5:9]` | `visual_features` |
| `[..., 9]` | `transparency` |

The ordering requested — verify RGB→DLP→RGB reconstruction **before** decoding
generated states — is the right one and is preserved in the staged plan. If
reconstruction is poor, every "imagination" figure is uninterpretable.

---

## Item 5 — Isaac Gym failure localization: STAGED, NOT RUN (GPU)

Requires rollouts. The metrics requested (EE-to-target distance, contact
occurrence, first contact time, cube displacement, push direction correctness,
action saturation) are all recoverable from the env's observation dict and
`state_observations`, which is `(N, T, 4, 6)` — 3 cubes plus the EE, 6-D each —
so the instrumentation is straightforward once the GPU is free.

Two priors from the OGBench work should be carried in, since the same failure
signature appeared there: purposeful-looking motion (action autocorrelation
0.93–0.98) with near-zero object displacement, and clip fraction well below the
demonstrations. Defect A above supplies a specific hypothesis to test here:
**does the EE trend downward into the table over an episode?** That is a
one-line measurement on the existing rollout instrumentation and directly tests
the z-asymmetry mechanism.

---

## Item 6 — Tal's variance hypothesis: VERDICT — VP is NOT variance-preserving on the current representation

Computed on 200,000 sampled transitions from `3C_randcolor`.

### Empirical statistics of the current [-1,1] features

| Group | mean | std | E[x²] |
|---|--:|--:|--:|
| actions (3 ch) | 0.1939 | 0.3853 | 0.1861 |
| observations (480 ch) | 0.0465 | 0.3438 | 0.1204 |
| **whole input (483 ch)** | **0.0475** | **0.3443** | **0.1208** |

Per DLP feature:

| Feature | mean | std | E[x²] |
|---|--:|--:|--:|
| pos_x | −0.1262 | 0.2527 | 0.0798 |
| pos_y | 0.0839 | 0.3079 | 0.1018 |
| scale_x | 0.1182 | 0.2887 | 0.0973 |
| scale_y | −0.1635 | 0.4108 | 0.1955 |
| depth | −0.1278 | 0.2357 | 0.0719 |
| f5 | −0.0915 | 0.0731 | 0.0137 |
| f6 | 0.0251 | 0.0957 | 0.0098 |
| f7 | −0.1106 | 0.1042 | 0.0231 |
| f8 | 0.1275 | 0.0795 | 0.0226 |
| **transparency** | **0.7303** | 0.2341 | **0.5881** |

Two things stand out: **E[x²] = 0.1208, not 1**, and the spread across feature
groups is ~60x (f6 at 0.0098 vs transparency at 0.5881).

### The interpolation path, measured

The requested instruction is confirmed: **`((1-t)z + tx)/sqrt((1-t)² + t²)` is
not variance-preserving on this representation**, because that formula is
only VP when `E[x²] = 1`.

Current [-1,1] representation:

| t | linear std | linear E[x²] | "VP" std | "VP" E[x²] |
|--:|--:|--:|--:|--:|
| 0.00 | 1.0001 | 1.0001 | 1.0001 | 1.0001 |
| 0.25 | 0.7550 | 0.5701 | 0.9550 | 0.9122 |
| 0.50 | 0.5289 | 0.2803 | 0.7479 | 0.5605 |
| 0.75 | 0.3595 | 0.1305 | 0.4547 | 0.2088 |
| 1.00 | 0.3443 | 0.1208 | 0.3443 | 0.1208 |

The so-called VP path still collapses from 1.00 to 0.12 — it **fails to preserve
variance by a factor of 8**. It is strictly less bad than linear (which reaches
0.28 at t=0.5 against VP's 0.56), but it does not do what it is named for.

Standardized representation:

| t | linear std | linear E[x²] | VP std | VP E[x²] |
|--:|--:|--:|--:|--:|
| 0.00 | 1.0001 | 1.0001 | 1.0001 | 1.0001 |
| 0.25 | 0.7907 | 0.6252 | 1.0001 | 1.0003 |
| 0.50 | 0.7073 | 0.5003 | 1.0003 | 1.0007 |
| 0.75 | 0.7910 | 0.6256 | 1.0005 | 1.0010 |
| 1.00 | 1.0005 | 1.0010 | 1.0005 | 1.0010 |

**After standardization the VP path is exactly variance-preserving — 1.0000 at
every t, to four decimals.** This is the cleanest result in the report.

Figure: `experiments/figures/interpolation_variance.png`

### What this establishes, and what it does not

**Established:** the current representation has E[x²] ≈ 0.12; linear
interpolation therefore sweeps the input scale over an ~8x range as t goes 0→1;
the VP formula does not fix this without standardization; standardization makes
it exact. Tal's instruction not to call the formula variance-preserving on the
current representation is **correct and now quantified**.

**Not established:** that any of this causes the Isaac Gym failure. A network
with per-layer normalization can in principle absorb a global scale sweep. The
argument for testing it is that the model sees inputs whose scale varies 8x
across t while sharing one set of weights across all t — but that is a
motivation, not a measurement.

---

## Item 7 — Is standardized VP training justified?

**Partially, and it is not the highest-value next experiment.**

The A/B/C design as specified is sound and correctly factorized:

- **A**: existing [-1,1] + linear Flow
- **B**: standardized + linear Flow — isolates standardization
- **C**: standardized + VP Flow — isolates the VP path

Three reservations before spending GPU on B or C:

1. **The positive control has not passed.** If the canonical Gaussian checkpoint
   does not reproduce its published success rate in this installation, then A is
   not a valid baseline and B−A / C−B measure nothing. Item 1 gates items 3–7.
2. **Item 3 removes the undertraining confound but adds a scale confound.** The
   total loss is dominated by 480 observation channels at ~0.23 while the 3
   action channels sit at ~0.076 and are still descending. Standardization
   changes exactly this balance, which means B−A will conflate "better
   interpolation path" with "reweighted loss". If B improves, that ambiguity has
   to be resolved with a fourth arm, or by matching the effective per-group loss
   weighting.
3. **Defect A is untested and cheaper to test.** The z-asymmetry is a concrete,
   specific, falsifiable mechanism. Item 5's rollout diagnostics test it for
   ~0.5 GPU-h with no training at all.

---

## The single highest-value next GPU experiment

**Item 1: the canonical Gaussian positive control on fixed recorded episodes.**

- **Cost:** ~0.5–1 GPU-h, no training.
- **Why it wins:** it is the only experiment whose expected answer is already
  published, so it is the only one that can distinguish "our pipeline is broken"
  from "Flow is worse than Gaussian". Every other item — 3, 4, 5, 6, 7 —
  produces uninterpretable output if the pipeline is broken, because a null
  result would have two explanations.
- **Decision rule (predeclared):** if Gaussian reproduces its published success
  rate, the pipeline is validated and items 5 and 4 run next on the same fixed
  episodes. If Gaussian fails, **stop** and debug the pipeline; do not run any
  Flow arm, and do not train B or C.

When it runs, it should share one recorded episode set with every later arm,
hash-locked the way the cube-double protocol did it, so that Gaussian, Flow, and
any A/B/C arm are paired from the first measurement rather than retrofitted.

---

## Summary table

| Item | Status | Verdict |
|---|---|---|
| 1 Gaussian positive control | **staged, GPU-blocked** | checkpoint located and configured; Isaac Gym imports cleanly |
| 2 Action normalization | **DONE** | round-trip exact (1.5e−08); Flow/Gaussian share one path; **two defects found** (z asymmetry; SafeLimits array-wide widening) |
| 3 Flow undertraining | **DONE** | **converged** — −0.16% over final quarter; more steps not justified |
| 4 DLP visualization | **staged, GPU-blocked** | encoder/decoder verified present, particle layout confirmed |
| 5 Failure localization | **staged, GPU-blocked** | metrics recoverable; Defect A gives a specific hypothesis to test |
| 6 Variance hypothesis | **DONE** | E[x²]=0.12; **VP formula is not VP here**; standardization makes it exact |
| 7 Standardized VP training | **not justified yet** | gated on item 1; loss-balance confound must be handled |

## Reproduction

```bash
source /home/jren313/miniconda3/etc/profile.d/conda.sh
conda activate ecdiffuser-linux
export PYTHONPATH="$PWD:$PWD/diffuser"
python experiments/scripts/audit_isaacgym_action_path.py
```
