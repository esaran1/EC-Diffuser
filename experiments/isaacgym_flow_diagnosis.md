# Isaac Gym Flow diagnosis: control, math, and probability path

Date: 2026-08-20. Branch `fast-generative-policies`. Continues
`experiments/isaacgym_debug_investigation.md`; prior CPU-only findings are
consumed, not repeated.

All arms below run on **one recorded episode set**,
`experiments/isaacgym_episode_set_v1.pkl`, SHA256
`fb0a95a710ceb9fa2a36f73e1b4e36ff78563b8dddc8fdea4a662b849694afba`, 32
episodes of 3-cube PushCube with fixed initial and goal cube positions. Every
result file records that hash, so the Gaussian/Flow comparison is paired by
construction rather than retrofitted.

---

## 1. Gaussian positive control — **PASSES**

Canonical checkpoint
`ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt`,
EMA weights, 100 diffusion steps (verified: exactly 100 denoiser calls per
plan), H=5, 3 cubes, random colors.

| Metric | Value |
|---|--:|
| **Success (all 3 cubes placed)** | **30/32 = 93.75%** |
| 95% CI (Clopper-Pearson) | [79.2%, 99.2%] |
| Goal-success fraction | 0.979 |
| Mean object-goal distance | 0.0162 m (threshold 0.04) |
| Cubes placed / episode | 2.94 of 3 |
| Cubes moved / episode | 2.94 |
| Cubes moved *away* from goal | **0.00** |
| Contact rate | **100%** of episodes |
| Cubes contacted / episode | 2.94 |
| Mean first-contact step | 0.8 |
| Action clip fraction | 0.46% |
| Wall time | 425 s |

**The published reference is 0.894 ± 0.025** for EC-Diffuser on 3-cube
PushCube (arXiv:2412.18907, Table 1). Our 93.75% [79.2, 99.2] contains it.

Both non-successes still placed 2 of 3 cubes (goal-success fraction 0.667) with
all three contacted and moved toward goal — near-misses, not breakdowns.

**Verdict: the Isaac Gym pipeline is faithful.** Environment, config,
checkpoint, DLP encoder, normalization, controller and action semantics all
reproduce published behavior end to end. The hard gate in
`experiments/protocols/isaacgym_positive_control_v1.md` is cleared, and any
Flow deficit measured on this same episode set is attributable to the Flow arm
rather than to the pipeline.

This also **retires Defect A as a pipeline-level explanation**: the z-action
asymmetry documented in the prior report is present in exactly this pipeline,
and Gaussian achieves 93.75% through it. EE height is essentially unchanged
across an episode (1.0457 -> 1.0589), so the arm is not pressing into the
table. Defect A remains a real representational oddity, but it is not
sufficient to break the task.

---

## 1b. Flow arm on the identical episodes — **FLOW WORKS**

Checkpoint `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42`,
EMA weights, **4 solver steps** (verified: exactly 4 denoiser calls per plan),
same episode set hash, same policy class, same normalizer, same controller.

| Metric | Gaussian | **Flow** |
|---|--:|--:|
| Network calls per plan | 100 | **4** |
| **Success** | 30/32 = **93.75%** | 27/32 = **84.38%** |
| 95% CI | [79.2%, 99.2%] | [67.2%, 94.7%] |
| Goal-success fraction | 0.979 | 0.917 |
| Mean object-goal distance | 0.0162 | 0.0194 |
| Cubes placed / episode | 2.94 | 2.75 |
| Cubes moved / episode | 2.94 | 2.94 |
| Cubes moved closer | 2.94 | 2.91 |
| Cubes moved farther | 0.00 | 0.03 |
| Contact rate | 100% | **100%** |
| Cubes contacted / episode | 2.94 | 2.94 |
| Min EE-to-cube distance | 0.0259 | 0.0267 |
| Action abs mean | 0.109 | 0.115 |
| Action clip fraction | 0.46% | 0.33% |
| EE z start -> end | 1.046 -> 1.059 | 1.047 -> 1.057 |
| EE path length | 2.310 | 2.297 |

**Paired McNemar test: b=4, c=1, exact p = 0.375.** On 32 episodes the
difference between Gaussian and Flow is **not statistically significant**.

This is the single most important result in the investigation, and it
contradicts the working assumption carried in from the OGBench phase.

**Flow matching is not broken.** At **4 network calls instead of 100 — a 25x
reduction in inference cost — it reaches 84% success on 3-cube PushCube**, a
task where the published Gaussian reference is 89.4%. Contact behavior is
essentially identical to Gaussian: same 100% contact rate, same 2.94 cubes
contacted, same EE path length to within 0.6%, same action magnitude, and no
downward drift.

### Failure taxonomy (item 8), measured in Isaac Gym

All five Flow failures fall in category **6, "solves some objects but fails
composition"**:

| Episode | Category | Cubes placed |
|---|---|--:|
| 15 | insufficient push / partial composition | 1/3 |
| 18 | insufficient push / partial composition | 2/3 |
| 20 | insufficient push / partial composition | 1/3 |
| 23 | insufficient push / partial composition | 2/3 |
| 28 | pushed one cube wrong direction | 1/3 |

Not one failure is category 1-4. Flow never fails to approach, never fails to
contact, and (except once) never pushes the wrong way. In every failing episode
it contacted all three cubes and moved all three, with 2-3 of them moving
*closer* to goal. The deficit is finishing, not planning or contact.

**The OGBench contact-failure conclusion does not transfer**, and this report
does not reuse it. In OGBench the arm travelled metres while cubes moved
fractions of a millimetre. Here Flow moves cubes 0.1995 m on average — slightly
*more* than Gaussian's 0.1982 m — and closes 0.179 m of goal distance.


---

## 2. Primary-source verification against EC-Diffuser and ECRL

Checked against arXiv:2412.18907 (EC-Diffuser, ICLR 2025) and the official
repository in this tree.

| Paper fact | Our code | Agrees? |
|---|---|---|
| DLP features and actions normalized to [-1,1] | `ParticleLimitsNormalizer`, `SafeLimitsNormalizer` | **yes** |
| Jointly generates future states **and** actions | `concat([actions, observations])`, `sequence.py:157` | **yes** |
| Only the first action executed, MPC-style re-query | `exe_steps: 1`; policy returns `actions[0,0]` | **yes** |
| Generated DLP states decodable with the pretrained decoder | `dlp_utils.get_recon_from_dlps` | **yes** |
| Conditioning on current state and goal | `cond = {0: obs, H-1: goal}` | **yes** |
| Actions are EE deltas (dx, dy, dz) | 3-D xyz, wrapper appends `[0,0,0,-1]` | **yes** |
| Diffusion steps 100 for generalization tasks | checkpoint `n_diffusion_steps: 100` | **yes** |
| 12 layers / hidden 512 for generalization tasks | `n_layers: 12`, `hidden_dim: 512` | **yes** |
| Multiview DLP, 24 particles per view | 2 views x 24 = 48 particles, 10-D each | **yes** |

**One discrepancy, and it is benign.** The paper's Table 5 lists horizon 3;
this checkpoint and our configs use **H=5**. The checkpoint's own `args.json`
declares `horizon: 5`, so our evaluation matches the artifact we are actually
running, and the 93.75% result confirms the combination is correct. The
mismatch is between the paper's headline hyperparameter table and the released
generalization checkpoint, not between the paper and our code.

No other discrepancy was found. In particular the ablation the paper reports —
removing generated latent states drops 2-cube from 0.917 to 0.423 and 3-cube
from 0.894 to 0.529 — confirms that joint state+action generation is load
bearing, which our implementation does.

---

## 3. Flow mathematics, exactly as implemented

Read from `diffuser/diffuser/models/flow_matching.py`, not assumed:

| Element | Value | Source |
|---|---|---|
| Source variable | `x0 ~ N(0, I)` | line 247, `torch.randn_like` |
| Data variable | `x1` | line 268 |
| Interpolant | `x_t = (1-t)*x0 + t*x1` | line 272 |
| Velocity target | `x1 - x0` | line 274 |
| `t = 0` | noise | consistent with 272 |
| `t = 1` | data | consistent with 272 |
| Integration | forward Euler, `x += dt * v`, `t: 0 -> 1` | lines 369-371 |
| Time input to model | `t * time_scale`, `time_scale = 1000` | line 369 |

Training and sampling use the **same** time orientation: training samples
`t ~ U[0,1]` and regresses `x1 - x0`; sampling starts from `randn` at `t=0` and
integrates upward. **No formula in this report needs reversing** — the code
already matches the convention in Tal's clarification.

Conditioning is re-imposed after every Euler step (`_apply_conditioning`, line
372) and the velocity is masked on conditioned entries (line 370).

---

## 4. Representation statistics — item 11

Measured on 200,000 sampled transitions, machine-readable in
`experiments/feature_statistics.json`.

Current representation is **limits/min-max normalization to [-1,1]**. It does
not imply zero mean or unit variance, and it does not deliver either:

| Feature group | mean | std | E[x²] |
|---|--:|--:|--:|
| actions (x, y) | ~0.00 | 0.24-0.28 | 0.06-0.08 |
| action z | **+0.579** | 0.299 | **0.424** |
| z_p (position) | -0.02 | 0.28 | 0.079 |
| z_s (scale) | -0.02 | 0.38 | 0.146 |
| z_d (depth) | -0.13 | 0.24 | 0.072 |
| z_f (visual features) | -0.01 | 0.09 | **0.009** |
| z_t (transparency) | **+0.730** | 0.23 | **0.588** |
| **whole 483-D input** | **0.047** | **0.344** | **0.121** |

- Means are far from zero for two groups: action z (+0.579) and transparency
  (+0.730).
- **E[x²] = 0.121, not 1** — off by a factor of 8.
- Variance is **strongly heterogeneous across groups**: z_f at 0.009 versus
  transparency at 0.588, a 65x spread. Per-channel std ratio is 12.3x.
- **No near-constant channels** (min channel std well above 1e-3), so
  standardization is numerically safe here and needs no special-casing beyond
  the guard already implemented.

---

## 5. What the current linear path actually does — item 12

Measured, not inferred, on the real representation:

**The current path (A) reduces model input scale by 89.2% at its minimum.**
E[x²] falls from 1.00 at t=0 to 0.108, then ends at 0.121 at t=1.

The right panel of `experiments/figures/three_path_variance.png` shows this is
not confined to one group: actions, z_p, z_s, z_d, z_f and z_t all collapse
together. Transparency is the least affected (bottoming near 0.37) and z_f the
most (approaching 0.02).

So the network sees inputs whose scale varies roughly 8-9x across t while
sharing one set of weights across all t. That is a real, measured property of
the current setup.

---

## 6. Three paths compared offline — items 15, 16

| Path | Definition | E[x²] at t=0 | minimum over t | at t=1 |
|---|---|--:|--:|--:|
| **A** current | min-max + linear | 1.000 | **0.108** | 0.121 |
| **B** standardization only | standardized + linear | 1.000 | 0.500 | 1.000 |
| **C** standardization + VP | standardized + VP | 1.000 | **0.999** | 1.000 |
| **D** control (item 16) | min-max + denominator | 1.000 | 0.121 | 0.121 |

- **A experiences a severe scale collapse** — 89.2%.
- **B roughly halves it**: standardization alone lifts the mid-path minimum
  from 0.108 to 0.500. Linear interpolation between two unit-variance
  independent endpoints necessarily gives `(1-t)² + t²`, which bottoms at 0.5.
- **C actually preserves marginal variance**: max deviation from 1.0 across the
  whole grid is **0.00126**. No amplification anywhere.
- **D, the control, does not fix anything.** Applying the VP denominator to
  min-max features leaves the path ending at 0.121. Per the naming rule this is
  **not** variance preserving, because `x_minmax` is not unit variance. It is
  reported because it isolates that the denominator alone is not the mechanism —
  standardization is required for it to mean anything.

---

## 7. VP mathematics, verified three ways — items 17, 18

Path: `y(t) = ((1-t)z + t x) / s(t)`, `s(t) = sqrt((1-t)² + t²)`.

The velocity target is **not** `x - z`. Derived independently:

```
n(t)  = (1-t) z + t x
n'(t) = x - z
s'(t) = (2t - 1) / s(t)
dy/dt = (x - z)/s(t) - n(t)(2t - 1)/s(t)³
```

| Check | Result |
|---|---|
| Symbolic derivation (sympy) | **MATCH** |
| Autodiff agreement | max error 7.1e-15 |
| Finite differences | max error 5.0e-10 |
| Endpoint `t=0` equals noise | exact, 0.0 |
| Endpoint `t=1` equals data | exact, 0.0 |
| Unit marginal variance across t (standardized independent endpoints) | 0.996-0.999 |

**Conditioned dimensions (item 18).** Order of operations matters and is
quantified: interpolate, divide by `s(t)`, *then* re-impose the exact condition
values gives deviation **0.0**; imposing the condition into the numerator
before dividing corrupts it by up to **1.1**. Vanilla flow already applies
conditioning after each update, so a VP implementation must preserve that
ordering rather than fold conditioning into the interpolant.

Reproduce: `python experiments/scripts/verify_vp_math.py`.
Pinned by `tests/test_vp_interpolation_math.py` (12 tests).

---

## 8. Naming discipline — item 14

Three distinct operations, kept distinct throughout this report, the code, and
`experiments/feature_statistics.json`:

1. **limits/min-max normalization** — existing preprocessing, maps to [-1,1].
2. **empirical standardization** — `(x - mu)/sigma`, a proposed preprocessing
   ablation. Statistics from the training split only.
3. **variance-preserving normalized interpolation** — a proposed probability
   path, and the only one of the three called "variance preserving", applicable
   only to standardized features.

Standardization makes each *marginal* mean 0 and variance 1. It does **not**
make the 483-D trajectory vector an isotropic Gaussian: cross-feature
covariance is untouched.

---

## 10. DLP quality, then imagination — items 5 and 6

### 10.1 DLP reconstruction is trustworthy

RGB -> DLP encode -> DLP decode on real Isaac Gym frames spanning reset,
approach and contact configurations.

**Mean pixel MAE 1.8 / 255 (0.7%)**, range 1.6-2.2 across frames.

Figure: `experiments/figures/dlp_reconstruction.png`. Cube positions, cube
colours, the arm and the gripper are all preserved faithfully. **The decoder is
not the bottleneck**, so decoded imagination figures below are interpretable
rather than confounded by reconstruction error.

### 10.2 Flow's imagined futures are visibly degraded

Same current state, same goal, both arms, decoded through the same pretrained
DLP decoder with the same background latent.

Figures: `experiments/figures/imagination_ep0.png`, `imagination_ep1.png`,
`imagination_ep2.png`. These are the first three episodes of the fixed set, not
cherry-picked.

**Gaussian** imagines three distinct, well-formed cubes that keep their
identity and colour across the horizon, moving plausibly toward goal.

**Flow** imagines *smeared, duplicated cube clusters*. Where three cubes exist,
the decoded imagination shows roughly eight to ten overlapping blobs, stretched
into streaks, with colours blending toward muddy yellow/brown. Cube identity is
not preserved across horizon steps.

### 10.3 Quantified, not just visual

Measured in DLP particle space on the generated interior horizon steps
(`experiments/isaacgym_control/imagination_stats.json`, 48 samples per arm),
with the encoded *real* observation as a reference:

| Metric | real encoded | Gaussian | **Flow** |
|---|--:|--:|--:|
| Active particles | 24.0 | 24.0 | 24.0 |
| Particle spread | 0.480 | 0.518 | **0.403** |
| Nearest-neighbour distance | 0.0748 | 0.0508 | **0.0483** |
| **Visual-feature dispersion** | 0.566 | 0.716 | **0.417** |

Gaussian brackets the real statistics or slightly exceeds them. **Flow
under-disperses on every axis**: its particles sit closer together and, most
tellingly, its visual features are compressed to 0.417 against the real 0.566 —
a 26% reduction. That compression is precisely the colour smearing and identity
loss visible in the figures.

**This is the signature the probability-path hypothesis predicts.** A path that
loses 89% of its input scale mid-trajectory (section 5) should produce
under-dispersed, mean-collapsed outputs, and that is what is measured.

### 10.4 But the imagination degradation does NOT prevent the task

This is the crucial tension in the result, and it must not be smoothed over.
Flow's imagined states are clearly worse, yet Flow still achieves **84.4%
success against Gaussian's 93.75%, p = 0.375**. Whatever the imagination is
doing wrong, the executed first action remains good enough to solve the task
most of the time.

So imagination quality and task success are **partially decoupled** here. The
degradation is real and measurable, but this experiment does not establish that
it is what costs Flow the 9 points.


---

## 9. Compute

| Item | Cost |
|---|--:|
| Gaussian control, 32 episodes | 425 s |
| Offline path analysis, VP verification | CPU only |
