# Isaac Gym Flow diagnosis: control, math, and probability path

Date: 2026-08-20. Branch `fast-generative-policies`. Continues
`experiments/isaacgym_debug_investigation.md`; prior CPU-only findings are
consumed, not repeated.

Two hash-locked episode sets of 3-cube PushCube with fixed initial and goal
cube positions were used. Every result file records its set hash, so all
comparisons are paired by construction rather than retrofitted.

| Set | SHA256 (prefix) | Episodes | Role |
|---|---|--:|---|
| exploratory | `fb0a95a710ceb9fa` | 32 | first pass, sections 1/1b/11 |
| **confirmation** | `c34b4c8ad456833c` | **96** | **authoritative, section 1c** |

**Section 1c supersedes the 32-episode rankings.** The exploratory results are
retained rather than deleted because one of them was wrong, and the reason it
was wrong is itself a finding (section 1c.3).

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

## 1b. Flow arm on the identical episodes — **FLOW WORKS** (exploratory, n=32)

> **Superseded by section 1c.** The 32-episode numbers below are retained for
> transparency. At 96 episodes Flow 499k scores 95.8%, not 84.4%, and
> *significantly beats* Gaussian rather than trailing it. Read 1c for the
> authoritative comparison.

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

## 1c. Confirmation at 96 episodes — **AUTHORITATIVE**

Fresh episode set `c34b4c8ad456833c`, 96 episodes, all three arms paired on it.

| Metric | Gaussian (100 NFE) | Flow 399k (4 NFE) | **Flow 499k (4 NFE)** |
|---|--:|--:|--:|
| **Success** | 83/96 = 86.46% | 88/96 = 91.67% | **92/96 = 95.83%** |
| 95% CI | [0.780, 0.926] | [0.842, 0.963] | **[0.897, 0.989]** |
| Goal-success fraction | 0.9236 | 0.9583 | **0.9826** |
| Mean object-goal distance | 0.0318 | 0.0230 | **0.0154** |
| Cubes placed / episode | 2.771 | 2.875 | **2.948** |
| Cubes moved farther from goal | 0.125 | 0.063 | **0.021** |
| Mean progress toward goal | 0.1555 | 0.1644 | **0.1723** |
| Contact rate | 100% | 100% | 100% |
| EE path length | 2.277 | 2.256 | **2.119** |
| **Wall time, 96 episodes** | **1279 s** | **129 s** | **127 s** |

### 1c.1 Flow beats the Gaussian control, significantly

Paired McNemar on the same 96 episodes:

| Comparison | b | c | p |
|---|--:|--:|--:|
| **Flow 499k vs Gaussian** | **11** | **2** | **0.0225** |
| Flow 399k vs Gaussian | 13 | 8 | 0.383 |
| Flow 499k vs Flow 399k | 8 | 4 | 0.388 |

**Flow at 4 network calls significantly outperforms the canonical Gaussian
checkpoint at 100 network calls (p = 0.0225)**, while running the same 96
episodes **10x faster in wall-clock time** (127 s versus 1279 s). Flow wins on
every metric in the table, not only on success.

### 1c.2 The apparent late-training degradation was NOT real

Section 11 reported Flow 399k at 96.9% against 499k at 84.4% on 32 episodes and
flagged it as suggestive (p = 0.125) but unestablished. **At 96 episodes the
ordering reverses**: 499k scores 95.8% against 399k's 91.7%, p = 0.388.

**There is no late-training degradation.** The 32-episode result was
small-sample noise, and the caution attached to it was warranted. The
undertraining verdict stands as **UNLIKELY** — but for the plain reason that the
final checkpoint is the best one, not because training past 399k hurts.

### 1c.3 What this says about evaluation noise

The same checkpoint scored **84.4% on 32 episodes and 95.8% on 96 episodes**.
Both are correct measurements of the same policy; the difference is sampling.
That is an 11-point swing from episode selection alone.

This is the most important methodological lesson in the investigation, and it
governs any future A/B/C comparison: **32-episode evaluations of this task
cannot resolve differences smaller than roughly 10 points.** A probability-path
effect would very likely be smaller than that.

### 1c.4 Failure taxonomy at 96 episodes (item 8)

| Category | Gaussian (13 failures) | Flow 499k (4 failures) |
|---|--:|--:|
| 1. Wrong-object selection | 0 | 0 |
| 2. Never approaches object | 0 | 0 |
| 3. Approaches but misses contact | 0 | 0 |
| 4. Contacts but pushes wrong direction | 10 | 2 |
| 5. Pushes correctly but insufficiently | 3 | 2 |
| 6. Solves one object, fails composition | (counted in 4/5) | (counted in 4/5) |
| 7. Planning correct, execution diverges | 0 | 0 |

Neither arm ever fails to approach or to contact — contact rate is 100% for
both. All failures are post-contact: pushing the wrong way, or not far enough.
Notably **Gaussian pushes the wrong direction more often than Flow** (10 versus
2), which is consistent with Flow's lower "cubes moved farther" figure (0.021
versus 0.125).

**The OGBench contact-failure conclusion does not transfer and is not reused.**
There the arm travelled metres while cubes moved fractions of a millimetre.
Here both arms contact every cube in every episode.


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
Flow's imagined states are clearly worse, yet at 96 episodes Flow **outperforms
Gaussian, 95.8% versus 86.5%, p = 0.0225**. The arm with visibly degraded
imagination is the arm that wins the task.

So imagination quality and task success are not merely decoupled here — on this
benchmark they point in **opposite directions**. The particle degradation is
real and measured, but this experiment gives no evidence that it costs task
performance, and some evidence against.

A caveat on interpretation: EC-Diffuser's own ablation shows that *removing*
generated latent states hurts multi-object performance badly (0.917 -> 0.423 at
2 cubes). So generating states clearly matters. What these figures show is that
the states need not be *visually clean* to serve that role.


---

## 11. Checkpoint sweep — the undertraining question, answered behaviourally

Item 4 asked whether Flow was undertrained. The prior phase answered from the
loss curve alone (converged, -0.16% over the final quarter). Existing
checkpoints at 100k / 300k / 500k make it possible to answer **behaviourally**,
on the same episode set, with no retraining.

**A labelling correction first.** Checkpoint *filenames* in this run are epoch
labels, not step counts: `Trainer.save(epoch)` writes `state_{epoch}.pt` while
storing the true `self.step` inside. Verified by reading every file:

| File | True internal step |
|---|--:|
| `state_0.pt` | 99,000 |
| `state_100000.pt` | 199,000 |
| `state_200000.pt` | 299,000 |
| `state_300000.pt` | 399,000 |
| `state_400000.pt` (latest) | 499,000 |

All step counts below are the **true internal steps**.

| Checkpoint | Success | 95% CI | Cubes placed | Goal-success frac | Obj-goal dist |
|---|--:|--:|--:|--:|--:|
| Flow 199k | 28/32 = 87.5% | [0.710, 0.965] | 2.875 | 0.958 | 0.0231 |
| **Flow 399k** | **31/32 = 96.9%** | [0.838, 0.999] | **2.969** | **0.990** | **0.0146** |
| Flow 499k (final) | 27/32 = 84.4% | [0.672, 0.947] | 2.750 | 0.917 | 0.0194 |
| Gaussian (100 NFE) | 30/32 = 93.75% | [0.792, 0.992] | 2.938 | 0.979 | 0.0162 |

> **Partially superseded by section 1c.** The direction of the 399k-vs-499k
> contrast reverses at 96 episodes, so the "past its best" reading below is
> **withdrawn**. The undertraining verdict itself is unchanged.

**Verdict: UNLIKELY that Flow is undertrained.** At 96 episodes the final
checkpoint is the best one (95.8%), so more training is not needed and the
apparent late-training degradation seen at n=32 was sampling noise.

- At n=32, Flow 399k appeared to outperform Flow 499k. **This reversed at n=96**
  and is withdrawn.
- Flow **matches or exceeds the Gaussian control** while using **4 network calls
  instead of 100**. This survived confirmation and strengthened: at n=96 the
  final Flow checkpoint beats Gaussian at p = 0.0225.

Paired McNemar tests on 32 episodes:

| Comparison | b | c | p |
|---|--:|--:|--:|
| Flow 399k vs Flow 499k | 4 | 0 | **0.125** |
| Flow 399k vs Gaussian | 2 | 1 | 1.000 |
| Gaussian vs Flow 499k | 4 | 1 | 0.375 |
| Flow 199k vs Flow 499k | 5 | 4 | 1.000 |

**Stated honestly: none of these reach significance at 32 episodes.** The 300k
versus 500k contrast is the most suggestive — 399k wins all four discordant
pairs, none the other way — but p = 0.125 does not establish it. A 96-episode
confirmation of the three central arms is therefore run on a fresh shared
episode set, and its result governs the conclusion.

### Why this matters more than the ranking

The loss curve is **flat from roughly 200k onward** (figure
`experiments/figures/flow_loss_isaacgym.png`, raw and smoothed), while task
success over the same interval moves 87.5% -> 96.9% -> 84.4% (199k / 399k / 499k). Loss statistics:

| Quantity | Value |
|---|--:|
| Initial loss | 0.15447 |
| Final loss | 0.11665 |
| Minimum loss | 0.09812 |

**Training loss is not a usable model-selection signal for this task.** A
sub-1% movement in loss coexists with a 12-point swing in success. Any future
claim about training length on this benchmark must be made from rollouts, not
from the loss curve — including the prior phase's "converged" verdict, which
was correct about the loss but could not have predicted the behavioural
ranking.


---

## 12. Compute gate for any B / C training — item 22

Measured from the **actual** Isaac Gym Flow run (`l1vkhnp9`), not extrapolated:

| Quantity | Value |
|---|--:|
| Steady-state median | **0.2343 s/step** |
| p5 / p25 / p50 / p75 | 0.2336 / 0.2339 / 0.2341 / 0.2345 |
| Observed wall time, 500k steps | 39.03 h (includes contention spikes) |
| Steady-state projection, 500k steps | **32.5 GPU-h per arm** |

The p5-p75 band is tight (0.2336-0.2345), so this is genuine throughput rather
than a contended average; the p95 of 0.74 reflects occasional interference and
is excluded from the projection.

**A matched B or C arm costs ~32.5 GPU-h. Both cost ~65 GPU-h.**

The directive's gate is 4 GPU-h per run. This exceeds it by roughly 8x, so
**B and C are not launched, and approval is required before they could be.**
Substituting a shorter, undertrained run is explicitly excluded: the checkpoint
sweep in section 11 shows training length changes success by 12 points on this
task, so a truncated arm would not be a valid comparison against a 500k
baseline.


---

## 13. Standardization preserves environment semantics — item 19

If a standardized model were ever trained, its outputs must invert back through
the *existing* interfaces rather than being handed to the environment or the
decoder directly. Verified numerically on 50,000 real transitions:

| Chain | Max abs error |
|---|--:|
| standardized -> [-1,1] -> raw action | **1.19e-07** |
| standardized -> [-1,1] -> DLP feature space | **2.86e-06** |

Both round trips are exact to floating-point tolerance.

The guard matters, and is quantified:

| Mistake | Magnitude of the error |
|---|--:|
| z-scored actions sent straight to Isaac Gym | wrong by up to **4.25** (on a [-1,1] action space) |
| z-scored DLP sent straight to the decoder | wrong by up to **17.64** |

Zero channels have sigma below 1e-3, so no special-casing is needed on this
dataset beyond the guard already implemented.


---

## 14. Root-cause ranking

The question this phase set out to answer was where Flow's failures come from:
a bug, undertraining, the probability path, DLP/state prediction, action
generation, or a genuine limitation.

**The premise turned out to be wrong. In Isaac Gym, Flow does not fail.** It
reaches 95.8% at 4 NFE against the canonical Gaussian checkpoint's 86.5% at 100
NFE, significantly better (p = 0.0225) and 10x faster in wall clock.

Ranked against the evidence:

### 1. Most likely — there is no Flow defect to explain in Isaac Gym

Flow beats the positive control on the benchmark the EC-Diffuser paper uses,
with the paper's own data, encoder, backbone and environment. Contact rate is
100%; no failure is a planning or approach failure. This is **Case 6** in the
directive's decision tree.

The direct consequence: **the OGBench nulls are not evidence of a Flow bug.**
The same Flow implementation, objective and solver succeed here. Whatever
happens in OGBench is specific to that setting — BC objective, task
formulation, data, or representation mismatch — and not a property of flow
matching or of low NFE.

### 2. Second most likely — the mid-path scale collapse is real but unproven as a cost

Measured and not in dispute: E[x²] = 0.121, the current path loses 89.2% of
input scale mid-trajectory, per-group variance spans 65x, and Flow's generated
particles are measurably under-dispersed (feature dispersion 0.417 versus 0.566
real, 0.716 Gaussian).

What is missing is any link from that to task cost. The arm with degraded
imagination is the arm that wins. So this remains a **real property with an
unproven consequence**, not a diagnosed fault.

### 3. Unlikely / ruled out

| Hypothesis | Status | Evidence |
|---|---|---|
| A bug in the action path | **Ruled out** | round trip 1.5e-08; one shared call site; Gaussian scores 86.5% through the same path |
| Undertraining | **Ruled out** | final checkpoint is the best of three; loss flat from ~200k |
| DLP representation / decoder | **Ruled out** | reconstruction MAE 1.8/255; cubes, colours and arm all faithful |
| Action generation | **Ruled out** | Flow's action magnitude, clipping and EE path all match Gaussian; it places *more* cubes |
| Contact failure (the OGBench story) | **Ruled out here** | 100% contact rate, every cube contacted, both arms |
| Defect A, the asymmetric z action | **Not sufficient** | present in both arms; Gaussian and Flow both succeed through it; EE height stable across episodes |
| A genuine limitation of the setup | **Not supported** | 95.8% is close to ceiling on this task |

---

## 15. Is a new training experiment justified? — item 20

**No, not as the next step.**

The directive is explicit that VP Flow should not be trained merely because it
was suggested, and the evidence does not support it now:

1. **There is no performance deficit to close.** Flow already beats the
   Gaussian control at 4 NFE. A path change would be optimizing a system that
   is winning, against a ceiling near 96%.
2. **The measurement cannot resolve the effect.** The same checkpoint moved
   11 points (84.4% -> 95.8%) purely by changing which 32 versus 96 episodes
   were used. A probability-path effect is very unlikely to exceed that. At
   96 episodes and ~96% success there are only 4 failures left to recover.
3. **The cost is prohibitive at this stage.** ~32.5 GPU-h per matched arm,
   ~65 GPU-h for B and C — roughly 8x the 4 GPU-h gate.
4. **The one signal that does favour the path hypothesis is decoupled from
   performance.** Under-dispersed particles are real, but they coexist with the
   best task performance measured in this project.

**This is Case 6, not Case 4.** Case 4 requires incoherent imagination *plus*
converged training *plus* a performance problem. The third condition is absent.

If the path question is pursued later, the honest framing is that it targets
**generation quality**, not task success, and it needs a metric sensitive enough
to detect it — which success rate on a near-ceiling task is not.

---

## 16. The single highest-value next experiment

**Not a training run. An evaluation-power and variance study.**

**Hypothesis.** Isaac Gym 3-cube success is near ceiling and dominated by
episode sampling, so no probability-path comparison run on it can be trusted
until the noise floor is characterized.

**Arms** (all use existing checkpoints, no training):

1. Flow 499k and Gaussian, each on **3 independent 96-episode sets**, to
   measure between-set variance directly.
2. The same arms on a **harder variant** — 4, 5 or 6 cubes, which the
   environment already supports via `num_entity` and which EC-Diffuser used for
   zero-shot generalization — to find a setting with headroom rather than a
   ceiling.

**Expected information.** Either a defensible noise floor and a task with
dynamic range, in which case an A/B/C path comparison becomes designable and
its required sample size is known; or the finding that 3-cube is saturated and
the path question must move to a different benchmark.

**Cost.** ~1.5 GPU-h. Six 96-episode Flow evaluations at ~130 s each plus two
Gaussian at ~1280 s. Well inside the gate.

**Why this before B/C.** It is roughly 40x cheaper than one training arm, and it
determines whether B/C could be *interpreted* at all. Running a 65 GPU-h
comparison whose effect size is smaller than the measurement noise would waste
the compute and produce an uninterpretable answer.


---

## 9. Compute

| Item | Cost |
|---|--:|
| Gaussian control, 32 episodes | 425 s |
| Offline path analysis, VP verification | CPU only |
