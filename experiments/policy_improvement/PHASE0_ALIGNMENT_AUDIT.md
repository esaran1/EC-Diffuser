# Phase 0 — Code / Temporal Alignment Audit

Zero GPU. No training. No code modified. All claims cite file:line.

Context: advisor (Tal) feedback redirects the project to policy improvement
(MeanFlow, loss balance, state-vs-action schedules, imagination). The
evaluation-methodology work is preserved but secondary (§16 of the brief).

---

## 1. The 5-step tensor — exact semantics

**Horizon is 5, and it is the ONLY horizon.** `horizon: 5` appears in both the
`diffusion` (training) and `plan` blocks of every pandapush config
(`diffuser/config/pandapush_flow_single_gpu.py:86,101,164`). Tal's "the original
EC-Diffuser environment computes 5 time steps" is **correct**.

The 100 seen elsewhere is a different quantity: the 3-cube **episode length**
(`env_config/n_cubes/IsaacPandaPushConfig.yaml:11`, `episodeLength: [30,50,100]`),
surfaced as `env.horizon` (`isaac_env_wrappers.py:88`) and bounding the rollout.
The policy replans every env step (`exe_steps: 1`, `pandapush_flow.py:150`).

**Tensor layout: `[batch, 5, 483]`, feature axis `[action(3) | observation(480)]`**
— action leads (`diffuser/diffuser/datasets/sequence.py:154`,
`np.concatenate([actions, observations], axis=-1)`). obs 480 = 48 particles × 10
features (48 = 24 particles × 2 views, `multiview: True`).

**Conditioning clamps observations only, at rows 0 and 4**
(`GoalDataset.get_conditions`, `sequence.py:158-165` → `{0: obs[0], 4: obs[-1]}`;
applied by `helpers.py:140-143` / `flow_matching.py:211-215` as
`x[:, t, action_dim:] = val`). **No action is ever conditioned.** At inference the
same structure is fed live: `{0: observation, 4: goal}` (`eval_utils.py:99`).

### The per-timestep table

| t | STATE (dims 3:483) | ACTION (dims 0:3) |
|---|---|---|
| **0** | env time `start`. **Conditioned** (current obs). Not predicted. **Not in loss** (masked). | env time `start`. Free, **predicted**. In loss, **weight 10**. **This is the executed action.** |
| **1** | env `start+1`. Predicted. In loss, weight 1. | env `start+1`. Predicted. In loss, weight 1. |
| **2** | env `start+2`. Predicted. In loss, weight 1. | env `start+2`. Predicted. In loss, weight 1. |
| **3** | env `start+3`. Predicted. In loss, weight 1. | env `start+3`. Predicted. In loss, weight 1. |
| **4** | **NOT env `start+4` — it is the episode GOAL** (~100 steps away). **Conditioned**. **Not in loss** (masked). | **Zero placeholder, no physical meaning.** Free, predicted. **In loss, weight 1.** |

Weights: `loss_discount=1` normalizes the temporal vector to exactly `[1,1,1,1,1]`
(`flow_matching.py:133-134`); `dim_weights = ones(483)` since `loss_weights: None`;
the single override is `weights[0, :action_dim] = 10` (`flow_matching.py:137`).
`loss_type: "l1"`.

Masking (flow path): `_make_conditioning_mask` (`flow_matching.py:217-221`) zeroes
only the **observation** slice at rows 0 and 4; loss is
`(error * active_weights).sum() / mask.sum()` (`:280-287`).

## 2. The offset Tal was referring to

**It is NOT a state/action pairing shift.** Row `i` carries `obs[start+i]` and
`action[start+i]` — the same absolute env timestep, sliced with one index range
(`sequence.py:139-144`). There is no `[1:]`/`[:-1]` misalignment between the streams.

**I verified the source data alignment empirically (CPU, 300 episodes):**

| hypothesis | correlation |
|---|--:|
| A: `action[t]` executed **at** `obs[t]` (predicts `eef[t+1]−eef[t]`) | **+0.973** |
| B: `action[t]` led **into** `obs[t]` (predicts `eef[t]−eef[t−1]`) | +0.254 |

(per-axis: x 0.982/0.240, y 0.993/0.271; z is degenerate — eef height is constant
at 1.0, so its variance is zero and the correlation is NaN, as expected.)

**Reading A confirmed. There is no source off-by-one.**

**The real defect is row 4, and it is a genuine temporal inconsistency.** The
dataset slices `start : actual_end-1`, giving only **4** real (obs, action) pairs.
Row 4 is a **hybrid**:
- its observation is the episode **goal** (`sequence.py:126-129`), typically ~100
  env steps away — **not** env time `start+4`;
- its action is a **normalized zero placeholder** (`sequence.py:143`).

The zero action is **not masked** from the loss — the mask only covers the obs
slice. So the model is explicitly trained to emit a meaningless zero action at
row 4.

**Quantified (arithmetic on the frozen weight structure, no new experiment):**

| component | active weight | share |
|---|--:|--:|
| total active | 1482.0 | 100% |
| all observations (rows 1-3 only) | 1440.0 | 97.17% |
| all actions | 42.0 | 2.83% |
| — row 0 `a₀` (weight 10) | 30.0 | — |
| — rows 1-3 real actions | 9.0 | — |
| — **row 4 zero placeholder** | **3.0** | **7.14% of all action weight** |

**7.14% of the entire action learning signal is spent teaching the model to
output a physically meaningless zero.** This is the concrete thing "not
offsetting the actions and the timesteps" should fix.

## 3. Minimal no-offset modification

Three candidates, smallest first. **None changes canonical behavior unless
explicitly enabled** — all are opt-in flags defaulting to current behavior.

**N1 — mask the row-4 action out of the loss (smallest, ~3 lines).**
Extend `_make_conditioning_mask` (`flow_matching.py:217-221`) to also zero
`mask[:, -1, :action_dim]`. Removes the zero-action artifact without touching the
dataset or the tensor shape. Recovers 7.14% of action loss weight.
*Risk: minimal. Does not fix the row-4 obs being a distant goal.*

**N2 — contiguous 5-step window, goal supplied separately (true "no offset").**
Change `sequence.py:139-144` to slice `start : start+H` so row 4 is genuinely env
time `start+4`, and stop overwriting row 4's obs with the goal. This makes all 5
rows temporally consistent. **But** it removes goal conditioning from the tensor,
which the architecture and `eval_utils.py:99` depend on — so the goal must be
re-supplied (e.g. as a separate conditioning input). *Risk: moderate; changes
the model's conditioning contract; requires matching the inference path.*

**N3 — H=6: 5 contiguous steps + goal row.** Keep the goal row but stop
overloading a real timestep with it. *Risk: changes tensor shape and cost;
larger blast radius.*

**Recommendation: N1 for the first screening arm** (it is the true minimal
change and isolates the placeholder artifact), with **N2 as the second arm** if
N1 moves anything. Do not combine them in one run — we need attribution.

## 4. Architecture: can the model express separate state/action times?

**No, not today.** `_time_embedding` takes `time` of shape `[batch]` — **one
scalar per sample** — and adds it to *every* token:
`x_proj = x_cat + t_embed[:, None, None, :]` (`pint.py:190-191`).

Critically, `AdaLNPINTDenoiser._time_embedding` **explicitly raises** on interval
conditioning: `raise ValueError("AdaLNPINTDenoiser does not support interval
conditioning")` (`pint.py:139-142`). So the canonical Flow denoiser cannot accept
an `(r,t)` pair.

**The good news for §11-12:** action and state are already **separate tokens**.
The action is token index 0 with its own `action_projection` and
`action_encoding`; state particles are tokens 1..N (`pint.py:165-187`). A
token-type-conditioned time embedding is therefore a *surgical* change: add
`t_action` to slot 0 and `t_state` to slots 1+, instead of one broadcast `t_embed`.

---

## 5. MeanFlow implementation status — ALREADY IMPLEMENTED AND CORRECT

This changes the plan: Phase 2 is **not** "implement MeanFlow". It is "run the
PandaPush training that has never been run."

**What exists** (`diffuser/diffuser/models/fast_generation.py`):

| Class | line | What it is |
|---|--:|---|
| `ImprovedMeanFlow` | 80 | **improved MeanFlow (iMF)**, arXiv:2512.02012, boundary variant |
| `AuxiliaryImprovedMeanFlow` | 300 | iMF Eq. 12 + paper's auxiliary marginal-velocity head |
| `ShortcutModel` | 424 | Frans et al. shortcut model, arXiv:2410.12557 |

Note it targets **improved** MeanFlow, not Geng et al.'s original.

**Objective (verified correct line-by-line).** JVP via `torch.func.jvp` with
tangent `(v_θ, 0, 1)`; `.detach()` on the JVP term only, so the average branch
still trains; compound target `average + (t−r)·derivative.detach()`; target
`noise − data`. Adaptive weighting `S/stopgrad((S+ε)^p)` with `p=1, ε=0.01`,
guarded to `loss_type='l2'`.

**Time sampling.** Logit-normal (`time_mean=-0.4, time_std=1.0`); `r,t` from two
draws coupled by min/max so `r ≤ t`; `boundary_probability=0.5` — half of every
batch is plain flow matching, as an **exact per-batch count** (a deliberate prior
correction over Bernoulli draws).

**Inference.** Genuine few-step: `x ← x − (t−r)·u(x,t,t−r)`, one network call per
step; `steps=1` is true single-call generation.

**Conditioning preserved.** Same `_apply_conditioning` / `_make_conditioning_mask`
as the standard Flow path, re-imposed after **every** solver step; state and
action remain **jointly** generated in one tensor.

**Tests: 49 passed CPU-only** (`test_improved_meanflow.py`,
`test_auxiliary_improved_meanflow.py`, `test_shortcut_model.py`,
`test_interval_temporal.py`). Substantive — they check the Eq. 12 compound target
against a hand-computed value, the adaptive-weighting formula, stop-gradient
placement, boundary reduction to flow matching, and exact denoiser call counts.

**Params (PandaPush):** Flow `AdaLNPINTDenoiser` **60,649,640**;
`IntervalAdaLNPINTDenoiser` **62,749,352** (+3.46%, the deep-copied `interval_mlp`).
Auxiliary head adds ~24% train-time params, inference unchanged.
Measured OGBench cost: boundary 0.0857 s/step, auxiliary 0.1172 s/step (**+37%**).

### Prior runs — extensive, disciplined, and the outcome is negative

**Almost all prior evidence is OGBench puzzle, not PandaPush.**

- Stability screen (1000 steps): canonical lr 8e-5 **diverged**; only **half_lr
  4e-5** was eligible (18.37 → 14.72 held-out).
- Paired confirmation (5000 steps): **boundary iMF interval L2 got worse**
  (35.79 → 40.03) — it learned flow matching and failed the interval part, which
  is the entire point. **Auxiliary iMF −50.1%** (35.77 → 17.86).
- 3-seed replication: auxiliary reliably beats boundary on live weights, **but the
  EMA gap nearly vanishes** (13.46 vs 14.14, only 4.8%). EMA weights are what gets
  deployed, so any writeup must lead with the EMA number.
- **Downstream: total failure.** 0/9 successes and zero return at 4 NFE for both
  variants; >79% of executed actions required clipping. An `action_weight=10`
  ablation fixed the clipping (clip fraction 0.797 → 0.078) but **still zero
  successes**.
- Failure taxonomy: `insufficient_training: UNRESOLVED (only 5000 steps)` and
  `data_goal_stitching: PLAUSIBLE` — 99.2% of OGBench conditioning goals lie
  beyond the modeled trajectory endpoint (a dataset pathology, not a MeanFlow one).
- **The decisive 50,000-step run was predeclared and NEVER EXECUTED.** So
  undertraining vs structural failure remains unseparated.

**PandaPush iMF: smoke only.** Best checkpoint is **201 steps**
(`improved_meanflow_smoke_mb8/state_200.pt`); its eval is all-zero success across
16 episodes and `denoiser_calls: 0` (NFE instrumentation inactive). Compare the
standard Flow baseline at **step 499000**. **There is no scientifically usable
PandaPush MeanFlow checkpoint.**

### Two caveats before any long run

1. **Deterministic boundary mask.** `boundary = arange(batch_size) < boundary_count`
   (`fast_generation.py:174-175`) always selects the **first k rows** of a
   minibatch, not a random subset. **I verified the training loader uses
   `shuffle=True` (`diffuser/diffuser/utils/training.py:93`)**, so row index is
   independent of content and the mask is unbiased. Residual note: with
   `gradient_accumulate_every=8` every microbatch takes the same fixed split;
   under shuffling this is still an unbiased 50%, but it is worth a one-line
   assertion in the run log.
2. **Lead with EMA numbers**, not live weights, for any auxiliary-vs-boundary claim.

**Verdict: a correct, well-tested MeanFlow implementation with no working
end-to-end result yet.** The recorded failures are plausibly attributable to a
5000-step budget and an OGBench-specific data pathology — but that is a
hypothesis, and PandaPush (which does not share the goal-stitching problem, since
its goals are the episode goal by construction) has never been tested.
