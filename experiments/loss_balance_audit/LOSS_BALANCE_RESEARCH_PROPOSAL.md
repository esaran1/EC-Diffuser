# Loss-balance audit and research proposal

Date: 2026-08-26. No training launched. No canonical file modified.
Diagnostic code isolated under `experiments/loss_balance_audit/`.

**Running-work check.** Seed-44 training had **already completed** when this task
began (final logged step 499,900; checkpoints at internal steps
99k/199k/299k/399k/**499000**, EMA present, zero errors; terminal SHA256
`c2c13f557aca7cf0eeb29b7baa572cf62a0027021e90ef75a7ca059b3f0e2bd3`). The GPU was
idle, so the small diagnostic below was safe to run. The frozen seed-44
evaluation has **not** been run and is untouched by this work.

---

## 1. Executive conclusion

**PLAUSIBLE BUT NOT PROVEN.**

Tal is right about the mechanism and right that it is worth checking: the
reduction is `(error * weights).sum() / n_unconditioned`, a **global weighted
mean over coordinates**, so semantic blocks contribute in proportion to their
coordinate count. Under equal per-coordinate error, state receives **97.2%** of
the scalar loss and action **2.8%** — a 34:1 ratio — and with *measured*
residuals the actual split is **99.4% / 0.6%**, a 165:1 ratio. That is a real,
quantified imbalance in the loss.

But the inference "therefore state dominates optimization" does **not** survive
measurement. Gradient norms with respect to the model parameters are
**action 0.115 vs state 0.074** (median ratio state/action **0.88**, range
0.19–2.26 across batches). Action, carrying 0.6% of the loss, produces
**comparable or larger** parameter gradients, because `action_weight=10` is
applied to 3 coordinates at t=0 and those few coordinates generate a large,
high-variance signal. Loss share and gradient share point in opposite
directions, so the premise "state is over-weighted in optimization" is not
established.

Two further findings constrain the proposal. First, **`action_weight=10` is not
an EC-Diffuser design choice**: the paper's objective (§4.3) is a plain L1 with
no action weighting, and the coefficient is inherited verbatim from Janner's
Diffuser, whose docstring reads *"coefficient on first action loss."* Second,
and most restrictive, **DLP particles are not objects** — 48 latent keypoints,
all active, whose indices are not temporally stable — which invalidates
Tal's per-entity normalization as literally posed and rules out naive
coordinate-wise latent metrics.

---

## 2. Exact current loss

From `diffuser/diffuser/models/flow_matching.py:242-286`:

```
x0 ~ N(0, I),  t ~ U[0,1]
x0, x1 <- apply_conditioning(x0), apply_conditioning(x1)
x_t = (1-t) x0 + t x1,  then apply_conditioning(x_t)
v*  = x1 - x0                                   (target velocity)
E   = |v_theta(x_t, cond, t*1000) - v*|         (L1, elementwise)

L = sum_{b,h,d} [ E_{b,h,d} * W_{h,d} * M_{b,h,d} ]  /  sum_{b,h,d} M_{b,h,d}
```

- `W` = `loss_weight_matrix`, shape `[H, D]`, built by `_make_loss_weights`
- `M` = `conditioning_mask`, shape `[B, H, D]`, **False only on observation
  channels of conditioned timesteps**; action channels are never masked
- The denominator counts **unconditioned elements**, not weighted elements

**The reduction is a global weighted mean over coordinates.** It does *not*
average semantic blocks separately.

---

## 3. Current tensor anatomy

Verified from `diffusion_config.pkl` of the seed-42 checkpoint and the dataset.

| Quantity | Value | Source |
|---|--:|---|
| Batch (microbatch) | 32 | `trainer_config.pkl` (grad-accum 2 → effective 64) |
| Horizon `H` | 5 | config |
| `action_dim` | **3** | config; EE delta (dx, dy, dz) |
| `observation_dim` | **480** | config |
| `transition_dim` `D` | **483** | 3 + 480 |
| Particles | **48** = 2 views x 24 | `dlp_push_6C/hparams.json` `n_kp_enc=24` |
| Features/particle | **10** | `[pos_xy(2), scale_xy(2), depth(1), visual(4), transparency(1)]` |
| Target tensor | `[32, 5, 483]` | `GoalDataset.__getitem__` |
| Conditioned timesteps | `t=0` (current obs), `t=H-1=4` (goal) | `get_conditions` |
| Masked coordinates | 2 x 480 = **960** obs coords | `_make_conditioning_mask` |
| Unconditioned (denominator) | **1455** per sample | 2415 − 960 |

Shape algebra: `D = D_action + D_obs = 3 + (48 particles x 10 features) = 483`.

**The EEF is not a separate entity in the model input.** It appears in
`state_observations` (4 entities x 6 dims) which is bookkeeping only; the model
consumes the 480-D DLP vector plus 3 action dims.

---

## 4. Effective weighting table

Equal-error thought experiment, per sample (`experiments/loss_balance_audit/`):

| Group | Coords | Weight/coord | Group weight | % of scalar loss |
|---|--:|--:|--:|--:|
| action t=0 | 3 | **10.0** | 30.0 | **2.02%** |
| action t=1..4 | 12 | 1.0 | 12.0 | 0.81% |
| obs t=0 (conditioned) | 480 | 0.0 | 0.0 | 0.00% |
| **obs t=1,2,3 (predicted)** | **1440** | 1.0 | **1440.0** | **97.17%** |
| obs t=4 (conditioned goal) | 480 | 0.0 | 0.00 | 0.00% |
| **Total** | 2415 | | 1482.0 | 100% |

**Action total 2.83% · state total 97.17% · ratio 34.3 : 1.**

`action_weight=10` on 3 of 1455 active coordinates raises action's share from
~1.03% to 2.83%. It does **not** equalize the blocks and was never intended to.

---

## 5. Measured current loss contributions

8 batches of 32, frozen seed-42 EMA weights, training-distribution data
(`measure_loss_balance.py`, results in `loss_balance_measurements.json`):

| Quantity | Action | State | Ratio state/action |
|---|--:|--:|--:|
| mean \|error\| per coordinate | 0.0605 | 0.2309 | **3.82** |
| **share of actual scalar loss** | **0.60%** | **99.40%** | **165.2** |

State residuals are ~3.8x larger per coordinate *and* 96x more numerous, so the
measured imbalance (165:1) is **worse** than the equal-error prediction (34:1).

---

## 6. Gradient contribution audit

Measured, same 8 batches, `torch.autograd.grad` with **no optimizer step**:

| Quantity | Action | State | Ratio state/action |
|---|--:|--:|--:|
| ‖∂L_g/∂(model output)‖ | 0.00215 | 0.00461 | 2.15 |
| **‖∂L_g/∂θ‖ (all params)** | **0.1149** | **0.0744** | **0.65** |

Per-batch parameter-gradient ratios: 0.64, 1.37, 0.93, 2.26, 1.87, **0.19**,
0.36, 0.83 — median **0.88**, range **0.19–2.26**.

**This is the pivotal measurement.** Despite holding 0.6% of the loss, the
action block produces parameter gradients of comparable magnitude to the state
block, and with far higher variance (sd 0.088 on mean 0.115 vs sd 0.013 on mean
0.074). Loss share is a poor proxy for optimization pressure here.

---

## 7. Temporal weighting

`loss_discount=1`, and `discounts = discounts / discounts.mean()` makes the
temporal vector exactly `[1,1,1,1,1]`. So time enters only through **how many
coordinates survive the mask**:

- **Actions**: active at all 5 timesteps (15 coords), weighted `[10,1,1,1,1]`.
- **Observations**: active at only **3 of 5** timesteps (t=1,2,3 = 1440 coords);
  t=0 and t=4 are conditioned and contribute zero.

So the horizon does not itself create the imbalance — the denominator already
divides by the active count. Increasing `H` would increase state coordinates
faster than action coordinates (each step adds 480 state vs 3 action), so the
imbalance **grows with horizon**, but at H=5 the effect is already what §4 shows.

---

## 8. Entity-level interpretation — the key restriction

**DLP particles are NOT objects.** Measured:

- 48 particles (2 views x 24), `n_kp_enc=24` fixed regardless of cube count.
- **All 48 are active in 100% of frames** (transparency mean 0.989, fraction
  >0.9 is 1.0000). There is no sparse 3-slot object decomposition.
- The DLP was trained on a **6-cube** environment (paper Appendix B) and is used
  unchanged for 3-cube data.
- **Particle indices are not temporally stable**: mean index-wise position change
  t→t+1 is **0.395**, while the mean pairwise particle spread is **0.345**. The
  index-wise change *exceeds* the spread, so index `i` at time `t` is not the
  same latent entity at `t+1`.

**Consequences.** (a) Tal's Candidate-2 "one weight per entity" is not
well-defined — dividing by 48 keypoints is arithmetically identical to the
current global mean over the state block, so it would change nothing. (b) Any
imagination metric based on coordinate-wise latent L1 across time is
**semantically invalid**; a permutation-invariant metric is required.

The paper's own §5.4 attention analysis (p = 8.4e-6) argues particles attend
consistently to the same object over time, but that is an *attention* statistic,
not index stability in the representation we regress on.

---

## 9. Why `action_weight=10` exists

- **Not from EC-Diffuser.** The paper's objective (§4.3) is
  `L = E[‖ε − ε_θ(x_t, t, c_g)‖₁]` with **no action weighting term** and no
  ablation of one.
- **Inherited from Diffuser.** `diffuser/diffuser/models/diffusion.py:115-126`
  docstring: *"action_weight : float — coefficient on first action loss."*
- Present in the **upstream pretrained checkpoint's own `args.json`**
  (`action_weight: 10`, commit `344ddeb162a3`), so our Flow arm faithfully
  reproduces upstream rather than introducing it.
- Its purpose is to emphasize the **first** action — the only one executed under
  receding-horizon control — not to balance dimensionality. It is applied at
  `weights[0, :action_dim]` only.

---

## 10. Literature findings

Targeted, primary-source where reachable.

| Work | Objective / weighting | Relevance |
|---|---|---|
| **EC-Diffuser** (arXiv:2412.18907v2, §4.3, Tables 4) | L1 on joint noise prediction, **no action weight**, reduction unspecified in text. Table 4: removing state generation drops 3-cube 0.894 → **0.529** | Direct upstream. Establishes state generation is load-bearing, and that no dimensional balancing was considered |
| **Diffuser** (Janner et al.) | Source of `action_weight`; "coefficient on first action loss" | Explains the 10 as receding-horizon emphasis, not balancing |
| [LeaP (arXiv:2606.17408)](https://arxiv.org/abs/2606.17408) | Flow loss + NLL + alignment loss, multi-term objective for generative robot policies | Shows multi-term flow objectives are current practice; weights hand-set |
| [UniJEPA (arXiv:2510.10642)](https://arxiv.org/html/2510.10642) | Future-state prediction in a **frozen encoder latent space** rather than pixels, with dedicated attention | Closest analogue to "predict future latent state as auxiliary"; supports latent-space metrics over pixel metrics |
| [ACT-JEPA (arXiv:2501.14622)](https://arxiv.org/html/2501.14622v4) | Joint action + abstract observation prediction | Same structural question of balancing action vs state heads |
| GradNorm / homoscedastic uncertainty weighting (Kendall et al.) | Learn per-task weights from gradient norms or learned variances | The principled alternative to hand-tuned λ, but adds a moving target |

**Answering the research questions directly.** I did **not** find prior robot-policy
work that normalizes a joint state–action generative loss by semantic-block
dimensionality, nor one that reports generated-state fidelity separately from
control success. Dimension/uncertainty/gradient balancing is well established in
generic multi-task learning; its *application to entity-centric generative robot
policies* appears unexplored. That is a narrow gap and, on its own, a thin one.

---

## 11. Is poor imagination plausibly caused by loss imbalance?

Ranked by evidence:

1. **The decoder exaggerates latent error (most likely, untested).** DLP
   reconstruction of *real* encodings is excellent (1.8/255 MAE), but generated
   particles may be slightly off the DLP prior manifold and render badly while
   carrying nearly correct task content. This predicts a large *visual* gap with
   a small *latent* gap — exactly the observed pattern.
2. **Low-NFE integration error (likely, partly measured).** Flow@4 uses 4 Euler
   steps; the decoded futures are the *interior* horizon slots, the least
   constrained part of the trajectory. Gaussian@100 gets 100 refinement steps.
   This is a sampler property, not a loss property.
3. **Loss imbalance (plausible, this document).** State carries 99.4% of the
   loss, which argues *against* state being under-trained. The imbalance that
   actually exists points the **opposite** way from Tal's intuition: state is
   over-represented in the loss, while action dominates the gradient.
4. **Genuine capacity/objective limit (open).** One network predicts both blocks;
   the action head may consume representation capacity.

**The naive reading — "state is under-weighted, so upweight it" — is
contradicted by the measurement.** Any candidate that increases state weight
would push further in the direction the loss already over-emphasizes.

---

## 12. Candidate losses

### Candidate 0 — current baseline (control)

```
L0 = Σ_{b,h,d} E·W·M / Σ M          W[0,:3]=10, else 1;  M masks obs at t∈{0,4}
```
Measured: action 0.6% of loss, ‖∇θ‖ ratio state/action 0.65.

### Candidate 1 — semantic-block mean normalization (**recommended primary**)

```
L1 = λ_a · mean_{active action coords}(E)  +  λ_s · mean_{active state coords}(E)
```
- **Solves**: removes raw coordinate count from the action/state trade-off; λ
  becomes an interpretable semantic knob rather than an artifact of 480 vs 3.
- **Assumes**: the two blocks are the right semantic units (defensible — unlike
  per-particle, which §8 rules out).
- **Not equivalent to C0**: C0's action share is 2.8% (equal-error) / 0.6%
  (measured); C1 with λ_a=λ_s=0.5 gives **20.8%**.
- **Scale**: L1 ≈ 0.146 vs L0 ≈ 0.230 → **0.63x**, so a matched-scale constant is
  required (Phase 18).
- **Risk**: raising action weight ~35x could destabilize the already
  high-variance action gradient (sd/mean = 0.77).
- **Novelty**: low as a technique; the *diagnosis* is the contribution.

### Candidate 2 — per-entity normalized state loss (**rejected on evidence**)

```
L_state = (1/N) Σ_i mean_{features of entity i}(E)
```
With all 48 particles active and equal 10-D size, this is **arithmetically
identical** to `mean_{state}(E)`. It would change nothing. It only becomes
meaningful with a genuine object decomposition, which §8 shows does not exist.
**Do not implement.**

### Candidate 3 — per-timestep × per-group normalization

```
L3 = (1/|T_pred|) Σ_t [ λ_a mean(action err at t) + λ_s mean(state err at t) ]
```
Because `loss_discount=1` normalizes to a uniform temporal vector and the
denominator already counts active elements, this is **equivalent to C1 at H=5**.
It only diverges if timesteps have unequal active counts. **Keep as a
formalization, not a separate experiment.**

### Candidate 4 — scale-normalized loss

```
L4 = λ_a mean(E_a / s_a) + λ_s mean(E_s / s_s),   s_g = target residual scale
```
Measured `s_state/s_action = 3.82`. This stacks a second normalization on top of
min-max-normalized data and risks amplifying noise in near-constant channels.
**Defer.**

### Candidate 5 — gradient-balanced (GradNorm-style)

Adapt λ so ‖∇θ L_a‖ ≈ ‖∇θ L_s‖. But the measurement says they are **already
near-parity** (median 0.88), so this would be close to a no-op while adding a
moving target. **Reject as first experiment.**

---

## 13. Recommended loss

**Primary: Candidate 1** (semantic-block mean, λ_a = λ_s = 0.5), with a
prospectively fixed scalar so the initial gradient norm matches baseline.

**Secondary: Candidate 4** (scale-normalized), only if C1 shows movement.

Rationale: C1 is the only candidate that (a) changes the actual quantity Tal
identified, (b) is well-defined given that particles are not objects, and (c) is
a single-line change with an interpretable knob.

---

## 14. Objective imagination metrics

**Not DLP dispersion** (unvalidated, ad hoc — explicitly excluded).
**Not coordinate-wise latent L1** — §8 shows particle indices are not temporally
stable, so index-wise comparison is meaningless.

Proposed, in priority order:

1. **Permutation-invariant (Chamfer / Hungarian-matched) particle error**
   between the generated future latent and the DLP encoding of the realized
   frame. Matches on `pos_xy` and compares full 10-D features. This is the only
   latent metric that respects the representation's actual symmetry.
2. **Predicted-vs-realized one-step error with a copy baseline.** A model that
   merely copies the current state must score worse than one that predicts
   dynamics; already scripted in `experiments/scripts/predicted_vs_realized.py`.
3. **Decoded-image error against the realized frame**, reported *alongside*
   latent error, precisely to test explanation #1 in §11 (decoder exaggeration).

---

## 15. Control metrics

Cheap screening: **held-out first-action L1** (the only executed action),
action-direction cosine error, and action-magnitude error, on held-out
training-distribution windows.

**Hard rule**: offline action error may not predict closed-loop success. No
candidate may be declared better on offline metrics alone; the eventual winner
must pass a paired Isaac Gym evaluation on the **frozen** H=100 episode sets.

---

## 16. Cheap screening protocol

**Stage A — no training (DONE, this document).** Loss decomposition, gradient
audit, particle-semantics check. Cost: ~5 GPU-min, already spent.

**Stage A2 — metric validation, no training (~0.3 GPU-h).** Implement the
Hungarian/Chamfer latent metric and compute it for the **existing** seed-42/43/44
checkpoints at Flow@1/@4. Purpose: establish whether the metric separates
Gaussian from Flow at all. **If it does not, the entire direction is dead and
no training is warranted.**

**Stage B — short paired training (~1.2 GPU-h/candidate).** C0 vs C1 at
**25,000 steps**. Chosen from the measured curves, not arbitrarily: the seed-42
loss is flat from ~200k, but action loss is still descending at 25k
(0.183→~0.13), so 25k is where an action/state re-weighting should first show.
Identical init seed, identical data order (`dataloader_seed` exists in the
current Trainer), identical LR/optimizer/architecture.

**Stage C — extend the survivor (~2.4 GPU-h).** Only if Stage B shows movement;
extend to 75k and check the difference persists.

**Stage D — small closed-loop (~0.6 GPU-h).** Frozen 3-cube H=100 set, Flow@1
and Flow@4, paired against seed-42.

**Stage E — full 500k.** Hard stop; requires explicit approval.

---

## 17. Compute budget

Measured throughput: **0.2339 s/step** (seed-43/44), evaluation ~130 s per
96-episode Flow arm.

| Stage | Per candidate | Candidates | Total |
|---|--:|--:|--:|
| A (done) | — | — | ~0.1 GPU-h |
| A2 metric validation | 0.3 h | 1 | **0.3 h** |
| B 25k steps | 1.63 h | 2 (C0, C1) | **3.3 h** |
| C 75k steps | 4.9 h | 1 | 4.9 h ⚠ |
| D closed-loop | 0.3 h | 2 | 0.6 h |
| **A2 + B only** | | | **~3.6 GPU-h** |

**A2 + B fits under the 4 GPU-h gate. Stage C and beyond require approval.**

Note: C0 must be retrained at 25k rather than reusing seed-42's early
checkpoint, because seed-42's `state_0.pt` holds internal step 99,000, not 25,000.

---

## 18. Falsification criteria

Tal's hypothesis is **wrong** if any of these hold:

1. Stage A2 shows the permutation-invariant latent metric does **not**
   distinguish Gaussian from Flow, i.e. the "worse imagination" is a decoder
   artifact (§11 explanation 1) — then loss balance is irrelevant.
2. Stage B shows C1 changes the latent metric by less than the between-seed
   spread already measured across seeds 42/43/44.
3. C1 improves the state metric but degrades held-out first-action error.
4. The gradient audit result generalizes — action already receives ≥ state
   gradient — implying the loss share was never the operative variable.

**Finding 4 is already partly true**, which is why the classification is
"plausible but not proven" rather than "strongly supported."

---

## 19. Relation to a future action-only ablation

The sharpest test of *why* state generation matters is EC-Diffuser's own Table 4
contrast (0.894 → 0.529 without state generation), reproduced in **our** pipeline
where imagination is measurable. If C1 moves the state metric without moving
control, that is evidence state prediction is an **auxiliary representation
shaper** (its fidelity not causally important) rather than a used world model.
`ConditionalFlowMatching` already exposes `action_only`/`obs_only`, so it is a
config change. **Not run. Not proposed for this round.**

---

## 20. Novelty assessment

- **Already known**: dimension/uncertainty/gradient loss balancing (GradNorm,
  Kendall et al.); joint state-action generative policies; that removing state
  generation hurts (EC-Diffuser Table 4).
- **Potentially novel**: the *measurement* that in an entity-centric generative
  policy, loss share (99.4% state) and gradient share (median 0.88 state/action)
  point in opposite directions; and that generated-state fidelity can diverge
  from control quality.
- **Definitely not enough for a paper**: "divide the loss by dimensionality."
  That is a one-line change with no measured deficit to fix.

---

## 21. Top-conference reviewer critique

1. **"You are fixing a problem you have not shown exists."** State holds 99.4%
   of the loss, so it is not under-trained; the gradient audit shows near-parity.
   Strongest objection.
2. **"Normalizing by dimensionality is not a contribution."** Standard practice
   in multi-task learning; needs a measured deficit and a mechanism to matter.
3. **"Your imagination metric is not validated."** Dispersion was ad hoc, and
   particle indices are not temporally stable, so the replacement metric must be
   justified before any claim rests on it.
4. **"Prettier latents are not the goal."** Flow already controls better than
   Gaussian while imagining worse. Improving imagination may be irrelevant.
5. **"N=3 training seeds."** Any loss-ablation claim inherits the same
   seed-variance problem already documented in the replication phase.

---

## 22. Exact recommended next experiment

**Stage A2 only: validate a permutation-invariant latent imagination metric on
existing checkpoints. ~0.3 GPU-h, no training.**

Compute Hungarian-matched (and Chamfer) generated-vs-realized particle error for
the **already trained** seed-42/43/44 Flow checkpoints at Flow@1 and Flow@4, and
for the canonical Gaussian, on a small number of frozen episodes.

Why this and not Stage B: every downstream claim depends on having a state-quality
metric that (a) respects permutation symmetry and (b) actually separates the two
models. If it does not separate them, the "worse imagination" is a rendering
artifact and **no loss change is warranted at all** — saving ~3.3 GPU-h and a
likely-null training comparison. It is the cheapest experiment that can falsify
the entire direction.

---

## 23. Approval gate

**Files that would change (Stage A2):**
- new: `experiments/loss_balance_audit/latent_imagination_metric.py`
- new: `experiments/loss_balance_audit/latent_metric_results.json`
- no change to any training code, config, or canonical result

**Compute:** ~0.3 GPU-h · storage <10 MB · no training

**Command shape:**
```bash
cd /home/jren313/EC-Diffuser-1
export PYTHONPATH="$PWD:$PWD/diffuser"
python experiments/loss_balance_audit/latent_imagination_metric.py \
    --checkpoints seed42 seed43 seed44 gaussian --nfe 1 4 --episodes 16
```

**If Stage B is later approved**, it would additionally touch
`diffuser/diffuser/models/flow_matching.py` (a new `loss_reduction` option
defaulting to current behaviour, so canonical training is byte-identical unless
explicitly enabled) at ~3.3 GPU-h.

**STOPPING HERE FOR APPROVAL.**
