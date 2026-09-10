# Canonical-Checkpoint Loss Fine-Tuning — design (ZERO GPU, not launched)

Replaces the abandoned 20k-from-scratch Phase-3 screen. Every arm starts from the
**same competent canonical 500k Standard Flow checkpoint**, so the question is the
one Tal actually asked:

> If we change only the loss formulation of an already competent policy, does its
> action generation or control improve or degrade?

---

## 1. Arms

| arm | terminal-action mask | λ_state | λ_action | first-action weight | role |
|---|---|--:|--:|--:|---|
| **A** | no | 1 | 1 | 10 | **essential control** — measures fine-tuning itself |
| **B** | **yes** | 1 | 1 | 10 | terminal zero-action supervision ablation |
| **C** | yes | 1 | **2** | 10 | moderate action emphasis |
| **D** | yes | 1 | **5** | 10 | stronger action emphasis |

**Frozen identical across arms:** initial checkpoint (`state_400000.pt`,
internal step 499000), architecture (512d, 12L, 8 heads, features_dim 10),
data, `dataloader_seed = 20260905`, training seed 42, optimizer type and betas,
LR schedule, batch 32 × accum 2, EMA decay, conditioning semantics, horizon 5,
`loss_type` l1, the five evaluation sets, and the frozen imagination examples.

**Attribution:** A vs B = terminal zero-action supervision. B vs C = 2× action
emphasis. B vs D = 5× action emphasis. **A vs C and A vs D are NOT pure λ
effects** (they also include the mask) and must not be read that way.

**Excluded:** two-time scheduling, N2 restructuring, first-action-weight changes,
MeanFlow.

## 2. Optimizer state — NOT recoverable, and this shapes the design

Verified: `state_400000.pt` contains **only** `{step, model, ema}`. It predates
the full-state save fix (`9b34ed8`). **Adam moments are unrecoverable.**

**All four arms must restart Adam identically, and this must be stated
explicitly in any writeup.**

### Why an Adam restart is a genuine hazard

At restart `m₀ = v₀ = 0`, so with bias correction the first update is

```
m̂₁ = g ,  v̂₁ = g²   ⇒   step₁ = lr · g/(|g| + ε) ≈ lr · sign(g)
```

**The first update has magnitude ≈ lr per parameter regardless of how small the
true gradient is.** A converged policy sitting in a flat minimum receives a
full-size sign-step kick on every weight. The `1/(1−β₂ᵗ)` correction is 1000× at
t=1, 100× at t=10, 10.5× at t=100, and only settles (1.58×) by t≈1000.

**This is precisely why arm A is not optional.** All four arms suffer it
identically, so B/C/D are read as **deltas against A**, never against the
canonical checkpoint.

### Predeclared mitigations (chosen, not tuned)

- **LR = 1e-5** — one-eighth of the canonical 8e-5, bounding the sign-step kick.
- **`lr_warmup_steps = 100`** — already supported by the Trainer
  (`training.py:63`, currently defaulting to 0), so the kick ramps in.
- **Arm A is itself the stop gate**: if A alone degrades badly, the design is
  invalid regardless of what B/C/D show.

## 3. LR pilot — bounded, not fishing

**One** short run: arm A only, 300 steps, at LR 1e-5, measuring diagonal replay
error at steps 0 / 100 / 300. **0.019 GPU-h.**

- If A's replay error stays within ~10% of the canonical value → proceed at 1e-5.
- If A degrades > 25% → **STOP and return**, do not silently lower LR.

Exactly one alternative (3e-6) is predeclared as the single permitted fallback.
Anything beyond that is hyperparameter fishing and requires approval.

## 4. Step budget

**2,000 steps per arm.** Rationale: the changed objective must have time to move
the policy, but the canonical model trained 500k steps, so 2k is 0.4% — enough to
perturb, not enough to retrain. Checkpoints at **500 / 1000 / 2000**, logging
LIVE and EMA separately.

## 5. Measured cost

Using the measured **0.230 s/step**:

| item | GPU-h |
|---|--:|
| LR pilot (arm A, 300 steps) | 0.019 |
| training, 4 arms × 2000 steps | 0.51 |
| offline diagnostics, 4 arms | 0.08 |
| five-seed control, 4 arms × 5 sets × ~140 s | 0.78 |
| **total** | **1.39** |

At 3,000 steps/arm this rises to 1.64 GPU-h. **Recommend 2,000** to stay near
1.4.

## 6. Diagnostics per arm

At each checkpoint: total loss; `action_loss` / `observation_loss` (Standard Flow
logs these **natively** — no reconstruction needed, unlike MeanFlow); weighted
contributions; action-gradient liveness; neutral replay action/state error for
**LIVE and EMA**.

At the final checkpoint: NFE 1/2/4/10/15 imagination on the **same frozen
examples** used for Standard Flow and MeanFlow, with row 4 labelled
**GOAL CONDITION**.

## 7. Five-seed control

E0/E1s/E2s/E3s/E4s on the final checkpoint of each arm, reported as **five
separate columns plus macro mean and SD** — never pooled into n=480.

**Contention protocol upgraded** per the Stage-A lesson: the boundary-only guard
missed a short-lived mid-set intruder during the Arm-A gate. Use **mid-set
polling** (5–10 s sampler recording max foreign memory per set); reject any set
with foreign contention at **any** point.

## 8. Stop gates

1. **LR pilot**: arm A degrades > 25% → STOP.
2. **Arm A after 2k steps**: if the canonical policy is materially damaged by
   fine-tuning alone, the whole design is invalid → STOP and report.
3. Any NaN/Inf → stop that arm, record.
4. Projected cost > 1.45 GPU-h → STOP and return.
5. Do **not** early-stop healthy arms on interim performance.

## 9. What this design fixes

Phase 3's failure was that the baseline scored **0/96**, so no loss difference
could be interpreted behaviourally — every arm sat at the floor. Starting from a
policy that already works means a difference in action generation or control is
**readable**. That is far more information per GPU-hour.

## 10. Honest limitations, stated up front

- **One fine-tuning seed.** Small differences will not be robust; multi-seed
  replication would be a separate gate.
- **Adam restart affects all arms**, and arm A can only measure its *average*
  effect, not guarantee it interacts identically with each modified objective.
- **2,000 steps is a perturbation, not convergence.** A null result means "this
  loss change does not quickly move a competent policy," not "this loss change
  is worthless."
- The five evaluation sets remain **evaluation**, not schedule- or
  hyperparameter-selection data.

## 11. Recommendation: **GO**, pending approval

1.39 GPU-h for a design that answers Tal's actual question with a valid control,
against Phase 3's 3.25 GPU-h that returned an uninterpretable floor. The Adam
restart is the main scientific risk and is explicitly controlled by arm A rather
than assumed away.
