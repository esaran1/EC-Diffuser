# Phase 3 — Standard Flow Loss Screen (DESIGN ONLY, zero GPU, not launched)

Prepared while the MeanFlow control retry completes. **No Phase-3 training has
been run.** MeanFlow final classification remains **PENDING**; the operational
decision is **NO-GO for further compute on that configuration**.

---

## 1. Arms

Four arms. Everything below the line is **frozen and identical across all four**.

| arm | terminal action mask | λ_state | λ_action | first-action weight | what it isolates |
|---|---|--:|--:|--:|---|
| **A** baseline | no | 1 | 1 | 10 | current objective, unchanged |
| **B** terminal mask | **yes** | 1 | 1 | 10 | removing the non-executed zero-action target |
| **C** mask + moderate | yes | 1 | **2** | 10 | doubling action emphasis |
| **D** mask + stronger | yes | 1 | **5** | 10 | 5× action emphasis |

**Frozen across arms:** architecture (`AdaLNPINTDenoiser`, 60,649,640 params),
data (`3C_randcolor`), training seed 42, LR 8e-5, Adam betas, batch 32 ×
accum 2 (effective 64), EMA decay 0.995, conditioning semantics (`{0: obs,
4: goal}`, observations only), horizon 5, `loss_type` l1, `loss_discount` 1,
the five evaluation seeds (E0/E1s/E2s/E3s/E4s), and the frozen imagination
examples.

**Excluded by instruction:** N2 trajectory restructuring, separate state/action
schedules, MeanFlow, any change to the first-action weight.

## 2. The implemented objective, written out exactly (§8)

Current code (`flow_matching.py:276-288`):

```
error            = |prediction − target_velocity|            # loss_type = l1
conditioning_mask[b, t, ad:] = False   for t ∈ {0, 4}        # observations only
active_weights   = W ⊙ conditioning_mask
denominator      = conditioning_mask.sum()                   # ELEMENT COUNT
loss             = Σ(error ⊙ active_weights) / denominator
```

with `W[t, j] = 1` everywhere except `W[0, 0:3] = 10` (the inherited
first-action weight).

**This is a weighted sum divided by an unweighted count — not a weighted mean.**
I checked whether that makes λ uninterpretable. It does not:

| quantity | value |
|---|--:|
| denominator (element count) | 1455 |
| Σ active weights | 1482.0 |
| action weight units | 42 |
| observation weight units | 1440 |
| measured `L_action` | 0.06981 |
| measured `L_state` | 0.22972 |
| reconstructed loss | 0.22937 (probe measured **0.22915** ✔) |
| **action share of numerator** | **0.879%** |

The reconstruction matches the measured probe to 4 decimal places, so the
arithmetic below is trustworthy.

**Why the action share is small is a coordinate-count fact, not a normalization
bug:** 12 action coordinates carry loss (3 at t=0 weighted 10, 9 at t=1..3
weighted 1) against 1440 observation coordinates (3 timesteps × 480). λ_action
is therefore a clean multiplier on the action block, and the arms below change
exactly one thing each.

### Arm B — what terminal masking actually does

Extend the mask to `mask[:, 4, 0:3] = False`.

| | A | B |
|---|--:|--:|
| denominator | 1455 | **1452** (−3) |
| action weight units | 42 | **39** (−3) |
| observation weight units | 1440 | 1440 |
| action share of numerator | 0.879% | **0.816%** |

**Note this carefully: masking slightly *reduces* the action share.** It removes
a supervision target rather than rebalancing anything. B is therefore a clean
ablation of *"is supervising the non-executed terminal zero-action target
helping or hurting?"* — exactly as specified — and it is **not** an action-
emphasis manipulation. C and D supply that separately.

### Arms C/D — λ_action on top of B

```
L = λ_state · Σ(err_state ⊙ W_state ⊙ mask_state) / denom
  + λ_action · Σ(err_action ⊙ W_action ⊙ mask_action) / denom
```

with `λ_state = 1` fixed, the first-action weight 10 remaining **inside**
`W_action` (untouched), and the terminal mask applied in all of B/C/D.

| λ_action | action share of numerator | multiple of arm B |
|--:|--:|--:|
| 1 (A/B) | 0.816% | ×1 |
| **2 (C)** | **1.619%** | ×2 |
| **5 (D)** | **3.952%** | ×5 |

**Are λ=2/5 meaningful but bounded?** Yes, and I verified the alternative:
forcing `λ_action·L_action ≈ λ_state·L_state` (parity) would need
**λ_action ≈ 121.5** — two orders of magnitude away, and precisely the
"equalize the scalars" move the brief forbids. λ=2 and λ=5 move the action
contribution by a factor of 2 and 5 while leaving it a minority term, which is a
bounded change around the current objective rather than a re-definition of it.

Recommendation: **keep λ = 2 and 5.** They are candidates that survived the
check, not inherited assumptions.

## 3. Screening budget (§9)

Measured Standard Flow throughput, uncontended: **0.1100 s/step at bs=32**
(Phase-1B probe).

| steps/arm | train (4 arms) | + diagnostics | + five-seed control | **total** |
|--:|--:|--:|--:|--:|
| 3,000 | 0.37 | 0.20 | 0.80 | 1.37 |
| **5,000** | **0.61** | **0.20** | **0.80** | **1.61** |
| 8,000 | 0.98 | 0.20 | 0.80 | 1.98 |
| 10,000 | 1.22 | 0.20 | 0.80 | 2.22 |

**Proposed: 5,000 steps per arm → ~1.61 GPU-h total**, well inside Tal's ~6-hour
framing. This is a **screen for learning-trajectory differences, not
convergence** — the canonical Flow checkpoint trained 500k steps, so no arm here
is expected to match it, and none will be compared against it.

Not every arm needs equal budget in principle, but equal budget is the correct
choice here because the arms differ only in objective; unequal budgets would
confound the comparison.

**Checkpoints per arm:** 500 / 1,000 / 2,500 / 5,000, logging LIVE and EMA
separately.

**Predeclared final screening checkpoint: step 5,000** (the last). The five
evaluation seeds are run **only** on that checkpoint, per arm — they are the
control column, never a model-selection surface.

## 4. Per-arm diagnostics (all frozen, identical across arms)

**During training (cheap, offline):** total objective; `L_action` and `L_state`
as independently masked means; weighted contributions; action-gradient norm and
liveness; neutral replay action/state error at NFE4 — all for **LIVE and EMA**.

**At the final checkpoint:** NFE 1/2/4/10/15 imagination on the **same frozen
examples** as the Standard Flow and MeanFlow panels, row 4 labelled
**GOAL CONDITION**; then the five-seed control (E0–E4s) reported as five
separate columns plus mean and SD, each with a per-set `nvidia-smi` snapshot
before and after and the same contention gating that Phase 2 now uses.

## 5. What would make an arm interesting

Predeclared, so the read-out is not chosen after seeing results:

- **B beats A** on action replay error and/or imagination ⇒ the terminal
  zero-action target was hurting; the ablation resolves it.
- **B ≈ A** ⇒ the terminal target is inert at this budget; it is 7.14% of action
  weight but only 0.20% of the total objective, so a null is a plausible and
  reportable outcome.
- **C/D beat B** on action replay while state replay does not collapse ⇒ action
  emphasis is under-weighted in the current objective.
- **C/D degrade state replay or imagination** ⇒ the current balance is closer to
  right than dimension-counting intuition suggests.

Any arm producing non-zero five-seed control at 5k steps would be notable, but
**5k is a short screen** — near-zero control across all four arms would not by
itself condemn the direction, and would be reported as budget-limited rather
than as evidence against the objective changes.

## 6. Status

**DESIGN ONLY. Nothing launched.** Awaiting approval, and awaiting completion of
the outstanding MeanFlow control sets. No Phase-3 training, no 50k MeanFlow, no
schedules, no E1.
