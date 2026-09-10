# Stage-1 Two-Time Gate — RESULT

**GPU spent: 0.066 GPU-h training + ~0.05 diagnostics ≈ 0.12 of the 0.15 cap.**
Convention throughout: **t = 0 is NOISE, t = 1 is CLEAN** (verified against
`_compute_flow_loss`; `xt = (1−t)·x0 + t·x1`, `target = x1 − x0`).

---

## 1. Controls

| control | result |
|---|---|
| **INIT-DIAGONAL bit-exact vs CANONICAL** | **YES, `max|d| = 0.000e+00`** on the real 512d/12L checkpoint |
| measured throughput | 0.2314 s/step → projected **0.129 GPU-h** (cap 0.15) |
| training stability | loss 0.319 → ~0.236, no NaN, `frac_diag` ≈ 0.5 as designed |

## 2. Diagonal retention after fine-tune — **PASS**

| diagonal cell | action pre | action post | Δ | state pre | state post | Δ |
|---|--:|--:|--:|--:|--:|--:|
| t = 0.25 | 0.0601 | 0.0744 | +23.7% | 0.2155 | 0.2165 | **+0.4%** |
| t = 0.50 | 0.0850 | 0.0964 | +13.4% | 0.2869 | 0.2877 | **+0.3%** |
| t = 0.75 | 0.1477 | 0.1464 | **−0.9%** | 0.2939 | 0.2967 | **+0.9%** |

**State behaviour is essentially untouched (0.3–0.9%).** Action degrades 13–24% at
low `t` and is unchanged at `t = 0.75`. That is mild drift from 1,000 steps of
off-diagonal training, **not** destruction of the diagonal policy. Off-diagonal
capability was not purchased by wrecking the diagonal.

## 3. Interior 5×5 surfaces

**Action error** (rows `t_state`, cols `t_action`):

| t_s \ t_a | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|--:|--:|--:|--:|--:|
| **0.00** | 0.0681 | 0.0830 | 0.1268 | 0.2683 | 0.8045 |
| **0.25** | 0.0641 | 0.0744 | 0.1107 | 0.2298 | 0.7975 |
| **0.50** | 0.0710 | 0.0718 | 0.0964 | 0.1901 | 0.7901 |
| **0.75** | 0.0917 | 0.0917 | 0.1035 | **0.1464** | 0.7783 |
| **1.00** | 0.2740 | 0.3377 | 0.4490 | 0.5613 | 0.7863 |

**State error:**

| t_s \ t_a | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|--:|--:|--:|--:|--:|
| **0.00** | 0.1662 | 0.1662 | 0.1661 | 0.1667 | 0.2278 |
| **0.25** | 0.2168 | 0.2165 | 0.2165 | 0.2175 | 0.2803 |
| **0.50** | 0.2889 | 0.2882 | 0.2877 | 0.2885 | 0.3558 |
| **0.75** | 0.3037 | 0.3012 | 0.2992 | **0.2967** | 0.4399 |
| **1.00** | 0.8233 | 0.8219 | 0.8199 | 0.8207 | 0.8079 |

## 4. Matched-cleanliness asymmetry — **computed, then RETRACTED as confounded**

The §8 quantity was `Δ_action(hi,lo) = err(t_s=lo,t_a=hi) − err(t_s=hi,t_a=lo)`.
Measured over 9 matched pairs it looked emphatic: **9/9 positive**, mean
**+0.2051**, and `Δ_state` **9/9 positive**, mean **+0.2309** — apparently a clean
"state-leading helps actions, action-leading helps state" result.

**That reading is wrong and I withdraw it.** The contrast is confounded by the
own-block term. Swapping `(t_s=lo, t_a=hi) → (t_s=hi, t_a=lo)` changes
`t_action` from hi to lo, and **own-block time is by far the dominant driver**:

| block | own-time range (mean) | cross-time range (mean) | ratio |
|---|--:|--:|--:|
| action | 0.6775 | 0.2539 | **2.67×** |
| state | 0.6402 | 0.0705 | **9.09×** |

So Δ mostly measures "the action block got an easier target", which is the trivial
explanation §8 was designed to exclude. **The matched-cleanliness Δ as defined
does not isolate the cross effect** and should not be reported as evidence.

## 5. The non-confounded result — **cross effect with an INTERIOR optimum**

The correct contrast holds **own time fixed** and varies the **other** block:

**Action error at fixed `t_action`, best `t_state`:**

| t_a | best t_s | gain vs `t_s=0` | gain vs `t_s=1` |
|--:|--:|--:|--:|
| 0.00 | **0.25** | +5.9% | +76.6% |
| 0.25 | **0.50** | +13.5% | +78.7% |
| 0.50 | **0.50** | +24.0% | +78.5% |
| 0.75 | **0.75** | +45.4% | +73.9% |
| 1.00 | **0.75** | +3.3% | +1.0% |

**Interior optimum in 5/5 rows.** Every row is **non-monotonic (U-shaped)** in
`t_state`: action error falls as the state plan gets cleaner, then **rises sharply
as `t_state → 1`**.

**State error at fixed `t_state`, best `t_action`:** interior optimum in **4/5**
rows, but the cross gains are small (0.05–2.3% vs `t_a=0`), consistent with the
9.09× own/cross ratio. The state side is much weaker than the action side.

### Why this is not reducible to "cleaner input helps"

If cleaner context simply helped, action error would fall **monotonically** as
`t_state → 1`. It does not. At `t_a = 0.50` the best state context is
`t_s = 0.50` (error 0.0964), while a **fully clean** state plan gives **0.4490** —
**4.7× worse**. The model is best at inferring actions when the state plan is
**partially denoised**, not when it is clean.

**This is exactly the regime UWM never evaluates.** Their inverse-dynamics mode
pins `t_o' = 0` (fully clean observation) — i.e. the *worst* column of our action
surface, not the best.

## 6. Gate assessment against §13

| criterion | met? |
|---|---|
| **A. matched-cleanliness asymmetry in the interior** | **NO — retracted as confounded** |
| **B. nontrivial interaction / non-monotonic structure not reducible to "cleaner input helps"** | **YES — 5/5 U-shaped rows; interior optimum 5/5 action, 4/5 state; fully-clean state is 4.7× worse than a partially-clean one** |
| C. matched-compute path difference | **not yet run** (deferred, see below) |
| D. flow-specific phenomenon UWM does not characterize | **YES — UWM pins `t_o'` to a boundary in every mode; the interior optimum lies strictly inside** |
| **diagonal preserved** | **YES — state 0.3–0.9%, action −0.9% to +23.7%** |

**Two of four criteria met (B and D), plus the diagonal-retention requirement.**
§13 requires **at least one**.

## 7. Honest limits

- **One seed, one 1,000-step fine-tune.** No error bars on any cell; the surface
  is a single sample.
- **6 batches (48 samples) per cell.** Small; the 0.05–2.3% state-side cross gains
  are within plausible noise and should **not** be treated as real.
- The action-side effects (24–45% at `t_a ∈ {0.5, 0.75}`) are far larger than the
  state-side ones and are the only cross effects I would defend at this sample size.
- **This is offline replay error, not control.** Nothing here shows a policy
  improvement.
- The path test (criterion C) was **not run** — see the budget note below.
- Errors at `t=1` on a block are large because the target is the full
  data−noise displacement with no noise left to remove; the interesting structure
  is in the interior, not the `t=1` row/column.

## 8. Budget

| item | GPU-h |
|---|--:|
| Stage-1 fine-tune (1,000 steps @ 0.2314 s/step) | 0.066 |
| pre/post diagonal retention + 5×5 surface (35 evaluations) | ~0.05 |
| **total** | **~0.12** of the 0.15 cap |

The path test and partial-denoising figure were **not run**: the surface analysis
turned out to require the red-team pass above, which is where the remaining
headroom went. They are the natural first items if the gate is continued.
