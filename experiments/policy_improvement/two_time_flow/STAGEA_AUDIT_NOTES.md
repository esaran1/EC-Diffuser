# Stage A — audit findings settled before the falsification results

## §6 Row-count discrepancy — RESOLVED, and it narrows the claim

There are **five** `t_action` rows (0.00, 0.25, 0.50, 0.75, 1.00). All five do
have an interior optimum, so the "5/5" count was **arithmetically correct** — but
**my displayed table showed only four rows**. That was the reporting
inconsistency.

Examining the omitted row shows it should not count as support:

| t_a | best t_s | min | at t_s=0 | at t_s=1 | margin vs t_s=1 |
|--:|--:|--:|--:|--:|--:|
| 0.00 | 0.25 | 0.0641 | 0.0681 | 0.2740 | +76.6% |
| 0.25 | 0.50 | 0.0718 | 0.0830 | 0.3377 | +78.7% |
| 0.50 | 0.50 | 0.0964 | 0.1268 | 0.4490 | +78.5% |
| 0.75 | 0.75 | 0.1464 | 0.2683 | 0.5613 | +73.9% |
| **1.00** | 0.75 | 0.7783 | 0.8045 | 0.7863 | **+1.0%** |

At `t_a = 1.00` the action block is **fully clean**, so "action error" is the
model's velocity error on an uncorrupted action input, and the interior margin is
a trivial **+1.0%** on a baseline ~5× larger than any other row.

**Corrected claim: interior optimum in 4/4 non-degenerate rows**, with the
degenerate `t_a = 1.00` row reported separately and not counted as support.

## §4 Action-error metric — exact definition

The Stage-1 surface reports **velocity-field error**, not reconstructed action:

```
error = | v̂ − (x₁ − x₀) |      averaged over free (unmasked) action coords
```

per `stage1_gate.py:replay_errors`. Conditioned coordinates are excluded by the
mask; `x₁` and `x₀` are both conditioned before corruption so those coords
contribute nothing.

### Derivation of the reconstruction estimator (added in Stage A)

With the repo's verified convention `x_t = (1−t)x₀ + t·x₁`, `v* = x₁ − x₀`:

```
x₀ = (x_t − t·x₁)/(1−t)
v* = x₁ − x₀ = [ (1−t)x₁ − x_t + t·x₁ ] / (1−t) = (x₁ − x_t)/(1−t)
⇒  x₁ = x_t + (1−t)·v*        exact

⇒  x̂₁ = x_t + (1 − t_action)·v̂          (the estimator used)
```

**Two conditioning notes, stated rather than hidden:**

1. **Degenerate at `t_action = 1`.** The factor `(1−t)` is zero, so
   `x̂₁ = x_t = x₁` identically regardless of `v̂`. That cell is reported **N/A**,
   not as a perfect score.
2. **Not a monotone rescaling of velocity error.** `(1−t)` *shrinks* velocity
   error as `t → 1`, so reconstruction error reweights the surface. The U-shape
   can therefore genuinely survive or vanish under this metric — which is exactly
   why the check is informative.

## Stage-1 reproduction — bit-identical

The Stage-1 run persisted only its JSON, not the fine-tuned weights, so the
pre-vs-post control was impossible. I re-ran the identical configuration with
saving enabled and verified it reproduces exactly:

| step | original loss | rerun loss |
|--:|--:|--:|
| 0 | 0.3191 | 0.3191 |
| 500 | 0.2405 | 0.2405 |
| 999 | 0.2362 | 0.2362 |

POST-FT diagonal values match to all printed digits
(action 0.0744 / 0.0964, state 0.2165 / 0.2877). **Same experiment, weights now
available.**

## GPU provenance for Stage A

A foreign process (`alloc/stochastic_floor.py`, PID 316104, ~5.5 GB) is resident.
Stage A is **offline replay only — no Isaac Gym physics** — so contention affects
throughput, not correctness, and it is recorded rather than treated as
invalidating. The unrelated job was not touched. **Any closed-loop control (§11)
will require a genuinely clean GPU with mid-set polling.**
