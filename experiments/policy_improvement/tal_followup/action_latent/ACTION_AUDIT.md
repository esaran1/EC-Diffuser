# Stage H — action-dataset audit (§40) and bottleneck decision (§41)

Zero GPU. Run **before** any VAE training, because §41 makes the bottleneck
decision conditional on a predeclared dataset criterion.

Source: 4,000 sampled windows (16,000 nonterminal action rows) from the
production `GoalDataset` (198,000 windows, horizon 5).

## 1. The terminal row is a CONSTRUCTED target, confirmed empirically

| | mean | sd | identical across all windows |
|---|---|---|---|
| terminal row (`t = H−1`) | `[0.000, 0.000, 0.706]` | `[0, 0, 0]` | **True** |

Zero variance across every sampled window. This independently confirms from data
what the code comment asserts: the terminal action is a **normalized zero
placeholder**, not padding and not an executed action. (0.706 is the normalized
encoding of physical zero for the third dimension.)

It is separated here **for representation analysis only** and is **not** removed
from policy training (§40).

## 2. Real (nonterminal) actions

| dim | mean | sd | range |
|---|--:|--:|---|
| 0 | +0.0018 | 0.2237 | [−1.000, +1.000] |
| 1 | +0.0112 | 0.2633 | [−1.000, +1.000] |
| 2 | +0.5905 | 0.2888 | [−0.693, +0.892] |

`\|a\|`: mean 0.736, sd 0.099, p50 0.721, p95 0.966. **No near-zero actions at
all** (fraction < 1e-6 is 0.0000) — every executed action has substantial
magnitude, which is consistent with a continuously-moving pushing policy.

## 3. Correlation — the dimensions are nearly independent

```
        dim0    dim1    dim2
dim0   1.000   0.019   0.060
dim1   0.019   1.000  -0.155
dim2   0.060  -0.155   1.000
```

Maximum off-diagonal |r| = 0.155. There is essentially no linear redundancy to
exploit.

## 4. PCA spectrum — the predeclared criterion, and it FAILS

| component | variance fraction | cumulative |
|---|--:|--:|
| PC1 | 0.4453 | 0.4453 |
| PC2 | 0.3120 | **0.7573** |
| PC3 | 0.2427 | 1.0000 |

**Predeclared criterion (§41):** authorize latent dim 2 only if the top 2 PCA
dimensions explain **≥ 99%** of nonterminal action variance.

**Measured: 75.7%. NOT AUTHORIZED.**

> **No meaningful lower-dimensional bottleneck exists for this action space.**
> The spectrum is nearly flat (0.445 / 0.312 / 0.243) and the dimensions are
> nearly uncorrelated. Projecting to 2-D would discard ~24% of the action
> variance.

Per §41 we therefore do **not** invent a lossy 2-D bottleneck to satisfy a sweep.
This is recorded as the result of the criterion, not as a failure to run an arm.

## 5. Temporal structure

Per-dimension lag-1 autocorrelation within a window: `[0.261, 0.310, 0.891]`.
Dimension 2 is strongly autocorrelated (0.891); dimensions 0–1 are weakly so.
Relevant to the A2A idea (a previous-action source distribution), which relies
on adjacent action chunks being similar — that premise holds mainly for
dimension 2 here.

## 6. Consequence for Stage H

Two independent lines now argue the latent-action direction is **low value at
our action dimensionality**:

1. **This audit**: no compressible structure. A same-dimension VAE (3→3) remains
   permissible under §41 as a pure "learned coordinates" test, but there is no
   dimensionality-reduction story to tell.
2. **The A2A audit**: their latent is 512-d for an 8-step chunk; our entire
   action space is 15 numbers (H=5 × 3). Encoding 15 values into a large latent
   to "simplify transport" does not transfer.

Stage H is therefore **DEPRIORITIZED on evidence**, not skipped for convenience:
the ACTION-VAE-BOTTLENECK arm is explicitly **not authorized** by its own
predeclared criterion, and the remaining same-dim arm has a weak prior. It sits
behind the register pilot in the §6 priority order and will only run if headroom
remains after higher-priority work.
