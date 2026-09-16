# The `t_action = 1.0` column is a degenerate endpoint, not model failure

Noticed when the field heatmap made it visible: the rightmost column is high on
**both** arms (~0.81–0.94). This is expected by construction, and stating why
matters because it changes how the off-diagonal means should be read.

## The analysis

At `t_a = 1` the action input is the **clean action**:

```
x_a = 1·a + 0·ε_a = a
target = a − ε_a
```

The input contains **no information about ε_a** — it was integrated out. The best
possible prediction is `E[a − ε_a | a] = a − E[ε_a] = a`, leaving an irreducible
error of

```
E|ε_a| = sqrt(2/π) = 0.7979   for a standard normal
```

## Measured against that floor

| arm | `t_action = 1.0` column | mean |
|---|---|--:|
| CTL | 0.915, 0.922, 0.938, 0.935, 0.803 | 0.9029 |
| UWM | 0.819, 0.809, 0.813, 0.811, 0.822 | **0.8146** |
| **theoretical floor** | | **0.7979** |

The UWM arm sits essentially **at** the irreducible floor (+2.1%); the control is
13% above it. So even in the degenerate column the independent-time model is
closer to optimal — but neither column should be read as model error.

The same argument applies to the `t_state = 1.0` row for the state block.

## Robustness: the verdict is STRONGER without it

Recomputing the adaptation criterion with the degenerate endpoint excluded:

| metric | UWM | CTL | with column | without column |
|---|--:|--:|---|---|
| action off/diag | 3.667 | 13.703 | 1.558 vs 3.003 (1.9×) | **3.7× gap** |
| action corr(dist, err) | +0.435 | +0.791 | +0.356 vs +0.692 | wider |
| state off/diag | 1.256 | 2.236 | 1.145 vs 1.673 | wider |
| state corr(dist, err) | **+0.116** | +0.749 | +0.268 vs +0.654 | **much wider** |

> Including an unlearnable column **inflates both arms' off-diagonal means
> toward a common floor, which was masking the effect.** The ADAPTED verdict
> survives — and strengthens substantially — when it is removed.

## Which numbers the report uses

The **headline figures keep the full grid** (1.558 vs 3.003), because that is
what the registered criterion specified and changing it after seeing the data
would be exactly the post-hoc redefinition this program forbids.

The exclusion is reported here as a **robustness check that moves in the
favourable direction**, not as a replacement for the registered numbers.
