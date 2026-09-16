# §23 extension decision — 5000 → 10000 steps, AUTHORIZED

Registered rule (§23, written before the data):

> If adaptation is clearly complete by 2000: do not train to 5000 just because
> it was available. If off-diagonal quality is still improving strongly at 5000:
> you may extend ONCE to 10000 if projected compute fits.

## The measurement

UWM arm, action block, off/diagonal error ratio and distance-correlation across
training checkpoints — all on the **identical** 5×5 grid with identical replay
states, noise and conditioning:

| step | diag mean | off mean | **off/diag** | **corr(dist, err)** | corner (t_s=1, t_a=0) |
|--:|--:|--:|--:|--:|--:|
| 500 | 0.1971 | 0.5349 | 2.715 | +0.6985 | 1.4077 |
| 1000 | 0.2065 | 0.5166 | 2.502 | +0.7053 | 1.2990 |
| 2000 | 0.2137 | 0.4789 | 2.240 | +0.6947 | 0.9023 |
| **5000** | 0.2088 | 0.3254 | **1.558** | **+0.3564** | **0.2475** |
| *CTL (never saw off-diagonal)* | 0.1885 | 0.5660 | *3.003* | *+0.6920* | *1.5941* |

## Why this authorizes the extension

**Adaptation had NOT plateaued by 2000 — it was still improving strongly at
5000.** Relative change in the off/diag ratio over the last interval
(2000 → 5000) is **30.5%**, far from converged.

The `corr(dist, err)` column is the more striking evidence. Through step 2000 it
sits at **+0.695 / +0.705 / +0.695** — statistically indistinguishable from the
never-adapted control's **+0.692**. On that metric the model had learned
*essentially nothing* about off-diagonal structure by 2000. Only between 2000 and
5000 does it collapse to **+0.356**.

> Had we stopped at 2000 — the budget the prior loss program used — we would have
> concluded that training on off-diagonal times does not help, and we would have
> been wrong. The adaptation happens late.

That is itself a finding worth reporting, independent of the extension.

## Both arms are extended, not just the treatment

Extending only the UWM arm would destroy the matched control (§22): the control
must receive the identical optimizer budget, or any later difference confounds
"trained longer" with "trained off-diagonal". Cost is therefore 2 × 0.326 =
**0.652 GPU-h**.

## Budget check (§6), performed before launching

| item | GPU-h |
|---|--:|
| spent to date (pilot 0.6517 + diagnostics 0.0144) | 0.666 |
| this extension (both arms 5k → 10k) | 0.652 |
| closed-loop R=3 × 2 | 1.327 |
| Stage C direct-action train + R=3 | 0.990 |
| re-run diagnostics after extension | 0.020 |
| **projected total** | **3.655 of 10.0** |
| headroom | 6.345 |

Fits with room for the 3-seed confirmation (+3.958 → 7.613) if the pilot
warrants it.

**This is the ONE extension §23 permits. No further extension is authorized.**
