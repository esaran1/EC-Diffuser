# Stage G — register diagnostic (§35)

**Question:** is there evidence that EC-Diffuser's physical state/action tokens
are being used as generic computational scratch space, in the sense of Darcet et
al. 2024?

**Answer: NO artifact found — which was the predicted outcome.**

## 1. A metric error I made, and fixed

My first bimodality detector returned **1.000 at every time and for every token
population** — a degenerate result, not a finding. Diagnosis: it scored the
largest valley between the global peak and *any* other nonzero bin. In a long
right tail there is always some far bin with a count of 1–3 preceded by a
zero-count gap, giving depth `(1−0)/1 = 1.000`.

At `t = 0.50`, particles: peak1 held **470** counts, the "secondary peak" held
**3**. That is a sparse-tail artifact.

Fixed to require the secondary mode to hold at least **1% of the mass** before it
counts at all — consistent with the audit's note that Darcet's signature is ~2%
of tokens forming a *visible* mode, not any nonzero bin. Recorded rather than
silently corrected, because a metric that returns a constant is worse than one
that fails.

## 2. What the summary statistics already establish

These do not depend on the broken detector.

| t | population | mean | max/mean | p99/mean |
|---|---|--:|--:|--:|
| 0.00 | action | 1.870 | 2.09 | 1.99 |
| 0.00 | particles | 2.824 | 2.38 | 1.85 |
| 0.50 | action | 1.747 | 2.05 | 1.95 |
| 0.50 | particles | 2.344 | 2.53 | 2.03 |
| 1.00 | action | 5.709 | 2.00 | 1.96 |
| 1.00 | particles | 4.463 | 2.46 | 1.89 |

**Darcet's outliers carry roughly 10× the norm of the main mode.** Here
`max/mean` never exceeds **2.53** and `p99/mean` never exceeds **2.03**, across
every diffusion time and both token populations. There is no population of
tokens sitting an order of magnitude above the rest — the spread is an ordinary
unimodal one.

Per-particle persistence is equally flat: `max/median` of the per-particle mean
norm is **1.025 / 1.038 / 1.134** at t = 0.00 / 0.50 / 1.00. **No DLP particle
is a persistent high-norm outlier.**

## 3. The action token is not norm-dominant either

`action / particle` mean-norm ratio: **0.662** (t=0), **0.745** (t=0.5), **1.279**
(t=1). The action token's norm is *below* the particle mean for most of the
trajectory and only modestly above it at the clean end. Whatever aggregation the
action token performs, it does not show up as a norm anomaly.

(Per the audit, this cross-population ratio is reported as context only. The
diagnostic proper is **within** a homogeneous population — the 48 DLP particles —
because action and particle tokens differ in role for uninteresting reasons.)

## 4. Verdict

**NO-REGISTER-ARTIFACT.** Consistent with the predicted negative prior: the
registers phenomenon is a conjunction requiring ≥ViT-Large capacity, >⅓ of a long
training schedule, and a long sequence containing redundant background tokens.
EC-Diffuser has 60.6M params in 8 blocks, 49 tokens per horizon step, and every
DLP particle is a meant-to-be-distinct object slot. It satisfies none of the
preconditions — and it already has a designated global-aggregation slot, the
action token, which is functionally what a register is.

**Consequence for §36–38:** the register *implementation* pilot is **not
justified**. Adding register tokens now would be adding capacity with no
diagnosed pathology to fix, and per the audit any improvement would then be
uninterpretable — indistinguishable from "a slightly bigger model helped" —
because the absorption test (outlier behaviour moving into the register while
normal tokens are unchanged) has no outlier to absorb.

This frees the ~0.99 GPU-h the register pilot would have cost.

## 5. Caveat

Measured on the diagonal control at step 5000 (a 5k fine-tune of the canonical
policy), 32 windows × 5 horizon steps. The canonical model itself was trained for
499k steps, so the "long training" precondition is closer to satisfied for the
*pretrained* weights than this fine-tune suggests; however the capacity and
sequence-length preconditions fail regardless, and those are not affected by
training length.
