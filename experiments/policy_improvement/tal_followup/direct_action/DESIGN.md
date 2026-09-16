# Stage C — direct-action / query-token model: build specification

**This is a NEW construction, not a reproduction.** No prior direct-action
implementation, checkpoint or result exists in this repository (see
`DIRECT_ACTION_RECOVERY.md`). It must never be reported as reproducing a prior
finding.

## 1. What changed, minimally

| element | baseline | this model |
|---|---|---|
| state tokens | noised, generative | **unchanged** |
| action input | `action_projection(noised_action)` at token 0 | **learned query embedding** |
| action output | velocity `a − ε_a` | **the action directly** |
| state output | velocity `s − ε_s` | **unchanged** |
| loss weights / mask / denominator | production | **unchanged** |
| terminal supervision | active | **unchanged** |
| added parameters | — | **H × 512 = 2,560** (0.004% of 60.6M) |

`tie_across_horizon=False` gives one learned query per horizon position, matching
the existing token layout. Control consumes `actions[0,0]` MPC-style, so horizon
position 0 is the one that matters, but all H are supervised as in production.

## 2. Verified properties (CPU, zero GPU)

- **The action slice of the input has literally zero effect**: perturbing it by
  17× its scale changes the output by `0.000e+00`. This is the defining property
  — the model cannot be secretly reading the noised action.
- The state input still matters (`1.2e-1` under perturbation).
- All canonical weights load cleanly; only `action_query` is new.
- Targets are correctly blockwise: action target == clean action, state target ==
  velocity.
- Sampling integrates **only** the state; the action is read from the final model
  call, so it is never integrated.

## 3. What this model can and cannot answer

**Can answer (H1):** whether a generative state representation plus a learned
query token recovers most of the useful action mapping — i.e. what explicit
action denoising is actually buying.

**Cannot answer:** whether it is *equivalent* to any diffusion endpoint. Per
§15 a query token is architecturally distinct from `t_action=0` (noise) and
`t_action=1` (clean action). No equivalence is claimed.

## 4. Falsifiable prior from the literature

A2A (arXiv:2602.07322, RSS 2026) Fig. 10 ran the closest available comparison —
flow-in-latent vs deterministic regression-in-latent — and found:

> both reach high in-distribution success, but regression "fails to generalize to
> unseen scenarios."

**Predicted outcome by that analogy:** the query model should be *competitive
in-distribution* and *weaker out-of-distribution*. Our PandaPush evaluator uses
fixed in-distribution scenario sets, so the prediction is **parity or modest
degradation on our benchmark** — and a genuinely informative test of robustness
would need an OOD variant we do not currently have.

This is a prediction registered before the result, not an explanation offered
after it.

## 5. Interpretation rule

Per §49, a difference below ~3 pp is **UNRESOLVED / below practical routine
resolution** at R=3. Given the measured conservative RC95 of ±4.30 pp, a
"modest degradation" of the size Tal describes would very likely sit below what
this evaluator can resolve at R=3 — which must be stated plainly rather than
reported as parity.
