# Denominator causality — theory and optimizer audit (Stage A)

**Question.** The registered terminal-mask intervention (A → B) changes two
things at once: it removes terminal-action supervision from the numerator, and it
drops the active-coordinate denominator 46560 → 46464. Can that ~0.207%
normalization change plausibly explain a multi-point control difference?

**Stage A answer: the denominator change is an exact global loss scaling, and
under the actual optimizer it is scale-invariant up to an `eps` effect bounded by
2.06e-3 per coordinate (mean 8.2e-5).** That is a bound, not a measurement of the
realized divergence; Stage B measures that empirically.

## 1. The algebra

Define B′ = "terminal action masked (numerator identical to B), denominator
restored to A's unmasked 46560."

From `ConditionalFlowMatching._compute_flow_loss`:

```python
conditioning_mask = self._make_conditioning_mask(x1_local, cond)
weights          = self.loss_weight_matrix...unsqueeze(0)
active_weights   = weights * conditioning_mask.to(x1.dtype)
denominator      = conditioning_mask.sum()
loss             = (error * active_weights).sum() / denominator.to(x1.dtype)
```

The numerator `(error * active_weights).sum()` depends on the mask **structure**
(which coordinates are active), not on the count. B and B′ share the identical
mask structure, so they share the identical numerator N. Therefore

```
L_B  = N / 46464
L_B' = N / 46560
L_B' = (46464 / 46560) * L_B = c * L_B
```

with

```
c = 46464 / 46560 = 0.997938144...      1 - c = 2.0619e-3
```

**B′ differs from B by a constant global scaling of the objective — nothing
else.** Gradients scale identically: ∇L_B′ = c·∇L_B.

## 2. The three conditions, verified against production code (not assumed)

| condition | verification |
|---|---|
| mask structure fixed | `_make_conditioning_mask` builds the mask from `cond` keys and `mask_terminal_action` only; no data dependence |
| denominator constant across batches | `get_conditions` returns the **fixed** timestep set `{0, horizon-1}` for every sample (`sequence.py`), so the active-coordinate count never varies with data |
| no other denominator-dependent term | `grep denominator` in `flow_matching.py` returns exactly 3 lines: the definition, a zero-guard, and the single division. The `info` metrics use `_masked_mean`, computed independently of `denominator` |

All three hold. The scaling argument is structurally sound.

## 3. Optimizer audit — read from a real trained checkpoint

Not from defaults. `state_final.pt` (arm B, seed 42) `param_groups[0]`:

| field | value |
|---|---|
| optimizer | `torch.optim.Adam` (not AdamW) |
| lr | 1e-5 |
| betas | (0.9, 0.999) |
| **eps** | **1e-8** |
| **weight_decay** | **0** |
| amsgrad | False |
| maximize / capturable / fused | False / False / None |
| gradient clipping | `max_grad_norm = None` — **not enabled** |
| AMP / GradScaler / autocast | **none in the training path** |
| LR schedule | linear warmup 100 steps, then constant |

This is the clean case. With `weight_decay = 0` there is no decoupled term to
break invariance, and with no clipping there is no threshold whose crossing
depends on gradient magnitude.

## 4. Adam scale-invariance, and where it breaks

Under g′ = c·g, the Adam moments scale exactly:

```
m'_t = c * m_t          v'_t = c^2 * v_t          sqrt(v'_t) = c * sqrt(v_t)
```

(bias correction is identical, being a function of step count only). The update:

```
            m'_t                  c * m_t                     m_t
u'_t =  ---------------  =  -------------------   vs   u_t = -----------------
        sqrt(v'_t) + eps     c*sqrt(v_t) + eps               sqrt(v_t) + eps
```

**If eps were 0, u′ = u exactly and B′ would be byte-identical to B.** The entire
difference is the eps term. The per-coordinate relative deviation is

```
u'/u - 1  =  (c*s + eps) / (c*(s + eps)) - 1,      s = sqrt(v)
```

which → 0 when s ≫ eps, and → (1/c − 1) = 2.066e-3 when s ≪ eps.

**Measured on the real Adam state** (60,646,925 coordinates, arm B seed 42 at
step 2000):

| quantity | value |
|---|--:|
| sqrt(v) median | 8.50e-07 (85× eps) |
| sqrt(v) 10th pct | 9.89e-08 (9.9× eps) |
| sqrt(v) 1st pct | 1.16e-08 (1.2× eps) |
| **fraction with sqrt(v) < eps** | **0.83%** |
| fraction with sqrt(v) < 10·eps | 10.1% |
| **deviation from invariance — max** | **2.044e-03** |
| **deviation — mean** | **8.173e-05** |
| deviation — 99.9th pct | 1.664e-03 |

Consistent across all four seeds (max 2.04–2.06e-3, mean 7.8–8.5e-5;
`ADAM_SCALE_INVARIANCE.json`).

## 5. Why theory alone does NOT close the caveat

The mean per-coordinate deviation (8.2e-5) is ~15× smaller than the relative RMS
weight change the masking intervention itself produces over 2,000 steps
(~1.26e-3, B vs A). That is suggestive but **not** sufficient, for three reasons:

1. ~0.8% of coordinates sit below eps where the deviation reaches its 2.06e-3
   ceiling — small but not zero, and those are exactly the low-gradient
   coordinates whose updates are most eps-dominated.
2. The bound is per-step and per-coordinate; it says nothing directly about how
   deviations compound or cancel over 2,000 steps of coupled dynamics.
3. The analysis uses the Adam state at step 2000 as a proxy for the whole
   trajectory; early-training `v` is smaller, so eps is relatively *more*
   important near step 0 — precisely where warmup makes updates small anyway,
   but this is an assumption worth not leaning on.

**Therefore the caveat is closed empirically, by the Stage-B 500-step
deterministic probe, against the predeclared gate — not by this derivation.**
This document supplies the mechanism and the expected magnitude; the probe
supplies the measurement.

## 6. Predeclared expectation

If the theory is right, the Stage-B probe should show B′ diverging from B by far
less than B diverges from A, giving R_param and R_output ≪ 0.05. A materially
larger divergence would mean one of the three conditions above is violated in
practice, and would trigger the full four-seed B′ branch.
