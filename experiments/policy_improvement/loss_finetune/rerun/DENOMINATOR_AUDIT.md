# Denominator audit — terminal-action masking

**Question.** The A-vs-B production-path validation reports the loss denominator
changing 46560 → 46464 (difference 96 = batch 32 × 3 terminal-action
coordinates). Is that *exactly* the behaviour of the already-registered
terminal-action-mask implementation (case A), or an accidental extra effect on
top of a design that intended a fixed denominator (case B)?

**Verdict: case A — REGISTERED / INTENDED SEMANTICS. Proceed unchanged.**

This was recovered from the committed design and implementation history, not
chosen because one behaviour looked preferable. Three independent lines of
evidence converge.

## 1. The committed design registered this denominator, before the knob existed

`58d07c4` (2026-08-04, *"test: harden flow matching integration boundaries"*)
changed the normalizer and recorded the intent in the code:

```diff
-        denominator = active_weights.sum()
-        if denominator.item() <= 0:
+        denominator = conditioning_mask.sum()
+        if denominator.item() == 0 or not torch.any(active_weights > 0):
             raise ValueError("conditioning and loss settings leave no active elements")
-        loss = (error * active_weights).sum() / denominator
+        # GaussianDiffusion applies weights once and reduces by element count.
+        # Restrict that count to unconditioned elements when masking is present.
+        loss = (error * active_weights).sum() / denominator.to(x1.dtype)
```

The registered rationale is explicit: **restrict the element count to
unconditioned elements when masking is present**, for GaussianDiffusion parity.
Excluding masked coordinates from the denominator *is* the registered design.

## 2. The terminal-action knob did not touch the denominator

`16f08cb` (2026-09-05) introduced `mask_terminal_action` and `lambda_action`.
Its diff to `flow_matching.py` contains **no denominator, `conditioning_mask.sum`
or `active_weights` lines at all**. The knob adds exactly three lines inside the
pre-existing `_make_conditioning_mask`:

```python
if getattr(self, "mask_terminal_action", False) and self.action_dim:
    mask[:, self.horizon - 1, :self.action_dim] = False
```

It deliberately reuses the conditioning-mask mechanism and therefore inherits
normalization semantics fixed a month earlier. There is no separate,
later-introduced denominator effect to back out.

## 3. Prior registered calculation already behaves this way

Observation conditioning has always removed its coordinates from the denominator
through the same mechanism. For this batch (B=32, H=5, transition=483, A=3,
O=480), with `cond` at timesteps {0, 4}:

| stage | active coordinates | removed |
|---|--:|--:|
| all coordinates | 77280 | — |
| after observation conditioning (pre-existing) | **46560** | 30720 = 32×2×480 |
| after + terminal-action mask (the knob) | **46464** | 96 = 32×3 |

Computed values match the observed production-path run exactly (46560 / 46464 /
96). Terminal masking is **one registered mechanism applied to a different
slice**, not a new effect.

Corroborating: the Phase-3 commit registered the action share moving
0.879% → 0.816%. A *drop* is only consistent with supervision being genuinely
removed — the ablation framing — which is case A.

## Consequence to acknowledge in the causal wording

**A vs B changes two things, not one:**

1. **Terminal-action supervision** — the t=H−1 action slice (a normalized ZERO
   placeholder, not an executed action) is removed from the numerator. 96
   targets in a batch of 32.
2. **Active-coordinate normalization** — the denominator falls 46560 → 46464,
   a factor **46560/46464 = 1.002066**, i.e. a **0.207%** rescaling of the
   remaining mean loss.

Effect 2 is tiny, but it is real and it is an inherent consequence of the
registered masking mechanism, not a separable confound that could be removed
without departing from the committed design. Any A-vs-B causal claim should be
worded as the effect of *terminal-action target masking as registered*, which
bundles the 0.207% renormalization, rather than as the effect of removing
terminal supervision with all else exactly equal.

## Loss-matrix buffer provenance

`loss_weight_matrix` is a **configuration-derived registered buffer**. Being
registered, it is serialized into every checkpoint, so loading a canonical
checkpoint restores the canonical λ=1 matrix over a correctly λ-scaled one while
the scalar `lambda_action` attribute still reports the configured value (root
cause #2).

For these pinned scientific runs the required invariant is:

```
config -> constructor -> canonical checkpoint load
       -> rebuild_loss_weight_matrix() -> objective_guard -> optimizer step 0
```

The guard must run **after** the checkpoint load **and after** the rebuild.
This ordering is essential and is verified in the actual B/C/D process logs.

A future cleanup could make configuration-derived buffers non-persistent, or
otherwise prevent checkpoint state from overriding them. That refactor is
deliberately **not** performed during the pinned scientific runs.
