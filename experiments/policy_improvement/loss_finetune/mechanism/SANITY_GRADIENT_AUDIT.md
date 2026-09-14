# Trainer sanity-gradient quirk — audited

**Verdict: OUTCOME B. The residual sanity-check gradient is OBJECTIVE-DEPENDENT,
not a shared constant.** An earlier statement in this program described it as
"identical across arms / a shared constant, not a differential." **That was wrong
and is withdrawn.** This document records what was actually measured.

## 1. The path, traced exactly

`train.py` L153-160, *before* the warm-start load at L198:

```python
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])      # ONE example, batch size 1
loss, _ = diffusion.loss(*batch)        # the ARM'S OWN constructed wrapper
loss.backward()                         # leaves .grad populated
print('✓')
```

| question | answer |
|---|---|
| exact batch | `dataset[0]` — a **single** trajectory, batch size **1** |
| exact loss | `diffusion.loss` → `ConditionalFlowMatching._compute_flow_loss` |
| which object | the **arm-specific** wrapper from `diffusion_config` (L116) |
| terminal mask active? | **yes** — it is that arm's configured wrapper |
| `lambda_action` active? | **yes** |
| active-coordinate denominator active? | **yes** |
| weights at that moment | **random init** — warm start happens later (L198) |

`Trainer.train()` calls `zero_grad()` **after** `optimizer.step()`
(`training.py` L266), never before the accumulation loop. So step 1 of every
fine-tune computes:

```
step-1 gradient = g_test + g_batch1 + g_batch2
```

## 2. Measured: g_test differs by arm

Reproducing the sanity check exactly (same seed, same frozen flow-noise draw,
CPU, random-init weights as at L156):

| arm | sanity loss | ‖g_test‖ | rel. diff vs A | cos vs A |
|---|--:|--:|--:|--:|
| A | 0.8488690853 | 5.2007e-01 | — | — |
| B | 0.8486053944 | 5.2224e-01 | **2.25e-02** | 0.99975628 |
| C | 0.8727948070 | 5.5712e-01 | **2.50e-01** | 0.97329125 |
| D | 0.9453629255 | 7.7480e-01 | **9.50e-01** | 0.77778859 |
| B′ | 0.8468557000 | 5.2116e-01 | **2.22e-02** | 0.99975628 |

The differences scale with the size of the objective change, exactly as they must
if the sanity check uses each arm's own loss. λ=5 (arm D) moves it most.

**Therefore the correct wording is:**

> Every scientific arm went through the same production-path pre-training
> backward quirk, but the residual sanity-check gradient was objective-specific.
> It therefore forms part of the actual registered training trajectory rather
> than a common additive gradient.

## 3. Why this does not invalidate A/B/C/D/B′

- Every scientific run used the **same production entry point** (`train.py`), so
  the quirk is part of the registered protocol, applied uniformly in *form*.
- The intervention is *supposed* to change the objective; a sanity gradient that
  reflects the arm's objective is a consequence of the intervention, not an
  uncontrolled confound from outside it.
- Paired runs within a seed still differ **only** through their registered
  objective — the sanity gradient included, since it is computed from that same
  objective.
- All replication arms (four seeds × A/B, plus four B′) behaved identically in
  this respect.

What it does mean: the first optimizer step of each arm is not a clean gradient
of that arm's objective at the canonical checkpoint. It contains a one-off
contribution from an **untrained** network on **one** example. For scale, the
Adam gradient EMA at step 500 has L2 **1.84e-02**, while ‖g_test‖ ≈ **5.2e-01** —
roughly **28× larger** than a typical training gradient. Adam then carries that
into `m` and `v`.

This is a real blemish in the pipeline. It is uniform across arms in form and
arm-specific in content, and it is part of the trajectory every accepted result
was produced on.

## 4. Why it broke the Stage-B probe

`bprime_probe.py` drives the `Trainer` directly and never runs the sanity check,
so its step 1 used only `g_batch1 + g_batch2`. That single difference put the
probe on a different trajectory from the scientific runs immediately (probe vs
scientific at step 10: 1.535e-02, larger than either arm's own displacement from
canonical), which is why probe B(500) sits 1.70 away from scientific B(500) while
the entire scientific A-vs-B difference is 0.0996.

**Any future direct-Trainer probe must reproduce this behaviour** — or the
behaviour must be removed from `train.py` first — otherwise it is not measuring
the same training process.

## 5. Not fixed now; recommended for future work

Adding `optimizer.zero_grad()` (or `model.zero_grad()`) after the sanity check
and before training would create a **new training protocol**, making any later
arm incomparable with the completed A/B/C/D/B′ runs. The production path is
therefore preserved exactly for this study.

**Recommended for future experiments:** zero gradients explicitly after the
sanity check, or move `zero_grad()` to the top of the accumulation loop in
`Trainer.train()`. Both make step 1 a clean gradient of the configured objective
and let direct-Trainer probes reproduce `train.py` runs.
