# Phase 3 — Pre-launch pairing finding: the `dataloader_seed` gap

Recorded as part of the experiment record. **This was found and fixed before any
Phase-3 training began**, during the §4 pre-launch pairing verification.

---

## The finding

`diffuser/scripts/train.py` constructed the `Trainer` **without ever passing
`dataloader_seed`**. The Trainer's default is `None`, which means no seeded
generator is created:

```python
dataloader_generator = None
if dataloader_seed is not None:
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(dataloader_seed)
```

**Consequence had it gone unnoticed:** each of the four arms would have drawn a
**different shuffle order**, so A/B/C/D would have differed by the registered
loss change *and* by their training data sequence. The comparison would not have
been properly paired, and any A-vs-B or B-vs-C/D difference could have been
partly or wholly a data-order effect rather than a loss effect — with one
training seed and a short 20k screen, that confound could plausibly have
exceeded the effects being measured.

This is exactly the class of silent confound the §4 verification exists to catch,
and it would not have produced any error, warning, or visible symptom at run
time.

## The fix

1. Wired `dataloader_seed` through `diffuser/scripts/train.py`:
   `dataloader_seed=getattr(args, 'dataloader_seed', None)` — the default `None`
   preserves the previous behaviour exactly, so no existing config changes.
2. Set `dataloader_seed=20260905` in all four arm configs
   (`pandapush_flow_arm{A,B,C,D}.py`).

## Verification (`verify_paired_arms.py`, run before launch)

```
init_hash       : IDENTICAL
first_batches   : IDENTICAL      (first 3 batch identities across arms)
pred_hash       : IDENTICAL      (initial predictions, fixed input)
arm A: mask=False lambda=1.0  OK
arm B: mask=True  lambda=1.0  OK
arm C: mask=True  lambda=2.0  OK
arm D: mask=True  lambda=5.0  OK
RESULT: PAIRED - SAFE TO LAUNCH
```

Raw record: `experiments/policy_improvement/phase3_pairing.json`.

Also verified at the same time:
- `mask_terminal_action=True` leaves the weight matrix **identical to A** and
  removes exactly 3 action coordinates per sample from the loss mask — it is an
  **ablation**, not an emphasis change (action share 0.879% → 0.816%).
- `lambda_action` scales the **entire** action block including the inherited
  t=0 first-action weight: 10 → 20 at λ=2, 10 → 50 at λ=5.

## Standing lesson

A paired multi-arm comparison needs an explicit check that the *only* difference
between arms is the intended one. Verifying initial weights alone would not have
caught this — the model init was already identical; it was the **data stream**
that differed. Any future arm-based screen in this repo should run the same
three-way check (init hash, initial predictions, batch identities) before launch.
