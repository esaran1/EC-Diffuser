# Phase 3 — Final-checkpoint fix, validation, and revised budget

**Status: READY to rerun Arm A from step 0.** No B/C/D launched.

---

## 1. The defect

`Trainer.save()` fires on `step % save_freq == 0` and labels the file
`step // label_freq * label_freq`. The training loop runs steps `0 .. N-1`, so a
save scheduled at exactly step `N` **never fires**.

For Arm A (`n_train_steps=20000`, `save_freq=5000`, `n_saves=4` ⇒
`label_freq=5000`) the saves landed at 5k / 10k / 15k, and the **step-20,000
save never happened**. The model trained the full 20k — the log reaches step
19,900 — but the last persisted weights are from **step 15,000**. The final
5,000 steps of training were lost at save time.

This is a config/loop-semantics defect on my side, not a training failure.
MeanFlow suffered the same class of truncation (last save at step 19,000,
labelled 18981) but only ~1,000 steps, which is why it passed unnoticed.

## 2. The fix — infrastructure only

Added to `diffuser/scripts/train.py`, **after** the epoch loop completes:

```python
if getattr(trainer, 'contention_event', None) is None:
    trainer.save('final')
```

Properties, by construction:
- Periodic checkpointing is **unchanged**; `n_saves` / `save_freq` untouched.
- It runs **after** the training loop, so it cannot affect training, batch
  order, or any RNG stream consumed by training.
- It writes the full-state payload already implemented (model, EMA, optimizer,
  RNG, `batches_drawn`, meta).
- It is skipped on a contention exit, where an emergency checkpoint was already
  written — so an emergency checkpoint is never relabelled as a final one.
- It lives in `train.py`, so **A/B/C/D inherit it identically**.

I did **not** fix this by adjusting `n_saves`, because the loop semantics would
still leave the endpoint easy to miss on any future step count.

## 3. Validation (`validate_final_save.py`, CPU only)

Two otherwise-identical runs, one with the final save and one without:

```
=== 2. final save is training-inert ===
  model weights identical : True   max|d| = 0.000e+00
  EMA weights identical   : True   max|d| = 0.000e+00
  batch order identical   : True   (20 batches)
  step identical          : True   (10 vs 10)

=== 1. N-step run produces an N-step final checkpoint ===
  state_final.pt exists   : True
  configured N            : 10
  checkpoint internal step: 10

=== 3. saved weights == in-memory weights at termination ===
  model matches in-memory : True   max|d| = 0.000e+00
  EMA matches in-memory   : True   max|d| = 0.000e+00
  has optimizer state     : True
  has rng state           : True

RESULT: FINAL-SAVE FIX VALIDATED
```

All three §1 criteria pass, including the batch-order check that proves the
added save consumes no RNG and cannot perturb the paired-stream design.

## 4. Revised compute projection

Measured Standard Flow rate **0.230 s/step**; five-seed control ~131 s/set.

Per arm: training 1.28 + diagnostics 0.05 + control 0.18 = **1.51 GPU-h**.

**VALID PHASE-3 EXPERIMENT** (new A + B + C + D):

| | GPU-h |
|---|--:|
| 4 arms × 1.51 | **6.04** |
| approved cap | 6.50 |
| **verdict** | **WITHIN CAP** (0.46 margin) |

**GLOBAL COMPUTE LEDGER** (nothing hidden):

| item | GPU-h |
|---|--:|
| Arm A run whose final weights were lost | 1.28 |
| A@15k non-gating diagnostic | 0.23 |
| Standard Flow timing probe | 0.04 |
| valid A/B/C/D (projected) | 6.04 |
| **Phase-3 global total** | **7.59** |

The wasted 1.28 GPU-h is recorded here and **not** used to shrink the valid
experiment's scope. No arm is shortened.

## 5. ARM A @15K — NON-GATING DIAGNOSTIC

Explicitly **not** the registered 20k gate endpoint. Recorded for information
only; must not authorize B/C/D, and must not be compared against future 20k
B/C/D results as if budgets matched.

| metric (EMA, step 15,000) | value |
|---|--:|
| neutral replay action error | 0.1109 |
| neutral replay state error | 0.2035 |
| action gradients alive | yes (LIVE 0.0804, EMA 0.0766) |
| semantic action loss | 0.1538 |
| semantic state loss | 0.2436 |
| NFE10 / NFE15 imagination | recognizable discrete cubes preserving conditioning colours, but ghosted, duplicated, and not coherently goal-directed |
| five-seed diagnostic control | see below |

This is concerning and suggests 20k may still be short, but it **does not prove
the registered 20k gate would fail** — 5,000 further optimizer steps separate it
from the gate checkpoint.

## 6. Next step

**READY** to rerun Arm A from step 0 with:
identical initialization, training seed 42, `dataloader_seed=20260905`,
identical optimizer/config/architecture/data, identical 20k budget — the **only**
change being the checkpoint-saving infrastructure, which is validated as
training-inert.

Not resuming 15k → 20k: exact dataloader-order continuation is unavailable
(PyTorch `RandomSampler` limitation, documented in `RESUME_INFRA.md`), so a
resumed Arm A would differ from B/C/D in stream provenance and break the paired
design.

**No B/C/D until the corrected Arm A @20k gate is evaluated.**
