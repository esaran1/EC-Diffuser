# Phase 2 — MeanFlow viability run: STOPPED AT CAP BREACH

**Status: STOPPED before completion, per §2 of the approval.**
GPU consumed: **~0.05 GPU-h** (156 s of training + startup). Cap was 2.5 GPU-h.
E1 untouched. No hyperparameters tuned mid-run. No results are claimed.

---

## 1. What happened

The run launched correctly and trained. It was stopped at step 200 because the
**measured** runtime projected **above the approved hard cap**:

| projection | from measured 0.7800 s/step | cap |
|---|--:|--:|
| Stage A (20k) | **4.33 GPU-h** | — |
| Stage A+B (50k) | **10.83 GPU-h** | **2.5** |

§2 states: *"If measured runtime projects above 2.5: STOP."* Stage A alone
exceeds the total cap by 1.7×, so the run was terminated rather than silently
reduced in scope. Process killed, GPU verified released (135 MiB, no compute apps).

## 2. Why my earlier projection was wrong

Phase 1B measured **0.1100 s/step** for Flow at bs=32 and I applied the **+37%**
JVP overhead recorded in the prior OGBench work → 0.84 GPU-h for 20k.

Two things were wrong with that:

**(a) The +37% came from a different backbone.** The prior measurement was on
OGBench's **U-Net** (`AuxiliaryIntervalTemporalUnet`). This run uses the
**transformer** `IntervalAdaLNPINTDenoiser`. Measured like-for-like at equal
effective batch (64 samples/step):

| | s/step | per-sample |
|---|--:|--:|
| Flow (probe, bs=32×1) | 0.1100 | 0.00344 |
| Flow (canonical, bs=32×2 = 64) | ~0.2200 | 0.00344 |
| **iMF (bs=8×8 = 64)** | **0.7800** | **0.01219** |

**iMF costs 3.55× Flow per sample on this architecture**, not 1.37×.
`torch.func.jvp` through a 12-layer particle transformer is far more expensive
than through a U-Net.

**(b) A forced micro-batch reduction.** At the canonical `batch_size=32` the run
**OOMed** on the 16 GB RTX 4080 (`Tried to allocate 16.00 MiB … 30.44 MiB free`)
because the JVP holds a second forward graph. I moved to the **audited iMF
micro-batch shape** `batch_size=8, gradient_accumulate_every=8` — effective batch
**64, identical to canonical Flow's 32×2** — which is why the frozen iMF configs
use bs=8 in the first place. Smaller micro-batches use the GPU less efficiently,
compounding the JVP cost.

Neither factor was knowable from the Phase-1B Flow probe alone. **The correct
lesson: measure the target objective's step time, not a proxy's.**

## 3. What the 200 steps do show (recorded, not interpreted as a result)

| step | loss | meanflow_loss | unweighted | boundary_fraction |
|--:|--:|--:|--:|--:|
| 0 | 0.8569 | 0.8569 | 0.8409 | 0.5000 |
| 100 | 0.3209 | 0.3209 | 0.3162 | 0.5000 |
| 200 | 0.3153 | 0.3153 | 0.3097 | 0.5000 |

- **No NaN/Inf.** Loss decreased 0.857 → 0.315 over 200 steps.
- **`boundary_fraction: 0.5000` exactly at every log** — the §1 assertion holds
  empirically under `shuffle=True` + `gradient_accumulate_every=8`. The
  deterministic first-k mask concern from the Phase-0 audit is **resolved**.
- Memory footprint 7.1 GB at bs=8.

**This is 200 of 20,000 steps (1%). It is NOT evidence that MeanFlow works, and
no viability classification is made.** Early loss decrease is expected from any
non-broken objective.

## 4. Classification

**Not MF-GO, not MF-CONDITIONAL, not MF-NO-GO.** The gate was never reached, so
none of the three verdicts applies. The correct status is:

> **MF-UNTESTED — blocked on budget.** The viability question is unanswered.

Classifying this as MF-NO-GO would be wrong: nothing failed scientifically.

## 5. Options (each needs explicit approval)

| option | scope | measured cost | notes |
|---|---|--:|---|
| **1. Raise cap to ~4.5 GPU-h** | Stage A only, 20k steps | **4.33** | answers the 20k gate; 50k stays out of reach |
| **2. Reduce to 10k steps** | within current cap | **2.17** | fits 2.5; weaker than the 20k gate, and 10k is close to the 5k budget that already proved inconclusive on OGBench |
| **3. Raise cap to ~11 GPU-h** | full staged 20k → 50k | **10.83** | settles the undertraining question the OGBench work left open; still inside Tal's ~6 h *per-screen* framing only if counted separately |
| **4. Drop micro-batch further / bf16** | any | unmeasured | would change the optimization setup, breaking the "current formulation" attribution §1 requires. **Not recommended without a separate decision.** |

**My recommendation: Option 1 (4.5 GPU-h, Stage A to 20k).** It reaches a real
gate at a cost we have now actually measured, and preserves the staged design —
if 20k is healthy, Stage B can be approved separately with its true 6.5 GPU-h
increment visible rather than assumed.

## 6. What was NOT done

No loss balancing, no terminal-action masking, no λ screen, no schedule
experiment, no N2, no control evaluation on the five seeds, no E1. No
hyperparameters were tuned mid-run. The config
(`diffuser/config/pandapush_imf_viability.py`) is committed and deviates from
canonical Flow **only** in model/objective, LR 4e-5, micro-batch shape (with
effective batch preserved at 64), and checkpoint cadence.
