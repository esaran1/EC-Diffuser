# Phase 3 — budget re-estimate BLOCKS launch at every approved step count

**Returning for approval before launching, as instructed.** No Phase-3 training
has been run. Phase-3 registration (`c75d701`) is unchanged.

---

## 1. Measured Standard Flow training rate

The Phase-1B probe (0.1100 s/step) measured **forward+backward only, at bs=32
with no gradient accumulation, no optimizer step, no EMA update**. Canonical
Flow training runs **bs=32 × grad_accum=2** plus optimizer, EMA and checkpoint
I/O, so that probe understates a real training step and should not have been
used for a training projection.

I therefore ran a short real-training timing probe on a clean GPU
(`config.pandapush_flow_timing`, canonical config, 600 steps, checkpointing off):

```
  0: t=  0.3524      (startup)
100: t= 22.9379
200: t=  0.2357      (epoch boundary, not a 100-step interval)
300: t= 23.0676
400: t=  0.2361      (epoch boundary)
500: t= 22.9335
```

Steady 100-step intervals: **22.94 / 23.07 / 22.93 s** →

> **Measured rate: 0.230 s/step** (bs=32 × accum=2, real train loop)

This is 2.09× the Phase-1B probe figure — consistent with grad_accum=2 plus
optimizer/EMA overhead — and 1.10× *below* my pessimistic extrapolation, so the
earlier estimate was conservative but the correct number is still large.

The probe also confirms Standard Flow logs `action_loss` and `observation_loss`
natively (unlike `ImprovedMeanFlow`), so Phase-3 needs no reconstruction step.

Probe cost: ~0.04 GPU-h. Artifacts deleted; no checkpoints written.

## 2. Projected four-arm cost — over cap at every approved length

Diagnostics ~0.20 GPU-h + five-seed control 4 arms × ~0.20 = 0.80 ⇒ **+1.00
GPU-h** beyond training.

| steps/arm | training (×4 arms) | + diag/control | total | vs 5.5 cap |
|--:|--:|--:|--:|---|
| **30,000** (requested) | 7.67 | 1.00 | **8.67** | **OVER by 3.17** |
| **25,000** (fallback) | 6.39 | 1.00 | **7.39** | **OVER by 1.89** |
| **20,000** (approved floor) | 5.11 | 1.00 | **6.11** | **OVER by 0.61** |

**Every approved option exceeds the 5.5 GPU-h cap**, including the 20k floor
below which I was told not to go without returning. Hence this document.

For reference, what *would* fit:

| steps/arm | total | vs cap |
|--:|--:|---|
| 17,500 | 5.47 | fits (0.03 margin) |
| 15,000 | 4.83 | fits comfortably |

## 3. Options (each needs your decision)

| # | option | cost | trade-off |
|--:|---|--:|---|
| **1** | **Raise cap to 6.5 GPU-h, run 20k/arm** | 6.11 | keeps the approved floor; smallest deviation from the registered design. **Recommended.** |
| 2 | Raise cap to 9.0, run 30k/arm as requested | 8.67 | the full requested design; 1.6× the current cap |
| 3 | Keep 5.5 cap, drop to 17.5k/arm | 5.47 | fits, but below the 20k floor you set |
| 4 | Keep 5.5 cap, run 20k/arm on **3 arms** (A, B, D) | 4.83 | fits at full length, but loses the C-vs-D dose-response that distinguishes "more emphasis helps" from "too much hurts" |
| 5 | Keep 5.5 cap, 20k/arm, control on 3 of 5 seeds | 5.79 | still over, and weakens the primary endpoint. **Not recommended.** |

**Recommendation: Option 1.** 20k/arm at a 6.5 GPU-h cap is a 0.61 GPU-h
overrun on the current cap and preserves everything that makes the screen
interpretable — all four arms, full length at the approved floor, and the
complete five-seed primary endpoint. Option 4 is the best strict-5.5 choice but
sacrifices the C/D dose-response, which is a real part of the scientific
question.

I did not pick one and proceed, because every path either breaches the cap or
breaches the 20k floor.

## 4. Everything else remains as registered

Arms A/B/C/D, λ = 2 and 5, first-action weight 10 frozen, terminal masking
described as an ablation (action share 0.879% → **0.816%**, i.e. it *removes*
supervision and is **not** an emphasis arm), the A-vs-B / B-vs-C / B-vs-D
comparison structure, the paired-initialization verification, the diagnostic
schedule, the frozen imagination examples, the five-seed control reported as
five columns with macro mean and SD, and the decision hierarchy with control as
primary — all unchanged from `c75d701`.

Checkpoints will be set to match whichever step count is approved
(e.g. 5k/10k/20k for a 20k target).
