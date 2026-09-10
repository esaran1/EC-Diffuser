# Phase 3 — Arm A @20k Budget-Validity Gate: RESULT

**Verdict: STOP before B/C/D.**

> **A 20k Standard Flow training budget is too short for an informative
> loss-formulation screen.**

This says nothing about arms B/C/D themselves. They remain designed, registered
(`c75d701`) and unrun.

---

## 1. Training endpoint — VERIFIED

| item | value |
|---|---|
| `state_final.pt` internal step | **20000** ✔ |
| training interrupted? | **no** — no contention event |
| GPU at launch | clean |
| wall clock | 1.28 GPU-h @ 0.230 s/step |
| seed / dataloader_seed | 42 / 20260905 |
| loss 0 → 1k → 19.9k | 0.8585 → 0.2608 → 0.2349 |

## 2. Offline diagnostics (EMA deployment weights)

| metric | Arm A @20k | Standard Flow @500k (context only) |
|---|--:|--:|
| semantic action loss | 0.1438 | — |
| semantic state loss | 0.2391 | — |
| neutral replay action error | **0.1012** | 0.0221 |
| neutral replay state error | **0.2041** | 0.0488 |
| action gradients alive | yes (LIVE 0.0886, EMA 0.0757) | — |

The 500k reference is **context, not an equal-budget competitor**.

**Imagination (NFE 4/10/15):** discrete blue/green cubes preserving the
conditioning colours, but persistent ghosting, spurious red/purple artifacts, and
— decisively — **the cubes barely move across t=1..3**, i.e. no goal-directed
progression.

## 3. Five-seed closed-loop control — the primary endpoint

| set | gen seed | success | n | status | foreign during |
|---|--:|--:|--:|---|---|
| E0 | 20260820 | **0.0000** | 96 | VALID | none |
| E1s | 20261820 | **0.0000** | 96 | VALID | none |
| E2s | 20262820 | **0.0000** | 96 | VALID | none |
| E3s | 20263820 | **0.0000** | 96 | VALID (rerun) | none |
| E4s | 20264820 | **0.0000** | 96 | VALID (rerun) | none |
| **macro mean** | | **0.0000** | | | |
| **SD** | | **0.0000** | | | |

Reported as five separate columns; never pooled into n=480.

### Provenance correction — E3s/E4s were invalidated and rerun

A queued diagnostic of mine fired twice while the control was still running,
because its wait condition did not match the control's process name. Timing:

| set | window | overlap |
|---|---|---|
| E0 | 20:16:56–20:19:15 | none |
| E1s | 20:19:15–20:21:32 | none |
| E2s | 20:21:32–20:23:50 | none |
| **E3s** | 20:23:50–20:26:11 | **diagnostic #1 killed ~20:24** |
| **E4s** | 20:26:11–20:28:29 | **diagnostic #2 killed ~20:27:50** |

Both intruders were short-lived and gone before the post-snapshot, so **the
boundary guard reported all five VALID when E3s/E4s were actually suspect.**

**This is a real limitation of the guard**: it samples `nvidia-smi` at set
boundaries and cannot observe a brief mid-set overlap. Detected by comparing kill
timestamps against per-set windows, not by the guard itself.

Both sets were marked `INVALIDATED_OVERLAP_MIDSET` (originals preserved in
`armA20k_diagnostics_SUSPECT_E3sE4s.json`) and rerun on a verified-clean GPU.
Both returned 0.0000 again, matching their suspect values — so the conclusion is
unchanged, but the record is now clean.

## 4. Gate decision under the frozen rule

The predeclared STOP condition was: *"Arm A remains catastrophically undertrained"*
— essentially zero control across all five sets, incoherent NFE10/15 imagination,
poor replay with no meaningful development trajectory.

All three hold:

- **control 0/96 on every one of five sets**, macro mean 0.0000, SD 0.0000;
- **imagination shows no goal-directed motion** at NFE 10 or 15;
- **replay ~4.6× (action) and ~4.2× (state) worse** than the trained reference,
  and it barely moved from 15k → 20k (0.1109 → 0.1012, 0.2035 → 0.2041).

**⇒ STOP. Do not run B/C/D at 20k.** A screen whose baseline scores zero on the
primary endpoint cannot resolve differences between loss formulations: every arm
would be compared at the floor.

## 5. What this does and does not license

**Licensed:** 20k is too short a Standard Flow budget for this screen.

**NOT licensed:** any conclusion about terminal-action masking or λ_action. Those
hypotheses are untested. The Phase-3 design stands unchanged.

**Context for a future decision:** the canonical checkpoint trained **500k** steps;
20k is 4% of that. Reaching a budget where the baseline is non-degenerate would
cost far more than the 6.5 GPU-h cap allowed — which is itself an important
finding about the cost of loss-formulation screening in this codebase, and part of
why the two-time **warm-start** route (start from the competent 500k checkpoint,
zero added parameters, bit-exact diagonal) is now the more compute-efficient
direction.

## 6. Compute ledger

| item | GPU-h |
|---|--:|
| Arm A v1 (final weights lost to save-cadence bug) | 1.28 |
| A@15k non-gating diagnostic | 0.23 |
| Arm A v2 (stopped by contention guard at step 1801) | 0.12 |
| **Arm A v3 (valid, 20k)** | **1.28** |
| A@20k diagnostics + 5-seed control + E3s/E4s rerun | ~0.30 |
| Standard Flow timing probe | 0.04 |
| **Phase-3 total** | **~3.25** |

Under the 6.5 GPU-h cap. B/C/D unspent (~4.5 GPU-h would have been required).
