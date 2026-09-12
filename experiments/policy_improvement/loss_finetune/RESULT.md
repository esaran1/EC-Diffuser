# Canonical-Checkpoint Loss Fine-Tuning — RESULT

**Classification: LOSS-FT-INVALID.**
A configuration-plumbing defect made all four arms train the *same* objective.
Arm A and the canonical anchor are valid and retained; B/C/D are void.
**Stopped at the compute cap rather than re-running.**

---

## 1. State recovery after the ~22-hour idle

| check | finding |
|---|---|
| processes | **none alive** (exact-cmdline scan, not substring `pgrep`) |
| GPU | idle, 79 MiB, 0% |
| run locks | none held |
| Arm A training | **complete**: internal step 2000, full state present |
| Arm A control | **complete and VALID**, written 2026-09-10 22:39 |

**The science had finished; only the waiter was stuck.** No GPU compute was
consumed during the idle window, and none is charged for it.

## 2. Waiter bug — root cause and fix

**Root cause:** completion was inferred from a process-name substring match. After
Arm A saved its final checkpoint and released its lock, the pattern could never
match again, so the poll loop spun indefinitely while finished results sat on disk.

**Fix:** `train.py` now writes `RUN_COMPLETE.json` **only after** reloading
`state_final.pt` and verifying its internal step, recording run dir, PID, final
step, step-match flag, checkpoint SHA, end timestamp and exit status. The new
sequential runner advances on that sentinel plus an explicit step-match check —
never on process names. One launch path; atomic lock retained.

**Validated:** B, C and D each produced a sentinel with `step=2000, matches=True`,
and the runner advanced through all three arms unattended with no stalls.

## 3. THE DEFECT THAT INVALIDATES B/C/D

All four final checkpoints are **byte-identical**:

```
arm A: 965e7e32fe19cf01   arm C: 965e7e32fe19cf01
arm B: 965e7e32fe19cf01   arm D: 965e7e32fe19cf01
```

Pairwise weight comparison: **relative RMS 0.000000e+00** for every pair, model
and EMA. And the stored `loss_weight_matrix` is identical across arms
(sum 2442.0, t0 action weight 10.0) when C should be **2×** and D **5×** on the
action block.

**Root cause:** `diffuser/diffuser/configuration.py::diffusion_wrapper_kwargs`
builds an **explicit whitelist** of wrapper arguments. It never included
`mask_terminal_action` or `lambda_action`, so both silently fell back to the
constructor defaults (`False`, `1.0`) in every arm. **All four arms trained arm
A's objective.**

**Why the pre-launch pairing check missed it:** `verify_pairing.py` constructed
the diffusion wrapper *directly*, passing `mask_terminal_action` and
`lambda_action` as explicit kwargs — bypassing the very function that drops them.
It verified the configs were *parsed* correctly, not that the values *reached the
model through the training path*. It reported PAIRED, correctly for what it
tested and uselessly for what mattered.

**Fix applied and verified:** the whitelist now forwards both knobs, confirmed
end-to-end by rebuilding each arm exactly as `train.py` does:

```
arm A: mask=False lambda=1.0     arm C: mask=True lambda=2.0
arm B: mask=True  lambda=1.0     arm D: mask=True lambda=5.0
```

**Invalid artifacts preserved**, not deleted:
`data/panda_push/lossft_arm{B,C,D}_INVALID_kwargs_bug/`, plus the corresponding
logs, diagnostics JSONs, and runner log.

**Consequence:** the B/C/D "results" below are **not** loss-formulation results.
They are four independent evaluations of the *same* Arm-A policy, which is why
their spread is informative about evaluator noise and nothing else.

## 4. Compute ledger (GPU-active only; idle never charged)

| stage | GPU-h | status | reason |
|---|--:|---|---|
| canonical 5-seed anchor + offline | 0.229 | **VALID** | |
| LR safety pilot (+ diagnostics) | 0.055 | **VALID** | |
| Arm A training (2000 steps) | 0.103 | **VALID** | |
| Arm A diagnostics + 5-seed control | 0.232 | **VALID** | |
| B/C/D training (3 × 2000) | 0.309 | **INVALID** | kwargs whitelist bug |
| B/C/D diagnostics + control | 0.696 | **INVALID** | same bug |

| | GPU-h |
|---|--:|
| **A. valid scientific** | **0.619** |
| **B. invalid / wasted** | **1.005** |
| **C. total consumed** | **1.624** |

**Cap 1.85 → headroom 0.226. Re-running B/C/D correctly costs 1.005, projecting
2.629 — over cap by 0.779.** Per the directive I stopped rather than shorten arms
or exceed the cap.

## 5. Canonical anchor (VALID)

| E0 | E1s | E2s | E3s | E4s | macro | SD |
|--:|--:|--:|--:|--:|--:|--:|
| 0.7500 | 0.7500 | 0.8125 | 0.7396 | 0.8021 | **0.7708** | 0.0338 |

All VALID, zero mid-set contention. Offline: LIVE action 0.03143 / state 0.20497;
**EMA action 0.02689 / state 0.20532**.

## 6. LR pilot (VALID)

**PASSED at 1e-5** on every predeclared threshold: no NaN/Inf; no gradient
explosion; LIVE action **0.957×** canonical; LIVE state **1.000×**; no clipping
pathology; NFE10/15 imagination clean. **Fallback 3e-6 not run and remains
unrun.** LR frozen at 1e-5. Pilot directory deleted so it could not leak into A.

## 7. Arm A — VALID, and it PASSES the damage gate

**Endpoint:** internal step 2000; model, EMA, optimizer, RNG, meta, batches_drawn
all present; meta confirms LR 1e-5, warmup 100, betas (0.9, 0.999), batch 32 ×
accum 2, `dataloader_seed` 20260905.

**Training trajectory** (LIVE):

| step | total | action_loss | observation_loss |
|--:|--:|--:|--:|
| 0 | 0.2240 | 0.0608 | 0.2248 |
| 500 | 0.2291 | 0.0605 | 0.2299 |
| 1000 | 0.2202 | 0.0426 | 0.2214 |
| 1500 | 0.2243 | 0.0517 | 0.2254 |
| 1900 | 0.2380 | 0.0626 | 0.2386 |

**Five-seed control, all VALID, zero mid-set contention:**

| | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| canonical | 0.7500 | 0.7500 | 0.8125 | 0.7396 | 0.8021 | 0.7708 | 0.0338 |
| **Arm A** | 0.6875 | 0.8229 | 0.8229 | 0.7396 | 0.8021 | **0.7750** | 0.0597 |

**Damage gate (thresholds predeclared):**

| # | criterion | threshold | measured | verdict |
|--:|---|--:|--:|---|
| 1 | macro ≥ 0.60 × canonical | 0.46250 | 0.7750 | **PASS** |
| 2 | EMA replay action ≤ 1.50 × canonical | 0.04033 | 0.02788 (1.037×) | **PASS** |
| 3 | imagination not collapsed | — | clean discrete cubes at NFE10/15 | **PASS** |
| 4 | no NaN/Inf | — | none | **PASS** |

**ARM A PASSES.** Fresh-Adam restart plus 2,000 baseline steps did **not** damage
the competent policy (+0.0042 macro vs canonical). Drift from canonical: model
0.0081, EMA 0.0229 relative RMS.

**This is a genuine and useful result**: the warm-start fine-tuning design is
sound. It was the loss plumbing that failed, not the experimental design.

## 8. An unintended but useful by-product: evaluator noise

Because B/C/D were byte-identical to A, their four control runs are **independent
evaluations of one fixed policy**:

| run | E0 | E1s | E2s | E3s | E4s | macro |
|---|--:|--:|--:|--:|--:|--:|
| A | 0.6875 | 0.8229 | 0.8229 | 0.7396 | 0.8021 | 0.7750 |
| "B" | 0.7500 | 0.8125 | 0.8125 | 0.8125 | 0.7812 | 0.7938 |
| "C" | 0.7083 | 0.7812 | 0.8333 | 0.8021 | 0.7708 | 0.7792 |
| "D" | 0.7396 | 0.7708 | 0.8229 | 0.7396 | 0.8438 | 0.7833 |

**Same weights, macro spread 0.7750–0.7938 ≈ 1.9 pp; per-set spread up to 7.3 pp
(E3s 0.7396–0.8125).** This is Isaac Gym physics nondeterminism, consistent with
the frozen evaluator-resolution work. **Any real loss effect must clear roughly
2 pp at the macro level to be distinguishable at n=5 sets.** That is a directly
useful design constraint for the re-run — and it was free.

## 9. Red-team findings

| check | result |
|---|---|
| all finals really step 2000? | **yes**, verified from checkpoints and sentinels |
| identical initial model/EMA hashes? | yes |
| batch streams paired? | yes — identical step-0/100/200 losses across arms |
| any evaluation contention? | **none**, zero mid-set events in all 20 sets |
| **did λ scale the intended block?** | **NO — this is the defect** |
| **did terminal masking apply?** | **NO — never reached the model** |
| apparent winner driven by one set? | moot — no valid contrast exists |
| could extra training explain it? | moot |

The red-team pass is what caught the defect: identical SHAs across arms were
impossible under the intended design, and checking rather than trusting the
sentinels exposed it.

## 10. Claim ledger

**Established:**
- Canonical five-seed anchor: **0.7708 ± 0.0338**, all VALID.
- LR 1e-5 with 100-step warmup is safe for warm-start fine-tuning.
- Arm A passes the damage gate: **fresh-Adam restart + 2k steps does not damage
  the canonical policy** (0.7750 vs 0.7708).
- Evaluator noise on this protocol: **~1.9 pp macro spread for identical weights**.
- `diffusion_wrapper_kwargs` is a whitelist that silently drops unlisted wrapper
  arguments — now fixed and verified end-to-end.

**Cannot conclude:** anything whatsoever about terminal-action masking or
λ_action. Arms B/C/D never ran their objectives.

**Withdrawn:** nothing (no loss claim was ever made).

**Null:** not applicable — this is an invalidation, not a null result.

## 11. Classification: **LOSS-FT-INVALID**

Not because fine-tuning failed — Arm A shows it works — but because a
configuration-plumbing defect meant the loss variants were never tested.

## 12. Recommended next action (ONE, not launched)

**Re-run B, C, D only**, with the whitelist fix, at **1.005 GPU-h**, reusing the
existing valid canonical anchor and Arm A. This needs a cap increase from 1.85 to
approximately **2.7 GPU-h total consumed** (1.624 already spent + 1.005 redo +
small margin), or equivalently a fresh ~1.1 GPU-h allocation for the redo.

Before relaunching, add a **runtime assertion** that the constructed wrapper's
`mask_terminal_action` and `lambda_action` match the config, and log
`loss_weight_matrix.sum()` at step 0 — so a silent objective mismatch can never
again survive to a finished run. The pairing check must exercise the real
`train.py` construction path, not a parallel one.
