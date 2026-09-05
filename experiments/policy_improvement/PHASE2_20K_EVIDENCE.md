# Phase 2 — 20k MeanFlow run: evidence record (NOT YET CLASSIFIED)

Run completed. **No classification is made in this document** — the five-seed
closed-loop control endpoint is part of the predeclared gate and its first
attempt was invalidated by GPU contention (see §2).

---

## 1. Run provenance — valid and uninterrupted

| item | value |
|---|---|
| launch gate passed | 2026-09-03T18:11:38-04:00 |
| GPU at launch | **387 MiB, 0% util** — clean |
| commit | `082f338a09b870f359462401d62317b6ea05348d` |
| exit | **rc=0**, no contention event |
| steps | **20,000** (final log line 19,900) |
| final checkpoint | `state_18981.pt` (last save under `n_saves=60` label rounding) |
| wall clock | **3.175 GPU-h** @ 0.572 s/step |
| NaN/Inf in training | **none** |
| `boundary_fraction` | exactly 0.5000 at every log |

**No emergency resume occurred.** Training was continuous from step 0, so the
data-order caveat from the resume infrastructure does **not** apply here.

Phase-2 compute ledger: 0.374 (aborted attempts) + 3.175 (valid run) +
~0.2 (diagnostics) = **~3.75 GPU-h** of the 5.0 cap.

## 2. Five-seed control — first attempt INVALIDATED

E0 and E1s were evaluated in the window **13:56:36 – 14:01:05** on 2026-09-05.
A foreign process (**PID 1174180**, `python`, **6652 MiB**) was observed on the
GPU at ~13:54 and was gone by 14:02:17. **The evaluation window overlaps the
foreign process's presence**, so neither set can be shown to have run
uncontended.

Per the standing correction — GPU execution/scheduling differences can alter
contact-rich Isaac Gym physics outcomes, so contention is a potential
**experimental confound**, not merely a throughput effect — both results
(E0 0.0000, E1s 0.0000) are **marked INVALID for classification**, not averaged
in, and not statistically corrected. The run was stopped before E2s/E3s/E4s.

**Rerun:** all five sets are being re-evaluated from scratch on a verified-clean
GPU. The harness now records an `nvidia-smi` snapshot **immediately before and
after each set**, aborts a set that starts contended, and marks a set
`INVALID_CONTENDED_DURING` if a foreign process appears mid-set. Only sets
marked `VALID` enter the mean/SD.

## 3. Current evidence (offline diagnostics — contention-permitted per protocol)

Offline diagnostics do not invoke Isaac Gym physics, so contention affects
throughput only. They were additionally **reproduced bit-identically** on the
clean GPU (LIVE objective 0.26062 both times), which rules out numerical
inconsistency.

### Training: rapid initial drop, then flat for ~19k steps

| step | loss |
|--:|--:|
| 0 | 0.8569 |
| 1,000 | 0.2653 |
| 2,500 | 0.2704 |
| 5,000 | 0.2781 |
| 10,000 | 0.2674 |
| 15,000 | 0.2548 |
| 19,900 | 0.2640 |

The objective falls rapidly over the first ~1k steps and is then **essentially
flat through 20k** — the value at 19,900 is within noise of the value at 1,000.

### Imagination: incoherent at every tested NFE

Panels: `phase2/mf20k_imagination_{EMA,LIVE}_ep{0,1}.png`, same frozen examples
and identical conditions as the Standard Flow panel.

- **NFE1, 2, 4, 10, 15 — all incoherent.** Smeared and duplicated cubes,
  ghosting, and colours inconsistent with the conditioning (blue/green inputs
  yielding red/yellow/purple objects).
- **Including NFE10 and NFE15**, where frozen Standard Flow is coherent on these
  same examples.
- The DLP-reference and Gaussian control rows established in Phase 1A that this
  decode path renders good samples faithfully, so a decoder/post-processing
  explanation is **ruled out**.

### Offline neutral replay (EMA deployment weights)

| metric | MeanFlow 20k (EMA) | Standard Flow @ NFE4 (frozen) |
|---|--:|--:|
| action replay error | **0.1164** | 0.0221 |
| state replay error | **0.2283** | 0.0488 |

LIVE is comparable to EMA (0.1275 / 0.2275), i.e. **no large LIVE/EMA
discrepancy** of the kind seen in the prior OGBench auxiliary-iMF work.

### Action gradients

Alive under both LIVE (0.0731) and EMA (0.0648) — the action branch is receiving
signal. This is **not** a dead-branch failure.

## 4. Instrumentation note — the NaN is a probe bug, not a training pathology

My probe read `action_loss` / `observation_loss` from the loss info dict. Those
keys are emitted by `ConditionalFlowMatching` but **not** by
`ImprovedMeanFlow`, which returns `meanflow_loss`,
`unweighted_meanflow_loss`, and `boundary_fraction`
(`fast_generation.py:238-242`). The NaNs are therefore missing-key artifacts.

Semantic action/state losses are being reconstructed offline from the saved
checkpoint. **The completed model is not modified or retrained** to obtain them.
If they cannot be reconstructed faithfully they will be reported **N/A**.

## 5. Status

**NOT CLASSIFIED.** Awaiting the uncontended five-seed control result.

The offline evidence is already strongly negative for **this tested viability
configuration**. Whatever the control shows, the wording will be:

> Boundary ImprovedMeanFlow with the audited current PandaPush formulation and
> LR 4e-5 did not establish viability at 20k steps.

and **not** "MeanFlow does not work" — one formulation and one optimization
configuration were tested.

---

## 6. Semantic losses reconstructed (§7 resolved — no N/A needed)

The NaNs were an instrumentation artifact, now fixed offline. `ImprovedMeanFlow`
exposes `meanflow_loss` / `unweighted_meanflow_loss` / `boundary_fraction`, and
its public `loss()` takes no kwargs — so the semantic split was recomputed by
calling `_compute_meanflow_loss(..., return_details=True)` (read-only) and
applying the model's own error definition
`|compound_velocity − target_velocity|` under its own conditioning mask.
**The trained model was neither modified nor retrained.** Fixed `(r,t)`/noise
draw across LIVE and EMA; 30 batches each.

| | LIVE | EMA |
|---|--:|--:|
| meanflow loss | 0.2720 | 0.2702 |
| **action loss** | **0.1684** | **0.1545** |
| **state loss** | **0.2693** | **0.2678** |
| action / state | 0.625 | 0.577 |
| action grad norm | 0.0852 | 0.0720 |
| action grad alive | **yes** | **yes** |

LIVE and EMA agree closely on every quantity — no deployment-weight pathology.

## 7. Five-seed control — attempt 2 (clean launch, per-set gating)

Launched with the GPU verified clean (no compute processes at 14:03:10). The
harness snapshots `nvidia-smi` immediately before and after **each** set.

| set | success | status | note |
|---|--:|---|---|
| **E0** | **0.0000** | **VALID** | n=96, 129 s, no foreign process before or after |
| E1s | 0.0000 | **INVALID_CONTENDED_DURING** | PID 1191611 (4762 MiB) appeared mid-set |
| E2s | — | **ABORTED_CONTENDED** | refused to start; foreign process present |
| E3s | — | **ABORTED_CONTENDED** | refused to start |
| E4s | — | **ABORTED_CONTENDED** | refused to start |

**n_valid = 1 of 5. Mean and SD are deliberately not computed.**

The gating behaved exactly as intended: rather than emitting five
contamination-confounded numbers, it produced one trustworthy value and
explicitly refused the rest. A retry loop is armed to re-run only the four
outstanding sets whenever the GPU is genuinely clean, merging results.

**Still NOT CLASSIFIED** — the predeclared gate requires the uncontended
five-seed control, and four sets remain outstanding.
