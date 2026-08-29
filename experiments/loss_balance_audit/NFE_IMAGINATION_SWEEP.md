# NFE vs imagination sweep — fixed-noise, no training

Date: 2026-08-29. No training. No loss modified. No canonical artifact overwritten.
Raw results: `nfe_imagination_sweep.json`. Figure:
`experiments/figures/nfe_imagination_vs_control.png`.

## 1. Sample set and pairing protocol

| Item | Value |
|---|---|
| Episode set | `experiments/isaacgym_episode_sets/replicate0_n96.pkl`, SHA `35144910b1471b7b` |
| Episodes used | 16 (first 16 of the frozen set) |
| Rollout steps per episode | 6 |
| current→future pairs per arm | **96** (16 x 6) |
| Task / horizon | 3-cube PushCube, H=100 |
| Noise seed | 777 (`torch.Generator`, CPU, one draw per episode-step) |
| Checkpoints | seed 42 `861dc344…`, seed 43 `c8e00ead…`, seed 44 `c2c13f55…`, all internal step 499000, **EMA** |
| Gaussian reference | `3C_adalnpintlarge_dlp_randcolor_H5_T100`, 100 steps |
| Data hash (feature stats) | `7abf83b8…` (training buffer, first 400 episodes) |

Target: the DLP encoding of the state the environment **actually reached** after
executing the policy's own first action (predicted-vs-realized, one step ahead).

## 2. Confirmation that z / current / goal are identical across NFE

`conditional_sample` draws its own noise internally (`flow_matching.py:358`) with
no injection point, so the diagnostic re-implements the **same** left-endpoint
fixed-step Euler loop with an externally supplied `x0`. **No training code was
modified.**

Two controls were run before the sweep:

| Check | Result |
|---|---|
| Diagnostic loop vs canonical `conditional_sample` (same manual seed, NFE=4) | **max abs diff 0.000e+00 — bit-identical** |
| Same `x0`, same NFE, run twice | **max abs diff 0.000e+00 — deterministic** |
| Same `x0`, NFE 1 vs 2/4/8/16 | endpoint displacement 0.713 / 1.136 / 1.370 / 1.493 — NFE genuinely changes the result |

Within each (episode, rollout step, training seed) **one** `x0` is drawn and
shared by all five NFE values; conditioning is built once from the same
observation and goal. The environment is advanced with the lowest-NFE action so
every arm sees one shared trajectory.

## 3. Metric

The already-validated, unchanged metric: **Hungarian matching on position only**
(dims 0:2), per-block errors reported in z-scored units, plus **Chamfer on
position** as the assumption-light primary. Sanity checks (permutation
invariance 0.0000, monotonicity, block separation, collapse detection) were
established previously and were not re-tuned. DLP dispersion is not used.

## 4. Results by seed and NFE (chamfer, position — lower is better)

| Seed | NFE1 | NFE2 | NFE4 | NFE8 | NFE16 |
|---|--:|--:|--:|--:|--:|
| 42 | 0.22710 | 0.12731 | 0.06544 | 0.04976 | 0.04707 |
| 43 | 0.23610 | 0.12554 | 0.06615 | 0.04984 | 0.04718 |
| 44 | 0.23322 | 0.12613 | 0.06403 | 0.04864 | 0.04678 |
| **mean** | **0.23214** | **0.12633** | **0.06521** | **0.04941** | **0.04701** |
| sd | 0.00375 | 0.00074 | 0.00088 | 0.00055 | 0.00017 |

Seed-to-seed spread is tiny (sd ≤ 0.004), so the curve is a property of the
training procedure, not of one checkpoint.

## 5-7. Aggregate, references, curve shape

| Method | State error | vs NFE1 | /Gaussian | /copy |
|---|--:|--:|--:|--:|
| **Gaussian @100** | **0.04004** | — | 1.00 | 0.54 |
| Flow @1 | 0.23214 | 0.0% | 5.80 | 3.10 |
| Flow @2 | 0.12633 | 45.6% | 3.16 | 1.69 |
| Flow @4 | 0.06521 | 71.9% | 1.63 | 0.87 |
| Flow @8 | 0.04941 | 78.7% | 1.23 | 0.66 |
| Flow @16 | 0.04701 | 79.7% | 1.17 | 0.63 |
| Copy-current baseline | 0.07484 | — | 1.87 | 1.00 |

Flow needs **NFE ≥ 4** merely to beat "copy the current state" (0.0652 < 0.0748).
At NFE 1 and 2 it is **worse than copying**, i.e. not predicting forward dynamics.

## 8. Paired improvements (per-episode, bootstrap 95% CI, pooled within seeds)

| Transition | Improvement | 95% CI | |
|---|--:|--:|---|
| 1 → 2 | +0.10581 | [+0.10085, +0.11088] | SIG |
| 2 → 4 | +0.06112 | [+0.05765, +0.06455] | SIG |
| 4 → 8 | +0.01580 | [+0.01446, +0.01712] | SIG |
| 8 → 16 | +0.00240 | [+0.00178, +0.00303] | SIG |
| **1 → 4** | **+0.16693** | [+0.16130, +0.17276] | SIG |
| **4 → 16** | **+0.01820** | [+0.01665, +0.01972] | SIG |

Every step helps, with sharply diminishing returns: 1→2 and 2→4 deliver 71.9% of
the total improvement; 8→16 delivers 1.0%.

## 9-10. Control vs imagination

**Provenance caveat, stated plainly: the control NFE curve is SEED 42 ONLY**
(3-cube NFE study, n=288 over three replicate sets, H=100 — the same frozen sets,
of which `35144910` is r0). The imagination curve is **three seeds**. These are
comparable in task, horizon and episode sets, but *not* matched in seed count,
and no three-seed control NFE curve exists.

| NFE | Control per-object (seed 42) | Imagination error (3 seeds) |
|---|--:|--:|
| 1 | 0.9051 | 0.23214 |
| 2 | 0.9375 | 0.12633 |
| 4 | 0.9479 | 0.06521 |
| 8 | **0.9595** | 0.04941 |
| 16 | 0.9317 | 0.04701 |

The control curve is **non-monotone** (NFE16 < NFE8 < NFE4), consistent with the
3–11 point evaluation noise floor established earlier, so its "asymptote" is not
well determined. Using its **maximum** (NFE8 = 0.9595) as the reference:

- Control reaches **97.7%** of its best value at **NFE 2** and **98.8%** at NFE 4.
- Imagination captures only **57.2%** of its total improvement by NFE 2, and
  **90.2%** by NFE 4; it needs **NFE 8** to reach 98.7%.

**Descriptive, post-hoc statistic (threshold chosen after seeing the data and
labelled as such):** control is within 95% of its best measured value from
**NFE 2** onward, whereas state prediction requires **NFE 4–8** to reach 95% of
its achievable improvement.

## 11. Vector-field / integration diagnostics

Seed 42, mean over the same paired samples:

| NFE | mean ‖Δx‖ per Euler update | cos(v_k, v_{k-1}) |
|---|--:|--:|
| 1 | 39.86 | n/a |
| 2 | 19.21 | 0.9730 |
| 4 | 9.49 | 0.9696 |
| 8 | 4.73 | 0.9851 |
| 16 | 2.36 | 0.9945 |

Update norm scales almost exactly as 1/NFE (39.86, 19.21, 9.49, 4.73, 2.36 —
each ~half the previous), i.e. the *total* path length is roughly conserved.
Consecutive velocity directions are highly but **not perfectly** aligned
(0.970–0.995), and alignment **increases** with finer steps, which is the
signature of a smoothly curved path being resolved better.

## 12. Is this consistent with Euler discretization error?

**Yes — consistent with, not proven to be.** The evidence:

- identical weights, identical `x0`, identical conditioning; only step size changes
- error decreases monotonically as step size shrinks, at every seed
- improvement plateaus (8→16 yields only 1.0% of the total)
- the velocity field is non-constant along the path (cos 0.970–0.995 < 1), so a
  single left-endpoint step cannot track it

This is the expected profile of first-order integration error on a curved ODE
path. It is **not** a mathematical proof, and no solver alternative was tried.

## 13. Classification: **D — FLOW PLATEAUS ABOVE GAUSSIAN**

Flow improves strongly with NFE (5.80x → 1.17x of Gaussian) but **plateaus at a
level clearly above Gaussian**: 0.04701 vs 0.04004, a residual gap of
**+0.00697 (1.17x)**. The seed sd at NFE16 is 0.00017, so the gap is **41.5x the
seed standard deviation** — far outside checkpoint noise.

This is exactly the two-effect case category D describes:
1. a **large** low-NFE discretization effect (0.232 → 0.047, 79.7% of the gap), and
2. a **small but robust** residual model/objective/representation gap (1.17x).

Category A is rejected because the curve plateaus rather than continuing; B is
rejected because most improvement is not complete by NFE 2 (only 45.6%); C is
rejected because improvement is strictly monotone and highly significant.

## 14. Residual gap at highest NFE

**+0.00697 chamfer (Flow@16 1.17x Gaussian@100).** Robust across seeds
(41.5x seed sd) but **small in absolute terms** — and note Gaussian uses 100 NFE
against Flow's 16, so the comparison is not iso-compute. A Flow@100 arm was not
run; that is an open, cheap question.

## 15. Revised status of the loss-reweighting hypothesis

**LOWER PRIORITY.**

79.7% of the Flow-vs-Gaussian imagination gap is closed by integration steps
alone, with the loss untouched. The residual 1.17x is real but small, and could
equally be objective, architecture, representation, or the un-matched compute
budget (16 vs 100 NFE). Reweighting the loss cannot address the dominant effect,
which is a sampler property.

Combined with the earlier finding that state already receives **99.4%** of the
scalar loss while action produces **1.32x** the state gradient — and that a
50/50 block objective would shift that ratio **30.5x the wrong way** — there is
no measured deficit that loss reweighting is well-positioned to fix.

## 16. Is "control and imagination need different inference budgets" supported?

**Yes, with the seed-count caveat.** Control is within 95% of its best measured
value from **NFE 2**; state prediction needs **NFE 4–8** for 95% of its
improvement, and is *worse than copying the current state* at NFE 1–2 where
control is already near its plateau. So the same trained model, on the same
scenes, requires roughly **2–4x more integration steps for accurate future-state
prediction than for competent control.**

Caveat: control is one seed, imagination is three. A three-seed control NFE curve
does not exist and was not run.

## 17. One recommended next experiment

**A second-order solver comparison at matched NFE — Heun/midpoint vs Euler, no
training, ~0.4 GPU-h.**

If the deficit is first-order discretization error, a second-order solver at
NFE 4 should approach Euler at NFE 8–16 while costing the same number of function
evaluations as Euler at 4 (Heun uses 2 evaluations per step, so Heun@2 steps = 4
NFE is the iso-compute comparison). This directly tests the §12 interpretation
and, if it holds, delivers better imagination *at unchanged inference cost*
without touching training at all — a strictly cheaper intervention than any loss
change.

It also sharpens the residual-gap question: if Heun closes most of the remaining
1.17x, the gap was integration; if it does not, the residual is genuinely
model/objective and the loss question returns with actual motivation.
