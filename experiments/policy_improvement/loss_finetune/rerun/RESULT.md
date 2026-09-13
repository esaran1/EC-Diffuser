# Corrected Canonical-Checkpoint Loss Fine-Tuning — RESULT

**Classification: LOSS-FT-NULL** (with one preliminary, unreplicated signal).

The corrected execution is scientifically valid: all four arms provably trained
their intended objectives. Over a 2,000-step warm-start fine-tuning window, from
one training seed, **neither terminal-action masking nor action-side loss
weighting produced a change in closed-loop control that this protocol can
resolve.** The A→B difference (+2.92 pp) is the one result that sits above the
observed identical-policy evaluation spread and is recorded as **preliminary and
unreplicated**, not established.

Code pinned at commit **`9c8e8e0`** for every scientific run.

---

## 1. Root cause — TWO defects, not one

**Root cause #1 (found previously): configuration whitelist.**
`diffuser/diffuser/configuration.py::diffusion_wrapper_kwargs` is an explicit
whitelist. It never listed `mask_terminal_action` or `lambda_action`, so both
silently fell back to constructor defaults. All four arms trained arm A's
objective and produced byte-identical checkpoints.

**Root cause #2 (found during this re-run's validation, BEFORE any GPU spend):
checkpoint buffer clobber.**
`loss_weight_matrix` is a **registered buffer**, so it is serialized into every
checkpoint. Warm-starting from the canonical checkpoint restored the canonical
λ=1 matrix **over** the correctly λ-scaled matrix the constructor had just built.
The scalar `self.lambda_action` attribute survived the load and still reported
the configured value.

Demonstrated directly for Arm D before launch:

```
AFTER canonical load : matrix sum 2442.0   lambda_action attr = 5.0   <-- mismatch
AFTER rebuild        : matrix sum 2610.0   action block 210.0 = 5 x 42.0
```

**Fixing only #1 would have reproduced the identical failure** — byte-identical
B/C/D — and wasted another ~1.0 GPU-h. Any check reading only scalar attributes
would have passed it.

## 2. Why the old pairing test missed defect #1

`verify_pairing.py` constructed the diffusion wrapper **directly**, passing
`mask_terminal_action` and `lambda_action` as explicit kwargs — bypassing the
very function that was dropping them. It verified the configs *parsed*
correctly, not that the values *reached the model through the training path*. It
reported PAIRED: correct for what it tested, useless for what mattered.

The same failure mode recurred in miniature during this work: a first version of
the per-timestep replay probe reimplemented the diagnostics' data loading instead
of calling it, and produced confidently wrong numbers (see §13). A verification
script that reimplements the path it checks can be confidently wrong.

## 3. The fix (production path)

- `configuration.py`: whitelist now forwards both knobs.
- `flow_matching.py`: new `rebuild_loss_weight_matrix()` recomputes the matrix
  from the configured knobs.
- `train.py`: calls it after warm-start load, for **both** model and EMA.

**Required invariant, verified by line order in `train.py`:**

```
config -> constructor -> canonical checkpoint load (L200)
       -> rebuild_loss_weight_matrix() (L212) -> objective_guard (L240)
       -> optimizer step 0 (L269)
```

**Buffer provenance recorded:** `loss_weight_matrix` is a configuration-derived
registered buffer, and checkpoint state can override it. Making such buffers
non-persistent is a sensible future cleanup; it was deliberately **not** done
during the pinned runs.

**§4 omission audit:** no other loss-relevant field is dropped.
`n_diffusion_steps` is reported as "omitted" but is a benign **alias** —
the constructor resolves `n_solver_steps` / `n_diffusion_steps` / `n_timesteps`
by precedence into one value, and the whitelist forwards the config value as
`n_timesteps=4`. Cleared, not waived.

## 4. Fail-loud safeguards (now mandatory and frozen)

`diffuser/diffuser/utils/objective_guard.py` — raises `ObjectiveMismatch` before
optimizer step 0; never warns and continues. Three independent layers:

1. **scalars** reached the wrapper;
2. **terminal mask genuinely in force**, probed through the wrapper's own
   `_make_conditioning_mask` — the exact function the loss uses;
3. **λ actually scaled the ACTUAL matrix**, checked against an *independently
   rebuilt* λ=1 reference. Not circular: dividing the observed action sum by λ
   would pass regardless of matrix contents. This layer is what catches the
   buffer clobber.

Negative controls confirmed the guard fires: it catches the buffer clobber, a
wrong λ, and a wrong mask, and passes all four arms after rebuild.

Per-run artifacts: `RUNTIME_OBJECTIVE_MANIFEST.json` (step 0) and
`RUNTIME_OBJECTIVE_MANIFEST_after10.json` (after 10 real optimizer steps), both
gated to PASS by the runner before diagnostics.

## 5. Pre-run validation (real production path)

`realpath_objective_test.py` drives `ArgsParser -> diffusion_wrapper_kwargs ->
utils.Config`, the path `train.py` uses. On one frozen batch, frozen noise,
frozen time:

| quantity | A | B | C | D |
|---|--:|--:|--:|--:|
| prediction sha16 | 795d8ef6… | 795d8ef6… | 795d8ef6… | 795d8ef6… |
| params sha16 (before) | 11a8fb13… | 11a8fb13… | 11a8fb13… | 11a8fb13… |
| total loss | 0.25499725 | 0.25549960 | 0.25685993 | 0.26094109 |
| unweighted action loss | 0.06590991 | 0.07941950 | 0.07941950 | 0.07941950 |
| unweighted observation loss | 0.25625703 | 0.25625703 | 0.25625703 | 0.25625703 |
| weighted action contribution | 0.00138206 | 0.00136038 | 0.00272077 | 0.00680191 |
| weighted observation contribution | 0.25361520 | 0.25413921 | 0.25413921 | 0.25413921 |
| terminal action contribution | 0.00002448 | 0.0 | 0.0 | 0.0 |
| loss denominator | 46560 | 46464 | 46464 | 46464 |

**Same model output, different objective** — exactly the distinction the failed
run lacked. λ ratios are exact: C/B = **2.000000**, D/B = **5.000000**.
Observation contribution frozen across B/C/D.

**One-step divergence sentinel:** all six pairs DISTINCT (previously B/C/D
identical). Gradient norms order B (0.152) < C (0.187) < D (0.342).

**§10 "A would remain A":** the corrected path builds A's matrix with sha16
`f6f56c359875b7f4`, **byte-identical to valid Arm A's stored matrix**. The fix
does not retrospectively alter A's objective; A remains the valid control and was
not re-run.

**Real-path pairing:** 36/36 checks PASS → PAIRED.

## 6. Provenance

| item | value |
|---|---|
| code commit (all scientific runs) | **`9c8e8e0`** |
| canonical checkpoint | `flow/3C_…seed42/state_400000.pt` |
| canonical matrix | sum 2442.0, action 42.0, obs 2400.0 |
| Arm A final (reused, not re-run) | step 2000, matrix sha `f6f56c359875b7f4` |
| B final | sha256:16 `d902ede7a73bb1a0`, step 2000, clean |
| C final | sha256:16 `dde609e413ea7ae0`, step 2000, clean |
| D final | sha256:16 `0a70d93a7eba9bcf`, step 2000, clean |
| seeds | train seed 42, `dataloader_seed` 20260905, eval NOISE_SEED 20260902 |
| shared settings | LR 1e-5, warmup 100, `step_start_ema`=0, batch 32 × accum 2, first-action weight 10 |

Invalid first-attempt checkpoints (all `965e7e32fe19cf01`) preserved under
`*_INVALID_kwargs_bug/`, including their imagination PNGs, which were renamed so
the corrected runs could not overwrite them.

## 7. Runtime objective manifests (what each arm ACTUALLY constructed)

| arm | mask | λ | matrix sum | action block | obs block | first-action wt | denominator | verdict |
|---|---|--:|--:|--:|--:|--:|--:|---|
| A | False | 1.0 | 2442.0 | 42.0 | 2400.0 | 10.0 | 46560 | pre-manifest¹ |
| B | True | 1.0 | 2442.0 | 42.0 | 2400.0 | 10.0 | 46464 | **PASS** |
| C | True | 2.0 | 2484.0 | 84.0 | 2400.0 | 20.0 | 46464 | **PASS** |
| D | True | 5.0 | 2610.0 | 210.0 | 2400.0 | 50.0 | 46464 | **PASS** |

¹ Arm A predates the manifest mechanism; its objective is verified by its stored
matrix sha, which the corrected path reproduces byte-identically.

**Element-wise matrix verification from the real trained checkpoints:**
C action side **exactly** 2× B (max deviation **0.0**); D action side **exactly**
5× B (max deviation **0.0**); C and D observation sides **exactly** equal to B's
(max deviation **0.0**). No unintended observation-side scaling.

## 8. Corrected training table

Step-0 values confirm pairing: identical unweighted losses (same weights, same
batch), different weighted totals.

| step | metric | A | B | C | D |
|--:|---|--:|--:|--:|--:|
| 0 | total (weighted) | 0.2240 | 0.2244 | 0.2258 | 0.2301 |
| 0 | unweighted flow | 0.2231 | **0.2236** | **0.2236** | **0.2236** |
| 0 | action_loss | 0.0608 | **0.0727** | **0.0727** | **0.0727** |
| 0 | observation_loss | 0.2248 | **0.2248** | **0.2248** | **0.2248** |
| 500 | action / obs | 0.0605 / 0.2299 | 0.0746 / 0.2298 | 0.0743 / 0.2299 | 0.0741 / 0.2300 |
| 1000 | action / obs | 0.0426 / 0.2214 | 0.0525 / 0.2214 | 0.0521 / 0.2213 | 0.0517 / 0.2214 |
| 1500 | action / obs | 0.0517 / 0.2254 | 0.0637 / 0.2254 | 0.0645 / 0.2255 | 0.0648 / 0.2255 |
| 1900 | action / obs | 0.0626 / 0.2386 | 0.0775 / 0.2386 | 0.0770 / 0.2386 | 0.0764 / 0.2386 |

A's action_loss is measured over a different active set (it includes t=H−1), so
A-vs-B/C/D action_loss is not directly comparable; B/C/D are mutually comparable.

**Pairwise final-weight drift (relative RMS, model, buffers excluded):**

| pair | drift | | pair | drift |
|---|--:|---|---|--:|
| A–B | 1.259518e-03 | | B–C | 2.790144e-03 |
| A–C | 2.825863e-03 | | B–D | 5.511703e-03 |
| A–D | 5.511066e-03 | | C–D | 3.888638e-03 |

All nonzero and finite (previously **exactly 0.0** for every pair), and drift
scales monotonically with λ.

## 9. Full control table

NFE = 4, five fixed scenario-generation seeds, n = 96 episodes each.
**All 20 sets VALID; zero mid-set foreign GPU memory in every set.**

| policy | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| canonical *(context only)* | 0.7500 | 0.7500 | 0.8125 | 0.7396 | 0.8021 | 0.7708 | 0.0338 |
| **A** — matched control | 0.6875 | 0.8229 | 0.8229 | 0.7396 | 0.8021 | **0.7750** | 0.0597 |
| **B** — terminal mask | 0.7500 | 0.8333 | 0.8229 | 0.8021 | 0.8125 | **0.8042** | 0.0324 |
| **C** — mask + λ=2 | 0.7083 | 0.7812 | 0.8229 | 0.7917 | 0.8229 | **0.7854** | 0.0469 |
| **D** — mask + λ=5 | 0.7083 | 0.7604 | 0.8333 | 0.8229 | 0.8125 | **0.7875** | 0.0524 |

## 10. Primary contrasts

**A → B — terminal zero-action supervision ablation**
macro 0.7750 → 0.8042, **+2.92 pp**.
per-set: E0 +6.25, E1s +1.04, E2s 0.00, E3s +6.25, E4s +1.04 pp.
**4/5 improved, 1 unchanged, 0 worse.** Leave-one-out macro delta stays positive
across all five drops (+2.08 to +3.65 pp) — not driven by a single set.

**B → C — 2× action-side weighting**
macro 0.8042 → 0.7854, **−1.88 pp**. 1 improved, 1 unchanged, 3 worse.

**B → D — 5× action-side weighting**
macro 0.8042 → 0.7875, **−1.67 pp**. 2 improved, 1 unchanged, 2 worse.

**C → D — descriptive strength comparison**
macro 0.7854 → 0.7875, **+0.21 pp**. 2 improved, 1 unchanged, 2 worse.

Canonical is context only. A is the matched fine-tuning control; canonical-vs-C
and canonical-vs-D are **not** used as causal contrasts.

## 11. Replay results

| policy | LIVE action | LIVE state | EMA action | EMA state |
|---|--:|--:|--:|--:|
| canonical | — | — | 0.02689 | 0.20532 |
| A | 0.02853 | 0.20527 | 0.02788 | 0.20533 |
| B | 0.17298 | 0.20531 | 0.17224 | 0.20533 |
| C | 0.17246 | 0.20518 | 0.17203 | 0.20530 |
| D | 0.17200 | 0.20532 | 0.17184 | 0.20527 |

**Replay action error is NOT comparable between A/canonical and B/C/D.** See §13.
Replay **state** error is comparable everywhere and is **flat across all arms**
(0.20527–0.20533) — stronger action weighting did **not** damage state
prediction. Replay is not equivalent to control and does not rank arms.

## 12. Imagination

NFE10/NFE15 panels are clean, discrete, well-formed cubes for **B, C and D**,
including D at λ=5 — the most aggressive arm. B and D are visually
near-indistinguishable at every NFE. The blur at NFE1/2/4 is the expected
few-step flow behaviour and matches canonical. **No qualitative collapse in any
arm.** No NaN/Inf anywhere; action gradients alive in all arms.

## 13. Per-timestep replay decomposition (EMA)

| arm | t=0 *(executed)* | t=1 | t=2 | t=3 | **t=H−1** | all-t *(as reported)* |
|---|--:|--:|--:|--:|--:|--:|
| A | 0.01634 | 0.03228 | 0.04047 | 0.04970 | **0.00062** | 0.02788 |
| B | 0.01640 | 0.03263 | 0.04043 | 0.04961 | **0.72214** | 0.17224 |
| C | 0.01631 | 0.03239 | 0.04012 | 0.04921 | **0.72214** | 0.17203 |
| D | 0.01622 | 0.03205 | 0.03988 | 0.04890 | **0.72214** | 0.17184 |

**The entire 6.4× difference lives at t=H−1.** A learns to emit the normalized
zero placeholder (0.00062); B/C/D, having no supervision there, stop doing so
(0.72214, identical to five decimals). At t=0..3 all four arms agree to ~3
decimals, and the **executed** action error at t=0 is essentially unchanged
(0.01622–0.01640, ~1.1% spread).

So masking removed a learned behaviour that only ever applied to a non-executed
placeholder slot. It did not degrade the actions the policy actually takes.

Consequence: the predeclared damage-gate criterion "EMA replay action ≤ 1.5×
canonical" **cannot be applied to B/C/D** — different active set. Control,
imagination and NaN checks still apply, and all pass.

**Methodological note.** A first version of this probe rebuilt the dataset from
the arm config and used seed 20260905, producing numbers that contradicted the
diagnostics (A t=H−1 = 0.81093). That was the probe being wrong: the diagnostics
loads via `utils.load_diffusion` with an explicit `override_dataset_path` and
uses NOISE_SEED = 20260902. Once the probe imported the diagnostics' own loader
and seed it reproduced the reported values exactly. Recorded because it is the
same lesson as the original defect.

## 14. Repeated-evaluation variability — DESCRIPTIVE

From the first attempt's byte-identical checkpoints: four independent
evaluations of **one fixed policy** under Isaac Gym's nondeterministic physics.

| repeat | E0 | E1s | E2s | E3s | E4s | macro |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 0.6875 | 0.8229 | 0.8229 | 0.7396 | 0.8021 | 0.7750 |
| 2 | 0.7500 | 0.8125 | 0.8125 | 0.8125 | 0.7812 | 0.7937 |
| 3 | 0.7083 | 0.7812 | 0.8333 | 0.8021 | 0.7708 | 0.7792 |
| 4 | 0.7396 | 0.7708 | 0.8229 | 0.7396 | 0.8438 | 0.7833 |

macro range **1.87 pp**; empirical SD **0.80 pp**; per-set ranges E0 6.25, E1s
5.21, E2s 2.08, E3s 7.29, E4s 7.29 pp.

**This is DESCRIPTIVE EVALUATOR VARIABILITY, not formal inferential
uncertainty.** It was not predeclared, and four repeats do not estimate the
evaluation distribution precisely. It is **not** a significance threshold: no
claim here rests on a rule that "anything below N pp is zero."

> Differences of the same order as the observed repeated-evaluation variability
> are difficult to interpret confidently from a single training seed and one
> simulator realization per arm.

## 15. Red-team findings

| check | result |
|---|---|
| identical initialization across arms? | **yes** — params sha `11a8fb13…`, identical step-0 unweighted losses |
| identical batch stream? | **yes** — `dataloader_seed` 20260905; step-0 unweighted losses identical to 4 dp |
| did the mask actually apply? | **yes** — denominator 46560→46464, terminal contribution →0 |
| did λ actually scale the action block? | **yes** — action block 42/84/210, exact 2×/5× element-wise |
| observation weighting frozen? | **yes** — 2400.0 in all arms, max element deviation 0.0 |
| action gradients altered as expected? | **yes** — EMA action_grad ratio to B: C 1.81, D 4.36 |
| stronger action weight damaged state prediction? | **no** — state_loss 0.2249 flat; replay state flat |
| correct final step? | **yes** — all arms internal step 2000, sentinels `step_matches=true` |
| hidden resumption? | **no** — fresh Adam, `trainer.step=0`, one launch path, atomic lock |
| evaluation contention? | **none** — 0 MiB foreign mid-set in all 20 sets |
| is A→B driven by one set? | **no** — leave-one-out +2.08…+3.65 pp, all positive |
| is A→B larger than identical-policy spread? | **partly** — macro +2.92 pp exceeds the 1.87 pp range and B's 0.8042 exceeds the highest repeat (0.7937), **but only 1/5 sets (E1s) falls outside its identical-policy per-set range** |
| LIVE vs EMA agreement? | **yes** — same ordering and magnitudes |
| qualitative imagination collapse? | **none**, including D at λ=5 |
| could extra training explain A→B? | **no** — identical step count, LR, schedule, batches |

**The honest tension in A→B:** the macro effect is coherent in direction and not
single-set-driven, yet per-set it mostly sits *inside* the range one identical
policy spans across repeats. A macro mean aggregates five sets and so is more
stable than any one set, which is why the macro can clear the macro spread while
individual sets do not clear theirs. This is suggestive, not resolved.

## 16. Claim ledger

**Established (this program):**
- Two independent configuration-plumbing defects existed; both are fixed and
  verified through the production path.
- All four corrected arms provably trained their intended, distinct objectives:
  exact matrix structure, exact λ ratios, all-pairs weight divergence.
- Arm A passes the restart-damage gate: fresh-Adam restart + 2,000 steps does
  not damage the competent policy (0.7750 vs canonical 0.7708).
- Action-side weighting up to λ=5 does **not** damage state prediction or
  imagination over this window.
- `replay_action_error` is not comparable between masked and unmasked arms; the
  difference is entirely the non-executed t=H−1 slice.
- Descriptive evaluator variability on this protocol: 1.87 pp macro range,
  0.80 pp empirical SD for one identical policy.

**Preliminary (single seed, unreplicated):**
- Terminal-action target masking (A→B) **may** improve closed-loop control
  (+2.92 pp macro, 4/5 sets up, none down). Requires multi-seed replication
  before it can be called an effect.

**Null / unresolved over this window:**
- B→C (−1.88 pp) and B→D (−1.67 pp): action-side weighting produced **no
  improvement** over masking alone, and the differences are on the scale of
  simulator variability — **unresolved**, not demonstrated harm.
- C→D (+0.21 pp): no distinguishable difference between the tested strengths.

**Cannot conclude:**
- Whether any λ is optimal, or whether λ helps at other windows/seeds/budgets.
- Whether the terminal target is "wrong". **It is not called a bug**; §13 shows
  it is a learned placeholder behaviour on a non-executed slot, and removing it
  is an ablation whose benefit is unconfirmed.

**Withdrawn:** nothing.

## 17. Compute accounting

| category | GPU-h |
|---|--:|
| previous valid | 0.619 |
| **previous invalid / wasted** | **1.005** |
| previous total | 1.624 |
| new corrected rerun — B/C/D train + diag | 0.999 |
| new corrected rerun — per-timestep replay probes | 0.133 |
| **new rerun total** | **1.133** *(cap 1.15, headroom 0.017)* |
| new rerun invalid / wasted | 0.000 |
| **CUMULATIVE TOTAL** | **2.757** *(ceiling 2.774)* |

Within both caps. Idle waiting and CPU-only validation are not charged.

Honest note: the probe line includes a first probe run whose numbers were invalid
(wrong loader/seed, §13) and had to be repeated. That consumed part of the
headroom; it is counted, not hidden.

## 18. Classification: **LOSS-FT-NULL**

The corrected execution is valid, so LOSS-FT-INVALID is retired. It is not
LOSS-FT-GO: the only candidate signal is a single contrast from one training
seed that clears the macro-level identical-policy spread but not the per-set
spread on 4 of 5 sets, while both weighting interventions were flat-to-negative.
That does not meet the bar of "meaningful enough, after red-teaming and
simulator-variability context, to justify replication across additional seeds"
as a demonstrated effect — though see §20.

## 19. Tal's question, answered plainly

> **Did changing terminal-action supervision or action-vs-state loss weighting
> meaningfully change an already competent EC-Diffuser policy over this
> 2,000-step fine-tuning window?**

**Essentially no — with one caveat worth a second look.**

**Action-vs-state loss weighting: no.** Multiplying the entire action-side
objective by 2× or 5× measurably changed the optimization — action gradients
scaled 1.8× and 4.4×, and the weights moved further from the starting point in
proportion to λ — yet closed-loop success did **not** improve. Both weighted arms
came out slightly *below* masking alone (−1.9 and −1.7 pp), by margins on the
scale of this simulator's run-to-run variability. They also did no harm: state
prediction and imagination were untouched. So the model was **not** being held
back by under-weighting of actions in a way this intervention could fix.

**Terminal-action supervision: possibly, but unproven.** Removing supervision on
the final-step action target — a normalized zero placeholder that is never
executed — gave the largest and most consistent gain seen here: +2.92 pp macro,
better on 4 of 5 scenario sets and worse on none, and it does not depend on any
single set. Against that: the gain is from **one training seed**, and on 4 of 5
individual sets it falls within the spread a *single identical policy* produces
across repeated evaluations. The diagnostics confirm the model genuinely stopped
emitting the placeholder (t=H−1 error 0.0006 → 0.72) while the actions it
actually executes were unchanged.

**Bottom line for the project:** loss re-balancing is not the lever that makes
this policy better. Terminal-target masking is the only candidate worth another
look, and it needs multi-seed replication before it counts as anything. One
training seed cannot settle it.

## 20. Next scientific action — exactly ONE, not launched

**Replicate A vs B only, across three additional training seeds** (matched
warm-start fine-tunes from the same canonical checkpoint, same 2,000-step
window, same five-set evaluation protocol), to determine whether the +2.92 pp
terminal-masking effect survives seed variation.

Rationale: it is the only contrast with a coherent, non-single-set signal, and
seed variation is the dominant unaddressed confound. Dropping C and D keeps the
cost to roughly 6 × 0.167 = **~1.0 GPU-h** (3 seeds × 2 arms, train + diagnostics
+ control). λ sweeps, first-action-weight changes, longer fine-tunes and full
retraining are **not** recommended: this program found no evidence that action
weighting is the operative lever.

**Not launched.** Requires a new allocation and explicit approval.
