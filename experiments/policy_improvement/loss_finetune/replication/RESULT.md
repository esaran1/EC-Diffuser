# Terminal-Action Mask — Paired Four-Seed Replication

**Classification: TERMINAL-MASK-REPLICATED** (preliminary; four fine-tuning
seeds, one pretrained policy, one task).

Across four paired fine-tuning runs from the **same** competent canonical
checkpoint, removing terminal zero-action supervision improved closed-loop
control by a mean of **+2.24 pp**, positive in 3 of 4 seeds, exactly zero in the
fourth, and **negative in none**. The aggregate is not driven by the original
seed. No state-prediction, imagination, or executed-action cost was observed.

---

## 1. Claim corrections carried forward

**Screening classification corrected** from LOSS-FT-NULL to **LOSS-FT-GO —
NARROWLY FOR TERMINAL-MASK REPLICATION**. The predeclared rule set GO when a
signal is coherent enough *to justify replication*, not when an effect is
established; the earlier NULL conflated "not established" with "not worth
replicating."

**Action-weighting language softened.** The claim "the model was not being held
back by under-weighted actions" was too broad for one seed, λ ∈ {1,2,5} and a
2,000-step window. It now reads: *in this single-seed 2,000-step fine-tuning
experiment, increasing the action-side loss weight to λ=2 or λ=5 did not improve
closed-loop control relative to terminal masking alone.* C and D are **not**
re-run and their conclusion is unchanged.

## 2. Replication protocol

| item | value |
|---|---|
| source checkpoint (FIXED, all replicates) | `flow/3C_…_seed42/state_400000.pt` (canonical step 499000) |
| replication unit | **paired A/B fine-tuning run** |
| fine-tuning stochastic seeds | 42 *(original)*, 43, 44, 45 |
| dataloader seeds | 20260905 *(original)*, 20260906, 20260907, 20260908 |
| arm order | original: historical · r1: **B→A** · r2: **A→B** · r3: **B→A** |
| shared per pair | model & EMA init, fresh Adam, LR 1e-5, warmup 100, `step_start_ema`=0, 2000 steps, batch 32×accum 2, first-action weight 10, architecture, evaluator |
| only intended difference | terminal-action supervision + inherited 0.207% renormalization |

These are **fine-tuning stochastic seeds, not independent pretrained policies.**
Every replicate warm-starts from the same canonical checkpoint. The pretrained
`flow/…_seed43` and `flow/…_seed44` models exist in this repo and were
deliberately **not** used.

The balanced A/B ordering avoids a systematic A-always-first temporal confound.
A and B never ran concurrently.

## 3. Provenance and guards

| item | value |
|---|---|
| code commit (all replication runs) | **`59d132e`** |
| screening/rerun commit | `9c8e8e0` |
| per-replicate configs | `pandapush_lossft_arm{A,B}_seed{43,44,45}.py` |
| runtime manifests | **PASS** at step 0 and after 10 real optimizer steps, for all 7 new arms |
| final internal step | **2000** for all 7 new arms, `step_matches=true` |
| postrun pairing | **PASS** for all three replicates |

All pinned fail-loud safeguards retained unchanged: production-path config
propagation, post-load loss-matrix rebuild, `objective_guard` (scalar + actual
mask + independently rebuilt λ=1 reference + matrix-level), runtime manifest,
10-step recheck, early divergence checks, atomic run lock, `RUN_COMPLETE`
sentinel, true final-step validation, continuous contention monitoring.

**Preflight pairing, all three replicates: 19/19 checks PASS**, denominators
A=46560 / B=46464, ratio **1.002066116**.

**Postrun A→B weight drift** (nonzero, so the arms genuinely diverged; matrices
identical, as they must be since masking acts on the conditioning mask):
r3 = 1.166530e-03; r1 and r2 likewise nonzero.

### One hard gate fired (zero GPU cost)

The first launch died at **4 seconds** on r1 arm B and the runner stopped the
whole program. `train.py` parses arguments **twice** — `'diffusion'` then
`'plan'` — and `add_extras` re-applied `--dataloader_seed` to the plan config,
which has no such key. My preflight had exercised only the first parse: the same
class of error as the original defect, verifying part of a path and assuming the
rest. Fixed with per-replicate config files (no pinned code touched), and
`verify_pair.py` now exercises **both** parses. Failed log preserved as
`replication_runner_attempt1_plumbing.log`.

## 4–7. Per-replicate five-set control

All 40 sets across all 8 arms: **VALID**, `n_valid_sets=5`, **zero** mid-set
foreign GPU memory.

**Original — fine-tuning seed 42**

| arm | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| A | 0.6875 | 0.8229 | 0.8229 | 0.7396 | 0.8021 | 0.7750 | 0.0597 |
| B | 0.7500 | 0.8333 | 0.8229 | 0.8021 | 0.8125 | 0.8042 | 0.0324 |
| **B−A** | +0.0625 | +0.0104 | 0.0000 | +0.0625 | +0.0104 | **+0.0292** | |

B>A on 4, tied 1, B<A 0.

**Replicate 1 — fine-tuning seed 43**

| arm | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| A | 0.7292 | 0.7708 | 0.8229 | 0.7604 | 0.7812 | 0.7729 | 0.0341 |
| B | 0.7292 | 0.8125 | 0.7604 | 0.7708 | 0.7917 | 0.7729 | 0.0316 |
| **B−A** | 0.0000 | +0.0417 | −0.0625 | +0.0104 | +0.0104 | **+0.0000** | |

B>A on 3, tied 1, B<A 1. **Exactly zero macro delta** — the one non-positive seed.

**Replicate 2 — fine-tuning seed 44**

| arm | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| A | 0.7292 | 0.6979 | 0.8021 | 0.7396 | 0.7812 | 0.7500 | 0.0417 |
| B | 0.7917 | 0.7708 | 0.8021 | 0.7708 | 0.8125 | 0.7896 | 0.0186 |
| **B−A** | +0.0625 | +0.0729 | 0.0000 | +0.0312 | +0.0312 | **+0.0396** | |

B>A on 4, tied 1, B<A 0.

**Replicate 3 — fine-tuning seed 45**

| arm | E0 | E1s | E2s | E3s | E4s | macro | SD |
|---|--:|--:|--:|--:|--:|--:|--:|
| A | 0.7604 | 0.7708 | 0.7812 | 0.8229 | 0.7812 | 0.7833 | 0.0238 |
| B | 0.7396 | 0.8333 | 0.8229 | 0.7708 | 0.8542 | 0.8042 | 0.0474 |
| **B−A** | −0.0208 | +0.0625 | +0.0417 | −0.0521 | +0.0729 | **+0.0208** | |

B>A on 3, tied 0, B<A 2 — positive macro despite two sets favouring A.

## 8. Four-seed summary

| fine-tune seed | A macro | B macro | **B−A (pp)** | A SD | B SD | B>A | tie | B<A |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 *(original)* | 0.7750 | 0.8042 | **+2.92** | 0.0597 | 0.0324 | 4 | 1 | 0 |
| 43 | 0.7729 | 0.7729 | **+0.00** | 0.0341 | 0.0316 | 3 | 1 | 1 |
| 44 | 0.7500 | 0.7896 | **+3.96** | 0.0417 | 0.0186 | 4 | 1 | 0 |
| 45 | 0.7833 | 0.8042 | **+2.08** | 0.0238 | 0.0474 | 3 | 0 | 2 |

| statistic | value |
|---|--:|
| paired deltas (pp) | +2.92, +0.00, +3.96, +2.08 |
| **mean** | **+2.24 pp** |
| median | +2.50 pp |
| sample SD | 1.68 pp |
| min / max | +0.00 / +3.96 pp |
| sign consistency | **3 positive, 1 zero, 0 negative** |
| DESCRIPTIVE paired 95% interval (n=4, t=3.182) | −0.43 to +4.91 pp |

The interval is **descriptive only**; n=4 is far too small for a hypothesis
test, and no significance cutoff is used anywhere in this report.

**Leave-one-seed-out mean paired delta** — the aggregate is **not** carried by
the original seed:

| dropped | mean |
|---|--:|
| original (42) | **+2.01 pp** |
| rep1 (43) | +2.99 pp |
| rep2 (44) | +1.67 pp |
| rep3 (45) | +2.29 pp |

## 9. Per-evaluation-set summary (mean B−A across seeds)

| set | mean B−A | range | seeds B>A |
|---|--:|---|--:|
| E0 | +2.60 pp | −2.08 … +6.25 | 2/4 |
| **E1s** | **+4.69 pp** | +1.04 … +7.29 | **4/4** |
| E2s | −0.52 pp | −6.25 … +4.17 | 1/4 |
| E3s | +1.30 pp | −5.21 … +6.25 | 3/4 |
| **E4s** | **+3.12 pp** | +1.04 … +7.29 | **4/4** |

The benefit is **not** confined to one set: E1s and E4s favour B in all four
seeds, E0 and E3s lean positive, and only E2s is slightly negative on average.

## 10. Offline action-metric decomposition (EMA)

Reported as directed: executed t=0 primary, nonterminal secondary, terminal
placeholder separate. The whole-horizon aggregate is **not** used as an A/B
headline.

| seed | arm | **t=0 (executed)** | nonterminal (t=0..3) | **t=H−1 (placeholder)** |
|---|---|--:|--:|--:|
| 42 | A | 0.01634 | 0.03470 | 0.00062 |
| 42 | B | 0.01640 | 0.03477 | 0.72214 |
| 43 | A | 0.01628 | 0.03449 | 0.00061 |
| 43 | B | 0.01637 | 0.03459 | 0.72214 |
| 44 | A | 0.01626 | 0.03435 | 0.00058 |
| 44 | B | 0.01643 | 0.03441 | 0.72214 |
| 45 | A | 0.01600 | 0.03417 | 0.00065 |
| 45 | B | 0.01616 | 0.03424 | 0.72214 |

Replicates cleanly across all four seeds: A learns to emit the zero placeholder
(≈0.0006), B does not (0.72214). **Executed action quality is essentially
unchanged** — and in every seed B's t=0 error is *marginally larger* than A's.
The control gain is therefore **not** explained by better executed-action
reconstruction, which is a genuinely interesting loose end (§14).

## 11. State prediction and imagination — no tradeoff

| seed | arm | EMA state_loss | EMA replay state | EMA action_grad |
|---|---|--:|--:|--:|
| 42 | A / B | 0.224888 / 0.224889 | 0.20533 / 0.20533 | 0.074828 / 0.072921 |
| 43 | A / B | 0.224879 / 0.224877 | 0.20533 / 0.20532 | 0.074963 / 0.073070 |
| 44 | A / B | 0.224874 / 0.224880 | 0.20535 / 0.20541 | 0.074113 / 0.073652 |
| 45 | A / B | 0.224863 / 0.224864 | 0.20529 / 0.20528 | 0.074417 / 0.073236 |

Flat to 5 decimal places. Imagination panels at NFE10/15 show clean discrete
cubes for every replicate. No NaN/Inf; action gradients alive throughout.

## 12. LIVE vs EMA

LIVE and EMA agree in ordering and magnitude in every arm (e.g. seed 43 A:
t=0 LIVE 0.01641 / EMA 0.01628; B: 0.01650 / 0.01637). No arm tells a different
story under LIVE than under EMA.

## 13. Simulator-variability context — DESCRIPTIVE

Four repeated evaluations of one **byte-identical** policy (from the invalid
first attempt) gave a macro range of **1.87 pp**, empirical SD **0.80 pp**, and
per-set ranges up to **7.29 pp**.

The observed paired deltas (mean +2.24, SD 1.68 pp) are of **comparable
magnitude** to that evaluator variability. With four seeds, between-seed training
variability and physics/evaluator variability cannot be cleanly decomposed. What
distinguishes the result from noise is not magnitude but **sign structure**: an
evaluator-noise-only explanation would produce negative deltas roughly as often
as positive, and **none of the four seeds favoured A**.

This is context, **not** a threshold. No "must exceed N pp" rule is applied.

## 14. Red-team findings

| check | result |
|---|---|
| objective manifests correct in every run? | **yes** — PASS at step 0 and after 10 steps, all 7 new arms |
| all finals true step 2000? | **yes**, `step_matches=true` |
| hidden resumes? | **no** — fresh Adam, `trainer.step=0`, one launch path, atomic lock |
| A/B paired within each replicate? | **yes** — 19/19 preflight checks per pair |
| contention? | **none** — 0 MiB foreign mid-set in all 40 sets |
| any seed where B harms control? | **no** — 3 positive, 1 exactly zero, 0 negative |
| driven by one E set? | **no** — E1s and E4s positive in 4/4; only E2s slightly negative |
| driven by the original seed? | **no** — leave-one-out mean stays +1.67…+2.99 pp |
| executed t=0 action quality changed? | **no** — unchanged, in fact marginally worse for B |
| nonterminal action quality changed? | **no** — flat |
| state prediction degraded? | **no** — flat to 5 dp |
| imagination degraded? | **no** — clean at NFE10/15 in every replicate |
| clipping changed? | no clipping pathology; gradients finite throughout |
| LIVE vs EMA disagree? | **no** |
| magnitude vs repeated evaluator variation? | **comparable** — the sign structure, not the size, carries the result |

**The honest weak points.** (i) n=4 with an effect comparable in size to
evaluator noise; the descriptive interval includes zero (−0.43 to +4.91 pp).
(ii) One seed (43) showed exactly **+0.00** — a genuine null, not a small
positive. (iii) The mechanism is unexplained: executed-action accuracy does not
improve, so *why* control improves is unknown. (iv) One pretrained policy, one
task, one 2,000-step window.

## 15. Compute ledger

| category | GPU-h |
|---|--:|
| screening valid | 0.619 |
| **screening INVALID / wasted** | **1.005** |
| corrected rerun valid | 1.133 |
| replication — 6 arms (train + diagnostics + control) | 2.134 |
| replication — post-hoc probes *(incl. one mis-targeted pass, repeated)* | ≤0.043 |
| **replication total** | **≤2.177** *(cap 2.45, headroom 0.273)* |
| **GRAND TOTAL (all stages)** | **≤4.934** |

Within the 2.45 cap. The probe line is a conservative upper bound: it charges the
entire 155 s window between program end and the last probe output — which also
contains CPU-only analysis — to GPU.

**Two honest notes.** (i) A first probe pass was mis-targeted: the script
hardcoded the seed-42 run directory, so all three "replication" probes silently
re-measured the original seed and returned byte-identical numbers. Caught because
identical values across three different runs are impossible. Fixed with a
`--seed` argument and re-run; the wasted pass is counted, not hidden. (ii) I
initially reported this study as **over cap at 2.734 GPU-h** using a guessed
6-minutes-per-probe figure. The probes actually take ~11 s each. The corrected
figure is 2.177 and within cap; the erroneous overage report is recorded here
rather than quietly dropped.

## 16. Final classification: **TERMINAL-MASK-REPLICATED**

Against the predeclared criteria:

| criterion | status |
|---|---|
| mean paired B−A remains positive | **yes**, +2.24 pp |
| direction positive in ≥3 of 4 seeds | **yes**, 3 positive, 1 zero, **0 negative** |
| not driven solely by the original seed | **yes**, leave-one-out +1.67…+2.99 pp |
| no major state / imagination degradation | **yes**, both flat |

All four are met. "REPLICATED" here means *the effect recurred under paired
stochastic replication*; it does **not** mean established, mechanistically
understood, or general. n=4, one pretrained policy, one task, one window, and an
effect size comparable to evaluator noise.

## 17. Tal's question, updated

> **Does removing terminal zero-action supervision reproducibly improve an
> already competent EC-Diffuser policy across stochastic fine-tuning runs?**

**Yes — reproducibly, modestly, and for reasons we do not yet understand.**

Across four paired fine-tuning runs from the same competent policy, masking the
terminal action target improved closed-loop success by **+2.24 pp on average**.
It helped in three seeds (+2.92, +3.96, +2.08 pp), did exactly nothing in the
fourth (+0.00), and **hurt in none**. Two of the five scenario sets improved in
every single seed. Dropping any one seed leaves the average positive.

The honest caveats: the effect is about the same size as the run-to-run noise of
the simulator itself, so a four-seed descriptive interval still includes zero.
What makes it more than noise is the **sign pattern** — pure evaluator noise
would send roughly half the seeds negative, and none went negative.

The most interesting part is what did **not** change. B's executed action — the
only action the robot actually takes — is no more accurate than A's; if anything
it is a hair worse. State prediction and imagination are unchanged. So the gain
is **not** "the policy predicts actions better." Something about removing a
supervised, never-executed zero target changes the policy's closed-loop behaviour
in a way that offline action accuracy does not capture. That mechanism is the
open question.

Strictly, the tested intervention is the **registered terminal-mask
intervention**: removing terminal supervision *together with* its inherited
0.207% active-coordinate renormalization. A 0.207% change in loss normalization
seems an unlikely driver of multi-point behavioural gains, but this experiment
did not separate the two, so that remains an assumption rather than a finding.
And the terminal zero target is still **not** called a bug — it is a learned
placeholder behaviour on a non-executed slot, and removing it is an ablation
whose benefit is now replicated but unexplained.

## 18. Next scientific action — exactly ONE, not launched

**Separate the two components of the registered intervention**: run a paired
third arm, B′, that masks the terminal action *and* restores the original
active-coordinate denominator (i.e. normalizes by the unmasked count, 46560),
across the same four fine-tuning seeds.

Rationale: this is the single cheapest experiment that converts the main causal
caveat into a measurement. If B′ ≈ B, the 0.207% renormalization is irrelevant
and the effect is attributable to removing the placeholder supervision itself —
which is the claim we actually want to make, and currently cannot. If B′ ≈ A,
the story inverts entirely and a 0.207% normalization change is doing the work,
which would be a far more surprising and important finding. Either outcome is
informative, and it directly addresses the strongest objection a reviewer will
raise. Cost ≈ 4 × 0.333 ≈ **1.33 GPU-h** (four seeds, one new arm each, reusing
all existing A and B runs).

**Not launched.** Requires a new allocation and explicit approval.
