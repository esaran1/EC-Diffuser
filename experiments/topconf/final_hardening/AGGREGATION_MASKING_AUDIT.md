# Aggregation Masking Audit

CPU only. E1 untouched. All numbers from frozen calibrated artifacts.

**Question.** Can an evaluator appear highly stable at an aggregate/leaderboard
level while remaining unable to resolve the policy differences researchers use it
to claim?

**Answer from our data: yes, and by more than an order of magnitude.**

---

## 1. Aggregation hierarchy (§3)

Legitimacy rule: we aggregate only units sharing an estimand. 3-cube and 4-cube
are **different benchmarks** and are not pooled, except at Level 3b which is
labelled illegitimate and shown only as a diagnostic.

| Level | Unit | run-to-run SD | range | n_scen | n higher-level units |
|---|---|--:|--:|--:|--:|
| 0 | individual scenario outcome | **17.95 pp** | — | 96 | 8 realizations |
| 1 | one 96-scenario arm score | **2.88 pp** (0.60–5.35) | 1.04–10.42 pp | 96 | 3 realizations |
| 2 | arm averaged over 3 checkpoints | **1.61 pp** (0.20–2.23) | 0.35–4.17 pp | 96 | 3 ckpt × 3 runs |
| 3 | task-regime aggregate | **0.42 pp** (3c 0.64, 4c 0.20) | 0.35–1.27 pp | 96 | 9 / 6 arms |
| 3b | grand mean across regimes *(estimands differ — diagnostic only)* | 0.45 pp | 0.90 pp | 96 | 15 arms |

Level 1 SE per arm: 0.35–3.09 pp.

**Contraction: 17.95 → 0.42 pp, a factor of 43, from the same realizations.**

## 2. Same raw data, two questions (§4)

| | quantity | value |
|---|---|--:|
| **A. aggregate stability** | task-regime score run-to-run SD | **0.20–0.64 pp** |
| **B. local resolution** | per-contrast R=1 spread, all 12 contrasts | **median 10.42 pp** (5.21–14.58) |
| | per-contrast R=1 estimate SD (like-for-like) | median 3.57 pp |
| | contrasts whose R=3 CI excludes zero | **0 / 12** |
| | contrasts admitting both directions at R=1 | **9 / 12** |

No contrast selection: all 12 calibrated comparisons are used.

## 3. Masking factor (§10)

Reported as a ratio **and** as raw quantities, per instruction.

- Like-for-like (SD ÷ SD): **5.6× (3-cube), 17.8× (4-cube)**
- Spread ÷ aggregate SD: 16.3× (3-cube), 52.0× (4-cube) — dimensionally valid
  (both pp) but numerator is a 9-view range and denominator a 3-run SD, so the
  like-for-like ratio is the defensible one.

We do **not** name this as a new metric. It is a ratio of two measured spreads.

## 4. Counterfactual leaderboard (§5)

Entries are our own policy arms (checkpoint × NFE). **These are not different
published algorithms and are not presented as such.** Evaluation diagnostic only.

| regime | entries | aggregate score SD | entries changing rank | max rank movement | pairwise order changes | adjacent-pair order changes |
|---|--:|--:|--:|--:|--:|--:|
| 3-cube | 9 | **0.64 pp** | **9/9** | 3 places | 10/36 | **5/8** |
| 4-cube | 6 | **0.20 pp** | **6/6** | 4 places | 8/15 | **4/5** |

The 4-cube leaderboard has a **different top entry in all three realizations**
while its aggregate score moves 0.20 pp.

## 5. Resolution-aware partial order (§6)

Not novel statistics — a presentation of the pre-calibrated resolution. An edge
A > B is drawn only when the observed gap exceeds the calibrated half-width.

| regime | pairwise distinctions | supported at R=3 | disappear at R=3 | supported at R=1 | disappear at R=1 |
|---|--:|--:|--:|--:|--:|
| 3-cube (res. 5.15 / 8.93 pp) | 36 | 14 (39%) | **22 (61%)** | 3 | **33 (92%)** |
| 4-cube (res. 6.84 / 11.85 pp) | 15 | 4 (27%) | **11 (73%)** | 0 | **15 (100%)** |

A conventional total ordering asserts all 36 and all 15 distinctions.

## 6. Why both facts are true simultaneously (§9)

Not a contradiction, and not merely "averaging reduces variance."

Aggregate score over N scenarios, M arms: `Var ≈ w/(N·M·R)` — it contracts in
**both** N and M. A treatment-effect contrast between two arms cannot borrow the
M-fold averaging: the arms are what is being compared, so
`Var[Δ̂] = 2w/(N·R)`. Averaging over arms **suppresses the reported instability
while leaving the comparison's uncertainty untouched** — and the factor 2 makes
the contrast strictly noisier than either arm's score.

Terminology: **global stability** (does the benchmark number move between reruns?)
versus **local resolution** (can the benchmark separate two close entries?). These
are different estimands. Global stability is necessary but **not sufficient** for
local resolution, and the gap between them grows with the number of units averaged.

## 7. RoboDojo primary-source audit (§7, §8, §15)

Source: arXiv:2607.04434, §6.4.1 "Simulation Evaluation Stability", Table 7
(read via ar5iv; the arxiv HTML renderer truncates that subsection).

| item | finding |
|---|---|
| "stability" defined? | Never formally. Operative sense: small SD of success/score on repeated eval across GPU devices and seeds |
| GPUs | **three RTX 4090s** — one GPU *model* |
| runs/seeds | 3 seeds × 3 GPUs = **9 runs**, on **3 policies** |
| SD granularity | **per-capability-dimension** and **overall aggregate**. **Not per-task** |
| same scenarios repeated? | Yes — *"we use layout 0 of each task"*, a fixed standardized layout |
| policy randomness repeated? | **NOT STATED** |
| simulator realization isolated? | **NOT STATED** — simulator nondeterminism, policy stochasticity and seed effects are confounded in one SD; no variance decomposition |
| max task-level variation | **NOT STATED** — finest grain is per-dimension |
| headline number | **1.1 pp** = largest cross-run SD of success rate, **per capability dimension**, maximized over policies × dimensions. Overall-average SD ≤ **0.5 pp** |
| learned policies ranked? | Yes — 30 policies, 42 sim + 18 real tasks, public leaderboard |

**Does RoboDojo establish (A) aggregate leaderboard reproducibility, (B) fine-grained
policy-comparison resolution, or both?**

**A only.** The paper reports descriptive means and SDs; it contains no
significance test, CI on a policy gap, paired comparison, or minimum-detectable-
difference analysis. §6.4.1 establishes that a single policy's aggregate score is
stable across reruns, then infers it is *"suitable for fair leaderboard
comparison"* — a statement about **differences between policies**. That step is
not tested.

**Stated fairly:** their benchmark may genuinely be extremely stable, and nothing
here suggests otherwise. We do **not** claim any RoboDojo ranking is unreliable,
and we do not claim their 1.1 pp is wrong. The point is conceptual: **aggregate
reproducibility and pairwise effect resolution answer different questions**, and
only the first was measured. Two structural limits on how far 1.1 pp can be
carried, both stated in the paper's own terms: it is measured on one GPU model,
and the SD itself rests on ~9 runs across 3 policies, so "largest observed SD" is
a low-confidence bound.

**§8 — safe question.** Because the per-task and per-run values are not published,
we **cannot** responsibly compare adjacent leaderboard gaps against their
reported variation: the required sample structure is absent, and converting their
SD into a CI would be exactly the error we are warning against. We therefore
**decline the §8 comparison** and state only the motivation: aggregate
reproducibility does not by itself certify that adjacent entries are separable.

## 8. External raw-artifact availability (§18)

**RoboDojo: no.** No `results/`, `data/`, or `logs/` directory in the repo; no
companion results repo in the org; docs expose no downloadable JSON/CSV; the
Table 7 per-run numbers are not published machine-readably. The pipeline
aggregates to a leaderboard table locally and requires Isaac Sim + GPU to
regenerate. **A CPU-only third-party reanalysis of their stability data is not
possible from published artifacts.** We did not scrape or infer unpublished data.

GPUSimBench: see §9 below.

## 9. GPUSimBench overlap check (§16) — NOVELTY SURVIVES

Source: arXiv:2607.13059 v1 (2026-07-06), "GPUSimBench: Towards Scalable and
Reliable GPU-Accelerated Simulators in Embodied AI". Seven simulators (IsaacLab,
ManiSkill, Genesis, Madrona, MuJoCo Warp, MJX, MuJoCo Playground), single RTX 5070.

Note: no public repo exists (`GPUSimBench` GitHub search returns 0). The paper
promises release "on the project website" but names no URL. **No raw artifacts.**

What the experiments actually measure: (1) throughput/memory on free-falling cubes
and a Franka Panda **driven by random actions**, rendering disabled; (2) sim-to-real
physical consistency via Earth Mover's Distance; (3) determinism via two EMD-based
variability metrics. **No learning, no policies, no success rates.**

| claim | verdict |
|---|---|
| 1. learned-policy rankings | **NO** — zero trained policies; it ranks *simulators*, not policies |
| 2. aggregation masking | **PARTIAL, opposite direction** — aggregation is their *method*, not a hazard studied. Nearest sentence concerns distributional dispersion across simulator internals, not averaging hiding comparison resolution |
| 3. global stability vs local resolution | **NO** — statistical apparatus is mean ± SD over 10 runs; no CI, significance test, effect size, or uncertainty on any *difference* |
| 4. fixed-scenario treatment-effect uncertainty | **PARTIAL on the premise, NO on the claim** — they do run R=10 repeats of identical scenarios ("we fix random seeds and disable all task-level randomization"), but with no policy there is no policy-vs-policy difference and no error bar on a contrast |

**Verdict: does not own any of the four claims.** It is a *supporting citation* for
our premise — independent, multi-simulator evidence that repeated identical-seed
GPU runs are not bit-reproducible — rather than a competitor. Caveats: v1,
unreviewed, single GPU, 10 runs, and several "0.00 ± 0.00" entries are explicitly
precision-limited rather than proven-zero.

## 10. Strongest defensible claim

> Aggregate benchmark reproducibility and pairwise policy-comparison resolution
> are different estimands, and the first does not imply the second. On our frozen
> data the same physics realizations that hold a task-regime aggregate score to
> **0.20–0.64 pp** run-to-run leave **9 of 12** calibrated policy contrasts
> admitting both directions, change the rank of **every** leaderboard entry, and
> leave **0 of 12** contrast CIs excluding zero. Evaluations used to rank policies
> should therefore report comparison resolution alongside aggregate stability.

Deliberately **not** claimed: that any external benchmark is unreliable; that any
published ranking is wrong; that aggregation masking has been demonstrated in any
system other than ours.

## 11. Verdict against the §14 bar

The §14 STRONG criterion was: *our data would satisfy a leaderboard-style
aggregate stability criterion while the majority of close constituent comparisons
remain directionally unstable.*

Both halves hold, with one comparison deliberately weakened.

Our task-regime aggregate SD is **0.20–0.64 pp** — the same *order* as the
sub-percentage-point figures that are reported to certify benchmark stability.
We do **not** claim ours is lower than RoboDojo's, and the numbers must not be
ranked against each other: our SD rests on **3** realizations versus their 9, and
an SD estimated from n=3 falls in [0.08, 0.96] pp 95% of the time when the truth
is 0.5 pp. The two are also structurally different — ours varies *only* the
physics realization (policy noise paired via CRN) on one device, while theirs
confounds simulator nondeterminism, policy stochasticity and seed across three
devices, on different tasks, simulators and policies.

The defensible statement is therefore about **magnitude class, not ordering**: an
evaluator can post a sub-percentage-point aggregate SD — the kind of number used
to justify leaderboard comparison — while 9/12 of its constituent contrasts admit
both directions and 61–73% of its leaderboard distinctions are unsupported at R=3.

**STRONG. KEEP.**
