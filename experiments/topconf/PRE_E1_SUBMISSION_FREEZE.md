# Pre-E1 Submission Freeze

**Status: FROZEN.** Written before E1 is observed, so that E1's outcome cannot
retro-fit the paper's claims. No new statistics; every number is drawn from
already-frozen artifacts.

E1 is staged, unrun, and untouched. Zero GPU was used to write this.

---

## 1. Primary paper claim (FROZEN)

> Aggregate benchmark reproducibility and pairwise policy-comparison resolution
> are distinct evaluation properties. In our contact-rich GPU manipulation
> benchmark, aggregation produces sub-percentage-point task-regime variability
> while close policy contrasts from the same evaluation data remain unresolved
> under a single simulator realization.

This is a statement about **our** benchmark and about two estimands. It is not a
criticism of RoboDojo or any external benchmark, and must never be written as one.

## 2. Core empirical numbers (FROZEN)

**Abstract / introduction carries exactly four numbers.** Everything else is
main-text or supplementary. Headlining every available percentage weakens all of
them.

| # | Number | Role |
|---|---|---|
| 1 | **0.64 pp / 0.20 pp** task-regime aggregate SD (3-cube / 4-cube) | establishes global stability |
| 2 | **9 / 12** contrasts admit both directions at R=1 | establishes local failure |
| 3 | **10.42 pp** median within-contrast R=1 range | gives the magnitude |
| 4 | **~8 pp → ~4.5 pp** predicted 95% half-width, R=1 → R=3 | gives the remedy |

Pairing 1 with 2 is the paper. They come from the same realizations, and that is
the whole point.

**Main text, not abstract:** 12 unique calibrated comparisons; max range 14.58 pp;
8/8 low-ratio contrasts with an opposite-direction view vs 1/4 above ratio 1;
0/12 CIs excluding zero; 61%/73% of leaderboard distinctions unsupported at R=3.

**Demoted to supplementary:** the 14.8% per-view sign-reversal rate. It is
descriptively correct but its unit of analysis is the nested view, and the
contrast-level statistic (9/12) is strictly more defensible.

## 3. Primary method claim (FROZEN)

> A same-policy repeatability pilot can estimate within-scenario simulator
> uncertainty before a policy comparison and conservatively predict the effect
> scale for which directional conclusions become reliable.

**SUPPORTED** — held-out checkpoints (σ from checkpoints excluding the predicted
comparison); observed 3→4 cube transfer (ratio 1.01 in that direction);
sign/direction reliability, monotone 66.7% → 97.2% in the held-out ratio.

**NOT SUPPORTED, and must be stated as such** — practical-category certification
(flat 72–78%, a negative result we report); universal simulator transfer;
arbitrary tasks; adaptive allocation (dropped); exact effect equivalence.

The word is **conservative**, never *accurate*: the predictor over-estimated
observed spread on all 12 comparisons (mean +47.7%), and 4→3 transfer is 1.91.

## 4. Aggregate-stability claim (FROZEN)

> Aggregate reproducibility does not imply fine-grained comparison resolution.

Correct implication:

> Aggregate score stability alone is insufficient to quantify the smallest
> pairwise policy effect the evaluator can support.

**Forbidden phrasings:** "aggregate benchmarks are unreliable"; "leaderboards with
low SD are invalid"; "RoboDojo's stability claim is insufficient." None is
supported, and each attacks a claim nobody made.

## 5. Leaderboard result status (FROZEN)

**Consequence and illustration — not a primary contribution.**

Allowed statements: every tested entry changes rank across R=1 realizations; the
4-cube leaderboard has a different top entry in each of the three realizations;
many adjacent orders change (5/8 and 4/5 adjacent pairs).

**Required acknowledgement, stated in the same breath:** near-tied total orderings
are inherently sensitive, and reordering among near-ties is expected rather than
surprising. Entries are our own arms (checkpoint × NFE), never presented as rival
published algorithms.

The scientific contribution is the **calibrated local resolution**. Ranking
volatility on its own would be a much weaker paper, and we do not lead with it.

## 6. RoboDojo positioning (FROZEN)

**Concession, stated plainly and early:**

> RoboDojo establishes strong aggregate reproducibility under its evaluation
> protocol.

Our work asks a different question: **what policy-effect magnitude can an
evaluator resolve?**

**Prohibited:** numerically comparing their SD against ours as if identically
constructed. They are not — theirs is 9 runs across 3 devices confounding
simulator, policy and seed variation; ours is 3 realizations with policy noise
paired via CRN on one device. An SD from n=3 spans [0.08, 0.96] pp when the truth
is 0.5. We claim **magnitude class, not ordering**.

Use as motivation for reporting **both** quantities. Nothing more.

## 7. GPUSimBench positioning (FROZEN)

GPUSimBench owns **GPU simulator nondeterminism / divergence**. We do **not**
claim discovery of that phenomenon, and cite them as independent multi-simulator
evidence for our premise.

Our contribution is strictly downstream: task outcomes; learned-policy
comparisons; the fixed-benchmark estimand; effect-resolution consequences;
predictive same-policy calibration; the aggregate-vs-local distinction. They run
random actions and no learned policies, and compute no uncertainty on any
difference.

## 8. STEP / N-score positioning (FROZEN — permanent correction)

Their assumptions are **not** violated by simulator nondeterminism. Their i.i.d.
assumption is on the **scenario draw**; realization noise is absorbed into the
per-scenario Bernoulli outcome.

They target **population-level** policy comparison under scenario sampling. Our
**fixed-benchmark** question contains an inner simulator-realization layer. These
are different estimands, and the methods are **complementary**.

Do not imply superiority. An earlier internal claim that nondeterminism "violates
i.i.d." was wrong and is permanently withdrawn.

## 9. E1 branch — predeclared consequence (FROZEN BEFORE OBSERVATION)

**If E1 is POSITIVE**, it adds exactly one result:

> Scenario-to-GPU-slot assignment, while holding policy, scenario, policy noise,
> num_envs, and evaluator settings fixed, can propagate to downstream
> learned-policy task outcomes.

A memorable hidden-evaluator-variable example. It **does not replace** the core
aggregate-vs-resolution thesis, and does not become the headline. E2 may then be
*proposed*, never automatically run.

**If E1 is NULL**, we state:

> Under fixed parallelism, static scenario-to-slot reassignment was not a
> material source of the observed evaluator variability.

No narrative damage — the thesis never depended on it. **No follow-up packing
search**, no variant hunting. Write the paper.

Both branches are committed before observation. Neither changes §1–§8.

## 10. Abstract skeleton — current evidence only (FROZEN)

Valid whether E1 is positive or null. E1 is deliberately absent.

1. Policy comparisons on fixed simulated benchmarks are reported as success-rate
   differences, with evaluator reliability certified by aggregate run-to-run
   stability.
2. On GPU physics, repeating an identical scenario with an identical policy and
   identical policy noise changes outcomes: results bifurcate (0.018 m vs 0.304 m
   against a 0.04 m threshold) rather than jitter, concentrated in
   contact-sensitive scenarios.
3. Aggregating over scenarios, checkpoints and arms drives task-regime
   variability to 0.20–0.64 pp while, from the *same* realizations, 9 of 12
   calibrated policy contrasts admit both directions under a single realization
   (median range 10.42 pp).
4. A same-policy repeatability pilot estimates this within-scenario uncertainty
   in advance and conservatively predicts the effect scale at which directional
   conclusions become reliable (held-out sign agreement 66.7% → 97.2%).
5. The same calibration does **not** certify effect *magnitude* (flat 72–78%);
   we therefore recommend reporting comparison resolution alongside aggregate
   reproducibility, with scope limited to one simulator and task family.

## 11. Top-conference verdict, judged NOW (pre-E1)

**RSS** — modal score **weak accept**. Distribution roughly: 10% strong accept,
35% accept, 35% weak accept, 20% reject. Strength: it is a measurement paper with
a falsifiable procedure, an honest negative result, and a corrected error trail.
Weakness: single simulator, single task family, no new policy capability.

**CoRL** — same modal score, marginally better distribution (CoRL is historically
more receptive to evaluation-methodology work). Roughly 12% strong accept, 38%
accept, 33% weak accept, 17% reject.

**Single strongest remaining reviewer objection:**

> "This is one simulator and one task family. You have shown that *your* Isaac Gym
> benchmark has poor comparison resolution; you have not shown that this is a
> general property of GPU-simulated policy evaluation, and the recommendation to
> report resolution is generic advice that follows from standard statistics."

This objection is largely correct, and the honest response is scope-limiting
rather than rebuttal. Our two external anchors (PushT breadth; GPUSimBench's
multi-simulator nondeterminism) mitigate but do not answer it.

**How much would a positive E1 address it?** **Partially, and not on the axis that
matters most.** E1 would show that a purely *administrative* evaluator variable —
slot assignment, with everything else fixed — propagates to task outcomes. That
converts "our simulator is noisy" into "hidden evaluator variables exist and are
not obviously enumerable," which is a qualitatively stronger and more memorable
claim, and would likely move the modal review from weak accept to accept.

But it is still one simulator. E1 does **not** deliver cross-simulator generality,
and we should not pretend it does. The generality objection can only be fully
answered by work outside this submission's scope.

## 12. Stop rule (BINDING)

After this document:

- **No more cache mining.**
- **No more new CPU analyses.**
- **No new statistics.**
- **No paper-thesis changes before E1.**

Wait for the GPU. **The next scientific information must come from E1.**
