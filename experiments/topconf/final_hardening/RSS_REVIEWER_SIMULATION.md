# RSS reviewer simulation (§26, §30, §31)

## Score if submitted TODAY (current evidence only)

**Expected distribution: 4 / 5 / 6 — modal 5 (borderline), mean ≈ 5.0.**

### Why a reviewer gives 6 (weak accept)
- ICC = 0.285: **71.5% of outcome variance lies below the nominal trial unit**, a
  precise statistical statement that directly contradicts the i.i.d.-trial
  assumption in STEP (RSS 2025) and N-SCORE (RSS 2026).
- The **reliability curve** (61% → 78% → 96% → 100% sign correctness vs
  effect/noise ratio) is a usable empirical rule, not an anecdote.
- **25% practical-conclusion flips** — changes what you would deploy.
- The **retroactive audit**: 9 conclusions, 5 revised, 1 survived — self-correction
  is unusually credible evidence.
- Direct tension with **RoboDojo's** published "suitable for fair leaderboard
  comparison" claim.
- Two task regimes + a deterministic contrast (PushT) + an external policy.

### Why a reviewer gives 4 (weak reject)
- **No new method.** STEP, N-SCORE and Active Selection all contribute
  algorithms; we contribute a measurement.
- **"Just add repetitions."** The prescription is standard practice.
- **Legacy simulator.** Isaac Gym Preview 4 is deprecated; GPUSimBench already
  covers modern stacks.
- **Effects are small.** The comparisons that flip are mostly ≤3 pp, so
  "of course they flip."
- **One task family, one policy family** in the calibrated results.

## The single result that most moves the score

**E1/E2 — slot permutation.** It converts the contribution from *"noisy
experiments need repetition"* (engineering hygiene) to *"an evaluation
configuration choice that nobody reports is a hidden experimental variable"*
(a finding). Nothing else on the table does this: E3-E6 deepen or replicate.

Expected post-E1/E2 distribution, if positive: **5 / 6 / 7, modal 6.**
If E1 is null: **4 / 5 / 6, modal 5** — unchanged, but with a closed objection
and a cleaner causal story ("not packing; execution order").

## §31 — clearing RSS without a new statistical method

STEP, N-SCORE and Active Selection each contribute a *procedure*. We contribute
an *empirical failure mode of the assumption those procedures rest on*. That is
publishable at RSS only if the phenomenon is large, general and consequential:

- **large** — 71.5% of variance below the trial; 10.4 pp same-arm spread
- **general** — two task regimes, three checkpoints, three policy contrasts, plus
  a deterministic contrast case showing when it does *not* apply
- **consequential** — 24% sign / 25% practical-conclusion flips, and five of our
  own conclusions revised

That is a defensible but not overwhelming case — hence modal 5 today. E1 is what
plausibly pushes it to modal 6.

## Ten attacks and whether we can answer

| # | attack | answer | strength |
|--:|---|---|---|
| 1 | "GPUSimBench already showed this" | They run **zero policies**, measure cm not decisions, and **never permute slots** | **strong** |
| 2 | "RoboDojo shows ≤1.1 pp — you're wrong" | They vary **GPU device**, average over 3 seeds, and use **one layout per task**; structurally blind to within-scenario bifurcation | **strong** |
| 3 | "Just use more rollouts" | The contribution is the *magnitude*, the *estimand*, and the reliability curve — not the prescription | adequate |
| 4 | "Hierarchical bootstrap is standard" | Conceded explicitly; we claim no statistical novelty | adequate |
| 5 | "Of course tiny effects flip sign" | **The reliability curve** quantifies exactly when: reliable above ~1.0σ, coin-flip below ~0.5σ | **strong** |
| 6 | "Isaac Gym is legacy" | GPUSimBench independently documents the same regimes in Isaac Lab/ManiSkill; we cite rather than replicate | adequate |
| 7 | "Only NFE variants of one policy" | Reliability curve generalizes over effect size; PushT adds an independent policy/simulator | **partial — E5 would close it** |
| 8 | "No new algorithm" | True. RSS accepts measurement papers if the phenomenon is consequential | **the core risk** |
| 9 | "One environment family" | Two task regimes + PushT contrast + external policy | adequate |
| 10 | "Engineering, not research" | Answer: a public benchmark asserts leaderboard validity on a blind design, and 25% of practical conclusions flip | adequate-to-strong |

Attack 8 is the one that cannot be fully answered without either a method or a
new phenomenon. **E1/E2 is the new phenomenon.**
