# Claim ledger (updated after final hardening, 2026-08-31)

Evidence levels: L0 code/math · L1 offline in-distribution · L2 single-checkpoint
closed-loop · L3 three-checkpoint replication · L4 cross-task/benchmark · L5 real.

## SUPPORTED — evaluation (the paper's core)

| claim | level | evidence | safe wording |
|---|---|---|---|
| Isaac Gym GPU physics is nondeterministic under identical state + actions + no policy | L0/L1 | divergence 5.9e-4 at t=0 → 0.92 at t=100 | *"in our tested configuration"* — do **not** claim discovery; cite GPUSimBench |
| **71.5% of outcome variance lies below the nominal trial unit** | L3 | ICC = 0.285, R=8 same arm | **new headline** |
| identical repeats of one policy span 10.4 pp | L2 | 8 repeats, SD 4.23 pp | — |
| ~24% of reconstructed single-realization views disagree in **sign** | L3/L4 | 108 nested views, 12 comparisons | must state the hierarchy; **not** N=108 independent |
| **~25% disagree in practical conclusion** (±5 pp categories) | L3/L4 | 27/108 | more consequential than sign |
| **sign reliability is monotone in effect/noise ratio** (61→78→96→100%) | L3/L4 | 12 comparisons, both regimes | **answers "of course tiny effects flip"** |
| resolution 7.55/4.36/3.38/2.67 pp at R=1/3/5/8 | L2 | paired, hierarchical | — |
| failures bifurcate, not threshold jitter | L2/L4 | 0.0184 vs 0.3036 (16.5×); 4-cube 0.0214 vs 0.1041 | — |
| **contact *timing* variability distinguishes sensitive scenarios** | L2 | within-episode SD 0.081 vs 0.030 (2.7×) | **new** |
| instability persists at 4 cubes | L4 | sensitive 41%→53%, sign flips 25%→22% | persistence, **not** a compositional law |
| PushT/pymunk deterministic → no physics replication needed | L1 | fixed-action repeat | *"in our tested configuration"* |

## SUPPORTED — policy (now secondary)

| claim | level | wording |
|---|---|---|
| NFE4 > NFE1, 3-cube | L3 | directional across 3 checkpoints, mean +6.0 pp |
| NFE2 ≈ NFE4, 3-cube | L3 | *no reliable difference*; ±5 pp equivalence **not** established |
| NFE2−NFE1 ≈ +4.3 pp | L3 | cross-campaign contrast on shared scenarios/CRN, **not** a directly paired physics experiment |
| 4-cube: NFE4 > NFE2 by +4.5 pp | L3 | operating point does **not** transfer; **ceiling confound must be stated** |
| PushT: NFE100 > NFE2 | L2 | −4.6 pp success [−10.6,+1.6], reward −0.024 [−0.044,−0.005]; 50× calls, 37× latency |
| offline fidelity ≠ deployment tradeoff | L1/L2 | cross-policy | **not** "offline metrics mis-rank" |
| state/action semantic NFE asymmetry | L1 | 3 checkpoints | **case study**, not a universal principle |

## FALSIFIED / WITHDRAWN

- episode-only bootstrap CIs **under-cover** — **NO**: 96.8% at R=1 (conservative). The failure is **resolution**, not coverage, and not bias (R=1 is unbiased to 2 dp).
- "offline metrics mis-rank inference budgets" (universal form)
- "control saturates at NFE2 / peaks at NFE8 / declines at NFE16" as mechanism
- "Regime A strong replication" / the R=1 ~3 pp effect
- compression-induced covariate feedback as dominant mechanism (28% NFE-specific)
- dimensional loss reweighting — NOT MOTIVATED
- semantic/projected shortcut method — NO-GO
- PushT Stage-1 successes computed with `>0.95` (correct: `>= 1.0`)

## PROHIBITED WORDINGS

new generative model / loss / solver / adaptive-NFE / new bootstrap statistic ·
"two steps are always enough" · "offline metrics are useless" · "we discovered GPU
nondeterminism" · "we corrected three *published* results" (they were internal) ·
"N=108 independent comparisons" · "compositionality causes evaluator noise" ·
"NFE2 is equivalent to NFE4" · "hierarchical bootstrap is our method"
