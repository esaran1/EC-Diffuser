# Submission gap analysis (§33-§35)

## Paper scope ranking

| | scope | verdict |
|---|---|---|
| **C** | **D2 + hidden parallel-environment packing variable** | **best, IF E1 is positive** |
| A | Pure D2: "Hidden Physics Variability in GPU Robot Policy Evaluation" | **best if E1 is null** — solid, modal 5 |
| B | D2 + resolution framework/tool | A + a released audit toolkit; the toolkit is reproducibility value, not novelty |
| D | D2 + cross-policy-family (Gaussian vs Flow) | adds one datum the reliability curve already predicts |
| E | insufficient for top conference | **rejected** — the ICC/reliability-curve results clear a workshop bar comfortably and are borderline-plausible for RSS |

## §35 — three abstracts, compared

**(A) Current evidence only.**
> Contact-rich GPU simulation makes a fixed task scenario an unreliable
> evaluation trial. Measuring a joint state-action flow policy and an independent
> diffusion policy, we find 71.5% of outcome variance lies *below* the nominal
> trial (ICC 0.285); identical repeats of one policy on frozen scenarios span
> 10.4 pp; and reconstructed single-realization views disagree with calibrated
> estimates on 24% of signs and 25% of practical conclusions. Sign reliability is
> a monotone function of effect/noise ratio, collapsing below ~0.5σ. We calibrate
> resolution against replication and give a simulator-conditional protocol.

**(B) With a positive slot result.**
> …*and we show that the arbitrary assignment of scenarios to parallel GPU
> environment slots — a configuration choice no paper reports — can itself change
> the measured policy difference.*

**(C) With a null slot result.**
> …*and we show the instability is not an artifact of scenario-to-slot packing,
> localizing it to execution-order nondeterminism.*

**Assessment.** (B) is materially stronger — it adds a *new hidden variable*,
which is a finding rather than a measurement, and it is the one thing that
answers reviewer attack #8 ("no new algorithm"). (C) is a genuine but modest
improvement on (A): it closes a question a reviewer would otherwise raise.

Since (B) and (C) are both useful and the cost is **0.111 GPU-h**, E1 is worth
running. That is the whole argument.

## Figure plan (§34), red-teamed

| # | figure | claim | keep? |
|--:|---|---|---|
| F1 | contact bifurcation: identical actions, divergence 5.9e-4 → 0.92, with the t=7→8 discontinuity and contact-timing SD 0.081 vs 0.030 | physical mechanism | **yes** |
| F2 | variance hierarchy: between-scenario vs within-scenario (28.5/71.5), ICC, design effect | the estimand result | **yes — new headline** |
| F3 | reconstructed R=1 Δ vs calibrated Δ, per comparison, with sign/category flips | consequence | **yes — headline** |
| F4 | reliability curve: P(correct sign) vs \|Δ\|/σ | the general rule | **yes — answers attack #5** |
| F5 | 3-cube + 4-cube persistence | generality | yes, compact |
| F6 | slot-permutation result | hidden variable | **only if E1 positive** |
| — | PushT determinism contrast | conditional protocol | **appendix** |
| — | latency/utility Pareto (D1) | efficiency | **appendix** — D1 is no longer the thesis |

Six main figures, one subject.

## §32 — toolkit

A small evaluator-audit toolkit (same-arm repeatability probe → variance
decomposition → resolution estimate → nested contrast → sign-stability →
provenance report) is **worth releasing** for reproducibility and adoption, but
must **not** be presented as methodological novelty. It is standard statistics
packaged for a specific simulator failure mode.

## §25 — legacy-simulator risk

**Option A: literature support is sufficient.** GPUSimBench independently
documents the four stochasticity regimes across Isaac Lab, ManiSkill, Genesis,
MJX, Madrona and MuJoCo Playground. We cite it for cross-simulator generality and
confine our claim to the stack we measured. Porting EC-Diffuser (>4 GPU-h, high
risk) is not justified.
