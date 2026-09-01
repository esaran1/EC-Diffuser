# Final experiment ranking (§28-§29)

Standard: recommend a GPU experiment **only** if it plausibly (A) reveals a
hidden evaluator variable affecting policy conclusions, (B) demonstrates the
issue with a qualitatively different policy family, (C) shows survival in a
modern simulator, or (D) closes a likely-fatal RSS objection.

| # | experiment | hypothesis | novelty | closest prior | GPU-h | impl. risk | positive result | null result | changes acceptance? |
|--:|---|---|---|---|--:|---|---|---|:--:|
| **E1** | slot permutation, same arm | scenario→GPU-slot assignment changes outcomes beyond run-to-run noise | **unclaimed** | GPUSimBench observes index divergence, never permutes | **0.111** | medium (noise-row confound, testable at 0 GPU) | hidden evaluator variable | localizes noise to execution order; still strengthens paper | **YES** |
| **E2** | slot permutation × policy ranking | apparent Δ changes/reverses by packing alone | **unclaimed** | none | 0.212 (+E1) | same | **headline** | — | YES, conditional on E1 |
| E3 | `num_envs` sensitivity | parallelism changes success | unclaimed | GPUSimBench scales for throughput | ~0.5 | **high — multi-variable, uninterpretable** | ambiguous | weak | no |
| E4 | CPU-vs-GPU same task | GPU physics is the cause | partial | GPUSimBench covers CPU/GPU sims | ~1.0+ | **high — wrapper assumes CUDA; dynamics may differ** | strong causal | confounded by dynamics mismatch | marginal |
| E5 | Gaussian@100 vs Flow calibrated | instability is not specific to near-identical policies | modest | — | ~0.5 | low | closes "tiny effects only" | modest | **partially — see below** |
| E6 | modern-simulator validation | result survives Isaac Lab/ManiSkill | modest | GPUSimBench already covers these sims | >4 | very high (port EC-Diffuser) | strong | — | no — cite GPUSimBench instead |
| **E7** | **no more experiments** | — | — | — | **0** | — | — | — | — |

## Detail on the two that matter

### E1/E2 — the only candidate that adds a *new* phenomenon

Everything else deepens what we have. E1 asks a question no one has asked, is
cheap (0.111 GPU-h), and is informative in **both** directions. Its one risk —
the policy-noise row being slot-indexed — is identifiable and testable at zero
GPU via an identity-permutation bit-exactness assertion.

### E5 — addresses a real objection, but is weaker than it looks

The objection *"instability may be specific to near-identical policies with tiny
treatment effects"* is legitimate. But our own **reliability curve already
answers it quantitatively**: sign reliability rises 61% → 78% → 96% → 100% with
effect/noise ratio. That *is* the general statement, and it is derived from
existing data. A Gaussian-vs-Flow point would add one large-effect datum at
~0.5 GPU-h; the curve already predicts it will be sign-stable. Low information
gain.

### E4/E6 — rejected on cost-benefit

E4's CPU path is blocked by CUDA assumptions in the wrapper, and CPU/GPU physics
need not be dynamically equivalent, so a difference would be uninterpretable.
E6 requires porting EC-Diffuser to a modern stack (>4 GPU-h, high risk); the
brief's option **A — literature support sufficient** applies, since GPUSimBench
independently documents the same regimes in Isaac Lab / ManiSkill / Genesis /
MJX. We cite rather than replicate.

## Recommendation

**Run E1 (0.111 GPU-h). Gate E2 on its outcome. Run nothing else.**
