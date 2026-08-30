# Retroactive correction: the three-seed fixed-H replication is PROVISIONAL

Applies to `experiments/audit/seed_replication.json` and every report that cites
its three-seed NFE4−NFE1 result.

## What was measured (raw values, unchanged and still valid)

| seed | 3cube Δ (n=288) | primary Δ | 95% CI | bootstrap p |
|---|--:|--:|---|--:|
| 42 | +0.04282 | +0.03181 | [+0.0022, +0.0617] | 0.0351 |
| 43 | +0.00463 | +0.03870 | [+0.0047, +0.0728] | 0.0255 |
| 44 | +0.03009 | +0.02461 | [−0.0042, +0.0539] | 0.0950 |
| **three-seed mean** | | **+0.0317** (SD 0.0070) | | |

These remain accurate records of what those runs produced.

## Why the interpretation is withdrawn

Every one of those numbers came from **R = 1 physics realization per episode**.
`ISAAC_GYM_NOISE_FLOOR.md` measures the single-realization paired resolution of
this evaluator at **~7.6 pp**.

**A ~3 pp effect is roughly 2.4× below the noise floor of the instrument that
measured it.** The bootstrap CIs above resample episodes but treat each
episode's single rollout as a fixed observation, so they do not include
simulator variance at all — which is why they look tighter than the true
uncertainty.

## Language withdrawn

- "Regime A strong three-seed replication"
- "F4 has a replicated ~3 pp control advantage"
- any algorithm-level interpretation of a ~3 pp R=1 difference
- the per-seed bootstrap p-values (0.0351 / 0.0255) as evidence of a real effect

## Replacement wording

> Under single-realization evaluation, the fixed-horizon NFE4−NFE1 contrast
> measured +0.032 (SD 0.007 across three checkpoints). Because the evaluator's
> single-realization resolution is ~7.6 pp, this measurement cannot establish a
> ~3 pp behavioural effect, and the result is **provisional** pending
> repeated-physics evaluation.

Raw results are preserved. Only interpretation changes.
