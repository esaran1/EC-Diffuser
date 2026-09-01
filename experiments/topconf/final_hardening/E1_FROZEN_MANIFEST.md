# E1 frozen manifest

Experiment frozen 2026-09-01, before any E1 GPU result exists. Nothing below may
change unless a genuinely invalidating bug is discovered.

## Frozen artifacts

| artifact | role |
|---|---|
| `slot_permutation_e1.py` | runner: scenario-keyed CRN + slot permutation + 12 provenance assertions |
| `analyze_e1.py` | analysis, tested on synthetic data before results existed |
| `E1_ANALYSIS_PLAN.md` | endpoints, A/B/C thresholds, claim gate, E2 gate, §10 red-team, prior-art check |
| `e1_results/e1_design.json` | permutations + hashes |

## Frozen design

Flow **seed 42**, **NFE 4**, scenario set `replicate0_n96`
(sha `35144910b1471b7b0d50d17da18b01db0b5e61e21d7e16e4ef1aa266ee80d511`),
H=100, **num_envs = 16 (unchanged)**, three mappings: identity + perm1 + perm2,
permutation seed **20260915**.

## Frozen thresholds (from E1_ANALYSIS_PLAN.md §3)

- **A NUMERICAL ONLY** — ≤1 discordant scenario per permutation **and** |Δsuccess| < 2.0 pp → E1 NULL
- **B BEHAVIOURALLY DETECTABLE** — ≥2 discordant **or** |Δsuccess| ≥ 2.0 pp
- **C COMPARISON-RELEVANT** — success range across mappings ≥ 5.0 pp → E2 gate opens

## Frozen validation results (zero-GPU, already passed)

| check | result |
|---|---|
| identity permutation vs canonical noise draw | **0.000e+00** |
| every slot receives its scenario's noise row | **True** |
| noise multiset unchanged (marginal preserved) | **True** |
| naive slot-keyed counterfactual | **6.13** — confound real, removed |

## Red-team findings recorded (§ instruction)

**Finding 1 — inactive colour randomizer.** `isaac_panda_push_env.py:662` draws
`rand_indices = torch.randperm(self.num_colors)` inside `reset`, which would
consume the global torch RNG and shift the CRN stream. It is gated by
`if self.goal_reset and self.random_obj_colors`, and `goal_reset` is set to
**`False`** at line 238, so it **never executes** in this configuration. Had it
been active it would have confounded E1 by coupling the RNG stream to reset
ordering. No change made — recorded as a provenance condition that must be
re-checked if the environment config is ever altered.

**Finding 2 — dead partial-batch padding branch.** The original collation
contained a padding/duplicate-skip branch whose condition was written
ambiguously. With 96 scenarios and `num_envs = 16`, 96 = 6 × 16 exactly, so the
branch is unreachable. It was replaced with two explicit assertions
(`n_act == nE`, `scn not in recs`) **before** any result existed, so no E1 number
depends on the change. This was a latent collation hazard, not an observed
failure.

Also recorded: `reset_noise = torch.rand((len(env_ids), 9))` in `reset_idx` is
multiplied by `frankaDofNoise = 0.0`, so it perturbs no state (it does advance
the RNG, but identically for every mapping since it is drawn per reset, not per
slot).

## Status

Staged and frozen. **Not launched** — GPU occupied by unrelated jobs.
