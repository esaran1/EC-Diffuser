# Retroactive claim audit (§19)

Every closed-loop conclusion this project produced before evaluator calibration,
including those that **survived**. No cherry-picking.

| # | original experiment | original estimate | original conclusion | R | calibrated result | corrected conclusion | sign changed? | significance lost? | practical interpretation changed? |
|--:|---|--:|---|--:|---|---|:--:|:--:|:--:|
| 1 | NFE study, 3-cube, seed 42 (n=288, 3 replicates) | NFE1 0.8056 / NFE2 0.8681 / NFE4 0.8889 / NFE8 0.8993 / NFE16 0.8854 | "control saturates at NFE2, peaks at NFE8, declines at NFE16" | 1 | resolution at R=1 is 7.6 pp; the whole 2-16 spread is 3.1 pp | **fine ordering unresolved**; only NFE1-vs-rest survives | no | **yes** | **yes** |
| 2 | fixed-H three-seed replication | NFE4−NFE1 +0.0428/+0.0046/+0.0301, mean **+0.0317** (SD 0.0070) | "Regime A strong three-seed replication, ~3 pp advantage" | 1 | R=3 gives **+0.0602** (SD 0.0313) | direction survives; **magnitude was attenuated ~2×**, and the original CIs excluded simulator variance entirely | no | **yes** (p-values withdrawn) | yes (magnitude) |
| 3 | NFE4 vs NFE32, 3 seeds | +0.104/+0.010/−0.010, mean +0.0347 | "no reliable difference" | 3 | unchanged | **survives** | no | no | no |
| 4 | NFE4 vs NFE32 same-config rerun | 0.8854 vs 0.8021 on identical config | flagged as an 8.3 pp "anomaly" | 1 | falls inside the measured 10.4 pp same-arm spread | **not an anomaly — ordinary evaluator behaviour** | n/a | n/a | **yes** |
| 5 | 4-cube / 5-cube probes | various | used as compositional evidence | 1 | inadmissible under our own protocol | **withdrawn**; 4-cube re-run at R=3 | n/a | **yes** | **yes** |
| 6 | NFE1 vs NFE4, 3-cube | — | — | 3 | +9.03/+2.78/+6.25, mean +6.02 | directional across 3 checkpoints; 1 of 3 CIs excludes 0 | — | — | — |
| 7 | NFE2 vs NFE4, 3-cube | — | — | 3 | −0.69/−1.04/+1.74, mean ≈0 | no reliable difference; **±5 pp equivalence NOT established** | — | — | — |
| 8 | NFE2 vs NFE4, 4-cube H=100 | — | — | 3 | +7.30/+3.45/+2.85, mean +4.53 | operating point does **not** transfer (ceiling confound stated) | — | — | — |
| 9 | PushT Stage-1 screen (external) | NFE2 88%, NFE100 86% | "behavioural saturation at NFE2" | 1 | **success definition was wrong** (`>0.95` vs `>=1.0`); corrected: NFE2 48%, NFE100 72% | withdrawn; confirmatory n=500 gives 60.8 vs 65.4 | **yes** | **yes** | **yes** |

## Summary

- **9 conclusions audited; 5 materially revised or withdrawn; 1 survived intact
  (row 3); 3 are post-calibration results.**
- **Sign changes: 1** (row 9 — and that one was a metric-definition bug, not
  simulator noise; it is reported separately for honesty rather than counted as
  evidence for the evaluator thesis).
- **Significance/interpretation lost: 4** (rows 1, 2, 5, 9).
- Row 2 is the most instructive: the direction was right, the **magnitude was
  understated ~2×**, and the original bootstrap CIs looked *tighter* than the
  truth because they resampled scenarios while treating each single rollout as
  fixed — omitting the 71.5% of variance that lives below the scenario.

## Wording (§15 of the prior brief)

These were internal project conclusions, never externally published. Correct
phrasing:

> Simulator calibration forced us to withdraw or revise several previously frozen
> conclusions derived from conventional single-realization evaluation.

**Not** "we corrected three published results."
