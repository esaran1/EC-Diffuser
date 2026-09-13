# Contention event — Stage B first attempt INVALIDATED

**Both Stage-B probe runs were contaminated by a foreign GPU job and are void.**
They are preserved, not deleted, and their GPU time is charged to this program's
ledger.

## What happened

| time | event |
|---|---|
| ~18:44 | foreign job `RLinf/tools/g1_piston/probe_press_mechanics.py` (PID 1164330, 5,061 MiB) starts |
| 18:54:32 – 18:57:08 | **B probe runs — overlapping the foreign job** |
| ~18:58 | B′ probe launched, also overlapping |
| ~18:59 | contention detected; **my own B′ probe killed**, foreign job left untouched |

## Why it was not caught before launch

`bprime_probe.py` is a standalone script driving the production Trainer. It
inherits `objective_guard`, the warm-start rebuild and the manifest, but it does
**not** call `gpu_guard.require_uncontended()` and is not launched through
`run_replication.sh`, which is where the stable-clean gate lives. I checked the
GPU by hand immediately before the B launch and it read clean; the foreign job
appeared during the run, and nothing was polling for it.

This is the same failure mode as earlier in this project — a new script that
reimplements part of a validated path and silently omits a safeguard. The
protective machinery existed; the probe simply did not route through it.

## Invalidated artifacts (preserved)

```
probe_runs/pandapush_lossft_armB_seed42_INVALID_contended/            (500 steps, 4 ckpts)
probe_runs/pandapush_lossft_armBprime_seed42_INVALID_contended_partial/
```

They are void as measurements: contention perturbs timing, not arithmetic, but
the whole point of this probe is a tight parameter-divergence comparison, and an
uncontrolled co-tenant is exactly the sort of confound this project has been
burned by. They are not used for any reported number.

## Corrective action

The probe runner now (a) requires a stable-clean GPU before launch using the same
10-probe / 180 s condition as the pinned runners, and (b) polls continuously
during training, aborting and marking the run INVALID if a foreign process
appears mid-run. No unrelated job is ever killed; the probe yields instead.

## Compute charged

Both contaminated runs are counted as **invalid GPU time in this program's
ledger** (Stage A/B invalid). Approximately 0.043 + 0.017 ≈ 0.060 GPU-h.
