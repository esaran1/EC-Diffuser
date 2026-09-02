# Phase 2 (2nd attempt) — 20k run STOPPED: cap breach under GPU contention

**Status: STOPPED at step 1,200 of 20,000.**
Phase-2 GPU consumed to date: **0.374 GPU-h** (0.05 first attempt + 0.324 here).
Cap 5.0 GPU-h. E1 untouched. No hyperparameters tuned mid-run.
**No viability classification is made — MeanFlow remains MF-UNTESTED.**

---

## 1. Sequence of events

| step | wall | cumulative s/step | 20k projection (train + 0.35 diag) |
|--:|--:|--:|--:|
| 400 | 259 s | 0.648 | **~3.6 GPU-h** — inside cap |
| 1000 | 912 s | 0.912 | ~5.4 GPU-h |
| 1100 | 1047 s | — (inst. 0.900) | 5.37 |
| 1200 | 1158 s | 0.965 | **5.71–5.89** — **over cap** |

Three consecutive independent measurements (0.90, 0.97, 1.00 s/step) agreed that
20k needs **5.7–5.9 GPU-h**. Only ~16,800–17,300 steps fit inside 5.0.

§2 says: *"If updated runtime projection exceeds 5.0: STOP again."* I stopped
rather than silently truncating to ~17k, because reducing the approved target is
a scope change that is the advisor's decision, not mine.

## 2. Root cause: the GPU became contended mid-run — this is NOT a MeanFlow slowdown

At launch the GPU was clean (135 MiB, 0% util; verified before starting).
An unrelated job appeared **during** the run:

```
PID 209627  .venv-isaac/bin/python  4168 MiB
  scripts/ab_train.py --arm A --seed 1 --num-envs 32 --max-iterations 6000
  started 18:07  (my run started 18:02)
```

This is **`newton-r1`'s job**, not mine. I did not touch it, per the standing
instruction not to kill or interfere with unrelated jobs.

The timing history matches contention exactly:
- steps 0→400, uncontended: **0.577–0.648 s/step**
- steps 400→1200, contended: **0.90–1.00 s/step**

So the earlier 0.577 s/step reading was not "optimistic" — it was **correct for an
uncontended GPU**, and the run then lost ~40% throughput to a co-tenant. The
approval required running *"only on an uncontended GPU"*; that condition held at
launch and was broken externally mid-run.

**Revised uncontended estimate for 20k: ~3.2–3.6 GPU-h training, ~3.6–4.0 GPU-h
including diagnostics — comfortably inside the 5.0 cap.**

## 3. Resumability — checkpoint is INCOMPLETE for a clean resume

`state_999.pt` (step 1000) contains:

| key | present |
|---|---|
| `step` | ✔ (1000) |
| `model` | ✔ (286 tensors) |
| `ema` | ✔ (286 tensors) |
| **optimizer** | ✘ **absent** |
| scheduler / RNG | ✘ absent |

§3 requires model + EMA + **optimizer** + step (+ RNG where required) for a
resume to be protocol-compatible. **The optimizer state is not saved**, so
resuming would silently restart Adam's moment estimates — a real deviation at
step 1000 of 20000, and exactly the "stitch together an invalid partial run"
that §3 forbids.

**Therefore: a resumed run would have to be treated as a restart.** Recorded, not
worked around.

## 4. Engineering sanity only (§5) — NOT evidence of viability

| step | loss | unweighted | boundary_fraction |
|--:|--:|--:|--:|
| 0 | 0.8569 | 0.8409 | 0.5000 |
| 400 | 0.2988 | 0.2935 | 0.5000 |
| 1000 | 0.2653 | 0.2621 | 0.5000 |
| 1200 | ~0.27 | — | 0.5000 |

- No NaN/Inf. Loss fell 0.857 → ~0.265 and is still moving.
- `boundary_fraction` exactly **0.5000** at every log — the deterministic
  first-k mask concern stays closed.
- Step 0 loss reproduced the first attempt's 0.8569 **exactly**, confirming a
  deterministic restart.

**1,200 of 20,000 steps is 6%.** This establishes nothing about viability,
convergence, imagination, or control.

## 5. Classification

> **MF-UNTESTED — blocked on GPU contention.** Unchanged.

Not MF-NO-GO. Nothing failed scientifically; the run was stopped on a budget
rule triggered by an external co-tenant.

## 6. Options (each needs explicit approval)

| # | option | cost | notes |
|--:|---|--:|---|
| **1** | **Relaunch 20k when the GPU is genuinely free** (recommended) | **~3.6–4.0** GPU-h incl. diagnostics | the original plan at its true uncontended cost; needs a restart (see §3) |
| 2 | Relaunch now, accepting contention | 5.7–5.9 | **exceeds cap** — would need the cap raised to ~6.0 |
| 3 | Truncate to ~17k under contention | ~5.0 | fits, but changes the approved target and weakens the gate |
| 4 | Add optimizer state to checkpointing, then resume-capable runs | +small | a genuine improvement, but a code change to the training loop mid-experiment |

**Recommendation: Option 1.** The uncontended measurement (0.577–0.648 s/step)
is the honest cost, it fits the existing 5.0 cap with margin, and it needs no
scope or code change. The blocker is scheduling, not budget.

I did not wait for the co-tenant to finish because its `--max-iterations 6000`
run has an unknown remaining duration and I should not hold the decision open
indefinitely.

## 7. What was NOT done

No 50k extension, no loss-screen training, no schedule experiment, no
terminal-action masking, no λ change, no five-seed control evaluation, no
imagination panel, no E1. Config unchanged from the approved formulation.
