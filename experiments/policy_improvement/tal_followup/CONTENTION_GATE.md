# GPU contention — hard gate on further GPU work

**Status: BLOCKED on a foreign user job. No further GPU work launched.**

## What happened

At **2026-09-18T17:09:59**, ~9 minutes into the control arm's extension rerun, a
foreign GPU job appeared:

```json
{"iso": "2026-09-18T17:09:59",
 "foreign": [{"pid": 58013, "name": "python", "used_mib": 5872}]}
```

The production guard behaved **exactly as designed**:

```
[ gpu_guard ] contention detected; full-state checkpoint written, stopping cleanly.
[ run_lock ] released
```

It wrote `state_contention_2261.pt` — a full resumable state at step 2261 — and
exited cleanly. It did **not** continue training on a shared GPU, and it did
**not** touch the foreign job.

## The foreign job (NOT ours, NOT to be touched)

| field | value |
|---|---|
| command | `python -u -m alloc.delay_arm2` (under a `conda run -n dagr` wrapper) |
| PIDs | 63828 (worker), 63442 (wrapper) |
| GPU memory | 5872 MiB |
| CPU | ~1683% |

This is an unrelated user workload. Per the standing constraints it is **never**
killed, interrupted, or waited out by preemption. The only correct action is to
wait for the GPU to become clean on its own.

## Accounting

| item | GPU-h | class |
|---|--:|---|
| contended `ctl_ext` run (reached step 2200 of 5000) | **0.1547** | **PARTIAL / CONTENDED** |

Recorded against the cap, not hidden. **Program total: 1.1558 of 10.0 GPU-h.**

## State of the experiment

| arm | 5k | 10k extension |
|---|---|---|
| uwm | complete `a4f24d23c5350c4b` | **complete** `bc2027e4d112ac7f` |
| ctl | complete `6bb31652c5323f77` | **INCOMPLETE** — stopped at step 2261 of 5000 |

**The matched comparison at 10k is not yet available**, because the control has
not received the same optimizer budget as the treatment. Reporting a 10k UWM
result against a 5k control would confound "trained off-diagonal" with "trained
longer" — precisely the confound the matched control exists to prevent.

## What is NOT affected

The **5k results are complete, matched, and already analysed**. Every finding
reported so far — the ADAPTED verdict, the 6.4× inverse-model corner
improvement, the diagonal-is-best negative result, the late-adaptation curve —
rests on the 5k pair, which is intact.

The extension was an *additional* check on whether adaptation continues past 5k.
Its absence does not invalidate anything already established.

## Resumption rule

When the GPU is clean, the registered order is:

1. resume/rerun the control extension to 5000 steps (~0.33 GPU-h)
2. re-run field + replay diagnostics on both extended arms (~0.02)
3. closed-loop R=3, CTL-diagonal vs UWM-diagonal (1.327)
4. Stage C direct-action training + closed-loop R=3 (0.99)

The lease's stable-clean protocol (10 probes × 18 s) already enforces waiting for
a genuinely idle GPU, so the resumption is safe to arm and leave.

---

## Resume-validity audit — RESTART, do not resume

`state_contention_2261.pt` was audited against the seven components a valid
mid-extension resume requires:

| component | status | evidence |
|---|---|---|
| model | **restored** | 282 entries |
| EMA | **restored** | 282 entries |
| optimizer moments | **restored** | 269 param states with `exp_avg`, `exp_avg_sq`, `step` |
| LR scheduler / warmup | **restored** | warmup is a *pure function of `self.step`* (`training.py:238-241`), not a stateful object — restoring `step` restores the schedule exactly |
| local step | **restored** | 2261 |
| RNG states | **restored** | `torch_cpu`, `torch_cuda_all`, `numpy`, `python`, `dataloader_generator` |
| **dataloader position** | **NOT RESTORED** | see below |

**The dataloader position is not exactly recoverable.** Torch's `RandomSampler`
draws a fresh per-epoch base seed via
`torch.empty((), dtype=int64).random_(generator=g)` and derives the permutation
from it internally. That seed cannot be reconstructed from the generator state.

The codebase documents this itself (`diffuser/utils/training.py`):

> "That per-epoch seed is NOT recoverable from the generator state, so restoring
> the generator does not reproduce the in-flight epoch's order... we do **NOT**
> claim bit-exact continuation of the data order."

**Decision: 6 of 7 is not enough.** Resuming would give the control arm a
different data trajectory from the treatment arm, defeating the matched control.
The control extension **restarts from the same 5k parent**. The 0.1547 GPU-h is
accepted as waste.

The partial run is **preserved, not deleted**, at
`data/panda_push/uwm_ctl_ext_PARTIAL_contended/` with a `WHY_PARTIAL.txt`
recording why it is invalid as a scientific arm.
