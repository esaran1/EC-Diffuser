# Phase 2 Infrastructure — Full-State Checkpointing + Contention Guard

Infrastructure only. **No change to model, loss, optimizer hyperparameters,
data-order semantics, gradient accumulation, batch size, objective, or the
MeanFlow formulation.** All validation ran on **CPU** (0 GPU-h) except one
launch-refusal test that exited before allocating any GPU memory.

---

## 1. Full-state checkpoint fields implemented

`Trainer.save()` (`diffuser/diffuser/utils/training.py`) now writes:

| field | present | notes |
|---|---|---|
| `step` | ✔ | optimizer step counter |
| `model` | ✔ | live weights |
| `ema` | ✔ | EMA/deployment weights |
| `optimizer` | ✔ | **new** — Adam moments (`exp_avg`, `exp_avg_sq`) |
| `rng.torch_cpu` | ✔ | **new** |
| `rng.torch_cuda_all` | ✔ | **new** — all CUDA devices |
| `rng.numpy` | ✔ | **new** |
| `rng.python` | ✔ | **new** |
| `rng.dataloader_generator` | ✔ | **new** |
| `batches_drawn` | ✔ | **new** — batches consumed |
| `meta` | ✔ | seed, LR, betas, warmup, accum, batch size, version |

**LR scheduler:** none exists as an object — warmup is computed inline from
`self.step` (`training.py:236-241`), so restoring `step` restores it exactly.
**AMP/GradScaler:** not used anywhere in this trainer. Both correctly absent.

`Trainer.load()` restores every field, and is backward compatible: older
checkpoints simply lack the keys and load as before.

## 2. Continuous-vs-resume validation result

`experiments/policy_improvement/validate_resume.py`, CPU, tiny model/data.
PATH 1 trains K+M=10 steps continuously; PATH 2 trains K=6, saves, reloads into
a **fresh Trainer**, trains M=4 more.

**Test 1 — with the live shuffling dataloader:**

| quantity | result |
|---|---|
| step | BIT-EXACT (10 vs 10) |
| EMA | BIT-EXACT (max\|d\| 0.000e+00) |
| model | DIFFERS (max\|d\| 2.06e-03) |
| optimizer moments | DIFFERS (max\|d\| 2.66e-02) |

**Test 2 — identical data stream (isolates state restoration):**

| quantity | result |
|---|---|
| step | **BIT-EXACT** |
| model | **BIT-EXACT** (0.000e+00) |
| EMA | **BIT-EXACT** (0.000e+00) |
| optimizer moments | **BIT-EXACT** (0.000e+00) |

## 3. Is resume bit-exact? — **Yes for all saved state; no for data order**

The state restoration is provably exact: with an identical data stream, model,
EMA, optimizer moments and step all match to 0.000e+00.

**The one gap is the dataloader epoch permutation, and it is a PyTorch
limitation, not a missing field.** `RandomSampler` draws a fresh `base_seed`
when each epoch's iterator is created and derives the permutation from it
internally; that per-epoch seed is **not recoverable from the generator state**.
Verified directly (`/tmp/gendiag.py` probe): replaying from either the
pre-iterator **or** post-iterator generator state reproduces a *different* epoch
order than the original.

I attempted a fast-forward fix (replay `batches_drawn % batches_per_epoch`
batches after restore) and **it did not work**, for the same reason — the epoch
being replayed is already a different permutation. That attempt was removed and
replaced with an explicit comment documenting the boundary rather than left in
as a fix that does not fix anything.

**Practical consequence:** a resumed run continues with correct optimizer state
and correct weights, but sees a *different* (still valid, still shuffled) data
order than an uninterrupted run would have. For an emergency contention
checkpoint this is scientifically acceptable — it is one reshuffle, not a
semantic change — but it means **a resumed run is not bit-identical to a
continuous one**, and that must be stated whenever a resumed run is reported.

Per §D the approved 20k run **restarts from step 0** regardless, so this
limitation does not affect the scientific result.

## 4. Contention guard implemented

`diffuser/diffuser/utils/gpu_guard.py` (new) + wiring in
`diffuser/scripts/train.py`, enabled by `require_uncontended_gpu=True` in
`diffuser/config/pandapush_imf_viability.py`.

- **At launch:** `require_uncontended()` records an `nvidia-smi` snapshot
  (timestamp, GPU name, memory, utilization, every compute process) to
  `gpu_launch_snapshot.json` and **raises** if any foreign process holds
  ≥1000 MiB.
- **During training:** `ContentionMonitor.check()` runs between optimizer steps,
  rate-limited to once per 60 s. On detecting a foreign process it (1) writes a
  **full-state** checkpoint `state_contention_<step>.pt`, (2) records
  `contention_event.json`, (3) prints and exits the epoch loop cleanly.
- It **never** kills the foreign process, never continues at degraded
  throughput, and never changes the step target.

Emergency checkpoints are named `contention_<step>` and are **not** analysis
checkpoints (§F).

## 5. Proof the guard does not affect training

Two trainers on an identical deterministic data stream, 8 steps each: one with
**no** guard attribute, one with the guard attached and `period_s=0.0` so it
checks on **every** step.

```
guard fired      : False
steps            : 8 vs 8
model  identical : True   max|d| = 0.000e+00
ema    identical : True   max|d| = 0.000e+00
```

The guard performs no tensor work — it only shells out to `nvidia-smi` and
compares PID sets. By default it is absent entirely (`getattr(self,
'contention_monitor', None)`), so unguarded training is byte-identical to before.

**Live refusal test:** launching the real config right now, with the co-tenant
present, produced exactly the intended behaviour and allocated no GPU memory:

```
RuntimeError: GPU is contended at launch; refusing to start.
Foreign processes: [{"pid": 209627, "name": ".venv-isaac/bin/python",
                     "used_mib": 4168}]
```

## 6. GPU cleanliness status — **CONTENDED, NOT READY TO LAUNCH**

```
209627, .venv-isaac/bin/python, 4168 MiB     (newton-r1: scripts/ab_train.py)
GPU utilization 100%
```

This is an unrelated job. Not touched, not killed. **The 20k run cannot start
until it clears** — and the guard now enforces that automatically.

## 7. Measured uncontended projection

From the pre-contention segment of the aborted run (steps 0→400, verified clean
GPU at launch): **0.577–0.648 s/step**.

| item | GPU-h |
|---|--:|
| 20k training | 3.2–3.6 |
| five-seed control (5 × ~145 s) | 0.20 |
| loss probes, replay, imagination | 0.10 |
| **projected total** | **3.5–3.9** |

Against the **5.0 GPU-h** Phase-2 cap. Previously wasted attempts (0.374 GPU-h)
are recorded in the ledger but not deducted from the valid run's scope, per §E.

## 8. Compute ledger

| attempt | outcome | GPU-h |
|---|---|--:|
| 1st (bs=32 OOM → bs=8) | stopped, bad projection | 0.05 |
| 2nd (20k under contention) | stopped at step 1200 | 0.324 |
| infrastructure validation | CPU only | ~0.00 |
| **total wasted to date** | | **0.374** |

## 9. Status

**NOT READY — blocked on GPU availability only.**

Infrastructure is complete and validated. Every §A–§C requirement is met, with
the single documented exception that dataloader epoch order cannot be restored
bit-exactly (a PyTorch `RandomSampler` limitation, irrelevant to the approved
step-0 restart). MeanFlow remains **MF-UNTESTED**; nothing from the aborted runs
is interpreted scientifically.

Awaiting (a) the GPU clearing, and (b) explicit authorization to start the 20k run.
