# Phase 3 — Engineering / provenance incident log

**Engineering record only. Not scientific data.** No result in this file is used
as evidence for or against any arm.

---

## Incident 1: duplicate Arm A launch (2026-09-05 ~16:09)

**What happened.** I had queued a background waiter to relaunch Arm A once the
A@15k diagnostic control released the GPU. Seeing the GPU idle a few minutes
later, I also launched Arm A manually. Both fired within ~7 seconds
(PIDs 1222165 at 16:09:39 and 1222405 at 16:09:46), producing **two training
processes writing to the same log file and the same checkpoint directory**.

**Why it mattered.** Two trainers sharing `data/panda_push/phase3_armA/` would
interleave checkpoint writes and log lines, producing artifacts that could not be
attributed to a single training trajectory. Undetected, this would have silently
corrupted the gate run.

**Detection and recovery.**
1. Noticed two `scripts/train.py` PIDs in the process table within ~10 s.
2. Killed both python trainers; verified the process table and GPU were empty
   (142 MiB, 0%, no compute apps).
3. Deleted the partial artifacts — `data/panda_push/phase3_armA/` contained only
   `state_0.pt` plus config files, i.e. **no training progress** — and the
   187-line partial log.
4. Relaunched **once** via `setsid`, PID 1223016.
5. Verified single-writer: exactly one process holds the GPU; the only other
   `train.py` PID is its `num_workers=1` dataloader child (parent 1223016); and
   the log contains **exactly one** step-0 record.

**A secondary false alarm.** Immediately after the kills, `pgrep -f
"scripts/train.py"` appeared to show processes respawning. This was my own grep
pattern matching its **own shell wrapper command line**, not a respawn loop.
Confirmed by filtering on `bin/python diffuser/scripts/train.py`. No respawn
mechanism existed.

**Contamination assessment: none.** The valid run restarted from step 0 with the
frozen configuration, and its step-0 and step-100 losses reproduce the earlier
Arm A run exactly (0.8585 / 0.3754), confirming identical initialization and an
identical paired data stream.

**Root cause.** Two launch paths for one job. The waiter and the manual launch
were each individually correct; the fault was having both armed at once.

**Corrective action (deferred by instruction).** Before B/C/D, add a
single-run lock/sentinel per arm directory:
- atomically acquire a lock keyed to the run directory (`O_CREAT|O_EXCL`);
- refuse to launch if held, printing the holding PID and start time;
- record PID and start timestamp in the lock file;
- release on clean exit.

It must be **infrastructure-only** — no effect on training, RNG, dataloader
order, or the model — and will be validated as training-inert the same way the
final-save fix was, using a no-lock control run with bit-identical weights, EMA
and batch order. **Not applied to Arm A while it is running.**

## Incident 2: final checkpoint never saved (earlier, resolved)

See `PHASE3_FINALSAVE_FIX.md`. The step-`N` save never fired because the loop
runs `0..N-1`; Arm A's last persisted weights were from step 15,000 despite
training the full 20k. Fixed by an explicit post-loop `trainer.save('final')`,
validated bit-exactly training-inert, and active for the current run.

## Non-scientific status of superseded artifacts

| artifact | status |
|---|---|
| `data/panda_push/phase3_armA_NONGATING_15k/` | **ARM A @15K — NON-GATING DIAGNOSTIC.** Trained 20k, final weights lost; evaluated at step 15,000. E0–E4s all 0.0000, macro mean 0.0000, SD 0.0000. Must not authorize, block, or be merged with the registered gate. |
| duplicate-launch partial artifacts | deleted; contained no training progress |
| `flow_timing_probe` | deleted after measuring 0.230 s/step |
