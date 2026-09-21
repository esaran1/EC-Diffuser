# HARD GATE: GPU driver/library version mismatch — and a fail-open defect it exposed

**Status: all GPU work blocked. Requires a system-level reboot or module reload,
which is not mine to perform.**

## 1. The gate

```
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 615.71
```

| component | version |
|---|---|
| **loaded kernel module** (`/proc/driver/nvidia/version`) | **580.173.02** |
| **userspace NVML library** | **615.71** |

The driver was upgraded on disk while the old kernel module is still loaded.
`nvidia-smi` exits **18**. Resolution is a reboot or an `nvidia` module reload —
a sysadmin action. **Not attempted.**

## 2. What still works, and why that is dangerous rather than convenient

**CUDA compute is unaffected**: `torch.cuda.is_available()` is True, a matmul on
the RTX 4080 succeeds, `get_device_name` returns correctly. CUDA talks to the
kernel module directly and does not need NVML.

So training *would* run. **That is precisely the hazard**: our entire
contention-detection system — the thing that keeps us off other users' jobs —
depends on `nvidia-smi`, which is exactly what broke.

## 3. The latent defect this exposed (now fixed)

`gpu_lease.foreign_mib()` was *written* to fail closed:

```python
except Exception:
    return -1          # unknown: treat as dirty, never as clean
```

But it **failed open in practice**, and would have done so silently:

- `nvidia-smi` does **not raise** here. It exits nonzero and prints the error to
  **stdout**, not stderr.
- `subprocess.run` does not raise on a nonzero exit, so the `except` branch
  **never fired**.
- The parse loop found no valid rows and returned **0** — i.e. **"clean GPU"** —
  while monitoring was completely blind.

Measured before the fix: `foreign_mib()` returned **0**, not −1.

**Consequence had this gone unnoticed:** the lease would have reported a clean
GPU regardless of what was actually running, and could have launched our
training on top of a foreign user's job — violating the standing "never overlap,
never disturb foreign work" constraint. The three contention stops earlier in
this program are exactly the events this guard is supposed to catch.

## 4. The fix

| site | change |
|---|---|
| `gpu_lease.foreign_mib()` | check `returncode != 0` → return −1; also treat `"NVML"`/`"Failed"` in stdout as −1 |
| `gpu_guard._query()` | new `strict=` mode raising `MonitoringUnavailable` instead of returning `[]` |
| `gpu_guard.monitoring_ok()` | new: True only if `nvidia-smi` is actually answerable |
| `gpu_guard.require_uncontended()` | **refuses to launch** when monitoring is unavailable — a blind guard must fail closed |

Verified under the live mismatch:

```
foreign_mib()            = -1     (was 0)
monitoring_ok()          = False
require_uncontended()    -> MonitoringUnavailable, refuses to launch
```

## 5. Why no GPU work proceeds despite CUDA working

Running now would mean running **blind to contention**. Given that this program
has already been interrupted three times by a live foreign pipeline
(`alloc.delay_arm2`, `alloc.fixed_eval`, `alloc.fixed_eval_contiguous`), the
probability of colliding with someone else's job is not hypothetical.

**Correct action: wait for the driver state to be repaired.** The guards now
enforce this automatically rather than relying on me to remember.

## 6. State at the gate

| item | status |
|---|---|
| 5k matched pair (uwm + ctl) | **complete and valid** |
| Stage-E field/replay analysis at 5k | **complete** |
| UWM 10k extension arm | complete |
| control 10k extension | **never completed** — 4 contention stops |
| closed-loop R=3 (chunked) | **3 of 30 chunks VALID** before the session ended |
| direct-action training | not started |

Closed-loop chunks completed so far (UWM 5k, realization 1):
`E0 = 0.7604`, `E1s = 0.7812`, `E2s` VALID — all n=96.

**Program spend: ~1.83 of 10.0 GPU-h.** Nothing is lost: every completed chunk is
merged into its diagnostics JSON and will be skipped on resume.
