# Two-time sampler — derivation and unit tests (§25)

Zero GPU. The sampler must be derived, not assumed, before any training.

## 1. The model's parameterization

Verified from `flow_matching.py:_compute_flow_loss` (see
`../architecture/ARCHITECTURE_AUDIT.md` §5):

```
x_s(t_s) = (1 − t_s)·ε_s + t_s·s        target v_s = s − ε_s
x_a(t_a) = (1 − t_a)·ε_a + t_a·a        target v_a = a − ε_a
```

`t = 0` is noise, `t = 1` is clean. Each modality's predicted velocity is the
derivative of *that modality's* interpolant **with respect to its own time**:

```
dx_s/dt_s = s − ε_s = v_s          dx_a/dt_a = a − ε_a = v_a
```

## 2. The chain rule — why the naive update is wrong

Parameterize a path by `u ∈ [0,1]` with `t_s = f_s(u)`, `t_a = f_a(u)`. Then

```
dx_s/du = (dx_s/dt_s)·(dt_s/du) = v_s · f_s'(u)
dx_a/du = (dx_a/dt_a)·(dt_a/du) = v_a · f_a'(u)
```

**Each modality's velocity must be scaled by its OWN time's derivative.** The
baseline scalar-time Euler step `x ← x + dt·v` is the special case
`f_s = f_a = u`, where `f_s' = f_a' = 1`.

Feeding arbitrary `t_s, t_a` into the old update would integrate `v` against
`du` instead of `dt`, silently mis-stepping any path whose schedule is not the
identity — e.g. on a state-first axis path where `f_a' = 0` over the first
segment, the naive update would still advance the action. Explicit forward Euler:

```
x_s ← x_s + (f_s(u_{k+1}) − f_s(u_k)) · v_s(x, t_s, t_a)
x_a ← x_a + (f_a(u_{k+1}) − f_a(u_k)) · v_a(x, t_s, t_a)
```

Using exact schedule *increments* rather than `f'(u)·Δu` keeps endpoints exact
even for piecewise schedules with kinks.

## 3. Correctness requirements on any path

1. **Endpoints**: `f_s(0) = f_a(0) = 0` (both fully noisy) and
   `f_s(1) = f_a(1) = 1` (both clean).
2. **Monotone** and within [0,1] — no un-noising then re-noising.
3. **No hidden access to clean targets.** The action channel must never be
   initialized from, or nudged toward, the true action. Only the model's own
   velocity moves it.
4. **Compute accounted fairly** — a two-stage axis path naturally uses more model
   calls (§27).

## 4. Analytic unit test

If the model is replaced by the **exact** velocity field for a known pair
`(ε, x1)` — i.e. `v_s ≡ s − ε_s`, `v_a ≡ a − ε_a`, constant in `t` — then the ODE
is linear and forward Euler is **exact regardless of step count or schedule**,
because the integrand does not depend on the state:

```
x_s(u) = ε_s + f_s(u)·(s − ε_s)   ⇒   x_s(1) = s exactly
x_a(u) = ε_a + f_a(u)·(a − ε_a)   ⇒   x_a(1) = a exactly
```

**This is the decisive test**: with an oracle velocity field, every registered
path must land on the clean target to floating-point tolerance, for any number of
steps. A sampler that fails this has a chain-rule or endpoint bug. A sampler that
passes it has the transport right, and any remaining error is the model's.

A second test uses a **state-dependent** field (`v = c·(x1 − x)`) where Euler is
*not* exact, to confirm error decreases at the expected first-order rate as steps
increase — this catches a sampler that accidentally hard-codes the analytic
solution.

## 5. Registered paths (§26)

| path | `f_s(u)` | `f_a(u)` | semantics |
|---|---|---|---|
| **P0 diagonal** | `u` | `u` | ordinary joint generation |
| **P1 state-leads** | `1−(1−u)²` | `u²` | state cleans early → plan-to-action |
| **P2 action-leads** | `u²` | `1−(1−u)²` | action cleans early → action-to-state |
| **P3 state-first axis** | `min(2u,1)` | `max(2u−1,0)` | state 0→1, then action 0→1 |
| **P4 action-first axis** | `max(2u−1,0)` | `min(2u,1)` | action first, then state |

All five satisfy requirements 1–2 by construction. P3/P4 are piecewise-linear
with a kink at `u = 0.5`; the increment form in §2 handles this exactly.

**Caveat carried from the UWM audit.** UWM's own endpoints ride *axis edges* with
one time pinned at a corner, which resembles P3/P4 — but UWM pins at a corner and
sweeps the other **without ever returning the pinned one to clean**. Our P3/P4
traverse both segments, so they are not UWM endpoints; they are interior/edge
*paths*, which UWM does not implement. That is the registered novelty and it must
be described precisely.

## 6. Unit-test results, and two tests I initially got WRONG

`test_two_time_sampler.py` — **all tests pass** after correcting two of my own
assertions. Both errors were in the tests, not the sampler, and are recorded here
rather than quietly fixed.

| test | result |
|---|---|
| schedule validity (endpoints, monotone, range), all 5 paths | PASS |
| all 5 paths distinct; P1 state-leads / P2 action-leads at midpoint | PASS |
| **oracle field lands on x1 exactly**, 5 paths × {1,4,17} steps | PASS, max err 2e-6 |
| blockwise chain rule: state at `t_s`, action at `t_a` after partial integration | PASS |
| naive shared-du update is detectably wrong | PASS, gap 1.17 |
| state-dependent field, first-order convergence | PASS, 17.1× over 16× steps |
| zero field leaves `x` bit-unchanged (no clean-target leakage) | PASS |

### Correction 1 — a constant field cannot expose the shared-du bug

My first version of the "naive update is wrong" test used the **oracle constant**
field. It reported a gap of 9.5e-7, i.e. no difference. That is correct
behaviour, not a sampler bug: with a constant velocity each block's increments
sum to 1 under either update, so both land identically. The test had no teeth.

Fixed by using a **time-dependent** field (`v_a` depends on `t_a`, `v_s` on
`t_s`), where integrating against `du` instead of the block's own `dt` genuinely
differs. The gap is now **1.17**.

### Correction 2 — I asserted the wrong limit for the convergence test

I asserted that Euler on `v = c·(x1 − x)` converges to `x1`. It does not. That
ODE solves to

```
x(1) = x1 + (x0 − x1)·e^(−c)
```

which for `c = 2` sits at `0.8647`, not `1.0`. My test was measuring error
against a limit the ODE never reaches, so "error" *grew* with step count as Euler
correctly approached the true solution. Fixed to measure against the analytic
solution; first-order convergence is then clean (4.31e-1 → 2.52e-2 over 4 → 64
steps, halving per doubling as expected).

**The sampler was right both times.** Recorded because a test that passes for the
wrong reason is worse than one that fails.
