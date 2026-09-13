# Replay action error is NOT comparable between A and B/C/D

**Observation.** Arm B's EMA replay action error is 0.17224, versus 0.02788 for
Arm A and 0.02689 for canonical — a ~6.4x jump — while the replay *state* error
is unchanged (0.20533 vs 0.20532).

**This is a metric artifact of the registered ablation, not a training failure
and not a behavioural collapse.**

## Why

`replay_error` in `experiments/policy_improvement/phase2_diagnostics.py` averages
the action error over **every horizon step**:

```python
ae.append(float((xs[:, :, :ad] - x[:, :, :ad]).abs().mean()))     # all of t=0..H-1
se.append(float((xs[:, 1:4, ad:] - x[:, 1:4, ad:]).abs().mean())) # state: t=1..3 only
```

The action term includes **t = H-1**, which is exactly the slice Arm B (and C, D)
no longer supervise. That target is a normalized ZERO placeholder and is trivially
predictable when supervised, so including it pulls A's and canonical's average
*down*. Once masked, the model has no reason to emit the placeholder there and the
all-timestep mean rises — mechanically, regardless of control quality.

Note the state term already restricts itself to `t=1:4`, i.e. the metric's own
author excluded conditioned/degenerate steps on the state side but not on the
action side.

## Corroborating evidence from training logs

At step 0 the A and B models are byte-identical (same canonical warm start, same
batch), yet the reported `action_loss` differs:

| step | A action_loss | B action_loss | observation_loss (A / B) |
|--:|--:|--:|--:|
| 0 | 0.0608 | 0.0727 | 0.2248 / 0.2248 |
| 500 | 0.0605 | 0.0746 | 0.2299 / 0.2298 |
| 1000 | 0.0426 | 0.0525 | 0.2214 / 0.2214 |
| 1500 | 0.0517 | 0.0637 | 0.2254 / 0.2254 |
| 1900 | 0.0626 | 0.0775 | 0.2386 / 0.2386 |

Identical weights at step 0 but different reported action loss ⇒ the difference is
the **metric's masked denominator**, not the model. Observation loss is unchanged
throughout, confirming λ=1 leaves the state side untouched.

## Consequences for the report

1. **Do not compare `replay_action_error` between A and B/C/D.** The quantity is
   computed over different active sets. Comparing them would manufacture a large
   spurious "effect".
2. **B vs C vs D replay action errors ARE mutually comparable** — all three mask
   the same terminal slice, so the metric's active set is identical across them.
3. Replay state error remains comparable everywhere (`t=1:4`, unaffected by the
   action mask).
4. Replay error is in any case **not** equivalent to closed-loop control, and is
   not used to rank arms.
5. The damage-gate criterion "EMA replay action <= 1.5x canonical", which was
   predeclared against Arm A, **cannot be applied to B/C/D** for the same reason.
   Closed-loop control, imagination and NaN checks still apply.

## Outstanding verification

A per-timestep decomposition of the replay action error (t=0..4 separately) would
confirm the increase is confined to t=H-1. That probe needs GPU and was deferred
so as not to overlap the pinned scientific runs. It is run after the program
completes and reported with the results.
