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
no longer supervise. That target is a normalized ZERO placeholder. A supervised
arm learns to emit it almost exactly; an arm that masks it has no reason to, so
the all-timestep mean rises — mechanically, regardless of control quality.

**Verified by per-timestep decomposition** (`REPLAY_PER_TIMESTEP.json`, EMA):

| arm | t=0 (executed) | t=1 | t=2 | t=3 | **t=H-1** | reported all-t |
|---|--:|--:|--:|--:|--:|--:|
| A | 0.01634 | 0.03228 | 0.04047 | 0.04970 | **0.00062** | 0.02788 |
| B | 0.01640 | 0.03263 | 0.04043 | 0.04961 | **0.72214** | 0.17224 |
| C | 0.01631 | 0.03239 | 0.04012 | 0.04921 | **0.72214** | 0.17203 |
| D | 0.01622 | 0.03205 | 0.03988 | 0.04890 | **0.72214** | 0.17184 |

The reported-style column reproduces the diagnostics values exactly. **The entire
difference lives at t=H-1**: A predicts the placeholder (0.00062), B/C/D do not
(0.72214, identical to five decimals across all three — they simply stopped
emitting it). At t=0..3 all four arms agree to about three decimal places, and
the **executed** action t=0 is essentially unchanged (0.01622-0.01640).

So masking removed a learned behaviour that only ever applied to a non-executed
placeholder slot. It did not degrade the actions the policy actually takes.

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

## Verification completed

The per-timestep decomposition above was run after the program finished, on a
free GPU, and confirms the increase is confined entirely to t=H-1. The useful
comparable quantities across ALL arms are the **executed action error at t=0**
(0.01622-0.01640, a spread of ~1.1%) and the mean excluding the terminal slice
(0.03426-0.03477).

**A methodological note worth keeping.** A first version of this probe rebuilt
the dataset from the arm config and used seed 20260905, and produced numbers
(A t=H-1 = 0.81093, matching B/C/D) that contradicted the diagnostics. That was
the probe being wrong, not the diagnostics: the diagnostics loads via
`utils.load_diffusion` with an explicit `override_dataset_path` and uses
NOISE_SEED = 20260902. Once the probe imported the diagnostics' own loader,
dataset override and seed, it reproduced the reported values exactly.

The lesson is the same one that produced this entire re-run: a verification
script that reimplements the path it is checking, instead of calling it, can be
confidently wrong.
