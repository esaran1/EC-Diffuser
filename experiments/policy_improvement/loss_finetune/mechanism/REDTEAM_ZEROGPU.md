# Red-team checks completed without GPU

These are the section-31 checks that can be answered from artifacts on disk.
GPU-dependent checks (physics averaging, B' divergence, action distribution) are
answered in the main report once those stages complete.

## Provenance and wiring

| check | result |
|---|---|
| did any diagnostic use seed-42 paths for all seeds? | **no** — every `*_diagnostics.json` records its own seed-specific `loadpath`; all 8 verified |
| are the four per-seed replay probes actually distinct? | **yes** — A t=0 values 0.01634 / 0.01628 / 0.01626 / 0.01600, four distinct values |
| correct A/B mapping? | **yes** — loadpath arm letter matches the diagnostics tag in all 8 files |
| checkpoint hashes correct? | **yes** — 8 distinct SHAs, all internal step 2000, all matrix sum 2442.0 |
| did gradient diagnostics mutate model/EMA state? | **no** — canonical checkpoint mtime unchanged (Aug 12); the script loads and never writes |

Note: A42's checkpoint SHA (`965e7e32fe19cf01`) coincides with the old invalid
B/C/D checkpoints. That is expected and correct: those invalid runs were
byte-identical copies of arm A's objective, which is exactly why they were void.

## Numerical validity

| check | result |
|---|---|
| are A-vs-B output differences above float noise? | **yes** — 7,856–12,175 × float32 eps |
| does the masked slice diverge most, as expected? | **yes** — terminal A-vs-B is 1.81–2.73× the t=0 difference in every seed |
| is t=4 being pooled back into an executed-action headline? | **no** — every table reports t=H−1 separately; `make_figures.py` shades and labels it |

## Scale context

The A-vs-B output difference at t=0 is **2.8–4.5%** of each arm's displacement
from the canonical starting point (1.0e-3 vs ~3.3e-2). A and B are far closer to
each other than either is to where they started. Any mechanism claim has to be
consistent with that: the fine-tuning displacement dominates, and the A/B
difference rides on top of it.

## Mechanism claims that did NOT survive

- **Gradient conflict**: refuted. cos(g_term, g_exec) is positive in every
  predeclared parameter group (+0.35 to +0.53, 4–17% of batches negative). The
  registered rule says positive ⇒ not supported. Reported as such; no search was
  made for a group that would show conflict.
- **Terminal target is padding**: refuted. It is a constructed constant in the
  goal-conditioning row, present in every window (see
  `TERMINAL_TARGET_PROVENANCE.md`). An intermediate analysis concluded "padding"
  and was wrong; recorded rather than quietly dropped.

## Magnitude of the terminal term

From the gradient study, component means on the production objective scale:

| component | mean loss |
|---|--:|
| L_state | 2.276e-01 |
| L_exec | 1.497e-03 |
| L_mid | 6.234e-04 |
| **L_term** | **3.280e-05** |

The terminal term is ~46× smaller than the executed-action term and ~7,000×
smaller than the state term, and its gradient is *aligned* with the executed
action (projection +0.08). That a term this small and this aligned could produce
a multi-point behavioural change is the central open puzzle, and it is a reason
for caution about the effect, not a mechanism for it.
