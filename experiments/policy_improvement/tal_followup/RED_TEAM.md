# Red-team — Tal-followup program

Adversarial checks against this program's own results. Anything that failed is
recorded, not removed.

## A. Could the two arms be secretly the same experiment? (the historical defect)

| check | result |
|---|---|
| final checkpoints byte-identical? | **No** — `a4f24d23c5350c4b` vs `6bb31652c5323f77` |
| either identical to canonical? | **No** — canonical is `861dc34434474455` |
| did the arms train on different objectives? | **Yes, proven at runtime** — `frac_diag` 0.0000 vs 1.0000; `abs_time_gap` ~0.33 vs 0.0000 in the training logs |
| did `p_diag` survive the kwargs whitelist? | **Yes** — `test_production_path.py` exercises the real `diffusion_wrapper_kwargs`, and the saved `args.json` of each executing run records its own `p_diag` |
| extension arms identical to their 5k parents? | **checked by sentinel** in `run_pilot.py`; FATALs if so |

This is the failure mode that silently invalidated the historical B/C/D arms. It
is now checked at three independent levels: config, runtime distribution, and
checkpoint bytes.

## B. Could the "adaptation" be an artifact of the metric?

| challenge | answer |
|---|---|
| Is the grid comparison confounded by different inputs? | **No** — identical `x1`, identical noise (fixed generator seed), identical conditioning at every cell, for both arms |
| Does the off-diagonal mean include an unlearnable cell? | **Yes, and it was found and analysed** — the `t_action=1.0` column sits at the irreducible floor E\|N(0,1)\| = 0.7979. Excluding it **strengthens** the verdict (3.667 vs 13.703). See `uwm/DEGENERATE_ENDPOINT.md` |
| Is the criterion post-hoc? | **No** — registered in `uwm/ADAPTATION_CRITERION.md` before the data was read, and the headline numbers still use the registered full-grid definition |
| Could the control simply be under-trained? | **No** — both arms had identical optimizer budgets (5000 steps, same LR, same schedule, same dataloader seed), and the extension applies to both |

## C. Could the diagonal-is-best result be a sampler bug?

This is the result that cuts against the interesting hypothesis, so it deserves
*more* scrutiny than the favourable one.

| check | result |
|---|---|
| does the path sampler reproduce the baseline on the diagonal? | **Yes, bit-identically** (max\|d\| = 0.000e+00) for both `path=None` and `P0_diagonal` |
| is the chain rule right? | **Verified** against an oracle constant field: every path lands exactly on the target for any step count |
| do non-diagonal paths get fewer model calls? | **No** — all paths use exactly 4, verified programmatically |
| could clean targets leak into a path? | **No** — a zero velocity field leaves `x` bit-unchanged |
| is the result consistent across LIVE and EMA? | **Yes** — CTL P0 0.0159 (EMA) / 0.0160 (model); the ordering is identical |

The diagonal-is-best finding survives every check. It is reported as a genuine
negative for the state-leading hypothesis.

## D. Claims NOT made, and why

| tempting claim | why it is not made |
|---|---|
| "independent-time training improves control" | **No closed-loop evidence yet.** §48: offline metrics do not support a control claim |
| "state-leading inference is better" | **Contradicted by our own offline data** — diagonal is best on both arms |
| "we reproduced Tal's direct-action finding" | **No prior artifact exists to reproduce**; it is a new build |
| "a query token is `t_action = 0`" | **Architecturally distinct** (§15); equivalence would have to be demonstrated |
| "registers would help" | **No artifact found**; adding them would be uninterpretable capacity |
| "the harder task would reveal the effect" | **Untested and unaffordable** (~33 GPU-h); and resolution, not task difficulty, is the binding limit |
| "a 2-D action latent is justified" | **Failed its own predeclared criterion** (75.7% vs ≥99%) |

## E. Errors I made during this program

Recorded because a program that reports no mistakes is not being audited.

1. **`utils.Config.load` does not exist.** Would have crashed four diagnostics
   *inside a GPU lease*. Caught by running the CPU-only action audit first.
2. **Two sampler unit tests were wrong** — a constant field cannot expose the
   shared-du bug; Euler on `v = c(x1−x)` converges to `x1+(x0−x1)e^{−c}`, not
   `x1`. The sampler was correct both times.
3. **The bimodality detector returned 1.000 everywhere** — a sparse-tail
   artifact scoring a 3-count bin against a 470-count peak as a "second mode".
   Fixed to require ≥1% of the mass. The register verdict does not depend on it.

## F. Open weaknesses (not defended)

- **One training seed.** Everything here is a pilot. §30 forbids treating it as
  evidence of general improvement.
- **One task.** Stage F is blocked, so generality across tasks is untested.
- **Offline-only so far.** The closed-loop gate is the decisive test and is
  pending.
- **Our two-time construction diverges from UWM's** (shared time MLP evaluated
  twice vs their joint MLP over the concatenated pair). Ours is zero-parameter
  and exactly diagonal-compatible; theirs is more expressive off-diagonal. A
  faithful-UWM embedding might adapt further or faster. Declared, not tested.
