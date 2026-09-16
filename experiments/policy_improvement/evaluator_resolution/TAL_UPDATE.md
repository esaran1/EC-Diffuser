# Tal update — what the PandaPush evaluator can actually resolve

Three slides, ~40–50 s of script each. Supporting figures in `figures/`.
Full detail and caveats in `RESULT.md`.

---

## Slide 1 — Two frozen policies, evaluated six times each

**Figure:** `fig1_macro_realizations.png`

| | canonical | cleanA42 |
|---|--:|--:|
| mean macro success | 79.10% | 78.82% |
| SD across 6 realizations | **1.23 pp** | **1.55 pp** |
| range (max − min) | 3.75 pp | 3.75 pp |

Nothing changed between repeats: same checkpoint bytes (SHA-verified), same five
scenario sets, same 96 episodes, same NFE=4, same seeds.

> **Script.** We took two frozen checkpoints and evaluated each one six times,
> changing nothing — same checkpoint bytes, same scenario files, same seeds, same
> NFE. If the evaluator were deterministic these would be two flat lines. They
> are not. Each policy wanders across a 3.75-point range, and the two traces
> cross each other repeatedly. The standard deviation is about 1.4 points. That
> is the measurement error on every PandaPush number this project has ever
> reported. And note the means: 79.10 versus 78.82 — a 0.28-point gap between two
> genuinely different policies, with a confidence interval of plus or minus about
> 1.7 points even at six repeats each. This benchmark cannot tell these two
> policies apart.

---

## Slide 2 — Three different numbers, which the old shorthand conflated

| | value | what it answers |
|---|--:|---|
| **A. largest observed repeat shift** | **3.75 pp** | an extreme of a finite sample |
| **B. within-checkpoint SD** | **1.40 pp** pooled | typical realization-to-realization spread |
| **C. 95% repeatability coefficient** | **±3.88 pp** pooled / **±4.30 pp** conservative | how far two *single* evaluations of one unchanged policy can differ |

Two SDs are in play: **pooled 1.40 pp** (both anchors, descriptive) and
**conservative 1.55 pp** (the noisier anchor alone). RC95 = 1.96·√2·s gives
±3.88 pp from pooled and ±4.30 pp from conservative. **All planning below uses
the conservative 1.55 pp**, so recommendations hold up if a future checkpoint
behaves like the worse of the two.

**Withdrawn:** "the evaluator noise floor is 3.75 pp." That was a maximum
reported as a standard deviation — wrong in kind, not just degree.

**Pseudo-treatment null** (`fig4`): split one checkpoint's own six realizations
3-vs-3, where the true effect is *exactly zero*.

- half of all splits produce an apparent effect **≥1 pp**
- the largest reaches **2.64 pp**

> **Script.** We had been saying the noise floor was 3.75 points. That number
> came from the largest shift we happened to observe, and reporting a maximum as
> a standard deviation is wrong in kind, not just degree. Here are the three
> numbers kept separate. The typical spread is 1.4 points. The figure that
> actually governs a single-evaluation comparison is the repeatability
> coefficient: two single evaluations of the *same unchanged policy* can
> legitimately differ by 3.9 points on the pooled estimate, or 4.3 on the
> conservative one we plan against. The cleanest demonstration is the
> pseudo-treatment test — we split one checkpoint's own six runs into a fake
> "baseline" and fake "treatment", where the true effect is exactly zero by
> construction. Half those splits show an effect of a point or more, and the
> largest reads 2.64 points. That is a treatment effect manufactured entirely out
> of noise, and it is the same size as the effects we were chasing.

---

## Slide 3 — Consequence, and what it costs to fix

**Every effect the loss program reported, against a single-evaluation SE of 2.19 pp:**

| effect | size | \|d\|/SE(R=1) | R for 80% power |
|---|--:|--:|--:|
| terminal mask, legacy R=1 | +2.24 pp | 1.02 | 8 |
| terminal mask, legacy R=2 | +0.39 pp | 0.18 | 249 |
| terminal mask, clean first-step | +1.46 pp | 0.67 | 18 |
| action weighting λ=2 | −1.88 pp | 0.86 | 11 |

**Cost of resolution** (measured: 0.2212 GPU-h per evaluation):

| target | R per policy | GPU-h per A/B |
|---|--:|--:|
| detect 1 pp (80% power) | 38 | **16.81** |
| detect 2 pp (80% power) | 10 | 4.42 |
| detect 3 pp (80% power) | 5 | 2.21 |
| **routine default R=3** *(precision, not power)* | 3 | **1.33** (±2.48 pp) |

**R=3 is a precision tier, not a detection tier.** It gives a ±2.48 pp interval;
it does *not* give 80% power at 2.48 pp. Effects below ~3 pp should be treated as
below routine resolution unless extra repeats are budgeted.

**Recommended next step (one, not run):** add *passive* episode-level success
logging and re-run both anchors at R=3 (~1.33 GPU-h), to determine whether the
~1.4 pp is driven mainly by a small set of nominally identical episodes flipping
— which would tell us whether more episodes buys resolution more cheaply than
more realizations. The logger must first be shown not to alter RNG consumption,
simulator stepping, action calls, episode ordering, or aggregate results.

> **Script.** Here is what that does to the loss program. Every effect we
> reported sits at or below the resolution of a single evaluation — the ratio
> column never exceeds 1.02. The headline 2.24-point result needed roughly eight
> repeats per policy to measure properly; we measured it with one. Those
> experiments were under-powered by about an order of magnitude in evaluation
> budget, which is the honest explanation for why the effect shrank to 0.39
> points the moment we averaged in a second realization. Resolving a one-point
> effect costs about 17 GPU-hours per comparison, so small-effect policy research
> on this benchmark is not practical at current cost. I recommend R=3 as the
> routine default — 1.3 GPU-hours for a plus-or-minus two-and-a-half-point
> interval. But be precise about what that buys: it is a precision tier, not a
> detection tier. Eighty percent power at three points still needs five repeats,
> and at two points, ten. Anything under roughly three points should be treated
> as below routine resolution unless we budget the extra repeats. The
> single next experiment I would run is cheap: add episode-level logging and
> repeat both anchors three times, to find out whether the 1.4 points comes from
> a handful of borderline episodes flipping or is spread across all of them. If
> it is a handful, more episodes fixes it cheaply. If it is spread, no episode
> count will help and we need a different benchmark. That fork decides whether
> this line of work is viable, so it should come before any new policy idea.
