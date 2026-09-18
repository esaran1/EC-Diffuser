# Tal update — what is action denoising buying us?

**Status: IN PROGRESS.** Slides marked ⏳ await empirical results.

---

## Slide 1 — Three audits changed the plan before we spent any GPU

**UWM is arXiv:2504.02792 (RSS 2025).** Reading the paper *and* the code changed
three things:

| what we assumed | what UWM actually does |
|---|---|
| times sampled with some diagonal mass | **plain product measure, NO diagonal mass** — our inherited code defaulted to 50% diagonal |
| our flow convention carries over | **UWM is DDPM with t=0 CLEAN** — the opposite of ours; corner semantics invert |
| interior paths are the obvious extension | **UWM trains the interior but never traverses it** — all four endpoints ride axis edges |

> **The interior of the 2D time square is unclaimed.** UWM's four inference
> modes each pin one modality's time at a corner and sweep the other. The only
> diagonal sampler in their repo is undocumented, visualization-only code. That
> gap is exactly what we registered as our Stage E.

**Script.** Before touching the GPU we read UWM properly — paper, appendix and
released code at a pinned commit. Three things came back that we would have got
wrong otherwise. Their training distribution has no deliberate diagonal mass at
all; the code is two independent randint calls, and our inherited implementation
defaulted to half the batch on the diagonal. Their time convention is inverted
from ours — they're DDPM where t=0 means clean, we're flow matching where t=0
means noise — so every corner semantic flips. And the thing we thought was the
obvious extension turns out to be genuinely unclaimed: they train the whole
two-dimensional time square but they only ever sample along its edges. That's
the gap we registered.

---

## Slide 2 — Two findings that closed stages without spending their budget ⏳

**Stage F (harder task) is BLOCKED, structurally.** Every trained checkpoint in
the repo is the 3-cube random-color task. Creating a baseline for any other task
costs **~33 GPU-h** — 3.3× the entire program cap — and a cheap 20k screen is
already known to give 0/96.

The honest reframing: our canonical task is *already* 3 objects at ~79%, neither
saturated nor floored. **The binding limit isn't task difficulty, it's evaluator
resolution** — RC95 ±4.30 pp. A 1–2 pp architectural effect is unresolvable on
*any* task at this budget.

**Stage H (latent action) failed its own predeclared criterion.** We fixed in
advance: authorize a 2-D bottleneck only if top-2 PCA ≥ 99% of action variance.
**Measured 75.7%** — spectrum nearly flat (0.445/0.312/0.243), dimensions nearly
uncorrelated. So we did not invent a lossy bottleneck to have something to run.

**Script.** Two stages closed on evidence rather than on GPU time. The
harder-task stage is blocked for a structural reason: we have no pretrained
policy for any task except the one we're on, and training one costs thirty-three
GPU-hours against a ten-hour cap. Worth saying plainly though — the task isn't
the bottleneck. We're already on three cubes at seventy-nine percent, and the
real limit is that our evaluator can't resolve anything under about three points
no matter which task we run it on. The latent-action stage failed a criterion we
wrote down before looking: we needed the top two principal components to explain
ninety-nine percent of action variance to justify compressing, and they explain
seventy-six. The action space is genuinely three-dimensional. So we didn't build
a bottleneck just to have an arm to report.

---

## Slide 3 — Independent-time training works; state-leading inference does not

**Two arms differing in exactly one value** (`p_diag`): same class, same
denoiser, same parameters, same canonical warm-start bytes, same optimizer and
data stream. Verified at runtime — the UWM arm logged `frac_diag = 0.0000`, the
control `1.0000`.

### It adapts, decisively

| action block | UWM | control |
|---|--:|--:|
| off-diagonal / diagonal error | **1.558** | 3.003 |
| corr(distance from diagonal, error) | **+0.356** | +0.692 |
| **inverse-model corner** (state clean, action noisy) | **0.2475** | 1.5941 → **6.4× better** |
| forward-dynamics corner | **0.2737** | 1.0833 → **4.0× better** |

The control — which never saw an off-diagonal time — reproduces the old warm-start
branch's failure signature. The trained model roughly halves it.

### But the diagonal is still the best path

| path (UWM arm, equal model calls) | action error @ step 0 |
|---|--:|
| **P0 diagonal** | **0.0160** |
| P2 action-leads | 0.0550 |
| P1 state-leads | 0.1169 |
| P3 state-first axis | 0.2508 |

> Independent-time **training** buys off-diagonal **robustness**. It does not buy
> a better **operating point**. Offline, state-leading inference is not
> supported.

### The finding I'd flag hardest

Through step 2000 the adaptation metric was **indistinguishable from the
never-adapted control** (+0.695 vs +0.692). It only moved between 2000 and 5000.

**At the 2,000-step budget we used in the loss program, we would have concluded
this does not work.**

**Script.** Two arms, one difference: whether the state and action diffusion
times are sampled independently or tied. Everything else identical, verified in
the training logs. The independent-time arm genuinely adapts — the corner where
the state is clean and the action is noisy, which is the inverse-model regime,
improves six-point-four times over the control. That's the regime Tal was asking
about, and it's real. But here's the part that argues against the exciting
version of the story: on both arms, the plain diagonal is still the best path to
actually generate an action. The non-diagonal paths are three to sixteen times
worse at equal compute. So independent-time training buys robustness across the
time square, not a better place to stand in it. And the thing I'd flag hardest:
through two thousand steps this experiment looked like a failure — the adaptation
metric was sitting right on top of the control. It only separated between two and
five thousand. At the budget we used for the loss experiments, we'd have called
this dead. Closed-loop control numbers aren't in yet, so none of this is a
performance claim — and a foreign job is currently on the GPU, so the last arm is
waiting rather than competing with someone else's work.

