# Loss-branch update

*Three slides. Full detail in `terminal_mask_mechanism/RESULT.md`.*

---

## Slide 1 — Loss weighting did not help

Fine-tuning the competent PandaPush policy for 2,000 steps, five fixed scenario
sets (n=96 each), NFE=4:

| arm | macro success |
|---|--:|
| baseline (A) | 0.7750 |
| terminal-target masked (B) | 0.8042 |
| masked + λ_action=2 (C) | 0.7854 |
| masked + λ_action=5 (D) | 0.7875 |

λ reached the objective — action gradients scaled 1.8× and 4.4× — and did no harm
to state prediction or imagination. It simply produced no control gain.

The only thing that looked interesting was the **terminal target**: a constructed
zero action attached to the goal-conditioning row, which the model never
executes.

> **Script (~40 s).** Tal, the first question was whether re-balancing action
> versus state loss would improve an already-competent policy. It didn't. Lambda
> equals two and lambda equals five both landed below the masked baseline, and
> we confirmed the weighting genuinely reached the optimizer — action gradients
> scaled by the expected factors. So weighting is not the lever. What did look
> promising was masking a constructed terminal zero-action target that sits on the
> goal-conditioning row and is never actually executed.

---

## Slide 2 — The mask signal did not survive resampling

The terminal mask initially replicated across four fine-tuning seeds:
**+2.24 pp**, three positive, none negative.

Then we ran the controls:

| control | result |
|---|--:|
| four seeds, one simulator realization | **+2.24 pp** |
| **same policies, one extra realization averaged in** | **+0.39 pp** |
| denominator control B′ (four seeds) | +1.82 pp — normalization is *not* the cause |
| clean first-step protocol (four seeds) | +1.46 pp, **SD 2.98**, one seed reversed |

**The number that settles it:** re-running the *same frozen checkpoint* on a new
physics realization moves macro success by **up to 3.75 pp** — larger than any
effect we were trying to measure.

We also found and fixed a real training bug along the way: a random-init sanity
gradient was leaking into the first optimizer step, and it was
objective-dependent. Removing it did not rescue the effect.

> **Script (~45 s).** The mask looked real — plus two point two four points across
> four seeds, three positive, none negative. So we controlled it properly. We
> re-evaluated the identical checkpoints on one additional physics realization,
> and the effect dropped to plus zero point three nine. We checked whether the
> loss-normalization side-effect explained it — it doesn't. And we found a genuine
> trainer bug where a random-initialization gradient contaminated the first update;
> fixing that didn't rescue it either. The decisive measurement is that the same
> checkpoint, re-evaluated, moves by up to three and three-quarter points. That's
> bigger than the effect. We were measuring noise.

---

## Slide 3 — Closing the branch, and what I'd do next

**Conclusion:** neither action weighting nor terminal masking produces a control
improvement this evaluator can resolve. Executed-action accuracy was unchanged
throughout. I'm closing the loss branch rather than pursuing it further.

**What we got that is solid:**
- A measured evaluator resolution limit: **~3.75 pp** of realization noise per
  policy.
- Seven production-path defects found and fixed — including one that had made
  four "different" loss arms train an identical objective.

**One next experiment:** characterise the evaluator before testing any further
policy change — how macro success converges as episodes and realizations
increase, on two frozen checkpoints, to derive the minimum detectable effect.

Every intervention we have tested lives in the 0–4 pp range. Until we can resolve
about 1 pp, we will keep producing effects that evaporate on resampling — which
is precisely what happened here.

> **Script (~40 s).** So I'm closing the loss branch. Neither weighting nor
> masking gives an improvement we can actually resolve, and executed-action
> accuracy never changed. What we do have is worth keeping: a measured resolution
> limit of roughly three point seven five points for this evaluator, and seven
> real pipeline defects fixed — one of which had silently made four different loss
> arms train the same objective. My recommendation is to characterise the
> evaluator next, before testing any further policy idea. Everything we have tried
> sits in the zero-to-four point range, so until we can resolve about one point,
> we will keep chasing effects that disappear when we resample.
