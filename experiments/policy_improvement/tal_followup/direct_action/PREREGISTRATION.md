# Stage C — direct-action pre-registration

Written **before** the model is trained. Fixes the interpretation so it cannot
be chosen after the fact.

## The question (Tal's H1)

> What is action denoising actually buying us?

Concretely: if the action can be produced by a learned query token conditioned
on a generative state representation — with **no noised action trajectory at
all** — how much control performance is lost?

## What is compared

| arm | action pathway | state pathway |
|---|---|---|
| **CTL** (matched diagonal control) | noised action → velocity head | noised, generative |
| **QAC** (direct action) | **learned query token → action directly** | noised, generative — *unchanged* |

Both warm-start from canonical, both use clean-first-step, matched LR/warmup/
EMA/schedule/dataloader seed. The only change is the action pathway.

`QAC` adds **2,560 parameters** (H × 512) to a 60.6M-parameter model — 0.004%.

## Registered predictions

From A2A (arXiv:2602.07322) Fig. 10, the closest published comparison
(flow-in-latent vs deterministic regression-in-latent):

> both reach high in-distribution success, but regression "fails to generalize
> to unseen scenarios"

Our evaluator uses **fixed in-distribution scenario sets**, so:

| prediction | registered before the result |
|---|---|
| **P1** | QAC will be **competitive** with CTL on our benchmark — within the evaluator's resolution |
| **P2** | The offline action-replay error of QAC will be **higher** than CTL's diagonal (it cannot exploit iterative refinement) |
| **P3** | If QAC is competitive, that is evidence **against** action denoising being load-bearing *on this benchmark*, and explicitly **not** evidence about OOD robustness, which we cannot test |

## Interpretation rule (§49), fixed now

| observed \|QAC − CTL\| closed-loop | reading |
|---|---|
| < ~3 pp | **UNRESOLVED / below practical routine resolution** — conservative RC95 is ±4.30 pp; R=3 half-width ±2.48 pp |
| ≥ ~3 pp, QAC worse | action denoising **is** contributing measurably on this benchmark |
| ≥ ~3 pp, QAC better | surprising; would need replication before any claim |

**A "competitive" result is the most likely outcome and is NOT a null result** —
it is the informative answer to H1, because it says the expensive mechanism is
not buying resolvable performance here.

## What this cannot establish

- **OOD robustness.** A2A's own evidence says that is where generation wins, and
  our five fixed scenario sets cannot probe it. Any claim about robustness would
  be unsupported.
- **Generality.** One training seed, one task.
- **Equivalence to a diffusion endpoint.** Per §15 the query token is
  architecturally distinct from `t_action ∈ {0, 1}`.

## Cost

Training 0.33 + closed-loop R=3 0.664 = **0.99 GPU-h**.
