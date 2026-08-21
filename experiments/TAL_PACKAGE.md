# Isaac Gym Flow investigation — summary for Tal

## Figures

| # | Path | What it shows |
|---|---|---|
| 1 | `experiments/figures/flow_loss_isaacgym.png` | Flow training loss, raw and smoothed, evaluated checkpoints marked |
| 2 | `experiments/figures/dlp_reconstruction.png` | RGB → DLP → RGB on real Isaac Gym frames |
| 3 | `experiments/figures/imagination_ep0.png` (also ep1, ep2) | Matched Gaussian vs Flow decoded imagination, identical current/goal |
| 4 | `experiments/figures/three_path_variance.png` | Input scale vs t: current path, standardized, standardized+VP |

## Findings

- **The pipeline is correct.** The canonical Gaussian EC-Diffuser checkpoint scores
  **86.5%** on 96 fixed 3-cube PushCube episodes (93.75% on an earlier 32-episode
  set), bracketing the paper's 89.4% ± 2.5. Contact rate 100%.

- **Flow is not broken — it wins.** On 96 identical episodes Flow reaches **95.8%
  using 4 network calls against Gaussian's 100**. Paired McNemar **p = 0.0225**, and
  it runs the same episodes **10× faster** (127 s vs 1279 s). Flow leads on every
  metric: cubes placed, goal distance, progress, cubes pushed the wrong way.

- **Action normalization is correct end to end.** Round-trip error 1.5e-08, and both
  arms share one inverse-transform call site. The z action range is asymmetric, so a
  zero output decodes to a small downward push — but both arms succeed through that
  same path, so it is not a fault.

- **Flow is not undertrained.** The final checkpoint is the best of three (199k
  87.5%, 399k 91.7%, 499k 95.8%). An apparent late-training degradation at 32
  episodes **reversed** at 96 and is withdrawn.

- **Training loss is useless for model selection here.** Loss is flat from ~200k
  while success moves 8 points across those checkpoints.

- **DLP reconstruction is trustworthy** (1.8/255 mean pixel error), so the
  imagination figures are interpretable.

- **Flow's imagined futures are visibly degraded — and it wins anyway.** Smeared,
  duplicated, colour-blended cube clusters where Gaussian keeps three distinct
  cubes; visual-feature dispersion 0.417 vs 0.566 (real) and 0.716 (Gaussian). Yet
  Flow beats Gaussian on the task. Imagination quality and success point in
  **opposite directions** here.

- **The representation is not unit variance and the current path collapses.**
  E[x²] = 0.121, per-group variance spans 65×, and min-max + linear loses **89% of
  input scale mid-path**. Standardization alone lifts the minimum to 0.500;
  standardization + VP holds 1.000. The min-max + denominator control does **not**
  fix it — standardization is what makes the denominator meaningful. VP target
  derived and verified (sympy, autodiff 7e-15, finite differences 5e-10).

- **Measurement noise dominates.** The same checkpoint scored 84.4% on 32 episodes
  and 95.8% on 96. **32-episode evaluations cannot resolve anything smaller than
  ~10 points** — almost certainly larger than any probability-path effect.

## NFE study — complete (1,728 episodes, 1.75 GPU-h, no training)

3 evaluation replicates x 96 episodes, all six arms paired within each set.
Model calls verified exactly by forward hook (1/2/4/8/16/100).

| Flow NFE | Success (n=288) | vs Gaussian | McNemar p | ms/episode-step |
|---:|--:|--:|--:|--:|
| 1 | 0.8056 | **−6.3 pts** | **0.0356** | 1.20 |
| 2 | 0.8681 | ±0.0 | 1.0000 | 2.39 |
| 4 | 0.8889 | +2.1 | 0.504 | 4.82 |
| 8 | **0.8993** | +3.1 | 0.281 | 9.78 |
| 16 | 0.8854 | +1.7 | 0.568 | 19.72 |
| Gaussian 100 | 0.8681 | — | — | **127.77** |

- **Minimum sufficient NFE is 2.** Flow at 2 calls ties Gaussian at 100 exactly
  (b=32, c=32, p=1.0) for a **53x latency reduction**. Recommended operating
  point is **4** — nominally best on every aggregate metric, 26x cheaper.
- **1 NFE is the only significant contrast in the whole study** and is genuinely
  worse. One call is too few; two suffice; past four there is no return.
- **Zero approach failures and zero contact failures in 1,728 episodes**, at
  every NFE. Contact rate is 1.0000 everywhere. Lowering NFE degrades *push
  direction*, not approach or contact.
- **The three replicates produced three different curve shapes** — flat, monotone
  rising, and rising-then-falling. Any one of them alone would have been
  reported confidently, and two would have been wrong. Only the pooled curve is
  quotable.
- **Evaluation noise floor is 3–11 points on 96 episodes.** Even n=288 resolves
  only ~6–7 points.

## 3-cube is saturated

Five of six arms sit within noise of each other (3.1-point spread across Flow
2–16, inside a 3–11 point noise floor), all with 100% contact rate. The task can
no longer discriminate. A **4-cube** probe needs **no retraining and no new
data** — DLP emits a fixed 24 particles/view regardless of cube count, cubes are
procedural — only a `--num-entity` flag. Minimum probe ~0.7 GPU-h.

## One unresolved question

**Why is one solver step enough to approach a cube correctly but not enough to push
it in the right direction?**

The failure taxonomy localizes the low-NFE deficit precisely: at 1 NFE, approach and
contact are perfect (0 failures in 288 episodes) but wrong-direction pushes rise from
18–24 to 41. So the extra integration step is not buying reachability or contact — it
is buying accuracy in the contact-phase action. That is a specific, mechanistic
question about what the second function evaluation contributes, and it is testable on
existing checkpoints without training.
