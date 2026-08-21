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

## NFE study (in progress at time of writing)

Paired study on 3 independent 96-episode evaluation replicates: Flow at
1/2/4/8/16 solver steps vs Gaussian at 100. Model calls verified by forward
hook; latency scales exactly linearly (1.21 ms/episode-step at 1 NFE to
19.72 at 16).

First replicate, Flow arms: **85.4 / 88.5 / 88.5 / 84.4 / 85.4%** at NFE
1/2/4/8/16 — flat within ~4 points across a **16x compute range**.

## One unresolved question

**Is 3-cube PushCube simply saturated for both objectives — and if so, on what task
does the probability path become measurable at all?**

At ~96% success with 4 failures left in 96 episodes, and 11-point swings from episode
sampling alone, this benchmark cannot detect a path effect even if one exists. The
next question is whether a harder setting (4–6 cubes, which the env already supports)
opens enough dynamic range to make the VP hypothesis testable — or whether it must
move to a different benchmark entirely.
