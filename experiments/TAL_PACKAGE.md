# Isaac Gym Flow investigation — summary for Tal

## Figures

| # | Path | What it shows |
|---|---|---|
| 1 | `experiments/figures/flow_loss_isaacgym.png` | Flow training loss, raw and smoothed, with the evaluated checkpoints marked |
| 2 | `experiments/figures/dlp_reconstruction.png` | RGB -> DLP -> RGB on real Isaac Gym frames |
| 3 | `experiments/figures/imagination_ep0.png` (also ep1, ep2) | Matched Gaussian vs Flow decoded imagination, identical current/goal |
| 4 | `experiments/figures/three_path_variance.png` | Input scale vs t for the current path, standardized, and standardized+VP |

## Findings

- **The pipeline is correct.** The canonical Gaussian EC-Diffuser checkpoint scores
  **93.75%** on 3-cube PushCube (32 fixed episodes), against the paper's 89.4% ± 2.5.
  Contact rate 100%, zero cubes pushed away from goal.

- **Flow is not broken.** On the identical episodes Flow reaches **84.4% using 4
  network calls against Gaussian's 100** — a 25x inference saving. Paired McNemar
  p = 0.375, so the gap is not significant. All Flow failures are partial
  composition: it contacts and moves all three cubes but does not finish them.

- **Action normalization is correct end to end.** Round-trip error 1.5e-08, and
  Flow and Gaussian share a single inverse-transform call site. One real oddity:
  the z action range is asymmetric, so a zero network output decodes to a small
  downward push — but Gaussian achieves 93.75% through that same path, so it is
  not the problem.

- **Flow is not undertrained — it is past its best.** Evaluating existing
  checkpoints: **199k -> 87.5%, 399k -> 96.9%, 499k (final) -> 84.4%.** Flow at
  399k matches or beats the 100-NFE Gaussian control at 4 NFE.

- **Training loss is useless for model selection here.** Loss is flat from ~200k
  onward while success swings 12 points over the same interval.

- **DLP reconstruction is trustworthy** (1.8/255 mean pixel error), so the
  imagination figures are interpretable.

- **Flow's imagined futures are visibly degraded**: smeared, duplicated,
  colour-blended cube clusters where Gaussian keeps three distinct cubes.
  Quantified — Flow's visual-feature dispersion is 0.417 against 0.566 for the
  real encoded observation and 0.716 for Gaussian. Yet this degradation does
  **not** stop it solving the task, so imagination quality and success are
  partially decoupled.

- **The current representation is not unit variance, and the path collapses.**
  E[x²] = 0.121, per-group variance spans 65x, and the current min-max + linear
  path loses **89% of input scale mid-trajectory**. Standardization alone lifts
  the mid-path minimum from 0.108 to 0.500; standardization + VP holds 1.000.
  The min-max + denominator control does **not** fix it, so the denominator alone
  is not the mechanism — standardization is what makes it meaningful.

## One unresolved question

**Is the 399k-vs-499k reversal a real late-training degradation, and if so, is the
mid-path scale collapse its mechanism — or is checkpoint variance alone enough to
explain a 12-point swing?**

This matters more than the VP question itself: if success varies this much across
late checkpoints of one run, then any A/B/C probability-path comparison needs
checkpoint selection and seed variance controlled first, or the path effect will
be confounded by exactly this.
