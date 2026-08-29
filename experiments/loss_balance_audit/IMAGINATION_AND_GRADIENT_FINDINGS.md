# Imagination metric, gradient consequences, and three-seed replication

Date: 2026-08-29. No loss training. No canonical artifact overwritten.

## 1. Frozen seed-44 replication

Six arms, all verified before analysis (cube count, H=100, record counts,
checkpoint `c2c13f55…` @ internal step 499000, EMA weights, measured calls ==
requested NFE, episode-set hashes identical to seeds 42/43).

| Arm | Success | Per-object | Calls |
|---|--:|--:|--:|
| 3cube F1 | 235/288 = .8160 | .9178 | 1.0 |
| 3cube F4 | 255/288 = .8854 | .9479 | 4.0 |
| 4cube F1 | 66/96 = .6875 | .8672 | 1.0 |
| 4cube F4 | 74/96 = .7708 | .9089 | 4.0 |
| 5cube F1 | 40/96 = .4167 | .7687 | 1.0 |
| 5cube F4 | 41/96 = .4271 | .7708 | 4.0 |

**D_44 = +0.0246** [−0.0042, +0.0539].

## 2. Three-seed result — FROZEN

| Seed | Δ 3-cube | Δ 4-cube | Δ 5-cube | **Mean Δ** |
|---|--:|--:|--:|--:|
| 42 | +0.0428 | **−0.0286** | +0.0813 | **+0.0318** |
| 43 | +0.0046 | +0.0677 | +0.0438 | **+0.0387** |
| 44 | +0.0301 | +0.0417 | +0.0021 | **+0.0246** |

**N = 3 independently trained Flow models. Mean +0.0317, SD 0.0070, range
[+0.0246, +0.0387].**

**Classification: A — STRONG THREE-SEED REPLICATION.** All three aggregates are
positive and of comparable scale. Category A explicitly permits task-level
heterogeneity, which is large: the 4-cube effect is negative for seed 42 and
positive for the other two; 5-cube spans 0.0021–0.0813 (39x). **No object count
carries the effect consistently, and no seed shows a monotone increase with
object count.**

No t-test is reported: with N=3 the power is negligible. Episode-level bootstrap
CIs are checkpoint-level; episodes are never pooled across seeds.

## 3. Exact DLP particle feature semantics

From `dlp_utils.get_dlp_rep:261-267` (encoder) and `get_recon_from_dlps:296-303`
(decoder), which agree:

| Dims | Meaning | Raw range | Raw std |
|---|---|--:|--:|
| 0:2 | `pixel_xy` — position | 1.90 / 2.40 | 0.269 / 0.369 |
| 2:4 | `mu_scale` | 1.88 / 1.23 | 0.276 / 0.260 |
| 4 | `mu_depth` | 5.40 | 0.643 |
| 5:9 | `mu_features` — appearance | **12.97–24.84** | 0.73–1.10 |
| 9 | `obj_on` — transparency | **0.076** | 0.0091 |

**Ranges span ~325x.** A raw 10-D Euclidean distance would be dominated by the
appearance channels, not position. Cross-dimension distance is therefore *not*
meaningful without per-dimension normalization.

## 4. Proposed metric

- **Matching on position only** (dims 0:2) — the one channel with common
  geometric meaning; matching on raw 10-D would be appearance-dominated.
- **Per-block reporting in z-scored units** (pos / scale / depth / vis / transp),
  never one summed heterogeneous number.
- **Chamfer on position** alongside Hungarian as an assumption-light check.
- Target: DLP encoding of the state the environment **actually reached** after
  executing the policy's own first action, with a **copy-the-current-state
  baseline**.

## 5. Metric sanity validation (real DLP states)

| Test | pos_z | chamfer |
|---|--:|--:|
| identity | 0.0000 | 0.00000 |
| **random particle permutation** | **0.0000** | **0.00000** |
| jitter ε=0.001 | 0.0026 | 0.00125 |
| jitter ε=0.005 | 0.0127 | 0.00603 |
| jitter ε=0.02 | 0.0476 | 0.02069 |
| jitter ε=0.08 | 0.1711 | 0.06225 |
| jitter ε=0.30 | 0.4846 | 0.14088 |
| appearance-only jitter | **0.0000** | 0.00000 |
| collapse to centroid | 0.7942 | 0.24652 |

**USABLE**: exactly permutation-invariant, strictly monotone in displacement,
block-separating, detects collapse.

## 6. Flow vs Gaussian objective comparison

16 episodes x 6 steps, frozen 3-cube set `35144910`, existing checkpoints only.

| Model | chamfer (pos) | pos_z | vis_z | copy baseline |
|---|--:|--:|--:|--:|
| **Gaussian @100** | **0.03981** | 0.1902 | 0.3281 | 0.07796 |
| Flow @4 s42 | 0.06517 | 0.2715 | 0.3656 | 0.07587 |
| Flow @4 s43 | 0.06018 | 0.2608 | 0.3506 | 0.07476 |
| Flow @4 s44 | 0.06313 | 0.2737 | 0.3748 | 0.07985 |
| Flow @1 s42 | 0.22808 | 0.7249 | 0.6570 | 0.07687 |
| Flow @1 s43 | 0.22553 | 0.7171 | 0.5748 | 0.07657 |
| Flow @1 s44 | 0.23260 | 0.7282 | 0.5714 | 0.07985 |

- **Flow@4 is 1.58x worse than Gaussian** (0.0628 vs 0.0398).
- **Flow@1 is 5.75x worse than Gaussian** and **3.64x worse than Flow@4**.
- Seed-to-seed CV of Flow@4 is **3.3%**; the Gaussian gap is **11.2x the seed SD**.
- **Copy baseline 0.0778**: Gaussian and Flow@4 beat it; **Flow@1 (0.229) is ~3x
  worse than simply echoing the current state** — at 1 NFE the model is not
  predicting forward dynamics at all in latent space.

## 7. Outcome: **A — OBJECTIVE DEGRADATION CONFIRMED**

Flow is consistently worse than Gaussian on a validated permutation-invariant
metric, across three independent training seeds, with a gap an order of
magnitude larger than seed variance. The visual impression was not a decoder
artifact.

## 8. Is poor imagination objectively established?

**Yes for the latent state target**, with two documented caveats:

1. **Multimodality (Phase 5).** The comparison is against the *one* realized
   future produced by the policy's own action, so it is a genuine
   predicted-vs-realized one-step check rather than distance to an arbitrary
   demonstration. Still, a valid alternative future would be penalized; the copy
   baseline partially controls for this, and Gaussian/Flow@4 beating it shows
   the metric rewards real dynamics prediction.
2. It measures **one-step** prediction. Longer-horizon imagination is untested.

## 9. Action/state gradient cosine similarity

10 batches x 32, frozen seed-42 EMA, no optimizer step:

```
cos(∇θ L_action, ∇θ L_state)
  mean -0.0447   median -0.0542   min -0.2810   max +0.1357
  negative (conflicting) in 70% of batches
```

**The two objectives mildly but consistently conflict.** Reweighting them is not
a neutral rescaling — it trades one against the other.

## 10. What λ_a = λ_s = 0.5 would actually do

| Objective | ‖∇θ L_a‖ | ‖∇θ L_s‖ | ratio a/s |
|---|--:|--:|--:|
| **Current** | 0.0913 | 0.0694 | **1.32** |
| **50/50 block means** | **1.4083** | 0.0351 | **40.18** |

**Action gradients amplify 15.4x, state shrinks 0.51x — a 30.5x shift in the
ratio.** Your objection is confirmed quantitatively: 50/50 would make an
already action-leaning optimization overwhelmingly action-dominated, in the
*opposite* direction from the measured state deficit. A global scalar cannot fix
this because it preserves the internal ratio.

**λ that removes dimensionality dependence while preserving the current gradient
ratio: λ_a = 0.0317, λ_s = 0.9683.**

## 11. Does Tal's loss-balancing hypothesis remain plausible?

**Partially — but the evidence has reframed it.**

Supported: there *is* a real, objectively measured state-prediction deficit
(§6-7), so "the state objective is not producing good state predictions" is
now established rather than visual.

Not supported: the *dimensional* framing. State already receives 99.4% of the
scalar loss; adding weight cannot be the fix. The action block, at 0.6% of the
loss, already produces **1.32x** the state gradient. The deficit coexists with
state dominating the loss — so the binding constraint is not the action/state
scalar split.

**The strongest competing explanation is now integration error, not loss
balance.** Flow@1 is 3.64x worse than Flow@4 *from the same weights*, with no
loss difference whatsoever. That single contrast shows most of the imagination
gap is a **sampler/NFE property**, not an objective property. Gaussian@100 uses
100 refinement steps; Flow@4 uses 4.

## 12. Recommended modified loss (if pursued)

**Primary: dimensionality-decoupled block loss at the gradient-preserving λ.**

```
L = λ_a · mean_{active action coords}(|v̂ - v*|) + λ_s · mean_{active state coords}(|v̂ - v*|)
     with λ_a = 0.0317, λ_s = 0.9683   (measured, ratio-preserving)
```

Justification: it removes the accidental dependence on 3-vs-480 coordinate
counts, making λ a semantic knob, **while holding the measured action/state
gradient ratio at its current 1.32** so the change is not confounded with a 30x
optimization shift. λ is set from training-distribution gradient statistics, never
from simulator success.

**Honest caveat:** at these λ the objective is *near-equivalent* to the current
one by construction. It is a clean scientific control — "does decoupling
dimensionality matter, holding gradient balance fixed?" — not a fix expected to
close the imagination gap.

**Secondary:** state-only upweighting is **not** recommended; state already
dominates the loss.

## 13. Cheap training experiment

Baseline C0 vs the ratio-preserving C1, **25,000 steps**, identical init seed,
identical data order (`dataloader_seed`), identical LR/architecture/optimizer.
Metrics: the validated chamfer/pos_z at Flow@1 and Flow@4, plus held-out
first-action L1. Rejection: no improvement in latent metric, or action error
degrades.

## 14. Compute

| Item | GPU-h |
|---|--:|
| Diagnostics (done) | ~0.5 |
| C0 + C1 @ 25k steps | **3.3** |
| Latent + action eval | 0.3 |
| **Total if approved** | **~3.6** |

## 15. One next action

**Do not run the loss experiment yet.** The measured evidence points at the
sampler, not the objective: Flow@1 → Flow@4 changes imagination error **3.64x
with an identical loss**, while the entire action/state rebalancing available to
us is constrained to hold gradients near their current ratio.

**Recommended next action: measure Flow imagination error at NFE 1/2/4/8/16 on
the existing checkpoints (~0.4 GPU-h, no training).** If the latent error keeps
falling toward the Gaussian level with more steps, the imagination gap is an
integration-budget property and a loss change is the wrong intervention — which
would save the 3.3 GPU-h and redirect the question to "how many steps does
*state* prediction need, versus the 2 that *control* needs?"

That contrast — control saturating at 2 NFE while state prediction needs far
more — would itself be a more interesting finding than a loss reweighting.
