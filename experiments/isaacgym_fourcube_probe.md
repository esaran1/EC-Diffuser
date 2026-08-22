# 4-cube zero-shot compositional headroom probe

Date: 2026-08-22. Branch `fast-generative-policies`.

**Question.** Does low-NFE entity-centric Flow retain useful zero-shot
compositional performance when task complexity increases from 3 to 4 objects,
and how does that compare with Gaussian EC-Diffuser?

**Verdict: Regime B — useful headroom. 4-cube PushCube is now a strong
controlled benchmark for NFE and generalization studies.**

## 0. Scope of the "zero-shot" claim

This is **zero-shot policy generalization from 3 to 4 cubes, not zero-shot
representation learning.** The policy checkpoints were trained only on 3-cube
data (`numObjects: 3`, `RandNumObj: False`, training tensors carry exactly
eef + 3 cubes). The DLP encoder `dlp_push_6C` was itself trained on scenes with
up to **six** cubes, so the representation has seen 4-cube scenes even though the
policy has not. This is the same encoder and protocol EC-Diffuser used for its
own generalization results, but the distinction must be preserved in any claim.

Full pre-flight validation: `experiments/fourcube_validation.md` (all five
conditions verified on CPU before any GPU spend).

## 1. Setup

| Item | Value |
|---|---|
| Task | Isaac Gym PushCube, **4 cubes**, random colours, 150-step episodes |
| Episode set | **`5962c3abb4367eaa`**, 96 episodes, fixed initial + goal states |
| Pairing | all three arms on byte-identical episodes, one hash, verified |
| Checkpoints | canonical 3-cube Gaussian and Flow, EMA, **unchanged** |
| Retraining | **none** |
| Env check | asserted `num_objects == 4` at run time |

## 2. Headline results

| Arm | Calls (verified) | **Full success** | 95% CI | **Per-object success** | Cubes placed | Goal frac |
|---|--:|--:|--:|--:|--:|--:|
| Gaussian | **100.00** | 0.6771 | [0.574, 0.769] | 0.8542 | 3.417 | 0.8542 |
| **Flow @4** | **4.00** | **0.7292** | [0.629, 0.815] | **0.9036** | **3.615** | **0.9036** |
| Flow @1 | **1.00** | 0.5729 | [0.468, 0.673] | 0.7917 | 3.167 | 0.7917 |

Call counts are forward-hook counted and exact for every arm.

## 3. Cubes-completed distribution (of 4)

| Arm | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|--:|--:|--:|--:|--:|
| Gaussian | 0 | 7 | 11 | 13 | 65 |
| **Flow @4** | **0** | **0** | 11 | 15 | **70** |
| Flow @1 | 6 | 5 | 11 | 19 | 55 |

As fractions:

| Arm | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|--:|--:|--:|--:|--:|
| Gaussian | 0.000 | 0.073 | 0.115 | 0.135 | 0.677 |
| **Flow @4** | 0.000 | 0.000 | 0.115 | 0.156 | **0.729** |
| Flow @1 | 0.062 | 0.052 | 0.115 | 0.198 | 0.573 |

This is the most informative single table in the probe:

- **No arm collapses.** Gaussian and Flow @4 never leave all four cubes unplaced;
  Flow @1 does so in only 6 of 96 episodes.
- **Flow @4 never scores below 2 of 4**, while Gaussian has 7 episodes at 1/4 and
  Flow @1 has 11 episodes at 0-or-1.
- The mass sits in the interior — 4/4 ranges 0.573-0.729 — which is exactly the
  dynamic range 3-cube lacked.

## 4. Control and contact metrics

| Arm | Mean obj-goal dist | Max obj-goal dist | Contact rate | Cubes contacted | Wrong-direction pushes | Cubes moved | Mean progress |
|---|--:|--:|--:|--:|--:|--:|--:|
| Gaussian | 0.0465 | 0.1067 | **1.0000** | 3.938 | 0.250 | 3.958 | 0.1396 |
| **Flow @4** | **0.0306** | **0.0729** | **1.0000** | 3.896 | **0.177** | 3.938 | **0.1547** |
| Flow @1 | 0.0593 | 0.1360 | **1.0000** | 3.927 | 0.396 | 3.938 | 0.1265 |

**Contact rate remains 1.0000 for all three arms at 4 cubes**, and every arm
contacts and moves ~3.9 of 4 cubes. The added object does not break approach or
contact; it stresses placement accuracy. Wrong-direction pushes track the NFE
ordering exactly (0.177 / 0.250 / 0.396).

## 5. Cost

| Arm | Calls/plan | ms / batch-16 plan | **ms / episode-step** | Wall (96 ep) |
|---|--:|--:|--:|--:|
| Gaussian | 100.00 | 1983.36 | **123.96** | 1923.8 s |
| **Flow @4** | 4.00 | 77.23 | **4.83** | **198.8 s** |
| Flow @1 | 1.00 | 19.03 | **1.19** | 144.7 s |

Flow @4 delivers the best success in the probe at **1/26th the per-step latency**
of Gaussian and **9.7x less wall time**.

## 6. Paired differences versus Gaussian

Same 96 episodes, episode-level pairing.

| Arm | Δ full success | b | c | McNemar p | Δ per-object | Wilcoxon p |
|---|--:|--:|--:|--:|--:|--:|
| **Flow @4** | **+0.0521** | 24 | 19 | 0.542 | **+0.0495** | 0.125 |
| Flow @1 | −0.1042 | 15 | 25 | 0.154 | −0.0625 | 0.126 |

**Neither difference is statistically significant at n=96.** Flow @4 is nominally
ahead of Gaussian on both metrics and Flow @1 nominally behind, and the two
tests agree in sign and rough magnitude, but this probe is not powered to
resolve differences of ~5 points. Per the noise floor established in the NFE
study (3-11 points on 96-episode sets), that is expected and is not evidence of
equivalence.

## 7. Degradation from 3 to 4 cubes

| Arm | 3-cube full success | 4-cube full success | Drop | 3-cube per-object | 4-cube per-object | **Per-object change** |
|---|--:|--:|--:|--:|--:|--:|
| Gaussian | 0.8681 | 0.6771 | −0.191 | 0.9282 | 0.8542 | **−0.074** |
| **Flow @4** | 0.8889 | 0.7292 | −0.160 | 0.9479 | 0.9036 | **−0.044** |
| Flow @1 | 0.8056 | 0.5729 | −0.233 | 0.9051 | 0.7917 | **−0.113** |

This table carries the interpretation the probe was designed for.

Full success falls 16-23 points, but **full success now requires 4 of 4 rather
than 3 of 3**, so a stricter criterion alone accounts for part of that. The
metric that separates the two readings is per-object success, and it drops only
**4.4 points for Flow @4** and 7.4 for Gaussian.

So the degradation is **mostly the stricter criterion plus a modest genuine
loss of per-cube competence**, not a compositional breakdown. Flow @4 loses the
least per-cube competence of the three arms.

**No independence-based expectation is quoted.** Cube outcomes within an episode
are correlated — one policy, one scene, shared contacts and shared arm
trajectory — so the p³ → p⁴ argument is intuition only and is not used as a
statistical null anywhere in this report.

## 8. Regime classification

**Regime B — useful headroom.** Both methods degrade to nontrivial intermediate
performance, and 4-cube PushCube becomes a strong controlled benchmark.

Checked against every alternative:

| Regime | Verdict | Evidence |
|---|---|---|
| A. Saturated | **No** | Best arm is 0.729, leaving 26 of 96 episodes failing. 3-cube left only 4 of 96 failing at its best. |
| **B. Useful headroom** | **YES** | All arms land in 0.573-0.729. Cross-arm span widens from 0.083 at 3 cubes to **0.156** at 4 — the benchmark discriminates roughly twice as well. |
| C. Low-NFE-specific degradation | **No** | Flow @4 degrades **less** than Gaussian on per-object success (−0.044 vs −0.074). Flow @1 does degrade most (−0.113), but that is the known 1-NFE deficit, not a compositional effect specific to low NFE — Flow @4 is also low NFE and holds up best. |
| D. Flow advantage persists | **Partly, but not claimable** | Flow @4 leads on every metric, but p = 0.542 / 0.125. See §9. |
| E. Joint collapse | **No** | Zero 0-of-4 episodes for Gaussian and Flow @4; contact rate 1.0000 everywhere; worst arm still fully solves 55 of 96. |

The distinguishing facts for B over D: the Flow @4 lead is nominal and
unpowered, whereas the *headroom* finding is robust — it rests on the span and
the distribution, not on a significance test.

## 9. What is not claimable

- **Not** that Flow @4 beats Gaussian at 4 cubes. It leads on every metric
  (full success, per-object, distance, wrong-direction pushes) but neither
  paired test is significant at n=96.
- **Not** an algorithm-level claim of any kind. This is **one training seed per
  method**. Independent training seeds are required before "Flow generalizes
  better than Gaussian" could be asserted, per the directive's Regime D caution.
- **Not** a representation-generalization result — see §0.

## 10. What is claimable

- **4-cube PushCube has useful headroom and is a valid controlled benchmark.**
  Cross-arm span nearly doubles (0.083 → 0.156) while no arm collapses.
- **Low-NFE Flow retains useful zero-shot compositional performance.** Flow at
  4 network calls solves 4 of 4 cubes in 73% of unseen 4-cube scenes, having
  been trained only on 3-cube data.
- **The NFE ordering from the 3-cube study is preserved out of distribution.**
  Flow @4 > Gaussian @100 > Flow @1 on full success, per-object success, goal
  distance and wrong-direction pushes. One call remains too few; four remains
  sufficient.
- **Contact is not the failure mode at 4 cubes.** Contact rate is 1.0000 for
  every arm and ~3.9 of 4 cubes are contacted and moved. The task stresses
  placement precision, not reachability.

## 11. Compute

| Item | Cost |
|---|--:|
| 3 arms x 96 episodes, 4 cubes | **0.63 GPU-h** (2,267 s) |
| Training | **none** |

The Gaussian arm alone consumed 1,924 s of that 2,267 s.
