# Protocol: is future-state prediction quality causally important for control?

Status: **predeclared, NOT run.** No training is authorized by this document.

## 1. The observation this exists to explain

From `experiments/isaacgym_flow_diagnosis.md` §10, measured on identical
current/goal pairs through the same pretrained DLP decoder:

> **Flow has worse decoded future-state quality but better executed control.**

Concretely:

| | real encoded | Gaussian (100 NFE) | Flow (4 NFE) |
|---|--:|--:|--:|
| Visual-feature dispersion | 0.566 | 0.716 | **0.417** |
| Particle spread | 0.480 | 0.518 | **0.403** |
| Nearest-neighbour distance | 0.0748 | 0.0508 | **0.0483** |
| **Isaac Gym success (96 ep)** | — | **86.5%** | **95.8%** |

Decoded Flow futures show smeared, duplicated, colour-blended cube clusters
where Gaussian keeps three distinct cubes. Yet Flow wins the task, significantly
(paired McNemar p = 0.0225).

## 2. What this does NOT license

Three readings are explicitly **not** supported and must not be asserted:

1. **"DLP is broken."** It is not. Reconstruction MAE is 1.8/255 on real frames;
   cubes, colours and arm geometry all survive the round trip.
2. **"DLP is irrelevant."** EC-Diffuser's own ablation (arXiv:2412.18907,
   Table 4) shows that *removing* generated latent states collapses 2-cube from
   0.917 to 0.423 and 3-cube from 0.894 to 0.529. Joint state generation is
   load-bearing in the published setting.
3. **"Imagination quality does not matter."** Our measurement compares two
   *different models* that differ in objective, solver, and NFE simultaneously.
   It cannot isolate the contribution of state-prediction quality.

The honest statement is narrower: **within this pair of models, decoded
future-state fidelity and control performance are not positively coupled.**
Something is generating good actions despite visibly degraded imagined states.

## 3. Competing explanations

| # | Hypothesis | Prediction |
|---|---|---|
| H1 | State prediction is an auxiliary task. Its *presence* helps representation learning; its *fidelity* is not what drives the action. | Degrading state quality further, at matched action loss, leaves control intact |
| H2 | Fidelity matters, but Flow is already above the threshold needed. The decoded blur is cosmetic in pixel space and the task-relevant content (cube centroids) is accurate. | Centroid error is small even though appearance error is large |
| H3 | The decoder exaggerates the defect. Flow's particles are valid but off the DLP prior's manifold, so the decoder renders them poorly while the latent still carries correct information. | A latent-space probe recovers cube positions accurately from Flow particles |
| H4 | Fidelity does matter and Flow wins *despite* it, for an unrelated reason (e.g. better action-head conditioning at low NFE). | Fixing Flow's state quality would improve it further |

H2 and H3 are cheap to separate and should be tested **before** any training.

## 4. Stage A — no training, ~1 GPU-h

Decide between H2/H3 and the rest by asking what the imagined particles actually
encode, rather than how they look.

**A1. Task-relevant content probe.** For each generated future state, extract
cube centroids from the particle set (high-transparency particles clustered by
position) and compare against the *realized* cube positions after executing the
action. Report centroid error in metres for Gaussian and Flow separately.

- If Flow's centroid error is comparable to Gaussian's, **H2/H3 hold**: the
  imagination carries correct task content and the visual degradation is
  cosmetic. The finding then becomes "decoder fidelity is not a proxy for
  task-relevant state accuracy", and no further ablation is needed.
- If Flow's centroid error is materially worse, the imagination is genuinely
  wrong about where cubes will be, and Stage B is justified.

**A2. Predicted-vs-realized dynamics.** Already scripted in
`experiments/scripts/predicted_vs_realized.py`: symmetric Chamfer distance
between the predicted next DLP state and the encoded realized frame, with
"copy the current state" as a baseline. An arm whose prediction error is not
below the copy baseline is not predicting dynamics at all.

## 5. Stage B — controlled training, only if Stage A says the imagination is wrong

The clean design varies **one** thing: how much the objective weights
state prediction relative to action prediction.

| Arm | State loss weight | Action loss weight |
|---|---|---|
| B0 | 1.0 (current) | current |
| B1 | 0.0 — actions only | current |
| B2 | boosted | current |

`ConditionalFlowMatching` already exposes `obs_only` / `action_only` and a
`loss_weight_matrix`, so B1 is reachable without architectural change. B1 is the
direct analogue of the paper's "no state generation" ablation, reproduced in
*our* pipeline where we can also measure imagination quality — the paper reports
only the success drop.

Prediction: if the paper's ablation reproduces (B1 much worse at 3 cubes) while
B1's *action* loss matches B0's, then state generation helps through
representation learning rather than through fidelity of the imagined rollout,
which supports **H1** and explains our observation without contradicting the
paper.

**Cost, from measured throughput** (0.2343 s/step): ~32.5 GPU-h per arm at 500k
steps. Two arms is ~65 GPU-h. **This exceeds the 4 GPU-h gate by ~8x and
requires explicit approval.** It should not be run on the strength of a visual
impression, which is exactly why Stage A comes first.

## 6. Ordering and gating

1. Stage A1 + A2 (~1 GPU-h, no training).
2. Only if A shows genuinely incorrect imagined dynamics, request approval for
   Stage B.
3. Any Stage B run must use >= 96 episodes per arm, per the evaluation noise
   floor established in `experiments/isaacgym_flow_diagnosis.md` §1c.3.

## 7. Explicitly excluded

No VP or standardization training. No OGBench. No MeanFlow or Shortcut. No new
algorithm. This protocol tests an existing observation; it does not propose a
method.
