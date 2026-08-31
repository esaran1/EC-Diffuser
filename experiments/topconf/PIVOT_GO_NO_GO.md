# PIVOT GO / NO-GO

2026-08-30 · commit `36256f8` · Deliverable §30
Compute this phase: **0.11 GPU-h** (3 rollouts × 96 eps + offline queries), inside
the 0.25 budget. No training. No new benchmark.

---

## 1. Executive verdict

**CONDITIONAL YES — but for a narrower paper than hoped, and only if cross-policy
breadth is added. Route D (B+C combined). The method route (A) is NO-GO.**

The compression-induced covariate-shift hypothesis is **partially supported**:
the occupancy shift and the disagreement amplification are both real and sizable.
But after controlling for action magnitude, **only ~28% of the effect is
NFE1-specific**; ~72% is a generic closed-loop-vs-replay gap shared by *every*
arm including NFE4. That is not enough to carry a method paper, and I will not
present the uncontrolled 3.2× number as if it were the whole story.

## 2. What Flash-WAM owns (2606.05254)

Modality-aware step distillation on joint video+action: `L = Lv + λa La` with a
linear-gradient-scaling consistency function for the action stream and a
variance-preserving one for video. Both streams collapsed to one step; no FM
retained. Also ran the mirror-image ablation (Eq. 15: consistency on video +
plain FM regularizer on action) and it **lost 24 points**. Covariate shift,
self-induced states, DAgger: entirely absent.

## 3. What SANTS owns (2605.27947)

All four of the observations we might have claimed, verbatim: "the best action
condition is not necessarily the fully denoised video"; refinement gains
"saturate or even reverse"; the optimum is "state-dependent"; scheduling is "for
downstream action quality rather than intermediate video fidelity."

**Its limits are our opening:** it varies *video* depth only (action denoising
frozen), its scan is **open-loop on a fixed demonstration dataset** (500 segments,
no rollout), and it explains the reversal by physical implausibility of late
frames — not by any distributional mechanism.

## 4. What Efficient-WAM owns (2606.10040)

Asymmetric per-modality budgets — but in the **opposite direction**: it asserts
"action generation requires precise multi-step sampling" while "future video only
needs coarse dynamic context" (video ~2 steps, action 5-10). It provides no
evidence for that assumption. Our measurement (action imitation best at NFE2,
state improving to NFE~32) contradicts the asserted direction.

## 5. What GPUSimBench owns (2607.13059)

Simulator-property benchmarking only: GPU-batched nondeterminism, four
stochasticity regimes, simulator-choice guidance. Quantifies nondeterminism as
**physical state divergence (EMD on cube positions, cm)** — never as task
success-rate spread. No policy-comparison statistics, no evaluation protocol, no
CI calibration, no replication allocation. RoboDojo (2607.04434) standardizes
evaluation infrastructure (50 episodes × 3 seeds, mean±sd) but reports no CIs and
never discusses simulator nondeterminism.

## 6. What is genuinely left for us

1. **Closed-loop mechanism for a phenomenon others observe only open-loop.**
   SANTS sees non-monotonic depth effects in open loop; FLASH (2605.15492)
   reports NFE1 (80%) *beating* NFE2 (54%) closed-loop with no mechanism at all.
2. **The inverted asymmetry** against Efficient-WAM's stated assumption.
3. **Nondeterminism-aware policy-comparison statistics** — bridging GPUSimBench
   (simulator-level physical divergence) to N-SCORE (2603.13616, which assumes
   **i.i.d. evaluation data** and does not address simulator nondeterminism).

## 7. Self-induced-state protocol

Seed 42 (largest NFE4−NFE1 gap, +9.03 pp), frozen `replicate0_n96`, CRN enabled.
Rolled out NFE1/2/4 closed-loop, caching **raw physical state** (3 cube xyz + EEF
xyz = 12 dims) every 5th decision → 1,920 states per arm. Then queried π₁, π₂, π₄
offline at every cached state with **shared Flow noise per state**. No simulator
stepping in the query. **No demo-action targets at self-induced states** (§4
respected): we measure disagreement, never correctness.

## 8. Occupancy-shift result

kNN distance (k=5) to NFE4-visited support, raw physical features:

| arm | overall | early (<20) | mid (20-60) | late (≥60) |
|---|--:|--:|--:|--:|
| **NFE1** | **0.10028** | 0.06411 | 0.10707 | **0.11158** |
| NFE2 | 0.05843 | 0.04937 | 0.06057 | 0.06083 |
| NFE4 | 0.02214 | 0.03701 | 0.02577 | 0.01108 |

**NFE1 sits 4.5× farther from the NFE4 support than NFE4 does**, NFE2 is cleanly
intermediate, and the NFE1 gap **grows over the episode** (0.064 → 0.112) while
NFE4's *shrinks* (0.037 → 0.011). The monotone NFE1 > NFE2 > NFE4 ordering
matches the control ordering.

## 9. Disagreement on replay states

`D₁₂ = 0.01145`, `D₁₄ = 0.02048`, `D₂₄ = 0.01514` (1,000 replay states).
Small — consistent with the earlier finding that the one-step action endpoint is
already accurate on demonstration states.

## 10. Disagreement on self-induced states

| distribution | D₁₂ | D₁₄ | D₂₄ |
|---|--:|--:|--:|
| demo | 0.01145 | 0.02048 | 0.01514 |
| **self1** | **0.03645** | **0.06642** | **0.04459** |
| self2 | 0.02847 | 0.05000 | 0.03370 |
| self4 | 0.02500 | 0.04473 | 0.03095 |

**D_self1 = 3.2× D_demo, and D_self1 > D_self2 > D_self4** — the §7 signature.

### The control that matters, and it cuts the claim down

Self-induced states carry larger actions (‖a‖ 0.282 vs 0.211), so part of the
gap is scale. Normalizing (`D₁₂/‖a‖`): demo 0.0543, self4 0.1012, self2 0.1095,
**self1 0.1196**.

| component | share of the demo→self1 gap |
|---|--:|
| generic closed-loop vs replay (demo→self4) | **72%** |
| **NFE1-specific (self4→self1)** | **28%** |

The effect survives (2.2× normalized), and the NFE1 ordering is intact — but
**most of it is not about NFE1 at all.** Any honest version of this paper leads
with the 28%, not the 3.2×.

### A second complication

Disagreement is **highest early** (0.070 at step<20) and *falls* late (0.023),
while occupancy shift *grows* late. So the simple feedback story — drift →
larger error → more drift — is **not** what the data show. Disagreement is
largest where the policy is farthest from the goal, not where it has drifted most.

## 11. Relation to failures

Not established. The per-episode success labels are cached, but with one seed and
n=96 the failure-localized analysis (§8) would be underpowered given a 4.4 pp
resolution at R=3. **Not attempted rather than reported weakly.**

## 12. Mechanism classification: **A/B boundary, reported as B**

- Occupancy shift: **present and ordered** (§8) → supports A.
- Disagreement amplification on self-states: **present and ordered** (§10) → A.
- But 72% of the amplification is not NFE-specific, and the temporal pattern
  contradicts the compounding narrative (§10) → pulls to **B**.

**Classification B — policy disagreement with occupancy shift, but the
NFE-specific component is a minority of the effect.** I decline to claim A.

## 13. Method novelty audit (would apply if A)

Nothing kills the idea, but two works wound it. **FA-OPD (2605.27095, ICML 2026)**
already owns on-policy distillation for robot policies — "the student is
supervised on the states it actually visits" — though its student is an MLP, not
a step-compressed sampler. **Trajectory-Consistent Flow Matching (2605.08511)**
owns drift-to-off-path-states, but *within one sampling trajectory*, not across
control steps. **Diff-DAgger** owns on-policy correction but never touches
compression. **SeFA-Policy** already names execution-time accumulation for
one-step policies.

Given classification B, the method would be "DAgger for few-step policies"
targeting an effect that is 28% NFE-specific. **NO-GO on Route A.**

## 14. Evaluation-paper novelty audit

Cleanest ground. GPUSimBench measures simulator divergence in cm, not
success-rate resolution; RoboDojo standardizes protocols without statistics;
N-SCORE does sequential policy comparison **assuming i.i.d. data** and does not
model simulator nondeterminism. Our contribution — success-rate resolution
(7.6 pp at R=1 → 2.7 pp at R=8), hierarchical episode/realization variance
decomposition, and adaptive replication — is unoccupied. **Adaptive allocation
(§20) still needs its own sequential-design novelty audit.**

## 15. Required cross-policy breadth

Mandatory for Route B/D. Cheapest credible axis: **EC-Diffuser Gaussian
(100 steps) vs Flow** — same repo, same evaluator, no install cost. That is one
family comparison, probably not enough alone; a public Diffusion Policy or
Consistency Policy checkpoint would be needed for a second. **Not yet scoped.**

## 16. Required cross-task breadth

3/4/5-cube exists in-repo (zero-shot *policy* compositional generalization; DLP
saw 6 cubes). Must use fixed-H controls. At R=3, ~0.7 GPU-h per NFE pair per
cube count.

## 17. Compute roadmap

| item | GPU-h |
|---|--:|
| replicate §8/§10 on seeds 43, 44 | 0.25 |
| Gaussian arm for cross-family | ~0.3 |
| adaptive-replication simulation (offline, on cached data) | ~0 |
| 4/5-cube calibrated pairs | ~2.8 |
| **total to a defensible Route D** | **~3.4** |

## 18. Best final paper thesis

> Fast generative robot policies are evaluated with offline imitation metrics and
> single-rollout simulation, and both mislead. In a joint entity-state/action
> flow policy, truncating the sampler produces *opposite* offline trends across
> semantic blocks; the closed-loop effect appears not in either offline metric but
> in the state distribution the compressed policy induces; and contact-rich GPU
> simulation contributes enough run-to-run variance to reverse small comparisons.
> We give a calibrated protocol separating model, rollout-distribution, and
> simulator uncertainty.

## 19. Five reviewer kill shots

1. *"SANTS already showed non-monotonic depth-vs-action-utility."* — True; ours is
   closed-loop and mechanistic, theirs open-loop and descriptive. Thin margin.
2. *"72% of your covariate shift isn't NFE-specific — so what is the finding?"*
   **The strongest objection. We currently answer it only by reporting it.**
3. *"GPUSimBench owns simulator nondeterminism."* — They measure cm of divergence;
   we measure success-rate resolution and inferential validity. Defensible.
4. *"One environment, one policy family."* — Currently fatal. §15/§16 required.
5. *"N=3 checkpoints, and your headline control effect had 1 of 3 CIs excluding
   zero."* — Honest answer: the calibration is the contribution; the NFE effect is
   directional.

## 20. GO / PIVOT / STOP

**PIVOT to Route D (evaluation + empirical), CONDITIONAL on cross-policy breadth.**

- **Route A (method): NO-GO.** Classification B, 28% NFE-specific, and FA-OPD
  owns on-policy distillation.
- **Route D: conditional GO** at ~3.4 GPU-h, contingent on §15 producing at least
  one credible second policy family. If it cannot, the honest outcome is
  **Route E — internal report**, not a padded single-environment paper.

**Next action requires approval: scope §15 (POLICY_BREADTH_FEASIBILITY.md), no
GPU.** I recommend deciding breadth feasibility *before* spending the 3.4 GPU-h,
because without it Route D is a case study.
