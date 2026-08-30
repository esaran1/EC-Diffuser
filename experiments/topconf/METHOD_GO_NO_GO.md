# METHOD GO / NO-GO

Date 2026-08-30 · commit `0799c93` · Deliverable §52

**DECISION: NO-GO on the projected-action-shortcut method as proposed.
PIVOT to the phenomenon + evaluation contribution, pending one cheap
falsification experiment (§I).**

---

## A. Strongest empirical phenomenon

Verified, replicated on **three independently trained checkpoints** (Level 1,
arm-neutral recorded replay; no model-authored targets):

| NFE | action error | state error |
|--:|--:|--:|
| 1 | 0.01785 | 0.21032 |
| 2 | **0.01717** (min) | 0.10761 |
| 4 | 0.02212 | 0.04881 |
| 32 | 0.04136 | **0.02493** (min) |
| 512 | 0.04592 | 0.02566 |

**The two semantic blocks of one joint generative tensor have optima 16× apart**
(action at NFE 2, state at NFE 32), and the action *degrades* monotonically past
2 while the state improves 8.4×. Closed-loop (Level 3, calibrated R=3, three
checkpoints): NFE1 is ~6 pp worse than NFE4; NFE2 vs NFE4 is ~0.

This phenomenon is real, replicated, and — per the audit — **not published for a
joint state-action policy with per-block supervised metrics**.

## B. Why existing methods do not already solve it

They largely do, or make it moot:

- **Flash-WAM (2606.05254)** already differentiates the *objective family* per
  modality on a joint video-action model: `L = Lv + λa La`, with a
  linear-gradient-scaling consistency function for actions and a
  variance-preserving one for video.
- **Efficient-WAM (2606.10040)** already allocates per-branch step budgets.
- **DreamZero-Flash (2602.15922)** already uses per-modality time distributions
  to enable 1-step action inference.
- **Dual-Stream VLA (2510.27607)** already states "vision needs more denoising
  steps than low-dimensional actions."

## C. Closest five prior works

1. **Flash-WAM** 2606.05254 — per-modality consistency functions, joint
   video+action, distilled. **Closest.**
2. **Efficient-WAM** 2606.10040 — per-branch NFE budgets. Notably assigns the
   **action** branch *more* steps (5-10) and video *fewer* (2) — the opposite of
   our asymmetry.
3. **DreamZero-Flash** 2602.15922 — per-modality noise-time distributions.
4. **Dual-Stream Diffusion VLA** 2510.27607 — earliest clean statement of the
   step-count asymmetry.
5. **SnapFlow** 2604.05656 — mixes FM and consistency objectives, but splits by
   **sample** (α=0.5), not by output subspace.

Resolved non-threats (I read both): **S-VAM 2603.16195** — "semantic" means
*scene semantics* (VFM features); its action expert is a **separate** DiT trained
with plain DDPM `L_A = E‖ε − ε_φ(a_j, F_agg, E, j)‖²`, not a joint tensor.
**MWM 2603.07799** — image-goal *navigation*; ICSD distills visual rollout, no
action subspace.

## D. Proposed method in one sentence

Train only the action projection of a joint entity-state/action Flow policy for
one-step consistency while retaining ordinary Flow Matching on the entity-state
projection, so that world-prediction supervision is preserved as a
representation-shaping auxiliary.

## E. Exact mathematical objective (as it would have been)

`L = L_FM(P_s) + λ_a · L_shortcut(P_a)`, with `P_a` the action token (index 0)
and `P_s` the 48 particle tokens; shortcut supervised by a frozen-EMA 2-step
teacher; step-size conditioning injected only into the action token via the
existing `_time_embedding(time, interval)` hook.

## F. What is genuinely novel

- The conjunction: *retaining* full multi-step FM on the state branch while
  compressing only the action branch (Flash-WAM collapses both).
- Doing it in an **entity-centric / object-factored** generator.
- **Compositional (4/5-cube) generalization evaluation of a low-step policy** —
  no fast-policy paper in the audit does this.

## G. What is not novel

- That different modalities need different step budgets (4 papers).
- Per-modality objective differentiation in a joint model (Flash-WAM).
- Consistency/shortcut/MeanFlow for fast manipulation (≥12 papers).
- Applying a subset objective by frequency band, transformer layer, sample, or
  timestep (FreqPolicy, BAC, SnapFlow, X-WAM).

## H. Mechanistic evidence — **this is where it fails**

Three findings, all Level 1, all three checkpoints:

1. **The one-step action endpoint is already accurate.** Replay action error is
   0.01785 (NFE1) vs 0.01717 (NFE2) — a 0.0007 gap, and on **seed 42 NFE1 is
   better** (0.01668 vs 0.01785). There is essentially no one-step action deficit
   to compress away.
2. **The action distribution is essentially unchanged.** 8 noises/condition
   (seed 44): pairwise spread 0.00837 (NFE1) vs 0.00713 (NFE2); mean action
   magnitude 0.20726 vs 0.20631. No mode collapse, no meaningful diversity gap.
3. **The action velocity barely reads the in-flight state** (0.78% response to a
   unit-scale state perturbation; 0.02% at 0.1). It is driven by the *clamped
   conditioning*, so there is little state→action refinement to amortize either.

I also **withdraw** my own earlier reading that `B(action-only) ≡ A(full 2-step)`
demonstrated action self-refinement: both consume the same `v1` from one model
call, so their action coordinates agree by arithmetic. That was a tautology of
the probe design, not evidence.

**Conclusion: the method targets a quantity that is not broken.** The ~6 pp
closed-loop NFE1 penalty is *not* explained by one-step action-endpoint error or
by action diversity on in-distribution replay.

### Direct contrary evidence from prior work

Flash-WAM ran the **mirror image** of our proposal as an ablation (their Eq. 15):
`L = Lv_LCM + λ_r · L_a_reg` — consistency on video, a plain flow-matching
regularizer on action. It fell **24 points below plain video-only LCM**, and they
conclude *"an auxiliary loss cannot substitute for distilling the action stream
when the action NFE budget is tight."*

That is not proof our direction fails — ours is the transpose (consistency on
*action*, FM retained on *state*) — but it is published evidence that mixing a
consistency objective on one stream with a plain-FM objective on the other is
fragile in exactly this setting. A reviewer will raise it.

## I. Cheapest falsification experiment

**Locate the ~6 pp NFE1 control penalty before building anything.** Hypothesis
H-A/H-B: the penalty arises off the replay distribution — on the self-induced
states a policy actually visits — not on recorded states.

Test, no training, ~0.15 GPU-h: roll out NFE2 closed-loop on the frozen
96 episodes, cache the *visited* states, then score NFE1 and NFE2 action/state
endpoints **at those visited states** against the NFE2 rollout's own continuation.
If the NFE1 action deficit appears there but not on recorded replay, the target
is off-distribution robustness (a different method); if it does not appear, the
+6 pp is likely evaluator noise (§1.4 had only 1 of 3 CIs excluding zero) and the
low-NFE story is *already complete* at NFE2.

**Either outcome is publishable and neither requires the proposed method.**

## J. Expected compute

Falsification: **~0.15 GPU-h**. The abandoned method path would have been
~30-60 GPU-h (post-training × 3 seeds + calibrated control + 4/5-cube).

## K. Reviewer kill shots

1. *"This is Flash-WAM with the mask inverted."* — Partly true; the differentiator
   is retention of the FM state branch, which we have not yet shown matters.
2. *"Flash-WAM already showed the FM-regularizer variant loses 24 points."*
3. *"Efficient-WAM gives the action branch MORE steps — your asymmetry is
   contradicted by prior work."* Our own data supports the opposite direction,
   but the closed-loop evidence (±5 pp margin, R=3) cannot yet adjudicate.
4. *"Your own diagnostics show the 1-step action is already accurate — what is
   the method fixing?"* **We currently have no answer. This is the kill shot.**
5. *"One environment, one task family."*

## L. Decision

**NO-GO** on projected action shortcut. Reasons, in order:

1. The motivating deficit does not exist in our own measurements (§H).
2. The governing insight is prior art (§B, §G).
3. Published contrary evidence for the closest variant (Flash-WAM Eq. 15).

**PIVOT** — the defensible contributions on current evidence are:

- **(1) The phenomenon:** semantic inference asymmetry in a joint state-action
  policy, with per-block supervised metrics, replicated on 3 checkpoints. Not
  published in this form.
- **(2) The evaluation methodology:** Isaac Gym GPU physics is nondeterministic;
  we measured the noise floor (R=1 ≈ 7.6 pp, R=3 ≈ 4.4, R=5 ≈ 3.4, R=8 ≈ 2.7),
  showed contact-triggered divergence to 23× the success threshold, and specified
  a hierarchical protocol. This retroactively invalidates sub-10 pp
  single-realization claims — **including several of our own** — and is directly
  useful to the fast-policy literature, none of which reports such calibration.
- **(3) The negative result:** offline action-imitation error does not predict
  closed-loop control (anti-correlated over NFE 1→8; +87% offline action
  degradation from NFE4→NFE32 with zero control cost).

Contribution (2) alone plausibly clears an RSS/CoRL evaluation-and-analysis bar,
and (1)+(2)+(3) is a coherent analysis paper. It is **not** currently a method
paper, and I will not manufacture one.

**No expensive training is started.** Next action is §I.
