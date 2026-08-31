# Paper-level roadmap

2026-08-31 · commit `56de3d4` · **Zero GPU used in producing this document.**
All new numbers below are re-analyses of cached artifacts.

---

# 1. Route decision: **D2** (evaluation study), with D1 as the motivating case

| route | novelty | coherence | breadth | burden | reviewer risk | venue fit | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| **D1** generative-inference study | 2/5 | 4/5 | 3/5 | med | **high** | ICLR/NeurIPS | **reject as primary** |
| **D2** policy-evaluation study | 4/5 | 5/5 | 4/5 | **low** | med | **RSS/CoRL** | **PRIMARY** |
| D3 combined "measurement stack" | 3/5 | **2/5** | 4/5 | high | high | — | reject |

**Why not D1 as primary.** The confirmatory gate materially weakened it. In
PushT, NFE100 is better offline **and** better on both closed-loop endpoints
(reward CI excludes zero). The surviving D1 claim — "the *magnitude* of offline
improvement doesn't specify the deployment tradeoff" — is close to the known
observation that more denoising has diminishing returns, and §26-1/§26-2 attack
it directly. Critical Interval MSE (2606.29898) already owns "action MSE fails to
predict deployment success."

**Why not D3.** §3 asked me to attack the unifying thesis. It is **three loosely
related observations** joined by the word "evaluation": semantic asymmetry is a
property of one joint model; the cross-policy Pareto result is about diminishing
returns; simulator nondeterminism is about GPU physics. Only the last two even
share a subject. A reviewer would read D3 as a portfolio, not a paper.

**Why D2 works.** It has a single subject (when is a measured policy difference
real?), its central evidence is already collected and unusually concrete, and its
findings *correct conclusions the project itself published* — the most persuasive
possible demonstration.

# 2. Final thesis

> In contact-rich GPU-simulated manipulation, the simulator contributes enough
> within-scenario variance that single-realization evaluation cannot resolve the
> small effects that fast-policy papers routinely report: across 81
> single-realization reconstructions of three calibrated inference-budget
> comparisons, **25% disagreed in sign** with the multi-realization estimate. We
> characterize the physical origin (contact-triggered divergence), calibrate the
> resolvable effect size as a function of replication, show a deterministic
> simulator that needs no replication, and demonstrate that the correction
> changes real conclusions about how much inference compute a generative policy
> needs.

# 3. Genuinely novel

1. **Sign instability quantified on real learned-policy comparisons.** 20/81
   (25%) of single-realization comparisons flip sign relative to R=3. Not a
   simulator-property benchmark — a statement about *conclusions*.
2. **Resolution calibration in success-rate units**: 7.55 / 4.36 / 3.38 / 2.67 pp
   at R = 1/3/5/8. GPUSimBench measures divergence in cm; nobody publishes the
   downstream effect-size resolution.
3. **Outcome bifurcation, not threshold jitter.** 39/96 episodes are
   physics-sensitive, **0/96 fail robustly**; on sensitive episodes failed
   realizations miss by 0.304 vs 0.018 when they succeed. Physics divergence
   changes outcomes qualitatively.
4. **A simulator-conditional protocol.** Isaac Gym needs replication; PushT
   (CPU pymunk, fixed dt, seeded reset) is deterministic and needs none. The
   prescription is *audit then replicate the layers that vary* — not "always
   repeat everything."
5. **Self-correction as evidence.** Three of our own published conclusions were
   overturned by the protocol (§17).

# 4. Merely known background

- GPU simulators are nondeterministic (GPUSimBench 2607.13059).
- Hierarchical/clustered bootstrap is standard statistics.
- More denoising has diminishing returns.
- Offline action MSE imperfectly predicts task success (2606.29898).
- Sequential/anytime policy comparison (N-SCORE 2603.13616), which assumes
  **i.i.d. evaluation data** — the assumption our nested physics layer violates.

# 5-7. Venue assessment

**RSS: PLAUSIBLE-to-STRONG.** RSS rewards measurement rigor and results that
change practice. The headline (25% sign flips) is exactly the kind of finding
that alters how the community runs evaluations. Weakness: no algorithm.

**CoRL: PLAUSIBLE.** Similar fit; CoRL is more method-hungry, so the evaluation
framing must be paired with the concrete low-NFE deployment consequence.

**NeurIPS: WEAK.** No new statistics (hierarchical bootstrap is standard), no
theory, and the empirical contribution is simulator-specific. §12's red team is
correct that "use more rollouts" is not a contribution; our defense is empirical
specificity, which is a robotics-venue argument.

**ICLR: WEAK.** Same, plus ICLR would want the generative-modeling angle (D1),
which the confirmatory gate weakened.

# 8. Direct NFE1 vs NFE2 — **NOT REQUIRED. Already answered at zero GPU.**

Both campaigns used the **same frozen episode set** (`35144910…`) and the **same
CRN bank** (base 20260905), verified. The contrast is therefore a cached
re-analysis:

| seed | p(NFE1) | p(NFE2) | Δ (2−1) | 95% CI (hierarchical) |
|--:|--:|--:|--:|---|
| 42 | 0.7951 | 0.8681 | **+7.3 pp** | [−0.7, +15.3] |
| 43 | 0.8090 | 0.8299 | **+2.1 pp** | [−5.9, +10.1] |
| 44 | 0.8472 | 0.8819 | **+3.5 pp** | [−3.5, +10.4] |
| **mean** | | | **+4.3 pp** (SD 2.7) | signs **+,+,+** |

Additivity is coherent: (2−1) + (4−2) = +4.3 + 0.0 ≈ **+6.0 pp** = the measured
(4−1). **The gain is in the second evaluation.** No CI excludes zero individually
— consistent with a ~4 pp effect against a 4.4 pp R=3 resolution — so the claim
stays *directional across three checkpoints*, which is what the paper needs.

**Decision: NO. Saves 0.59 GPU-h.**

# 9. Calibrated 4/5-cube breadth — **REQUIRED (4-cube only)**

This is the one genuine gap. Existing 4/5-cube results are R=1 and therefore
inadmissible under our own protocol — a reviewer will notice that we invalidated
sub-8 pp R=1 claims and then rely on them. Either recalibrate or drop the
compositional axis entirely.

Recommend **4-cube NFE2 vs NFE4 at fixed H=100, R=3, 3 seeds (0.64 GPU-h)**.
Fixed-H per §26 avoids confounding object count with horizon. 5-cube is
**NICE-TO-HAVE** (+1.27 GPU-h) — it doubles cost for a third point on the same
curve.

# 10. Evaluator analysis already possible at zero GPU — **most of it**

| analysis | status |
|---|---|
| A. R=1 pseudo-experiments from R=8 | **DONE** (spread 10.4 pp, SD 4.23) |
| B. sign-reversal frequency | **DONE — 20/81 = 25%** |
| C. naive-bootstrap undercoverage | **DONE — FALSIFIED, see below** |
| D. R scaling (1/2/3/5/8) | **DONE** (12.50/10.42/9.03/8.33/7.81 pp width) |
| E. contact localization | **DONE** (200× amplification, steps 2-10) |
| F. threshold vs bifurcation | **DONE** (0.018 vs 0.304) |
| NFE1-vs-NFE2 direct | **DONE** (§8) |

### A negative result I must report (§C)

I hypothesised that episode-only bootstrap CIs would **under-cover**. **They do
not.** Pseudo-coverage against the R=8 reference: **96.8% (R=1), 98.7% (R=3),
99.8% (R=5)** — at or above the nominal 95%.

The reason is instructive: an R=1 episode-only CI is **12.1 pp wide**. It covers
the truth *because it is too wide to resolve a 3-6 pp effect*. **The failure mode
is insufficient resolution, not anticonservative inference.** The paper must say
this plainly; claiming undercoverage would have been wrong.

# 11-12. Remaining experiments

| # | question | arms | task | seeds | R | GPU-h | claim unlocked | status |
|---|---|---|---|--:|--:|--:|---|---|
| A | is the 1→2 step where the gain is? | NFE1,2 | 3-cube | 3 | 3 | **0** | low-NFE boundary | **CACHED** |
| **B** | does the operating point transfer under compositional load? | NFE2,4 | **4-cube H=100** | 3 | 3 | **0.64** | compositional breadth | **MUST-HAVE** |
| C | third compositional point | NFE2,4 | 5-cube H=100 | 3 | 3 | 1.27 | marginal | NICE |
| D | evaluator analyses | — | cached | — | — | **0** | all of §10 | **DONE** |
| E | Gaussian fixed-budget comparator | G@100 | 3-cube | 3 | 3 | 0.33 | cross-family | **DROP** |
| F | nothing | — | — | — | — | 0 | — | fallback |

**E is dropped** (§23): the Gaussian arm has no valid NFE axis, and the
independent Diffusion Policy already supplies cross-family evidence. Its only
role would be a fixed-budget point that answers no question the paper asks.

# 13. Budget plans

**BUDGET S (≤1 GPU-h) — RECOMMENDED.** Run **B only** (0.64). Do not run C, E,
or any external policy. Yields: evaluation contribution complete + compositional
breadth at one object count.

**BUDGET M (≤2 GPU-h).** B + C (1.91). Adds the 5-cube point. Only worth it if
4-cube shows an *interesting* result (see gate below).

**BUDGET L (≤4 GPU-h).** B + C + a 4-cube NFE1-vs-NFE2 arm (0.89) = 2.80. I do
**not** recommend spending to the cap; there is no experiment worth the last
1.2 GPU-h.

# 14. Figures

1. **Contact-triggered divergence** — identical actions, no policy; cube divergence 5.9e-4 → 0.92 (23× threshold), 200× amplification at steps 2-10.
2. **Resolution vs R** — paired half-width 7.55/4.36/3.38/2.67 pp, with the effect sizes this project actually measured overlaid.
3. **Sign-instability (headline)** — distribution of single-realization Δ per contrast, zero line, calibrated estimate; 25% wrong-sign.
4. **Outcome bifurcation** — p_i histogram (57 robust-success, 39 sensitive, 0 robust-fail) + the 0.018/0.304 split.
5. **Simulator-conditional protocol** — Isaac Gym vs PushT determinism; when replication is and isn't needed.
6. **Consequence** — the corrected NFE curve (1→2→4) with calibrated CIs, plus the cross-policy latency/utility Pareto.

Six figures, one subject. Figure 2 of the old D1 plan (semantic asymmetry) moves
to a **supporting** role (§17-B).

# 15. Reviewer attacks

| # | attack | rebuttal | status |
|--:|---|---|---|
| 1 | "just diminishing returns" | D2 isn't about returns; it's about whether measured differences are real. | **strong** |
| 2 | "nobody expected action MSE to predict success" | Agreed — we cite 2606.29898 and demote this to background. | **strong (by concession)** |
| 3 | "GPUSimBench already showed nondeterminism" | They measure cm of divergence; we measure sign flips in policy conclusions. | **strong** |
| 4 | "hierarchical bootstrap is standard" | Conceded explicitly; the contribution is the measured magnitude, not the estimator. | **adequate** |
| 5 | "only two policies" | Two policies, two simulators with *opposite* determinism — the informative axis. | **adequate** |
| 6 | "semantic result is one model" | Demoted to a case study, not a headline claim. | **strong (by demotion)** |
| 7 | "NFE2 not formally equivalent" | Conceded; we show ±5 pp needs n≈116,000, i.e. unresolvable. | **strong** |
| 8 | "NFE100 actually wins in PushT" | Stated plainly; it is why D1 was demoted. | **strong** |
| 9 | "no new algorithm" | True. Defensible at RSS/CoRL, **not** at NeurIPS/ICLR. | **venue-limited** |
| 10 | "engineering best practice, not research" | **Weakest point.** Best answer: we didn't just recommend replication, we showed 25% of conclusions flip and corrected three of our own published results. | **needs the strongest writing** |

# 16. Titles

1. *When Is a Policy Difference Real? Simulator Nondeterminism in Contact-Rich Robot Evaluation*
2. *Sign Instability: Single-Realization Evaluation of Learned Manipulation Policies*
3. *Calibrating the Resolution of Robot-Policy Benchmarks*
4. *How Many Rollouts? Effect-Size Resolution Under Nondeterministic GPU Physics*
5. *Measuring Inference Budgets for Generative Policies Under Evaluator Noise*

**Preferred thesis (one sentence):** as §2.

# 17. Claim-ledger corrections

**Retire** (universal form): "offline metrics mis-rank inference budgets."

**Cross-policy — PARTIAL:** *Offline generative fidelity alone is insufficient to
determine the latency–behavior tradeoff of an inference budget.*

**PushT — SUPPORTED:** *NFE100 improves both offline action error (21%) and
closed-loop behavior relative to NFE2, but at 50× denoiser calls and ~37× planner
latency; the magnitude of offline improvement does not specify the deployment
tradeoff.*

**EC-Diffuser — SUPPORTED:** *Action replay error and closed-loop success can move
in different directions across inference budgets* (NFE2→NFE4: offline worsens,
control ≈ unchanged).

**Evaluation — SUPPORTED:** 25% sign disagreement; resolution 7.55→2.67 pp;
contact-triggered bifurcation; simulator-conditional protocol.

**Evaluation — FALSIFIED (report as such):** episode-only bootstrap CIs do
**not** under-cover (96.8% at R=1). The failure is resolution, not coverage.

**Prohibited claims:** new generative model / loss / solver / adaptive-NFE / new
bootstrap; "two steps are always enough"; "offline metrics are useless";
"GPU simulators are nondeterministic" (as *our* discovery); "Diffusion Policy
saturates at two steps"; "NFE1 is algorithmically worse."

# 18. **PAPER-CONDITIONAL**

A credible **RSS/CoRL** submission exists, conditional on Run B. Not NeurIPS/ICLR
without a method or theory, and I do not recommend inventing one to reach them.

# 19. Ordered queue

```
RUN 1 — 4-cube NFE2 vs NFE4, fixed H=100, R=3, seeds 42/43/44
  cost: 0.64 GPU-h
  why : the only inadmissible gap (existing 4/5-cube data are R=1);
        tests whether the operating point transfers under compositional load

DECISION GATE
  if |Δ| < ~4 pp with consistent signs   -> operating point transfers.
      Paper is complete. Run NOTHING further. Write.
  if Δ grows materially with object count -> "required inference compute scales
      with compositional demand" — a genuinely stronger result.
      THEN and only then consider RUN 2.
  if results are incoherent across seeds  -> drop the compositional axis,
      publish D2 without it. Run nothing further.

RUN 2 (conditional only) — 5-cube NFE2 vs NFE4, H=100, R=3, 3 seeds
  cost: 1.27 GPU-h
  why : confirms a *trend* in object count; pointless if RUN 1 shows transfer

NOT RUN: NFE1-vs-NFE2 (cached), Gaussian comparator, any third external policy,
         any additional NFE budget, any training.
```

# 20. Total additional GPU

**0.64 GPU-h** expected (Run 1 only). Worst case **1.91** if the gate opens
Run 2. Versus the ~4 GPU-h that was notionally available — most of the remaining
budget should go **unspent**.
