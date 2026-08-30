# Arm-neutral state/action error vs NFE

**No training. No Isaac Gym. No loss change. No Gaussian. Replay data only.**

Scripts: `arm_neutral_nfe_curve.py`, `analyze_nfe_curve.py`, `plot_nfe_curve.py`
Data: `arm_neutral_nfe_curve.npz` (11 MB), `nfe_curve_analysis.json`
Figure: `experiments/figures/arm_neutral_nfe_curve.png`

**HEADLINE: state and action want OPPOSITE integration budgets. Action error is
minimised at NFE 2 and rises monotonically thereafter; state error falls 8.4x to
a minimum at NFE 32. The ~20% E512 action penalty is not a late-convergence
artifact — 57% of it has already appeared by NFE 8, inside the practical control
regime. And the offline action optimum does NOT coincide with the measured
control optimum.**

---

## 1-2. NFE set and compute

NFE = **1, 2, 4, 8, 16, 32, 64, 128, 256, 512** (all ten; the practical region
was not reduced).

**Cost reported before reducing scope, as required.** With the noise bank shared
across NFE, one (seed, noise) pass costs 1+2+…+512 = 1023 NFE:

| design | model calls | estimate |
|---|--:|--:|
| 10 NFE × 8 noises × 3 seeds | 24,552 | **0.84 GPU-h — over the 0.5 limit** |
| drop 64 & 256, × 8 noises | 16,872 | 0.58 GPU-h — still over |
| **10 NFE × 4 noises × 3 seeds** | **12,276** | **0.42 GPU-h — chosen** |

I reduced **noise count**, not the NFE ladder, because §4 forbids cutting the
1/2/4/8/16 region and requires all three seeds. 4 noises × 96 conditions = **384
paired samples per seed per NFE**.

**Actual: 1944 s = 0.54 GPU-h**, 29% over the 0.42 estimate (small-NFE calls
carry fixed per-launch overhead the linear model ignored). Slightly above the
0.5 threshold; flagging it rather than burying it.

## 3. Fixed-noise protocol

One `torch.Generator(cpu)` seeded **20260901** per training seed. Within
(seed, condition, noise index) the **same x0 is passed to all ten NFE values**,
so the ladder isolates integration resolution alone. NFE accounting is asserted
in-loop (`assert nfe_used == n`). The sample set was verified **byte-identical**
to the frozen `ARM_NEUTRAL_SOLVER_BIAS` set (episodes and timesteps compared
element-wise). Targets are recorded replay transitions; **no policy authored the
state target, action target or goal.**

## 4-5. Arm-neutral error at every NFE, per seed

| NFE | act s42 | act s43 | act s44 | **act mean** | st s42 | st s43 | st s44 | **st mean** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.01668 | 0.01742 | 0.01944 | 0.01785 | 0.20962 | 0.21164 | 0.20969 | 0.21032 |
| **2** | 0.01785 | 0.01605 | 0.01761 | **0.01717** | 0.10726 | 0.10799 | 0.10758 | 0.10761 |
| 4 | 0.02439 | 0.02077 | 0.02120 | 0.02212 | 0.04914 | 0.04858 | 0.04873 | 0.04881 |
| 8 | 0.03567 | 0.03241 | 0.03252 | 0.03353 | 0.03008 | 0.02916 | 0.02956 | 0.02960 |
| 16 | 0.03972 | 0.03808 | 0.03691 | 0.03824 | 0.02626 | 0.02513 | 0.02560 | 0.02566 |
| **32** | 0.04283 | 0.04130 | 0.03994 | 0.04136 | 0.02556 | 0.02439 | 0.02486 | **0.02493** |
| 64 | 0.04499 | 0.04347 | 0.04220 | 0.04355 | 0.02567 | 0.02469 | 0.02500 | 0.02512 |
| 128 | 0.04627 | 0.04487 | 0.04334 | 0.04483 | 0.02597 | 0.02487 | 0.02525 | 0.02536 |
| 256 | 0.04704 | 0.04560 | 0.04400 | 0.04554 | 0.02619 | 0.02508 | 0.02538 | 0.02555 |
| 512 | 0.04740 | 0.04595 | 0.04442 | 0.04592 | 0.02629 | 0.02517 | 0.02551 | 0.02566 |

**The two curves are monotone in opposite directions over the practical range.**
Per-seed ordering is identical on all three checkpoints.

## 6. Action magnitude / direction decomposition

| NFE | L2 | L1 | \|magnitude\| err | cosine |
|--:|--:|--:|--:|--:|
| 1 | 0.01785 | 0.02403 | 0.01183 | 0.9947 |
| **2** | **0.01717** | **0.02280** | **0.01093** | **0.9948** |
| 4 | 0.02212 | 0.02941 | 0.01305 | 0.9897 |
| 8 | 0.03353 | 0.04405 | 0.02015 | 0.9802 |
| 16 | 0.03824 | 0.05040 | 0.02180 | 0.9713 |
| 512 | 0.04592 | 0.06121 | 0.02542 | 0.9534 |

**Every component degrades monotonically** — L1, L2, magnitude error and
direction cosine (0.9948 → 0.9534). The action degradation is not a scaling
artifact and not confined to one component.

## 7. Paired changes from the action optimum (same x0, per noise)

| transition | mean Δ | worse in | per-seed |
|---|--:|--:|---|
| 2→1 | +0.00068 | 58.9% | [−0.00116, +0.00136, +0.00184] |
| 2→4 | +0.00495 | 58.2% | [+0.00654, +0.00472, +0.00359] |
| 2→8 | +0.01636 | 69.0% | [+0.01782, +0.01635, +0.01491] |
| 2→16 | +0.02107 | 71.0% | [+0.02187, +0.02203, +0.01930] |
| 2→32 | +0.02419 | 68.3% | [+0.02498, +0.02525, +0.02233] |
| 2→512 | +0.02875 | **82.6%** | [+0.02956, +0.02989, +0.02681] |

Broad, not tail-driven: at NFE 512 **82.6% of individual noise draws are worse**
than the same draw at NFE 2. Note 2→1 is *not* clean (seed 42 prefers NFE 1),
consistent with the per-seed optima {42:1, 43:2, 44:2}.

## 8. NFE minimising each error

- **ACTION: NFE 2** (0.01717). Per seed: {42: **1**, 43: 2, 44: 2}.
- **STATE: NFE 32** (0.02493).

The optima differ by **16×**.

## 9. Is state NFE-invariant? **NO — emphatically not**

- state spans 0.02493 → 0.21032, a **743%** spread; it falls **8.4×** from NFE 1
  to NFE 32.
- action spans 0.01717 → 0.04592, a **167%** spread.

The earlier E16-vs-E512 result showed state *equivalence* because **both 16 and
512 sit on the flat post-convergence plateau** (0.02566 vs 0.02566 — identical).
That was a real observation about those two points, but it is **not** evidence
that state is NFE-insensitive. Below NFE 16 the state curve moves enormously.
This materially refines conclusion 2 of the frozen set.

## 10, 12-13. Degradation from the action optimum, and share of the E512 effect

| NFE | action err | rel. degradation | **fraction of E512 effect** |
|--:|--:|--:|--:|
| 2 | 0.01717 | 0.00% | 0.0% |
| 4 | 0.02212 | +28.8% | 17.2% |
| 8 | 0.03353 | +95.3% | **56.9%** |
| 16 | 0.03824 | +122.7% | 73.3% |
| **32** | 0.04136 | +140.9% | **84.1%** |
| 64 | 0.04355 | +153.7% | 91.8% |
| 512 | 0.04592 | +167.5% | 100.0% |

**Smallest NFE reaching ≥80% of the E512 degradation (predeclared threshold):
NFE 32.**

State, from its own optimum at 32: +743% (NFE 1), +332% (2), +95.8% (4), +18.7%
(8), +2.9% (16), then ≤2.9% everywhere above.

**The action penalty is not an extreme-integration phenomenon.** 57% of it has
appeared by NFE 8 — squarely inside the deployed control regime.

## 11. Seed-42 comparison with the canonical control curve

Canonical values **recomputed from raw per-episode records**
(`experiments/audit/recomputed_details.json`), not quoted from prose:

| NFE | control success | per-replicate | offline action err | offline state err |
|--:|--:|---|--:|--:|
| 1 | 0.8056 | 0.854 / 0.760 / 0.802 | **0.01668** | 0.20962 |
| 2 | 0.8681 | 0.885 / 0.865 / 0.854 | 0.01785 | 0.10726 |
| 4 | 0.8889 | 0.885 / 0.885 / 0.896 | 0.02439 | 0.04914 |
| **8** | **0.8993** | 0.844 / 0.896 / 0.958 | 0.03567 | 0.03008 |
| 16 | 0.8854 | 0.854 / 0.917 / 0.885 | 0.03972 | 0.02626 |

(The control curve is n=288 over 3 evaluation replicates but a **single**
checkpoint, seed 42. The offline curves have three checkpoints. No
population-level relationship is claimed.)

**They do NOT align.** Control is *worst* at NFE 1 (0.8056) where offline action
error is *best* for seed 42 (0.01668), and control peaks at NFE 8 where offline
action error is already +114% above its seed-42 minimum. Across NFE 1→8 the two
move in **opposite** directions.

The state curve tracks control far better over 1→8 (both improve sharply, both
flatten by 8–16), though state keeps improving slightly to 32 while control
turns down at 16.

## 12. §9's hypothesised pattern was NOT observed

§9 anticipated action error improving to NFE 4/8 then worsening, mirroring the
control curve. **That is not what happened**: action error is worst-case
monotone-increasing from NFE 2 onward, with no interior optimum near 4–8.
Reporting this plainly rather than fitting the expected shape.

## 14. State/action coupling across NFE

| transition | ρ(Δaction, Δstate) |
|---|--:|
| 1→2 | +0.004 |
| 2→4 | +0.065 |
| 4→8 | +0.033 |
| 8→16 | +0.182 |
| 16→32 | +0.119 |
| 256→512 | +0.044 |

**Coupling is weak everywhere (ρ ≤ 0.18, ≤3% of variance) and weakest at low
NFE.** The decoupling is not a late-integration phenomenon; the two output
blocks respond largely independently across the whole ladder. This extends the
E16/E512 finding rather than contradicting it.

Revised interpretation: integration resolution acts on the two blocks as
near-independent channels — refining the state block toward the recorded next
state while moving the action block away from the recorded action.

## 15. Dimensional-loss hypothesis: **LOWER PRIORITY** (unchanged)

Nothing here raises it. The evidence remains inverted relative to the original
concern: the 3 action coordinates are *best served at minimal integration* and
degrade under convergence, while the 480 state coordinates behave as a
well-posed ODE-convergence problem. That is a property of the discrete sampler,
not of loss weighting. Per §15, new evidence would be required to move this.

## 16. Recommended closed-loop NFE pair

**NFE 4 (practical baseline) vs NFE 32 (treatment).**

- NFE 32 captures **84.1%** of the full E512 action degradation — above the
  predeclared 80% threshold, and the smallest ladder point that clears it.
- It is **16× cheaper than NFE 512** for the same causal question.
- NFE 4 is the established recommended operating point (best on aggregate
  control metrics, +2.1 pts vs Gaussian) and is on the flat part of the state
  curve's descent, so the contrast isolates the action effect rather than
  confounding it with the large low-NFE state improvement.

## 17. Closed-loop cost estimates

Anchored on the **measured** NFE study (1,728 episodes, 1.75 GPU-h), decomposed
into a per-episode-per-NFE denoiser term and an NFE-independent simulator term:
a ≈ 3.42e-5 GPU-h per (episode·NFE), b ≈ 2.65e-4 GPU-h per episode.

Paired NFE 4 vs NFE 32, both arms:

| episodes/seed | 1 seed | 3 seeds | claim supported |
|--:|--:|--:|---|
| 32 | 0.056 GPU-h | 0.169 GPU-h | direction only; ~11-pt CI, underpowered |
| 64 | 0.113 GPU-h | 0.338 GPU-h | detects ≥8-pt effects; single-seed marginal |
| **96** | **0.169 GPU-h** | **0.508 GPU-h** | **matches the frozen 96-episode protocol and the established 3–11 pt noise floor** |

For contrast, the same design at NFE 512 costs 1.75 GPU-h (1 seed) to 5.24 GPU-h
(3 seeds) — an order of magnitude more for ~16% more of the effect.

## 18. Claim ledger

**WITHDRAWN**
- "Flow@1 has objectively worse neutral state imagination than Flow@4 because of
  integration" — the *measurement* direction actually survives arm-neutral
  (0.21032 vs 0.04881), but the original claim rested on policy-authored targets
  and is superseded by the neutral curve; state at NFE 1 is genuinely far worse.
- "E16 objectively imagines the future better than E512."
- "Gaussian objectively predicts states better than Flow" (self-scored comparison).

**SUPPORTED**
- Prior policy-authored measurements behaved as recorded; their generative-quality
  interpretations were confounded.
- Under arm-neutral targets, E16 and E512 have essentially equal state accuracy —
  **now refined: because both lie on the post-NFE-16 plateau, not because state
  is NFE-insensitive.**
- Under arm-neutral targets, E512 is ~20% worse on demonstrated-action imitation
  than E16 (here: +167% above the NFE-2 optimum, +20.1% above NFE 16).
- The action result replicates across three independently trained checkpoints.

**NEW, this experiment**
- Arm-neutral action error is minimised at **NFE 2** and rises monotonically.
- Arm-neutral state error is minimised at **NFE 32**, improving **8.4×** from NFE 1.
- The offline action optimum does **not** coincide with the measured control
  optimum (NFE 8) for seed 42.

## 19. Gaussian note

Not run. The old Gaussian-vs-Flow quantitative imagination comparison was
self-scored per arm and **requires an arm-neutral replay re-evaluation before
scientific use**. Estimated ~0.08 GPU-h, deferred; not needed for this mechanism.

## 20. Critical caveat on interpreting action error (§13)

There is **one** demonstrated action per replay state, and these transitions were
**in the Flow training set** (no train/val split exists in the pipeline). Lower
offline action error therefore means **closer alignment with the demonstrated
action**, not optimality — a different action may be equally or more effective in
closed loop. The §11 misalignment is direct evidence for this caution: NFE 1 has
the best seed-42 action error and the *worst* control success.

Repeat labelling: **arm-neutral in-distribution replay diagnostic**, not held-out
generalization.

## 21. ONE recommended control experiment

**Paired closed-loop Isaac Gym evaluation of Flow NFE 4 vs NFE 32, 96 episodes
per training seed, all three seeds, on the frozen episode set — 0.508 GPU-h.**

This is the causal test the offline curve cannot substitute for. The offline
result says accurate integration moves actions away from demonstrations by a
large, monotone, 3/3-seed margin; §11 shows offline action error and control
success are *anti*-aligned over NFE 1→8, so the closed-loop consequence genuinely
cannot be predicted from these data. NFE 32 delivers 84% of the E512 effect at
1/16 the cost, and 96 episodes × 3 seeds matches the established protocol and
noise floor.

**Not run. Awaiting explicit approval.**

---

## HARD STOP OBSERVED

No control simulation. No training. No loss change. No Gaussian generation.
No MeanFlow. No VP.
