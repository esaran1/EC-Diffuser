# Directional decomposition of the E16→E512 solver shift

**CPU only. No GPU. No training. No loss change. No new solver.** Runtime ~4 min.

Scripts: `direction_audit.py`, `plot_direction.py`
Data: `direction_audit.json` (on `multinoise_endpoints.npz`, within-run)
Figure: `experiments/figures/solver_bias_direction.png`

**Answer: Classification D — ORTHOGONAL SOLVER BIAS, with the target-axis
component ASSIGNMENT-UNSTABLE. Coarse Euler cannot honestly be called
"target-biased" on this evidence.**

---

## 0. SCOPE LIMITATION — parts of the requested analysis are BLOCKED without GPU

The multi-noise cache contains **only** the generated endpoints and the observed
future:

```
s{42,43,44}_euler16   (96, 8, 24, 10)
s{42,43,44}_euler512  (96, 8, 24, 10)
s{42,43,44}_real      (96, 24, 10)
```

It does **not** contain the current latent, the goal latent, or the action
channel. Therefore:

| section | status |
|---|---|
| §1-6, §11-13 (observed-future geometry) | **DELIVERED IN FULL** |
| §7 goal direction | **BLOCKED** — no goal DLP latent is cached anywhere |
| §8 current-state direction | **BLOCKED** — see below |
| §9 three-way decomposition | **PARTIAL** — only cos_future is computable |
| §14 action channel | **BLOCKED** — actions were not cached |

An earlier run (`cached_endpoints.npz`) does hold `current`, but it is from a
**different process**, and cross-process latents are not bit-reproducible. I
measured the mismatch rather than assuming it was tolerable:

| cross-run agreement of the observed future | mean | median | max |
|---|--:|--:|--:|
| s42 | 0.05939 | 0.04722 | 0.19676 |
| s43 | 0.07073 | 0.05736 | 0.17135 |
| s44 | 0.05832 | 0.04784 | 0.16133 |

That mismatch (~0.059) is **larger than the typical E16→observed-future distance
itself (0.0390)** and ~38× the effect under study (+0.00154). Substituting it
would manufacture the current-state direction outright.

**Per §17 I stopped rather than silently regenerating on GPU.** The goal/current/
action directions require a ~0.4 GPU-h regeneration that also caches those
tensors; that is not approved and is not done here.

## 1. Exact temporal semantics of the target

Read from the generating code:

- conditioning is `{0: current_obs, 4: goal_obs}` with `horizon = 5`
- the scored endpoint is taken at generated index **`[:, 1, view 0]`** — the
  **first predicted step**
- the environment is then advanced by **one** `env.step(action)`

**The imagination target is a ONE-STEP-AHEAD intermediate state, not the goal.**
The conditioned goal lives at generated timestep 4; we score timestep 1.

Consequence for §10: **only one prediction offset exists in this data**, so
stratification by prediction horizon is not possible. I stratified instead by
the available temporal axis — position within the episode (§9 below). The
question "does coarse Euler's advantage depend on prediction distance?" **cannot
be answered from these endpoints** and would need multi-horizon generation.

## 2. Particle-correspondence validation

E16/E512 share x0, model and noise, so their particle **slots** should correspond.
Verified rather than assumed:

| seed | slot-wise distance | optimally-matched | ratio | identity is optimal |
|---|--:|--:|--:|--:|
| 42 | 0.02682 | 0.02483 | **1.080** | 7.7% |
| 43 | 0.02710 | 0.02517 | **1.077** | 8.9% |
| 44 | 0.02598 | 0.02405 | **1.080** | 6.9% |

**Slot correspondence is meaningful but not exact**: using slots directly costs
only ~8% over optimal matching (so slots track each other through integration),
yet the identity permutation is strictly optimal in under 9% of cases. Slot-wise
differencing is therefore justified for the displacement `δ = p512 − p16`, which
is what §5 requires, and I use it there.

For the observed future, particle order is **not** semantically stable, so it is
Hungarian-matched. Anchoring conventions are compared below as required.

## 3. Displacement magnitude

Mean ‖δ‖ in position coordinates: **0.26650 / 0.27673 / 0.25759** (seeds
42/43/44), against a typical E16→observed-future distance of ~0.0428 (chamfer)
— note these are different norms (full 48-D position vector vs mean per-particle
chamfer) and are not directly comparable; the per-particle displacement is
0.02682 / 0.02710 / 0.02598, i.e. **~62% of the E16→GT chamfer distance**.

Accurate integration moves the state substantially — far more than the resulting
error change (+0.00154).

## 4-5. Alignment with the observed future, and the stability problem

| anchor | seed | cos_future | 95% CI | frac negative | ‖parallel‖ | ‖orthogonal‖ | \|para\|/‖δ‖ |
|---|---|--:|---|--:|--:|--:|--:|
| **E16-anchored** (primary) | 42 | **+0.0194** | [+0.0042, +0.0348] | 46.0% | +0.01166 | 0.25973 | **0.166** |
| | 43 | **+0.0297** | [+0.0151, +0.0442] | 47.5% | +0.01660 | 0.26940 | 0.158 |
| | 44 | **+0.0321** | [+0.0178, +0.0466] | 43.2% | +0.01468 | 0.25133 | 0.161 |
| **E512-anchored** | 42 | **+0.2286** | [+0.2090, +0.2483] | 20.3% | +0.10150 | 0.23318 | 0.292 |
| | 43 | +0.2343 | [+0.2155, +0.2529] | 20.7% | +0.10598 | 0.24233 | 0.285 |
| | 44 | +0.2284 | [+0.2105, +0.2459] | 18.2% | +0.09106 | 0.23055 | 0.280 |
| **independent** | 42 | +0.0194 | [+0.0041, +0.0346] | 46.0% | +0.01166 | 0.25973 | 0.166 |
| | 43 | +0.0297 | [+0.0152, +0.0443] | 47.5% | +0.01660 | 0.26940 | 0.158 |
| | 44 | +0.0321 | [+0.0174, +0.0467] | 43.2% | +0.01468 | 0.25133 | 0.161 |

**Two findings, and the second undercuts any strong directional story:**

1. **The shift is overwhelmingly ORTHOGONAL to the target axis.** Only
   **~16%** of the displacement lies along the direction toward the observed
   future (E16-anchored); ~84% is sideways. Sign of the aligned part is
   *positive*, i.e. E512 moves very slightly **toward** the observed future while
   nonetheless scoring worse — the degradation is not "moving away along the
   target axis".

2. **The result is assignment-unstable, exactly the failure mode §4 warned
   about.** E16-anchored gives cos ≈ +0.02 with 46% negative; E512-anchored gives
   cos ≈ +0.23 with 20% negative — an order of magnitude apart. Independent
   matching reproduces the E16-anchored numbers (as expected, since E16 anchors
   the reference frame). **Per §4 this is marked UNSTABLE and no directional
   conclusion is drawn from the cosine's magnitude.**

The E512-anchored number is inflated by construction: re-matching the target to
E512 partially absorbs E512's own movement into the assignment, so it is not
evidence of genuine target attraction.

## 6. Interpolation curve α: E16 → E512 (permutation-invariant metric)

| seed | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|--:|--:|--:|--:|--:|
| 42 | **0.04380** | 0.04475 | **0.04562** | 0.04561 | 0.04544 |
| 43 | **0.04296** | 0.04388 | **0.04456** | 0.04448 | 0.04419 |
| 44 | **0.04266** | 0.04361 | 0.04447 | **0.04451** | 0.04443 |

**Monotone increasing on all three seeds: FALSE.** Error rises to a **peak near
α = 0.5–0.75 and then comes back down**, so E512 is *not* the worst point on the
segment — the midpoint is. This is reported as required and it matters: the
displacement direction is **not** simply adverse. A straight-line move from E16
toward E512 first makes things worse than either endpoint, which is the signature
of a curved/orthogonal displacement in a set-valued metric, not of a clean bias
along a target axis.

## 7-8. Goal and current-state alignment — **NOT COMPUTED (blocked)**

See §0. No goal or current latent is available in a validly pairable form. I
decline to substitute cross-run tensors that differ by more than the effect size.

## 9. Stratification by episode position

(Prediction-horizon stratification is impossible — one offset only, §1.)

Paired E512−E16 by rollout step, all with bootstrap CIs:

| step | s42 | s43 | s44 |
|--:|--:|--:|--:|
| 0 | +0.00179 | +0.00106 | +0.00151 |
| 1 | +0.00115 | +0.00037 | +0.00159 |
| 2 | +0.00098 | +0.00174 | +0.00172 |
| 3 | +0.00166 | +0.00191 | +0.00144 |
| 4 | +0.00193 | +0.00104 | +0.00203 |
| 5 | +0.00234 | +0.00120 | +0.00231 |

**Positive at every one of 18 strata** (17 of 18 CIs exclude zero; s43 step 1 is
the exception). Mild tendency to grow later in the episode on seeds 42 and 44,
absent on 43 — not a robust trend.

## 10. Per-noise results

Per noise index (seed 42), Δ ranges +0.00112 to +0.00216 with E16 winning
62–69%. **All eight indices positive**; no single noise drives the effect.

## 11. Per-seed replication

Every quantity above replicates on all three independently trained checkpoints:
orthogonal fraction 0.158–0.166, non-monotone interpolation 3/3, positive
stratified effect 18/18, assignment instability 3/3.

## 12. Associations with Δ prediction error (Spearman, seed 42, descriptive)

| quantity | ρ | p |
|---|--:|--:|
| **cos_future** | **−0.282** | <1e-4 |
| **parallel_future** | **−0.276** | <1e-4 |
| orthogonal_future | −0.027 | 0.449 |
| displacement magnitude | −0.031 | 0.384 |

**This is the most informative result in the audit.** The *only* quantities
associated with the error change are the target-axis ones, and the sign is
**negative**: samples whose shift points *more* toward the observed future
degrade *less* (and vice versa). Orthogonal magnitude and raw displacement size
are uncorrelated (|ρ| < 0.04).

So the axis matters even though it carries only ~16% of the displacement — but
ρ ≈ −0.28 is still only **~8% of variance**. No causal claim.

## 13. Magnitude of the average solver-induced bias

| seed | ‖E[δ]‖ (signed, slot-wise) | E‖δ‖ (unsigned) | consistency ratio | E16→GT |
|---|--:|--:|--:|--:|
| 42 | 0.00360 | 0.02682 | **0.134** | 0.04380 |
| 43 | 0.00339 | 0.02710 | 0.125 | 0.04296 |
| 44 | 0.00350 | 0.02598 | 0.135 | 0.04266 |

The average signed displacement is **~13% of the typical unsigned displacement**
— i.e. the solver-induced shift is **largely inconsistent in direction** across
samples, with only a small systematic component. That systematic part
(‖E[δ]‖ ≈ 0.0035) is ~8% of the E16→GT distance: **small but consistent**, exactly
matching the "shifted centre" conclusion of the previous study while showing the
shift is not a single coherent direction in position space.

## 14. Action channel — **NOT COMPUTED** (not cached; §14 says skip if absent)

## 15. Classification: **D — ORTHOGONAL SOLVER BIAS**

Evidence:

- **~84% of the displacement is orthogonal** to the target axis (3/3 seeds).
- The target-axis component is **assignment-unstable** (cos +0.02 vs +0.23
  depending on anchor), so its magnitude is not trustworthy.
- The signed mean shift is only **13%** of the typical displacement — the
  direction is largely inconsistent sample to sample.
- The interpolation curve is **non-monotone**, peaking mid-segment, so the
  displacement direction is not straightforwardly adverse.
- What little association exists runs **opposite** to a simple "moves away from
  target" story: more target-aligned shifts degrade *less* (ρ = −0.28).

Not A: the degradation is **not** produced by E512 moving away along the
future-target direction — the mean parallel component is *positive* (toward the
target) while error still worsens.
Not B or C: goal and current-state directions are blocked, so those cannot be
claimed either way.
Not E: the underlying error effect is robust (18/18 strata, 8/8 noises, 3/3
seeds); it is the *geometric explanation* that fails, not the effect.

## 16. Strongest mechanistic interpretation supported

**Accurate integration displaces the generated particle configuration
substantially (~62% of the E16→GT distance) in a direction that is mostly
orthogonal to, and only weakly systematic with respect to, the observed future.
The resulting prediction degradation is real and highly replicable, but it is
not explained by simple semantic geometry in position space.**

The previous study's "shifted centre" conclusion survives and is refined: the
centre does shift (‖E[δ]‖ ≈ 0.0035, consistently signed), but that shift is a
small residual on top of a much larger, direction-inconsistent movement, and it
does not point cleanly at any of the semantic targets available to test.

## 17. Alternative interpretations still alive

- **Goal-directedness (§15-B) is untested**, not excluded. If E512 moves toward
  the conditioned goal at timestep 4 while scoring worse at timestep 1, the
  metric could be penalising a valid alternative trajectory. **This is the single
  most important open possibility and it is exactly what is blocked.**
- **Conservative/current-state bias (§15-C) is untested**, not excluded.
- The orthogonal movement may be semantically meaningful in a way position-only
  geometry cannot see (e.g. reallocation of particles among objects).
- The set-valued metric may itself penalise configurations that are physically
  reasonable; the non-monotone α curve is a hint that straight lines in particle
  space are not geodesics of this metric.

## 18. Can coarse Euler honestly be called "target-biased"?

**No — not on this evidence.** The displacement is 84% orthogonal, its aligned
component is assignment-unstable and *positive* (E512 drifts slightly toward the
target while scoring worse), and the mean signed shift is only 13% of typical
displacement. The defensible statement remains the empirical one:

> Euler@16 produces endpoints that score better against the observed one-step
> future than converged Euler@512 does, consistently across seeds, noises and
> episode positions — but the direction of the solver-induced shift is not
> explained by proximity to the observed future.

## 19. Is the imagination metric penalising valid alternatives?

**Cannot be determined here, and it remains a live concern.** Two concrete
reasons: (a) the goal-direction test that would address it directly is blocked;
(b) the non-monotone α curve shows the metric does not behave linearly along
straight paths between two plausible generated states. Combined with the standing
limitation that we have **one** observed future per context, this audit **cannot**
establish that E16's endpoints are objectively better futures — only that they
are closer to the single realised one.

## 20. Loss-hypothesis status: **LOWER PRIORITY** (unchanged)

Nothing here implicates the training objective. The effect remains a sampler
property, and the geometry is now shown to be poorly described by target
proximity, which if anything further removes any pointer toward the loss.

## 21. Exactly ONE next experiment

**Regenerate the multi-noise endpoints ONCE while additionally caching the
current latent, the goal latent, and the action channel — then complete §7, §8,
§9 and §14 on CPU.**

Cost: **~0.43 GPU-h** (the identical run already benchmarked), no training, no
new method, no protocol change — only the set of tensors written to disk grows.

Rationale: the primary observed-future geometry is now complete and returns
outcome D — the shift is mostly orthogonal and not explained by target proximity.
The two hypotheses that could explain it (goal-directed alternative trajectories,
§15-B; conservative current-state bias, §15-C) are precisely the ones blocked by
a caching omission, and §15-B carries the highest stakes in the whole
investigation: if E512 is moving toward the conditioned goal, then "E16 imagines
better" may partly be an artifact of scoring a one-step intermediate state. That
question should not be left open, and it is answerable for one modest run.

**This requires GPU generation, so it is NOT performed here.** Reported for
approval per §17.

---

## HARD STOP OBSERVED

No training. No GPU. No loss changes. No new solver. No policy modification.
