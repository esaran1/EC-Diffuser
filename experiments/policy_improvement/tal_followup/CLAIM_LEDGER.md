# Claim ledger — Tal-followup program

Updated as the program runs. Every claim carries its evidence level.

## ESTABLISHED (zero-GPU, verified from source or artifacts)

- **UWM is arXiv:2504.02792** (RSS 2025, Zhu et al., UW + TRI). Code audited at
  commit `b7d4067`.
- **UWM's time convention is the OPPOSITE of ours.** UWM is DDPM with t=0 CLEAN,
  t=T noise, on discrete integers in [0,100). Our flow convention is t=0 noise,
  t=1 clean. Corner semantics invert with the axis.
- **UWM trains with NO deliberate diagonal mass.** Verified in both paper
  (III-B, III-D, Appendix -A 2) and code (two independent `torch.randint`
  calls). The training measure is the plain product `U{0..99} ⊗ U{0..99}`.
- **UWM's time embedding is a joint MLP over the concatenated pair**, not
  additive per-modality embeddings (`DualTimestepEncoder`). Our zero-parameter
  construction diverges from this — declared, not hidden.
- **UWM never traverses the interior of the 2D time square.** All four paper
  endpoints pin one time at a corner and sweep the other (axis edges); the only
  diagonal sampler is undocumented, visualization-only code. Interior-path
  traversal is unclaimed by UWM.
- **EC-Diffuser token index 0 is the action**, 1..48 are DLP particles (2 views ×
  24). Traced in `pint.py:163-196`, not inferred from names.
- **Time reaches the denoiser by two paths**: additive per-token, and AdaLN via
  `c = action_embed + t_embed`. A two-time retrofit must handle both.
- **Our flow convention verified from production code**: `xt = (1−t)x0 + t·x1`
  with `x0 = randn`, target `x1 − x0` (`flow_matching.py:330-332`).
- **`time_mlp` is a single shared network** — which is what makes a
  zero-added-parameter two-time construction possible.
- **"3C" means three cubes** — verified from `env_config/generalization_num_cubes`
  (`numObjects: 3`, `RandColor: True`, `numColors: 6`), not from the name.
- **NO direct-action implementation, checkpoint, or result exists** anywhere in
  this repository — working tree, git history across all branches, or disk.
- **The two-time denoiser is bit-exact with the baseline on the diagonal**
  (max|d| = 0.000e+00 across 3 batches × 5 t-values) with **zero added
  parameters** (213,325 both).
- **The matched diagonal control (`p_diag=1.0`) reproduces the true baseline
  wrapper loss bit-identically** (1.9936090708 both) while sampling 100% on the
  diagonal. The two pilot arms differ **only** in `p_diag` and output prefix,
  verified programmatically.
- **The path-correct sampler is verified** against an oracle constant velocity
  field: every registered path lands on the clean target for any step count
  (max err 2e-6), first-order convergence confirmed against the analytic
  solution of a state-dependent field.
- **Measured evaluation cost: 0.2212 GPU-h** per five-set R=1 evaluation
  (carried from the completed evaluator-resolution program).

## PRELIMINARY / IN PROGRESS

- UWM pilot training (1 seed, 5k steps) — running.

## NULL / NEGATIVE

- **MeanFlow closure stands.** arXiv:2512.02012 diagnoses JVP gradient variance,
  not a deployment-boundary failure; improves already-working image models
  (FID 32.69 → 29.42) with no failed-run rescue anywhere; no control results.

## WITHDRAWN / CORRECTED

- **"Our MeanFlow failure was IVC-style deployment-boundary."** The IVC
  terminology (arXiv:2602.13810) refers to an **r = t** boundary, not the
  (r=0, t=1) deployment boundary. Two different boundaries were conflated. The
  honest statement is the observed symptom (0% closed-loop success at one-step
  deployment) with the mechanism explicitly unresolved.
- **Two of my own sampler unit tests were wrong** (a constant field cannot
  expose the shared-du bug; Euler on `v=c(x1−x)` converges to
  `x1+(x0−x1)e^{−c}`, not `x1`). The sampler was correct both times; the tests
  were fixed and the errors recorded in `TWO_TIME_SAMPLER.md`.

## EXPLICITLY NOT CLAIMED

- Tal's direct-action observation is **not** reproduced here, because no prior
  artifact exists to reproduce. If built, it must be reported as a **new**
  construction.
- A learned query token is **not** claimed equivalent to `t_action = 0` or
  `t_action = 1` (§15). Those endpoints contain noise or the clean action; a
  query token contains a learned constant.
- The registers phenomenon is **not** assumed to transfer. Prior is negative:
  EC-Diffuser is far below the ≥ViT-Large regime, has no redundant background
  tokens, and already has a global-aggregation slot (the action token).
