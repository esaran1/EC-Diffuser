# Improved MeanFlow source-to-code audit

## Scope and frozen sources

This audit concerns the boundary-condition iMF variant used for OGBench trajectory generation. It does not claim equivalence to the authors' full ImageNet system, which uses a different architecture, classifier-free guidance, and an auxiliary velocity head.

Primary sources:

- [Improved Mean Flows paper, arXiv:2512.02012](https://arxiv.org/abs/2512.02012), downloaded PDF SHA256 `19a2a27ed86cc813bb0943ab43e80f4a80cf725aa692bb2f7d8c33cbbc8e12b7`.
- [Official iMF repository](https://github.com/Lyy-iiis/imeanflow), audited JAX commit `bf60cd7cb653f6628e59d48034b333c5eba445e2` and official PyTorch branch commit `04687983e821b3ad01f54f03dc44194a33c20c54`.

## Equation mapping

For trajectory data `x`, Gaussian noise `e`, and `0 <= r <= t <= 1`:

1. Path: `z_t = (1-t)x + te`.
2. Sample-conditional velocity target: `v_c = e-x`.
3. Boundary marginal-velocity parameterization: `v_theta(z_t,t) = u_theta(z_t,t,t)`.
4. JVP: derivative of `u_theta(z_t,r,t)` along tangent `(v_theta,0,1)`.
5. Compound prediction: `V_theta = u_theta + (t-r) stopgrad(JVP)`.
6. Objective: regress `V_theta` to the network-independent target `e-x`, using the official per-sample adaptive L2 weighting.

The implementation matches iMF Eq. 12 and Algorithm 1 in path orientation, target sign, boundary parameterization, JVP tangent, stop-gradient placement, and reverse-time sampling update. Exact endpoint conditioning is applied before path construction and excluded from the regression mask. Scaling time inputs by 1000 is an architecture representation; automatic differentiation includes its chain rule.

## Findings

### Correct and retained

- The boundary-condition form is explicitly evaluated in the paper and is parameter-free at inference.
- The adaptive loss matches the official sum-over-sample reduction followed by detached `(loss + 0.01)^1` normalization.
- The fixed validation is stationary: held-out batches, noise, `r`, and `t` are replayed from fixed seeds, and the target `e-x` is independent of network weights.
- The observed rise is therefore meaningful. At 5,000 half-LR steps, boundary raw L2 improved `2.07470 -> 0.76492`, while interval raw L2 worsened `38.76824 -> 44.73591`. This agrees with the paper's recommendation to inspect `r != t` samples separately.
- EMA recovery does not invalidate the live-weight diagnostic; it is recorded as a separate result.

### Corrected source mismatch

The official sampler assigns exactly `int(batch_size * data_proportion)` examples to `r=t`. The local implementation used independent Bernoulli draws, causing 39–61% boundary fractions in microbatches of 32. It now uses the official exact per-batch proportion. This changes only iMF training-time sampling.

### Protocol difference, now configurable

The released iMF recipe uses Adam `(beta1,beta2)=(0.9,0.95)`, 10 warmup epochs out of 640, and EMA decay `0.9999`. The shared EC-Diffuser pilots used Adam defaults `(0.9,0.999)`, no warmup, and the repository EMA schedule. These were fairness choices, not equation bugs.

The Trainer now supports opt-in Adam betas and linear warmup while preserving all existing defaults. The next screen isolates beta2 and warmup at the already selected trajectory-policy learning rate; it does not silently import the ImageNet learning rate, batch size, EMA, architecture, or CFG system.

## Decision

Do not launch long iMF training. Run the predeclared three-arm, 1,000-step optimizer-dynamics screen in `imf_optimizer_dynamics_screen_v1.json`. Select only by held-out interval raw L2 under the frozen rule. Simulator outcomes are not used for selection. A 5,000-step confirmation requires a separate predeclaration after the screen.
