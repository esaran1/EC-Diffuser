# Improved MeanFlow objective audit

Status: the existing 1,000-step `improved_meanflow` pilot is retained as an engineering artifact but is not evidence for or against canonical iMF.

## Primary-source reference

- Paper: *Improved Mean Flows: On the Challenges of Fastforward Generative Models*, arXiv:2512.02012.
- Official JAX source: `Lyy-iiis/imeanflow` commit `bf60cd7cb653f6628e59d48034b333c5eba445e2`.
- Official PyTorch inference branch: commit `04687983e821b3ad01f54f03dc44194a33c20c54`.

## Equation-to-code audit

The EC-Diffuser wrapper implements the published boundary-condition variant of Eq. 12 / Algorithm 1:

1. `z_t = (1-t) x + t e`; conditional velocity target `e-x`.
2. The marginal-velocity JVP tangent is `v_theta(z_t,t) = u_theta(z_t,t,t)`, not the conditional target.
3. `JVP(u_theta; [v_theta, 0, 1])` differentiates with respect to trajectory and `t`, with zero tangent for `r`.
4. `V_theta = u_theta + (t-r) stopgrad(JVP)`.
5. Sampling uses `z_r = z_t - (t-r) u_theta(z_t,r,t)`.
6. Exact current/goal conditioning is masked from the loss and reimposed after every solver step.

This no-extra-head boundary variant is explicitly reported in Table 1(a) of the paper. The auxiliary velocity head is a separate published variant and is not silently approximated here.

## Discrepancy in the first OGBench pilot

The predeclared v1 pilot forced L1 for every method. Canonical iMF experiments use per-sample squared error with adaptive weighting

`L_i = S_i / stopgrad((S_i + 0.01)^1)`,

where `S_i` is the sum of active weighted squared errors for sample `i`. The original pilot omitted this weighting and logged only stochastic training batches. Its finite but rising loss therefore cannot be interpreted as an iMF optimization failure.

## Corrective protocol

- The old checkpoint/result remains immutable and labeled by its v1 protocol hash.
- `ImprovedMeanFlow` now exposes opt-in adaptive L2 weighting with the paper defaults (`p=1`, `epsilon=0.01`); legacy behavior remains load-compatible.
- Mathematical tests verify the compound target, stopped JVP branch, boundary case, exact adaptive formula, gradients, conditioning, checkpoint round-trip, and call counts.
- `ogbench_puzzle_state_extension_v1.json` predeclares the published boundary-condition iMF objective and fixed held-out validation before any extension run.
- No auxiliary-head result will be claimed unless that architecture is implemented and declared as a separate experiment.
