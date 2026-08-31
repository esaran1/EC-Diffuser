# Determinism / stochasticity audit — PushT

## Sources of randomness at evaluation

| # | source | status |
|---|---|---|
| 1 | env physics (`pymunk`, CPU, fixed `dt = 1/sim_hz`) | **deterministic** given state + actions |
| 2 | env reset | seeded (`np.random.RandomState(seed)`); frozen seeds 1000-1049 |
| 3 | initial action noise (`conditional_sample`) | **stochastic** — torch global RNG |
| 4 | DDPM scheduler variance noise per reverse step | **stochastic** — torch global RNG |
| 5 | image preprocessing at eval | deterministic (`crop_is_random` applies to training augmentation; eval uses centre crop) |

**Answer to the protocol's A/B/C/D question: B — policy diffusion stochasticity
only.** PushT physics is CPU `pymunk` with a fixed timestep and a seeded reset,
so there is no simulator-realization hierarchy analogous to Isaac Gym.

## Consequence — and this is a positive finding

**PushT does NOT require its own evaluation-noise calibration of the Isaac Gym
kind.** The Isaac Gym R=3 protocol is **not** imported here (per §23/§6). The
remaining variance is policy sampling, which is handled by averaging over the
frozen 50-episode set and reporting continuous `max_reward` alongside binary
success.

This makes PushT a **contrast case** for the evaluation-methodology claim:
nested simulator-realization variance is a property of contact-rich **GPU**
physics, not of robot simulation in general. That is a sharper, more falsifiable
statement than asserting it universally.

## CRN note

Exact common-random-number pairing across different `num_inference_steps` is
**not claimed**. Different schedule lengths consume different numbers of RNG
draws, so the draws cannot be aligned without altering the sampler — which the
protocol forbids doing merely for pairing. Policy sampling is therefore treated
as part of evaluation variance, with the same frozen episode set for every arm.
