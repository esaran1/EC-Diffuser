# `enriched_endpoints.npz` — cache schema

**Purpose: no further GPU rerun should ever be needed because a tensor was not cached.**

- File: `experiments/loss_balance_audit/enriched_endpoints.npz` (53 MB, compressed)
- **sha256: `82ad69c229a762aa3323adff222bf2b4b968834c8cd9bdc4da511f48ee89ef3c`**
- Generator: `enriched_multinoise.py` (1533 s wall)
- 45 keys, prefixed `s42_` / `s43_` / `s44_` by Flow training seed.

## Axis meanings

`(96, 8, 5, D)` = (condition, noise index, generated timestep, feature).
Conditions are ordered `rollout_step * 16 + episode`, i.e. 6 rollout steps × 16
episodes; `cond_id` records this as `rs{step}_ep{episode}`.

| key | shape | dtype | meaning |
|---|---|---|---|
| `cur_raw` | (96, 2, 24, 10) | f32 | conditioned current latent, **both views**, unnormalized |
| `goal_raw` | (96, 2, 24, 10) | f32 | conditioned goal latent, both views, unnormalized |
| `cur_norm` | (96, 480) | f32 | exactly what was passed to the model at t=0 |
| `goal_norm` | (96, 480) | f32 | exactly what was passed to the model at t=4 |
| `real_t1_raw` | (96, 2, 24, 10) | f32 | **observed** t=1 future (the supervised target) |
| `real_action` | (96, 3) | f32 | action actually executed by the env (E16 noise-0) |
| `cond_mask` | (5, 483) | bool | conditioning mask; **False on conditioned coords** |
| `{arm}_full_norm` | (96, 8, 5, 483) | f32 | full generated transition tensor, **model space** |
| `{arm}_obs_unnorm` | (96, 8, 5, 480) | f32 | observation channels, unnormalized, **all timesteps** |
| `{arm}_act_unnorm` | (96, 8, 5, 3) | f32 | action channels, unnormalized, all timesteps |
| `cond_id` | (96,) | U8 | `rs{rollout_step}_ep{episode}` |
| `x0_hash` | (96,) | U16 | sha256[:16] of the initial noise, for pairing provenance |

`{arm}` ∈ {`euler16`, `euler512`}.

## Normalization state

- `*_raw`, `*_unnorm` are in **data units** (what the metric consumes).
- `*_norm`, `*_full_norm` are in **model space** (normalizer applied).
- `full_norm` layout is `[action_dim=3 | observation_dim=480]` along the last axis.

## Particle feature layout (last axis of a `(...,24,10)` tensor)

`[0:2] position · [2:4] scale · [4:5] depth · [5:9] visual · [9:10] transparency`
(`latent_metric.POS/SCALE/DEPTH/VIS/TRANSP`). 24 particles = **one view**; the
raw tensors keep both views on axis 1, **view 0** is the one all metrics use.

## Timestep semantics

- generated **t=0** — conditioned to the current state
- generated **t=1** — the **scored imagination endpoint**
- generated t=2, t=3 — unconstrained interior
- generated **t=4** — conditioned to the goal

## Action semantics

Actions are **deltas** consumed by `env.step()`. `action[t]` drives the
transition **t → t+1**, so `action[0]` is the action that produces the scored
t=1 state, and it is the one compared against `real_action`.

## Conditioning semantics

`cond = {0: current, 4: goal}` applied to observation channels only; the
velocity is masked to zero there, so conditioned coordinates never move.

## Reproducibility

Noise bank: one `torch.Generator(cpu)` seeded **20260830** per training seed,
drawing 8 tensors per rollout step in order. Isaac Gym/DLP latents are **not
bit-reproducible across processes** — all paired analysis must use tensors from
within this one file.
