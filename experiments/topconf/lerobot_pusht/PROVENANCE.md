# Provenance — lerobot/diffusion_pusht external breadth experiment

## Model artifact

| item | value |
|---|---|
| repo | `lerobot/diffusion_pusht` (HuggingFace) |
| revision (pinned) | `84a7c23178445c6bbf7e1a884ff497017910f653` |
| last modified | 2025-03-06 |
| `model.safetensors` | 1,050,862,408 B · sha256 `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8` |
| `config.json` | 1,509 B · sha256 `d391a7bf488accd1c26b2043482f0060b0855b1ec236f3d7358486918472c0a5` |
| `train_config.json` | 5,939 B · sha256 `500ea79ba1bef13810219697ce60eca580a0f259f5c9bf4f847f8f843b06b14a` |

## Dataset

`lerobot/pusht` — 206 episodes, 25,650 frames, fps 10. Downloaded via
`LeRobotDataset`. **Total on-disk for model + dataset + cache: 1021 MB**
(approved cap 1.20 GB).

## Environment isolation

Separate conda env `lerobot-pusht`. **The EC-Diffuser env was not modified.**

See `env_lock.txt`. Key versions: python 3.10.21, torch 2.10.0+cu128 (CUDA 12.8),
diffusers 0.35.2, **lerobot 0.3.2** (see the version finding below), gym-pusht,
ffmpeg (conda-forge, required by torchcodec for dataset video decoding).

## Model / scheduler configuration (unchanged throughout)

`noise_scheduler_type: DDPM` · `num_train_timesteps: 100` ·
`beta_schedule: squaredcos_cap_v2` · `beta_start 1e-4`, `beta_end 0.02` ·
`horizon: 16` · `n_obs_steps: 2` · `n_action_steps: 8` ·
`num_inference_steps: null` (defaults to 100).

Only `num_inference_steps` is varied. Terminology used throughout:
**scheduler-supported reduced-step DDPM inference using the same pretrained
weights and DDPM scheduler family.** This is *not* described as truncation or
subsampling of the original ancestral Markov chain.

## Published reference (from the repo's own `eval_info.json`)

500 episodes, seeds **1000–1499** (contiguous, verified):
`pc_success = 65.4`, `avg_max_reward = 0.9551318575760519`,
`avg_sum_reward = 104.838`. Per-episode records are included, which allowed
seed-exact validation rather than a distributional comparison.

## Frozen closed-loop episode set

seeds **1000–1049** (n=50), `gym_pusht/PushT-v0`, `obs_type=pixels_agent_pos`,
`max_episode_steps=300`, success = `max(reward) > 0.95`.
Spec sha256 `86867a8e3d561451febfdb77ea395fe47e8692a6be3529f26ed262d19472623f`,
frozen in `frozen_episode_set.json` **before any arm was run**. The identical set
is used for all eight budgets.
