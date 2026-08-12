# Seed-control audit

- `ArgsParser.set_seed` seeds Python `random`, NumPy, CPU PyTorch, and every CUDA generator before model, dataset, trainer, and environment construction.
- Training seed therefore controls parameter initialization, shuffled DataLoader sampling (`num_workers=1` inherits a deterministically generated worker seed), flow/noise/time sampling, and optimizer trajectory. The repository does not enable PyTorch deterministic algorithms or set cuBLAS workspace determinism, so bitwise replay across hardware/library changes is not promised.
- Evaluation seed is passed independently on the command line and seeds the process before Isaac Gym construction. Cube/goal placement uses global Torch RNG; random-color/object-count branches also use global Torch/NumPy RNG.
- `SB3VecEnvAdapter.seed()` is a no-op and Isaac Gym receives no explicit simulator seed. Evaluation reproducibility is therefore defined as a fresh process per evaluation seed. Re-seeding an already-constructed environment is unsupported.
- Evaluation-seed repeats condition on one fixed trained checkpoint. They estimate scenario/simulator variability, not uncertainty from training.
- Independent training seeds require separate initialization, data order, optimization, and checkpoints. Final statistical inference uses training seed as the replicate.
