"""Batch-invariant per-episode policy noise.

Each (realization, scenario set, episode, control step) gets its own torch CPU
generator seeded through numpy SeedSequence (stable across processes and
platforms; no Python hash()). Noise for an episode therefore does not depend
on which other episodes share its batch, the batch size, or batch order."""
import numpy as np
import torch

NOISE_SEED_V2 = 20261024


def set_key(sha256_hex):
    """Stable integer id of a scenario set from its content hash."""
    return int(sha256_hex[:12], 16)


def episode_seed(realization, set_id, episode, step, base=NOISE_SEED_V2):
    ss = np.random.SeedSequence([base, int(realization), int(set_id), int(episode), int(step)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def batch_noise(realization, set_id, episodes, step, shape, device, dtype=torch.float32):
    """[len(episodes), *shape] noise; row i depends only on episodes[i]."""
    rows = []
    for e in episodes:
        g = torch.Generator(device="cpu").manual_seed(episode_seed(realization, set_id, e, step))
        rows.append(torch.randn(shape, generator=g, dtype=torch.float32))
    return torch.stack(rows).to(device=device, dtype=dtype)


COLOR_SEED = 20261025


def episode_colors(set_id, episode, num_colors, num_objects):
    """Per-episode cube color permutation: a property of the scenario (set,
    episode), shared by every config/realization/batch size. Same distribution
    as the env's own randperm(num_colors)[:num_objects], which instead draws one
    permutation per batch reset from the global torch RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([COLOR_SEED, int(set_id), int(episode)]))
    return rng.permutation(num_colors)[:num_objects].tolist()
