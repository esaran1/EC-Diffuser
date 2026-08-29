"""Permutation-invariant DLP state metrics, defined from representation semantics.

Design decisions, fixed BEFORE looking at any Flow-vs-Gaussian numbers:

  * Matching uses POSITION ONLY (dims 0:2). Position is the one channel with a
    common geometric meaning across particles, and matching on it is what makes
    the correspondence interpretable. Matching on raw 10-D would be dominated by
    the visual-feature channels, whose range is ~24 vs 0.08 for transparency.

  * Reported costs are separated by semantic block rather than summed into one
    heterogeneous number:
        pos   (dims 0:2)   - geometric
        scale (dims 2:4)
        depth (dim  4)
        vis   (dims 5:9)   - appearance
        transp(dim  9)
    Each is z-scored by per-dimension training statistics so "one unit" means
    the same thing across blocks.

  * Chamfer is reported alongside Hungarian as an assumption-light cross-check.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

POS = slice(0, 2); SCALE = slice(2, 4); DEPTH = slice(4, 5)
VIS = slice(5, 9); TRANSP = slice(9, 10)
BLOCKS = {"pos": POS, "scale": SCALE, "depth": DEPTH, "vis": VIS, "transp": TRANSP}


def training_stats(obs):
    """Per-dimension mean/std over training-distribution particles, shape (...,10)."""
    f = obs.reshape(-1, obs.shape[-1])
    return f.mean(0), f.std(0) + 1e-8


def hungarian_match(a, b):
    """Optimal particle correspondence by POSITION. a,b: (N,10) -> permutation of b."""
    cost = np.linalg.norm(a[:, None, POS] - b[None, :, POS], axis=-1)
    r, c = linear_sum_assignment(cost)
    return r, c


def block_errors(a, b, mu, sd, matched=True):
    """Per-block mean |error| after matching, in z-scored units."""
    if matched:
        r, c = hungarian_match(a, b)
        a2, b2 = a[r], b[c]
    else:
        a2, b2 = a, b
    az, bz = (a2 - mu) / sd, (b2 - mu) / sd
    return {k: float(np.abs(az[:, s] - bz[:, s]).mean()) for k, s in BLOCKS.items()}


def chamfer_position(a, b):
    """Symmetric Chamfer distance on position only (no correspondence assumed)."""
    d = np.linalg.norm(a[:, None, POS] - b[None, :, POS], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))
