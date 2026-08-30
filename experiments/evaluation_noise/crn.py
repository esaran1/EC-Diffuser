"""Common-Random-Numbers evaluation wrapper. DIAGNOSTIC ONLY.

Canonical training/sampling code is NOT modified. `conditional_sample` draws its
initial noise from the global torch RNG with no injection point, so CRN is
achieved by *seeding that RNG deterministically immediately before each policy
invocation*. The sampler then draws the identical tensor from the identical
distribution -- this changes the coupling between experimental arms, never the
marginal distribution of either arm.

Noise is keyed by (episode-batch, decision index), NOT by physical state, so the
two arms receive the same exogenous random sequence even after their
trajectories diverge (protocol section 7).
"""
import hashlib
import numpy as np
import torch

CRN_BASE_SEED = 20260905


def derive_seed(base, batch_start, decision):
    """Deterministic, order-independent seed derivation. Documented and frozen."""
    h = hashlib.sha256(f"{base}|{batch_start}|{decision}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


class CRNPolicyWrapper:
    """Wraps a canonical policy; seeds the RNG before each call. No other change."""

    def __init__(self, policy, base_seed=CRN_BASE_SEED, enabled=True):
        self.policy = policy
        self.base_seed = base_seed
        self.enabled = enabled
        self.batch_start = 0
        self.decision = 0
        self.seeds_used = []

    def new_batch(self, batch_start):
        self.batch_start = batch_start
        self.decision = 0

    def __call__(self, *args, **kwargs):
        if self.enabled:
            s = derive_seed(self.base_seed, self.batch_start, self.decision)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            self.seeds_used.append(s)
        self.decision += 1
        return self.policy(*args, **kwargs)

    # transparently expose the canonical policy's diagnostics
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "policy"), name)
