"""Pins the pairing and labelling guarantees of the Isaac Gym NFE study.

The study's validity rests on two claims that are easy to break silently:
every arm within a replicate must see byte-identical episodes, and the
requested NFE must equal the number of model calls actually made.
"""

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from diffuser.configuration import flow_sampling_kwargs
from diffuser.models import ConditionalFlowMatching, GaussianDiffusion


HORIZON = 5
ACTION_DIM = 3
OBSERVATION_DIM = 10


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward(self, x, cond, time):
        self.calls += 1
        return self.scale * x


def make_cond(batch=2):
    torch.manual_seed(0)
    return {
        0: torch.randn(batch, OBSERVATION_DIM),
        HORIZON - 1: torch.randn(batch, OBSERVATION_DIM),
    }


@pytest.mark.parametrize("requested", [1, 2, 4, 8, 16])
def test_requested_nfe_equals_actual_model_calls(requested):
    """The NFE axis is only meaningful if the solver honours the request.

    Note 16 exceeds the checkpoint's trained default of 4 solver steps, which
    the study deliberately evaluates, so it must work too.
    """
    model = CountingModel()
    flow = ConditionalFlowMatching(
        model=model, horizon=HORIZON,
        observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM,
        n_solver_steps=4,
    )
    cond = make_cond()

    model.calls = 0
    sample = flow.conditional_sample(cond, n_steps=requested)

    assert model.calls == requested
    assert torch.isfinite(sample.trajectories).all()


def test_flow_sampling_kwargs_overrides_flow_but_not_diffusion():
    """The step override must reach flow arms and leave Gaussian untouched.

    If it leaked into GaussianDiffusion the reference arm would silently stop
    running at its trained 100 steps.
    """
    flow = ConditionalFlowMatching(
        model=CountingModel(), horizon=HORIZON,
        observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, n_solver_steps=4,
    )
    assert flow_sampling_kwargs(flow, 16) == {"n_steps": 16}

    diffusion = GaussianDiffusion(
        model=CountingModel(), horizon=HORIZON,
        observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM,
        n_timesteps=10,
    )
    assert flow_sampling_kwargs(diffusion, 16) == {}


def test_episode_set_hash_detects_any_state_change():
    """Pairing is enforced by a hash over initial and goal states."""
    rng = np.random.RandomState(0)
    init = rng.randn(96, 3, 2).astype(np.float32)
    goal = rng.randn(96, 3, 2).astype(np.float32)

    def digest(i, g):
        return hashlib.sha256(i.tobytes() + g.tobytes()).hexdigest()

    baseline = digest(init, goal)
    assert digest(init.copy(), goal.copy()) == baseline

    # A single perturbed coordinate must change the hash.
    perturbed = init.copy()
    perturbed[41, 2, 1] += 1e-4
    assert digest(perturbed, goal) != baseline

    # Swapping two episodes changes it too: the set is ordered.
    reordered = init.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    assert digest(reordered, goal) != baseline


def test_replicates_are_generated_from_distinct_seeds():
    """Replicates must be independent sets, not the same episodes relabelled.

    The study seeds replicate r with 20260820 + 1000*r; this pins that the
    resulting seeds are distinct so the sets cannot coincide by construction.
    """
    seeds = [20260820 + 1000 * r for r in range(3)]
    assert len(set(seeds)) == 3

    draws = [np.random.RandomState(s).randn(96, 3, 2) for s in seeds]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.allclose(draws[i], draws[j])
