"""Regression tests for fixed held-out Phase 7 validation."""

from collections import namedtuple

import pytest
import torch
from torch import nn

from diffuser.scripts.train_phase7_pilot import fixed_validation, resolve_training
from diffuser.utils.arrays import set_global_device


Batch = namedtuple("Batch", "trajectories conditions")


class StochasticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.training_states = []

    def loss(self, trajectories, conditions):
        self.training_states.append(self.training)
        noise = torch.randn_like(trajectories)
        value = ((self.scale * trajectories + noise) ** 2).mean()
        return value, {"metric": value.detach()}


def test_fixed_validation_is_deterministic_and_side_effect_free():
    model = StochasticLoss()
    batches = [Batch(torch.ones(3, 2), {0: torch.ones(3, 1)})]
    set_global_device("cpu")
    try:
        rng_before = torch.random.get_rng_state().clone()
        first = fixed_validation(model, batches, seed=123, device=torch.device("cpu"))
        rng_after = torch.random.get_rng_state().clone()
        second = fixed_validation(model, batches, seed=123, device=torch.device("cpu"))
    finally:
        set_global_device("cuda:0")
    assert first == second
    torch.testing.assert_close(rng_after, rng_before)
    assert model.training
    assert model.training_states == [False, False]


def test_replication_seed_must_be_predeclared_and_exact():
    protocol = {
        "training": {"seed": 42, "learning_rate": 4e-5},
        "replications": {
            "43": {
                "training_overrides": {
                    "seed": 43,
                    "dataloader_seed": 200043,
                    "optimization_seed": 300043,
                }
            }
        },
    }

    resolved = resolve_training(protocol, replication_seed=43)
    assert resolved == {
        "seed": 43,
        "learning_rate": 4e-5,
        "dataloader_seed": 200043,
        "optimization_seed": 300043,
    }
    assert protocol["training"]["seed"] == 42
    with pytest.raises(ValueError, match="requires --replication-seed"):
        resolve_training(protocol)
    with pytest.raises(ValueError, match="not predeclared"):
        resolve_training(protocol, replication_seed=44)


def test_replication_seed_is_rejected_for_single_run_protocol():
    protocol = {"training": {"seed": 42}}
    assert resolve_training(protocol) == {"seed": 42}
    with pytest.raises(ValueError, match="does not declare"):
        resolve_training(protocol, replication_seed=43)
