"""Regression coverage for optimizer-step logging and finite-gradient guards."""

import pytest
import torch
from torch import nn

import diffuser.utils.training as training_module
from diffuser.utils.training import Trainer


class SequencedLossModel(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.losses = iter(losses)

    def loss(self, *_batch):
        value = self.anchor * 0.0 + next(self.losses)
        return value, {"metric": value * 10.0}


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0.0 + 1.0

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class NonfiniteGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def loss(self, *_batch):
        value = InfiniteGradient.apply(self.anchor)
        return value, {"metric": value.detach()}


def make_trainer(model, gradient_accumulate_every=2):
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.gradient_accumulate_every = gradient_accumulate_every
    trainer.dataloader = iter([()] * gradient_accumulate_every)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.step = 1
    trainer.update_ema_every = 10
    trainer.save_freq = 10
    trainer.log_freq = 1
    trainer.sample_freq = 0
    return trainer


def test_train_logs_mean_unscaled_microbatch_loss(monkeypatch):
    trainer = make_trainer(SequencedLossModel([2.0, 4.0]))
    logged = []
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    monkeypatch.setattr(training_module.wandb, "log", logged.append)

    trainer.train(1)

    assert len(logged) == 1
    assert logged[0]["step"] == 1
    torch.testing.assert_close(logged[0]["loss"], torch.tensor(3.0))
    torch.testing.assert_close(logged[0]["metric"], torch.tensor(30.0))
    assert not logged[0]["loss"].requires_grad
    assert not logged[0]["metric"].requires_grad


def test_train_rejects_nonfinite_gradients_before_optimizer_step(monkeypatch):
    trainer = make_trainer(NonfiniteGradientModel(), gradient_accumulate_every=1)
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    original = trainer.model.anchor.detach().clone()

    with pytest.raises(FloatingPointError, match="non-finite gradients at step 1"):
        trainer.train(1)

    torch.testing.assert_close(trainer.model.anchor.detach(), original)
