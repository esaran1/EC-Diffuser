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


def test_linear_warmup_uses_optimizer_step_index(monkeypatch):
    trainer = make_trainer(SequencedLossModel([1.0] * 8))
    trainer.train_lr = 0.1
    trainer.lr_warmup_steps = 3
    trainer.collect_step_diagnostics = True
    trainer.max_grad_norm = None
    trainer.step = 0
    trainer.save_freq = 0
    trainer.step_ema = lambda: None
    trainer.dataloader = iter([()] * 8)
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    monkeypatch.setattr(training_module.wandb, "log", lambda record: None)

    history = trainer.train(4)

    assert [record["learning_rate"] for record in history] == pytest.approx(
        [0.0, 0.1 / 3.0, 0.2 / 3.0, 0.1]
    )


def test_train_rejects_nonfinite_gradients_before_optimizer_step(monkeypatch):
    trainer = make_trainer(NonfiniteGradientModel(), gradient_accumulate_every=1)
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    original = trainer.model.anchor.detach().clone()

    with pytest.raises(FloatingPointError, match="non-finite gradients at step 1"):
        trainer.train(1)

    torch.testing.assert_close(trainer.model.anchor.detach(), original)


def test_train_returns_machine_readable_history_and_allows_disabled_saves(monkeypatch):
    trainer = make_trainer(SequencedLossModel([2.0, 4.0]))
    trainer.save_freq = 0
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    monkeypatch.setattr(training_module.wandb, "log", lambda record: None)

    history = trainer.train(1)

    assert history == trainer.train_history
    assert history[0]["step"] == 1
    assert history[0]["loss"] == pytest.approx(3.0)
    assert history[0]["metric"] == pytest.approx(30.0)
    assert history[0]["interval_seconds"] >= 0.0


def test_gradient_clipping_and_update_diagnostics_are_exact(monkeypatch):
    model = SequencedLossModel([])
    model.anchor.data.fill_(1.0)

    def loss(*_batch):
        value = (model.anchor - 3.0).square()
        return value, {"metric": value.detach()}

    model.loss = loss
    trainer = make_trainer(model, gradient_accumulate_every=1)
    trainer.max_grad_norm = 0.5
    trainer.collect_step_diagnostics = True
    monkeypatch.setattr(training_module, "batch_to_device", lambda batch: batch)
    monkeypatch.setattr(training_module.wandb, "log", lambda record: None)

    history = trainer.train(1)
    record = history[-1]
    assert record["gradient_l2_preclip"] == pytest.approx(4.0)
    assert record["gradient_max_abs_preclip"] == pytest.approx(4.0)
    assert record["gradient_l2_postclip"] == pytest.approx(0.5)
    assert record["gradient_max_abs_postclip"] == pytest.approx(0.5)
    assert record["parameter_l2"] == pytest.approx(1.0)
    assert record["update_l2"] == pytest.approx(0.05)
    assert record["update_to_parameter_ratio"] == pytest.approx(0.05)
    assert model.anchor.item() == pytest.approx(1.05)


def test_trainer_validates_optimizer_configuration():
    dataset = [torch.zeros(1)]

    with pytest.raises(ValueError, match="adam_betas"):
        Trainer(
            nn.Linear(1, 1), dataset, renderer=None,
            sample_freq=0, save_freq=0, adam_betas=(0.9, 1.0),
        )
    with pytest.raises(ValueError, match="adam_betas"):
        Trainer(
            nn.Linear(1, 1), dataset, renderer=None,
            sample_freq=0, save_freq=0, adam_betas=(0.9,),
        )
    with pytest.raises(TypeError, match="lr_warmup_steps"):
        Trainer(
            nn.Linear(1, 1), dataset, renderer=None,
            sample_freq=0, save_freq=0, lr_warmup_steps=True,
        )
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        Trainer(
            nn.Linear(1, 1), dataset, renderer=None,
            sample_freq=0, save_freq=0, lr_warmup_steps=-1,
        )

    trainer = Trainer(
        nn.Linear(1, 1), dataset, renderer=None,
        sample_freq=0, save_freq=0, adam_betas=(0.9, 0.95),
        lr_warmup_steps=16,
    )
    assert trainer.optimizer.defaults["betas"] == (0.9, 0.95)
    assert trainer.lr_warmup_steps == 16

    with pytest.raises(TypeError, match="dataloader_seed"):
        Trainer(
            nn.Linear(1, 1), dataset, renderer=None,
            sample_freq=0, save_freq=0, dataloader_seed=True,
        )
    seeded = Trainer(
        nn.Linear(1, 1), dataset, renderer=None,
        sample_freq=0, save_freq=0, dataloader_seed=42,
    )
    assert seeded.dataloader_seed == 42
