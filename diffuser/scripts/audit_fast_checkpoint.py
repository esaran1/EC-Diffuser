"""Repository-native reload and conditioned NFE audit for interval flow models."""

import argparse
import os

import torch

import diffuser.utils as utils
from diffuser.models import (
    ImprovedMeanFlow,
    IntervalAdaLNPINTDenoiser,
    ShortcutModel,
)
from diffuser.sampling import GoalConditionedPolicy
from diffuser.utils.arrays import set_global_device


METHODS = {
    "improved_meanflow": ImprovedMeanFlow,
    "shortcut": ShortcutModel,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--loadpath", required=True)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_device(args.device)
    checkpoint = os.path.join(args.loadpath, "state_{}.pt".format(args.epoch))
    raw = torch.load(checkpoint, map_location=args.device)
    assert set(raw) == {"step", "model", "ema"}
    assert raw["step"] == args.epoch

    experiment = utils.load_diffusion(
        args.loadpath,
        epoch=args.epoch,
        device=args.device,
        seed=42,
        override_dataset_path=args.dataset,
    )
    wrapper_type = METHODS[args.method]
    assert isinstance(experiment.diffusion, wrapper_type)
    assert isinstance(experiment.ema, wrapper_type)
    assert isinstance(experiment.diffusion.model, IntervalAdaLNPINTDenoiser)
    assert isinstance(experiment.ema.model, IntervalAdaLNPINTDenoiser)
    assert experiment.trainer.step == args.epoch
    for key, value in experiment.trainer.model.state_dict().items():
        assert torch.equal(value, raw["model"][key])
    for key, value in experiment.ema.state_dict().items():
        assert torch.equal(value, raw["ema"][key])

    batch = utils.batchify(experiment.dataset[0])
    conditions = batch.conditions
    observed = {}
    budgets = (1, 2, 4, 8)
    sample = None
    for requested in budgets:
        calls = [0]

        def count_call(_module, _inputs, _output):
            calls[0] += 1

        handle = experiment.ema.model.register_forward_hook(count_call)
        try:
            sample = experiment.ema(conditions, n_steps=requested, verbose=False)
        finally:
            handle.remove()
        assert calls[0] == requested
        assert sample.trajectories.shape == (
            1,
            experiment.ema.horizon,
            experiment.ema.action_dim + experiment.ema.observation_dim,
        )
        assert torch.isfinite(sample.trajectories).all()
        for timestep, condition in conditions.items():
            assert torch.equal(
                sample.trajectories[:, timestep, experiment.ema.action_dim:],
                condition,
            )
        observed[requested] = calls[0]

    unnormalized_conditions = {
        timestep: experiment.dataset.normalizer.unnormalize(
            value.detach().cpu().numpy(), "observations"
        )
        for timestep, value in conditions.items()
    }
    policy = GoalConditionedPolicy(
        diffusion_model=experiment.ema,
        normalizer=experiment.dataset.normalizer,
        preprocess_fns=[],
        horizon=experiment.ema.horizon,
        n_steps=1,
    )
    action, trajectories = policy(
        unnormalized_conditions, batch_size=1, verbose=False
    )
    assert torch.isfinite(torch.as_tensor(action)).all()
    assert torch.isfinite(torch.as_tensor(trajectories.actions)).all()

    print("checkpoint:", checkpoint)
    print("checkpoint_size_bytes:", os.path.getsize(checkpoint))
    print("trainer_step:", experiment.trainer.step)
    print("wrapper:", type(experiment.ema).__name__)
    print("denoiser:", type(experiment.ema.model).__name__)
    print("sample_shape:", tuple(sample.trajectories.shape))
    print("sample_finite: True")
    print("conditioning_exact: True")
    print("policy_action_finite: True")
    print("nfe_counts:", observed)
    print("FAST CHECKPOINT RELOAD AND NFE AUDIT: PASS")


if __name__ == "__main__":
    main()
