"""Repository-native reload, conditioned sampling, and solver-NFE audit."""

import os
import torch

import diffuser.utils as utils
from diffuser.models import ConditionalFlowMatching, AdaLNPINTDenoiser
from diffuser.sampling import GoalConditionedPolicy
from diffuser.utils.arrays import set_global_device


def main():
    set_global_device("cuda:0")
    root = "data/panda_push/flow_smoke/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
    checkpoint = os.path.join(root, "state_200.pt")
    raw = torch.load(checkpoint, map_location="cuda:0")
    assert set(raw) == {"step", "model", "ema"}
    assert raw["step"] == 200

    experiment = utils.load_diffusion(
        root, epoch=200, device="cuda:0", seed=42,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    )
    assert isinstance(experiment.diffusion, ConditionalFlowMatching)
    assert isinstance(experiment.diffusion.model, AdaLNPINTDenoiser)
    assert isinstance(experiment.ema, ConditionalFlowMatching)
    assert isinstance(experiment.ema.model, AdaLNPINTDenoiser)
    assert experiment.trainer.step == 200
    for key, value in experiment.trainer.model.state_dict().items():
        assert torch.equal(value, raw["model"][key])
    for key, value in experiment.ema.state_dict().items():
        assert torch.equal(value, raw["ema"][key])

    batch = utils.batchify(experiment.dataset[0])
    conditions = batch.conditions
    sample = experiment.ema(conditions, n_steps=4, verbose=False)
    assert sample.trajectories.shape == (1, 5, 483)
    assert torch.isfinite(sample.trajectories).all()
    for timestep, condition in conditions.items():
        assert torch.equal(sample.trajectories[:, timestep, 3:], condition)

    unnormalized_conditions = {
        timestep: experiment.dataset.normalizer.unnormalize(value.detach().cpu().numpy(), "observations")
        for timestep, value in conditions.items()
    }
    observed = {}
    for requested in (1, 2, 4, 8):
        calls = [0]
        def count_call(_module, _inputs, _output):
            calls[0] += 1
        handle = experiment.ema.model.register_forward_hook(count_call)
        policy = GoalConditionedPolicy(
            diffusion_model=experiment.ema, normalizer=experiment.dataset.normalizer,
            preprocess_fns=[], horizon=5, n_steps=requested,
        )
        _action, trajectories = policy(unnormalized_conditions, batch_size=1, verbose=False)
        handle.remove()
        assert calls[0] == requested
        assert torch.isfinite(torch.as_tensor(trajectories.actions)).all()
        observed[requested] = calls[0]

    print("checkpoint:", checkpoint)
    print("checkpoint_size:", os.path.getsize(checkpoint))
    print("trainer_step:", experiment.trainer.step)
    print("wrapper:", type(experiment.ema).__name__)
    print("denoiser:", type(experiment.ema.model).__name__)
    print("sample_shape:", tuple(sample.trajectories.shape))
    print("sample_finite:", bool(torch.isfinite(sample.trajectories).all()))
    print("conditioning_exact: True")
    print("nfe_counts:", observed)
    print("FLOW CHECKPOINT RELOAD AND NFE AUDIT: PASS")


if __name__ == "__main__":
    main()
