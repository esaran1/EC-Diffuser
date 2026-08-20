"""Item 7: compare each arm's predicted next DLP state against the realized one.

At state s_t the model generates a trajectory whose horizon slot 1 is its
prediction of s_{t+1}. We execute the model's own first action, encode the
frame that actually results, and compare.

Particle identity is NOT fixed across DLP encodings, so a naive index-wise
subtraction would be meaningless. Distances are therefore computed as a
symmetric Chamfer distance over the particle set, which is permutation
invariant.

A low error means the model's imagined dynamics are internally correct; a high
error with good task performance would mean a coherent policy sitting on top of
an incorrect world model.
"""

import argparse  # noqa: E402
import json
import os
import pickle

import isaacgym  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.configuration import flow_sampling_kwargs  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402

from isaacgym_control import ARMS, Args, EPISODE_SET, array_to_state_dict  # noqa: E402

N_PER_VIEW = 24


def chamfer(a, b):
    """Symmetric Chamfer distance between two (P, D) particle sets.

    Permutation invariant, which is required because DLP particle ordering is
    not stable between encodings.
    """
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    dist = torch.cdist(a, b)  # (P, P)
    return float(0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()))


def load_policy(arm, args, epoch="latest"):
    spec = ARMS[arm]
    experiment = utils.load_diffusion(
        spec["loadbase"], args.dataset, spec["loadpath"],
        epoch=epoch, seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    )
    return utils.Config(
        "sampling.GoalConditionedPolicy",
        diffusion_model=experiment.ema,
        normalizer=experiment.dataset.normalizer,
        preprocess_fns=[], verbose=False, horizon=args.horizon,
        **flow_sampling_kwargs(experiment.ema, spec["n_diffusion_steps"]),
    )()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    cli = parser.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length

    with open(EPISODE_SET, "rb") as handle:
        episodes = pickle.load(handle)
    keys = episodes["keys"]

    arms = {
        "gaussian": ("gaussian", "latest"),
        "flow_300k": ("flow", 300000),
        "flow_500k": ("flow", "latest"),
    }

    report = {}
    for label, (arm, epoch) in arms.items():
        policy = load_policy(arm, args, epoch)
        errors, baselines = [], []

        index = list(range(min(cli.episodes, env.num_envs)))
        index += [index[-1]] * (env.num_envs - len(index))
        obs = env.reset(
            set_init_states=array_to_state_dict(episodes["init"][index], keys, env.device),
            set_goal_states=array_to_state_dict(episodes["goal"][index], keys, env.device),
        )

        for _ in range(cli.steps):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            _, samples = policy({0: observation, args.horizon - 1: goal},
                                batch_size=1, verbose=False)

            predicted = np.asarray(samples.observations).reshape(
                env.num_envs, args.horizon, 2, N_PER_VIEW, 10
            )[:, 1, 0]  # horizon slot 1, front view
            current = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]

            obs, _, _, _ = env.step(samples.actions[:, 0])
            realized = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]

            for e in range(env.num_envs):
                errors.append(chamfer(predicted[e], realized[e]))
                # Baseline: how far the world moved in one step at all. If the
                # prediction error is not below this, the model is not
                # predicting dynamics -- it is at best copying the input.
                baselines.append(chamfer(current[e], realized[e]))

        report[label] = {
            "chamfer_pred_vs_realized": float(np.mean(errors)),
            "chamfer_current_vs_realized": float(np.mean(baselines)),
            "ratio_pred_over_copy": float(np.mean(errors) / np.mean(baselines)),
            "n": len(errors),
        }
        print(label, json.dumps(report[label], indent=2), flush=True)

    out = "experiments/isaacgym_control/predicted_vs_realized.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
