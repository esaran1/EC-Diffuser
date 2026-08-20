"""Quantify imagined-state coherence for Gaussian vs Flow.

The visual impression from the decoded figures is that Flow's imagined cubes
duplicate and smear while Gaussian's stay distinct. Visual impression is not
evidence, so this measures it in DLP particle space directly.

Metrics, all computed on the *generated* particles of the front view:

  particle_spread     mean pairwise distance among high-transparency particles
  n_active            count of particles above the transparency threshold
  nn_distance         mean nearest-neighbour distance among active particles
                      (small => duplicated particles piled on one another)
  feature_dispersion  std of visual features among active particles
                      (identity smearing shows up as reduced separation)
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
TRANSP = 0.5  # particles below this are "off" per the DLP transparency channel


def load_policy(arm, args):
    spec = ARMS[arm]
    experiment = utils.load_diffusion(
        spec["loadbase"], args.dataset, spec["loadpath"],
        epoch="latest", seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    )
    return utils.Config(
        "sampling.GoalConditionedPolicy",
        diffusion_model=experiment.ema,
        normalizer=experiment.dataset.normalizer,
        preprocess_fns=[], verbose=False, horizon=args.horizon,
        **flow_sampling_kwargs(experiment.ema, spec["n_diffusion_steps"]),
    )()


def describe_particles(particles):
    """particles: (24, 10) generated DLP set for one view at one horizon step."""
    transparency = particles[:, 9]
    active = particles[transparency > TRANSP]
    if len(active) < 2:
        return None

    positions = active[:, :2]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    off_diagonal = dist[~np.eye(len(dist), dtype=bool)]
    nn = np.min(dist + np.eye(len(dist)) * 1e9, axis=1)

    return {
        "n_active": int(len(active)),
        "particle_spread": float(off_diagonal.mean()),
        "nn_distance": float(nn.mean()),
        "feature_dispersion": float(active[:, 5:9].std(axis=0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    cli = parser.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length

    with open(EPISODE_SET, "rb") as handle:
        episodes = pickle.load(handle)
    keys = episodes["keys"]

    report = {}
    for arm in ("gaussian", "flow"):
        policy = load_policy(arm, args)
        rows = []
        # Also measure the *encoded real* observation as a reference ceiling.
        real_rows = []

        for start in range(0, cli.episodes, env.num_envs):
            index = list(range(start, min(start + env.num_envs, cli.episodes)))
            index += [index[-1]] * (env.num_envs - len(index))
            obs = env.reset(
                set_init_states=array_to_state_dict(episodes["init"][index], keys, env.device),
                set_goal_states=array_to_state_dict(episodes["goal"][index], keys, env.device),
            )
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            _, samples = policy({0: observation, args.horizon - 1: goal},
                                batch_size=1, verbose=False)

            generated = np.asarray(samples.observations).reshape(
                env.num_envs, args.horizon, 2, N_PER_VIEW, 10
            )
            real = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)

            for e in range(env.num_envs):
                stats = describe_particles(real[e, 0])
                if stats:
                    real_rows.append(stats)
                # Interior horizon steps only: 0 and H-1 are pinned conditions.
                for h in range(1, args.horizon - 1):
                    stats = describe_particles(generated[e, h, 0])
                    if stats:
                        rows.append(stats)

        def mean(rs, key):
            return float(np.mean([r[key] for r in rs]))

        report[arm] = {
            key: mean(rows, key)
            for key in ("n_active", "particle_spread", "nn_distance", "feature_dispersion")
        }
        report[arm]["n_samples"] = len(rows)
        report.setdefault("real_encoded", {
            key: mean(real_rows, key)
            for key in ("n_active", "particle_spread", "nn_distance", "feature_dispersion")
        })

    os.makedirs("experiments/isaacgym_control", exist_ok=True)
    out = "experiments/isaacgym_control/imagination_stats.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
