"""Items 5 and 6: DLP reconstruction quality, then decoded model imagination.

Order matters. Section A verifies RGB -> DLP -> RGB on real Isaac Gym frames.
Only if that is trustworthy does section B decode the *generated* DLP futures,
because a poor decoder would make every imagination figure uninterpretable.

Observations are multiview: the 48 particles the policy sees are two stacked
24-particle views (isaac_env_wrappers.py:551), so each view is decoded
separately with its own background latent.
"""

import argparse  # noqa: E402
import os
import pickle

# isaacgym binds CUDA before torch and raises if torch is imported first.
import isaacgym  # noqa: F401,E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.configuration import flow_sampling_kwargs  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from dlp_utils import extract_dlp_features_with_bg, get_recon_from_dlps  # noqa: E402

from isaacgym_control import ARMS, Args, EPISODE_SET, array_to_state_dict  # noqa: E402

OUT = "experiments/figures"
N_PARTICLES_PER_VIEW = 24


def decode_view(particles, z_bg, model, device):
    """Decode one view's (24, 10) particle set back to an RGB image."""
    particles = torch.as_tensor(particles, device=device, dtype=torch.float32)
    if particles.dim() == 2:
        particles = particles.unsqueeze(0)
    z_bg = torch.as_tensor(z_bg, device=device, dtype=torch.float32)
    if z_bg.dim() == 1:
        z_bg = z_bg.unsqueeze(0)
    return get_recon_from_dlps(particles, z_bg, model, device)


def section_a_reconstruction(env, n_frames=6):
    """RGB -> DLP -> RGB on representative Isaac Gym frames."""
    device = env.device
    model = env.latent_rep_model

    env.reset()
    frames, recons, errors = [], [], []

    # Sample frames across an episode: reset, approach, mid, late.
    action = np.zeros((env.num_envs, 3), dtype=np.float32)
    for step in range(n_frames):
        for _ in range(4):  # advance a few steps between captures
            action[:] = np.random.uniform(-0.6, 0.6, size=action.shape)
            _, _, _, infos = env.step(action)
        image = infos[0]["image"][0]  # front view, (3, H, W) uint8
        particles, z_bg = extract_dlp_features_with_bg(image, model, device)
        recon = decode_view(particles[0], z_bg[0], model, device)

        original = np.moveaxis(image, 0, -1).astype(np.uint8)
        frames.append(original)
        recons.append(recon)
        errors.append(float(np.abs(original.astype(float) - recon.astype(float)).mean()))

    fig, axes = plt.subplots(2, n_frames, figsize=(2.6 * n_frames, 5.4))
    for i in range(n_frames):
        axes[0, i].imshow(frames[i])
        axes[0, i].set_title(f"RGB frame {i}", fontsize=9)
        axes[1, i].imshow(recons[i])
        axes[1, i].set_title(f"DLP recon (MAE {errors[i]:.1f})", fontsize=9)
        for row in (0, 1):
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstruction")
    plt.suptitle(
        f"Isaac Gym DLP reconstruction — mean pixel MAE {np.mean(errors):.1f}/255",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(OUT, "dlp_reconstruction.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path, errors


def load_policy(arm, args):
    spec = ARMS[arm]
    experiment = utils.load_diffusion(
        spec["loadbase"], args.dataset, spec["loadpath"],
        epoch="latest", seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    )
    diffusion = experiment.ema
    policy = utils.Config(
        "sampling.GoalConditionedPolicy",
        diffusion_model=diffusion,
        normalizer=experiment.dataset.normalizer,
        preprocess_fns=[], verbose=False, horizon=args.horizon,
        **flow_sampling_kwargs(diffusion, spec["n_diffusion_steps"]),
    )()
    return policy, experiment


def section_b_imagination(env, args, episodes, n_show=3):
    """Decode each arm's generated DLP future for identical current/goal pairs."""
    device = env.device
    model = env.latent_rep_model
    horizon = args.horizon

    policies = {arm: load_policy(arm, args) for arm in ("gaussian", "flow")}
    keys = episodes["keys"]

    rows = {}
    for arm, (policy, _) in policies.items():
        obs = env.reset(
            set_init_states=array_to_state_dict(episodes["init"][:env.num_envs], keys, device),
            set_goal_states=array_to_state_dict(episodes["goal"][:env.num_envs], keys, device),
        )
        observation = obs["achieved_goal"].reshape(env.num_envs, -1)
        goal = obs["desired_goal"].reshape(env.num_envs, -1)
        conditions = {0: observation, args.horizon - 1: goal}
        _, samples = policy(conditions, batch_size=1, verbose=False)

        # samples.observations is unnormalized DLP space: (N, H, 48*10)
        generated = np.asarray(samples.observations)
        rows[arm] = generated.reshape(len(generated), horizon, 2, N_PARTICLES_PER_VIEW, 10)

    # Background latents from the true current frame, reused for every decode.
    front = env.env.obs_dict["media"][:, 0]
    _, z_bg = extract_dlp_features_with_bg(front, model, device)

    for episode in range(min(n_show, env.num_envs)):
        fig, axes = plt.subplots(2, horizon, figsize=(2.6 * horizon, 5.6))
        for r, arm in enumerate(("gaussian", "flow")):
            for h in range(horizon):
                recon = decode_view(rows[arm][episode, h, 0], z_bg[episode], model, device)
                axes[r, h].imshow(recon)
                axes[r, h].axis("off")
                label = "current" if h == 0 else ("goal" if h == horizon - 1 else f"imagined {h}")
                axes[r, h].set_title(f"{arm} · {label}", fontsize=8)
        plt.suptitle(f"Decoded imagination, episode {episode} (identical current/goal)", fontsize=12)
        plt.tight_layout()
        path = os.path.join(OUT, f"imagination_ep{episode}.png")
        plt.savefig(path, dpi=140)
        plt.close()
        print("wrote", path)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["a", "b", "both"], default="both")
    cli = parser.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    os.makedirs(OUT, exist_ok=True)

    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length

    if cli.section in ("a", "both"):
        path, errors = section_a_reconstruction(env)
        print(f"Section A: mean pixel MAE {np.mean(errors):.2f}/255 -> {path}")

    if cli.section in ("b", "both"):
        with open(EPISODE_SET, "rb") as handle:
            episodes = pickle.load(handle)
        section_b_imagination(env, args, episodes)


if __name__ == "__main__":
    main()
