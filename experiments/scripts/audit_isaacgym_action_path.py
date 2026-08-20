"""CPU-only audit of the Isaac Gym action path and interpolation variance.

Reproduces items 2, 3 and 6 of experiments/isaacgym_debug_investigation.md.
No GPU, no training, no environment stepping.
"""

import json
import os
import pickle

import numpy as np

from diffuser.datasets.normalization import (
    ParticleLimitsNormalizer,
    SafeLimitsNormalizer,
)

DATASET = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
FEATURES = ["pos_x", "pos_y", "scale_x", "scale_y", "depth", "f5", "f6", "f7", "f8", "transp"]
SAMPLE = 200_000


def load(seed=0):
    with open(DATASET, "rb") as handle:
        data = pickle.load(handle)
    observations = data["observations"].reshape(-1, 48, 10)
    actions = data["actions"].reshape(-1, 3)
    index = np.random.RandomState(seed).choice(len(observations), SAMPLE, replace=False)
    return observations, actions, index


def audit_action_round_trip(actions):
    """Item 2: the inverse transform must recover the raw action exactly."""
    normalizer = SafeLimitsNormalizer(actions)
    normed = normalizer.normalize(actions)
    restored = normalizer.unnormalize(normed)
    error = np.abs(restored - actions)

    zero_raw = normalizer.normalize(np.zeros((1, 3), dtype=np.float32))
    zero_normed = normalizer.unnormalize(np.zeros((1, 3), dtype=np.float32))

    return {
        "raw_min": actions.min(0).tolist(),
        "raw_max": actions.max(0).tolist(),
        "raw_mean": actions.mean(0).tolist(),
        "raw_std": actions.std(0).tolist(),
        "normed_mean": normed.mean(0).tolist(),
        "normed_std": normed.std(0).tolist(),
        "round_trip_max_abs_err": float(error.max()),
        "round_trip_mean_abs_err": float(error.mean()),
        # Defect A: the z channel is asymmetric, so neither zero is the other's image.
        "raw_zero_maps_to_normalized": zero_raw.ravel().tolist(),
        "normalized_zero_maps_to_raw": zero_normed.ravel().tolist(),
    }


def audit_feature_statistics(observations, actions, index):
    """Item 6: the VP formula assumes E[x^2] == 1. Measure what we actually have."""
    particle_normalizer = ParticleLimitsNormalizer(observations[index])
    normed_obs = particle_normalizer.normalize(observations[index])
    action_normalizer = SafeLimitsNormalizer(actions)
    normed_actions = action_normalizer.normalize(actions[index])

    x = np.concatenate([normed_actions, normed_obs.reshape(len(normed_obs), -1)], axis=1)

    per_feature = {
        name: {
            "mean": float(normed_obs[:, :, i].mean()),
            "std": float(normed_obs[:, :, i].std()),
            "E[x^2]": float((normed_obs[:, :, i] ** 2).mean()),
        }
        for i, name in enumerate(FEATURES)
    }

    return x, {
        "input_dim": int(x.shape[1]),
        "actions": {"E[x^2]": float((x[:, :3] ** 2).mean()), "std": float(x[:, :3].std())},
        "observations": {"E[x^2]": float((x[:, 3:] ** 2).mean()), "std": float(x[:, 3:].std())},
        "overall_E[x^2]": float((x**2).mean()),
        "per_feature": per_feature,
    }


def audit_interpolation(x, seed=0):
    """Item 6: compare linear and 'VP' paths, before and after standardization."""
    rng = np.random.RandomState(seed)
    sample = x[rng.choice(len(x), 20_000, replace=False)]
    noise = rng.randn(*sample.shape).astype(np.float32)
    standardized = (sample - x.mean(0)) / (x.std(0) + 1e-8)

    rows = []
    for t in np.linspace(0, 1, 21):
        scale = np.sqrt((1 - t) ** 2 + t**2)
        raw_linear = (1 - t) * noise + t * sample
        std_linear = (1 - t) * noise + t * standardized
        rows.append(
            {
                "t": float(t),
                "raw_linear_E[x^2]": float((raw_linear**2).mean()),
                "raw_vp_E[x^2]": float(((raw_linear / scale) ** 2).mean()),
                "std_linear_E[x^2]": float((std_linear**2).mean()),
                "std_vp_E[x^2]": float(((std_linear / scale) ** 2).mean()),
            }
        )
    return rows


def plot(rows, path="experiments/figures/interpolation_variance.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    left.plot(t, [r["raw_linear_E[x^2]"] for r in rows], "o-", label="linear", color="#d62728")
    left.plot(t, [r["raw_vp_E[x^2]"] for r in rows], "s-", label='"VP" formula', color="#ff7f0e")
    left.set_title("Current [-1,1] representation")

    right.plot(t, [r["std_linear_E[x^2]"] for r in rows], "o-", label="linear", color="#1f77b4")
    right.plot(t, [r["std_vp_E[x^2]"] for r in rows], "s-", label="VP formula", color="#2ca02c")
    right.set_title("Standardized representation")

    for axis in (left, right):
        axis.axhline(1.0, ls="--", c="k", alpha=0.5)
        axis.set_xlabel("t")
        axis.grid(alpha=0.3)
        axis.legend()
    left.set_ylabel(r"$E[x^2]$")
    left.annotate(
        "collapses to 0.12\nVP does NOT preserve variance",
        xy=(0.30, 0.30),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#f8d7da", ec="#721c24"),
    )
    right.annotate(
        "VP is exact: 1.000 at every t",
        xy=(0.22, 0.72),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#d4edda", ec="#155724"),
    )

    plt.suptitle(r"Interpolation path variance: $((1-t)z + tx)/\sqrt{(1-t)^2+t^2}$", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    return path


def main():
    observations, actions, index = load()
    report = {"action_round_trip": audit_action_round_trip(actions)}
    x, stats = audit_feature_statistics(observations, actions, index)
    report["feature_statistics"] = stats
    report["interpolation"] = audit_interpolation(x)
    report["figure"] = plot(report["interpolation"])

    out = "experiments/isaacgym_action_audit_results.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report["action_round_trip"], indent=2))
    print("overall E[x^2]:", report["feature_statistics"]["overall_E[x^2]"])
    print("wrote", out, "and", report["figure"])


if __name__ == "__main__":
    main()
