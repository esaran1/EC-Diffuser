"""Items 11, 12, 15, 16: characterize the representation and compare three paths.

Naming is kept strict throughout, per the directive:
  * limits/min-max normalization  -- what EC-Diffuser does today, maps to [-1,1]
  * empirical standardization     -- (x - mu) / sigma, a preprocessing ablation
  * variance-preserving interpolation -- the probability path

Only the third is ever called "variance preserving", and only when applied to
standardized features.
"""

import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffuser.datasets.normalization import ParticleLimitsNormalizer, SafeLimitsNormalizer

DATASET = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
OUT_FIG = "experiments/figures/three_path_variance.png"
OUT_JSON = "experiments/feature_statistics.json"

# DLP particle layout, verified against dlp_utils.get_dlp_rep / get_recon_from_dlps
GROUPS = {
    "z_p (position)": (0, 2),
    "z_s (scale)": (2, 4),
    "z_d (depth)": (4, 5),
    "z_f (visual features)": (5, 9),
    "z_t (transparency)": (9, 10),
}
SAMPLE = 200_000
EPS = 1e-6


def load(seed=0):
    with open(DATASET, "rb") as handle:
        data = pickle.load(handle)
    obs = data["observations"].reshape(-1, 48, 10)
    act = data["actions"].reshape(-1, 3)
    index = np.random.RandomState(seed).choice(len(obs), SAMPLE, replace=False)
    return obs[index], act[index]


def describe(values):
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "var": float(values.var()),
        "E[x^2]": float((values**2).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "frac_near_bounds": float((np.abs(values) > 0.99).mean()),
    }


def main():
    obs, act = load()

    # --- Item 11: statistics of the CURRENT limits-normalized representation ---
    particle_normalizer = ParticleLimitsNormalizer(obs)
    action_normalizer = SafeLimitsNormalizer(act)
    normed_obs = particle_normalizer.normalize(obs)          # (N, 48, 10)
    normed_act = action_normalizer.normalize(act)            # (N, 3)

    stats = {"representation": "limits/min-max normalization to [-1,1]", "groups": {}}
    for name, (lo, hi) in GROUPS.items():
        stats["groups"][name] = describe(normed_obs[:, :, lo:hi])
    for i, axis in enumerate("xyz"):
        stats["groups"][f"action_{axis}"] = describe(normed_act[:, i])

    x = np.concatenate([normed_act, normed_obs.reshape(len(normed_obs), -1)], axis=1)
    stats["whole_input"] = describe(x)
    stats["input_dim"] = int(x.shape[1])

    # Near-constant channels would make standardization unstable; check explicitly.
    per_channel_std = x.std(0)
    stats["near_constant_channels"] = int((per_channel_std < 1e-3).sum())
    stats["min_channel_std"] = float(per_channel_std.min())
    stats["max_channel_std"] = float(per_channel_std.max())
    stats["channel_std_ratio"] = float(per_channel_std.max() / max(per_channel_std.min(), EPS))

    # --- Item 13: empirical standardization, guarded against tiny sigma ---
    mu, sigma = x.mean(0), x.std(0)
    safe_sigma = np.where(sigma < 1e-3, 1.0, sigma)  # leave near-constant channels alone
    x_std = (x - mu) / safe_sigma
    stats["standardized_check"] = {
        "max_abs_mean": float(np.abs(x_std.mean(0)).max()),
        "max_abs_var_minus_one": float(np.abs(x_std.var(0) - 1.0).max()),
        "channels_left_unscaled": int((sigma < 1e-3).sum()),
        "note": (
            "Per-feature marginals are standardized. This does NOT make the "
            "trajectory vector an isotropic Gaussian -- cross-feature covariance "
            "is untouched."
        ),
    }

    # --- Items 12, 15, 16: the three paths, plus the min-max+denominator control ---
    rng = np.random.RandomState(0)
    pick = rng.choice(len(x), 20_000, replace=False)
    sample_raw, sample_std = x[pick], x_std[pick]
    noise = rng.randn(*sample_raw.shape).astype(np.float64)

    group_slices = {"actions": slice(0, 3)}
    for name, (lo, hi) in GROUPS.items():
        # Particle features are interleaved across 48 particles after flattening.
        cols = 3 + np.concatenate(
            [np.arange(p * 10 + lo, p * 10 + hi) for p in range(48)]
        )
        group_slices[name] = cols

    grid = np.linspace(0, 1, 41)
    curves = {k: [] for k in ("A_linear_minmax", "B_linear_std", "C_vp_std", "D_vp_minmax")}
    group_curves = {name: [] for name in group_slices}

    for t in grid:
        scale = np.sqrt((1 - t) ** 2 + t**2)
        a = (1 - t) * noise + t * sample_raw
        b = (1 - t) * noise + t * sample_std
        curves["A_linear_minmax"].append(float((a**2).mean()))
        curves["B_linear_std"].append(float((b**2).mean()))
        curves["C_vp_std"].append(float(((b / scale) ** 2).mean()))
        # Item 16: the denominator applied to min-max features. NOT variance
        # preserving, because x_minmax does not have unit variance.
        curves["D_vp_minmax"].append(float(((a / scale) ** 2).mean()))
        for name, cols in group_slices.items():
            group_curves[name].append(float((a[:, cols] ** 2).mean()))

    stats["paths"] = {
        "grid": grid.tolist(),
        **{k: v for k, v in curves.items()},
        "A_min_over_t": float(min(curves["A_linear_minmax"])),
        "A_mid_path_reduction_pct": float(
            100 * (1 - min(curves["A_linear_minmax"]) / curves["A_linear_minmax"][0])
        ),
        "C_max_abs_dev_from_one": float(
            max(abs(v - 1.0) for v in curves["C_vp_std"])
        ),
    }
    stats["group_curves"] = {"grid": grid.tolist(), **group_curves}

    with open(OUT_JSON, "w") as handle:
        json.dump(stats, handle, indent=2)

    # ------------------------------- figure -------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    left.plot(grid, curves["A_linear_minmax"], "o-", color="#d62728",
              label="A: min-max + linear (current)")
    left.plot(grid, curves["B_linear_std"], "^-", color="#1f77b4",
              label="B: standardized + linear")
    left.plot(grid, curves["C_vp_std"], "s-", color="#2ca02c",
              label="C: standardized + VP")
    left.plot(grid, curves["D_vp_minmax"], "x--", color="#ff7f0e",
              label="D: min-max + denominator (control)")
    left.axhline(1.0, ls="--", c="k", alpha=0.5)
    left.set_xlabel("t   (0 = noise, 1 = data)")
    left.set_ylabel(r"$E[x^2]$  (model input scale)")
    left.set_title("Model input scale along the probability path")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    for name, values in group_curves.items():
        right.plot(grid, values, label=name)
    right.axhline(1.0, ls="--", c="k", alpha=0.5)
    right.set_xlabel("t   (0 = noise, 1 = data)")
    right.set_ylabel(r"$E[x^2]$")
    right.set_title("Current path (A), per feature group")
    right.legend(fontsize=7)
    right.grid(alpha=0.3)

    plt.suptitle(
        "Three probability paths on the real EC-Diffuser representation "
        f"(current path loses {stats['paths']['A_mid_path_reduction_pct']:.0f}% of input scale)",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)

    print(json.dumps({k: stats["groups"][k] for k in list(stats["groups"])[:3]}, indent=2))
    print("whole input:", json.dumps(stats["whole_input"], indent=2))
    print("channel std ratio: %.1fx  near-constant channels: %d"
          % (stats["channel_std_ratio"], stats["near_constant_channels"]))
    print("A min E[x^2] over t: %.4f (%.1f%% reduction)"
          % (stats["paths"]["A_min_over_t"], stats["paths"]["A_mid_path_reduction_pct"]))
    print("C max |E[x^2]-1|: %.5f" % stats["paths"]["C_max_abs_dev_from_one"])
    print("wrote", OUT_JSON, "and", OUT_FIG)


if __name__ == "__main__":
    main()
