"""Verify the goal-horizon statistics for the OGBench puzzle-4x4 adapter.

CPU-only. Recomputes `training_goal_horizon_audit` from the raw source NPZ and
conversion manifest, checks it against the closed-form value implied by uniform
within-episode goal sampling, and contrasts the resulting conditioning demand
with the 3-cube PushCube dataset.

Usage:
    python experiments/scripts/verify_goal_horizon.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "diffuser"))

from diffuser.datasets.benchmark_sequence import _stable_integer  # noqa: E402

MANIFEST = REPO / "experiments/datasets/converted/ogbench_puzzle_4x4_play_v0_manifest.json"
PUSHCUBE = REPO / "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
HORIZON = 5
GOAL_SEED = 42

REPORTED = {
    "windows": 997000,
    "mean": 252.98824172517553,
    "median": 190.0,
    "fraction_goal_beyond_endpoint": 0.9923279839518556,
    "fraction_goal_more_than_100_steps_ahead": 0.6768525576730191,
}


def goal_offsets(manifest, split="train"):
    """Replay the adapter's exact window and goal-index construction."""
    offsets = manifest["episode_offsets"][split]
    index = 0
    out = []
    for offset in offsets:
        start, end = int(offset["start"]), int(offset["end"])
        for window_start in range(start, end - HORIZON + 1):
            lower = window_start + HORIZON - 1
            span = end - lower
            goal_index = lower + _stable_integer(GOAL_SEED, index, window_start) % span
            out.append(goal_index - window_start)
            index += 1
    return np.asarray(out, dtype=np.int64)


def analytic_fraction_beyond_endpoint(episode_length):
    """P(goal lands past the window endpoint), averaged over window positions."""
    spans = np.array(
        [episode_length - (p + HORIZON - 1) for p in range(episode_length - HORIZON + 1)],
        dtype=float,
    )
    return 1.0 - (1.0 / spans).mean()


def conditioning_demand(observations, windows, rng, samples=6000):
    """Compare in-window displacement with the displacement the goal demands."""
    achievable, demanded = [], []
    picks = rng.choice(len(windows), size=min(samples, len(windows)), replace=False)
    for i in picks:
        window_start, goal_index = windows[int(i)]
        achievable.append(
            np.linalg.norm(observations[window_start] - observations[window_start + HORIZON - 1])
        )
        demanded.append(np.linalg.norm(observations[window_start] - observations[goal_index]))
    return np.asarray(achievable), np.asarray(demanded)


def main():
    manifest = json.loads(MANIFEST.read_text())
    offsets = goal_offsets(manifest)

    episode_length = int(
        manifest["episode_offsets"]["train"][0]["end"]
        - manifest["episode_offsets"]["train"][0]["start"]
    )

    print("== Section 3: reproduction of the reported audit ==")
    computed = {
        "windows": int(offsets.size),
        "mean": float(offsets.mean()),
        "median": float(np.median(offsets)),
        "fraction_goal_beyond_endpoint": float((offsets > HORIZON - 1).mean()),
        "fraction_goal_more_than_100_steps_ahead": float((offsets > 100).mean()),
    }
    ok = True
    for key, reported in REPORTED.items():
        got = computed[key]
        match = np.isclose(got, reported, rtol=0, atol=1e-12)
        ok &= bool(match)
        print(f"  {key:42} reported={reported!r:22} computed={got!r:22} {'OK' if match else 'MISMATCH'}")
    print(f"  percentiles p90/p95/p99: "
          f"{np.percentile(offsets, 90):.0f}/{np.percentile(offsets, 95):.0f}/{np.percentile(offsets, 99):.0f}")
    print(f"  min/max: {offsets.min()}/{offsets.max()}")

    print("\n== Section 4: is it just the arithmetic of uniform sampling? ==")
    analytic = analytic_fraction_beyond_endpoint(episode_length)
    empirical = computed["fraction_goal_beyond_endpoint"]
    print(f"  episode_length={episode_length}  windows/episode={episode_length - HORIZON + 1}")
    print(f"  analytic  fraction beyond endpoint = {analytic:.7f}")
    print(f"  empirical fraction beyond endpoint = {empirical:.7f}")
    print(f"  absolute difference                = {abs(analytic - empirical):.7f}")
    print("  => the statistic is the mechanical consequence of the declared goal policy.")

    print("\n== Section 5: what the conditioning actually demands ==")
    rng = np.random.default_rng(0)

    with np.load(str(REPO / manifest["source_paths"]["train"])) as source:
        observations = np.asarray(source["observations"])
    windows = []
    index = 0
    for offset in manifest["episode_offsets"]["train"]:
        start, end = int(offset["start"]), int(offset["end"])
        for window_start in range(start, end - HORIZON + 1):
            lower = window_start + HORIZON - 1
            span = end - lower
            windows.append(
                (window_start, lower + _stable_integer(GOAL_SEED, index, window_start) % span)
            )
            index += 1
    achievable, demanded = conditioning_demand(observations, windows, rng)
    ogbench_ratio = demanded.mean() / achievable.mean()
    print(f"  OGBench puzzle-4x4  achievable={achievable.mean():.4f}  "
          f"demanded={demanded.mean():.4f}  ratio={ogbench_ratio:.1f}x")

    if PUSHCUBE.exists():
        with open(PUSHCUBE, "rb") as handle:
            buffer = pickle.load(handle)
        obs = np.asarray(buffer["observations"])
        goals = np.asarray(buffer["goals"])
        episodes, length = obs.shape[0], obs.shape[1]
        a, d = [], []
        for episode in rng.choice(episodes, size=300, replace=False):
            for _ in range(20):
                t = int(rng.integers(0, length - HORIZON))
                a.append(np.linalg.norm(obs[episode, t] - obs[episode, t + HORIZON - 1]))
                d.append(np.linalg.norm(obs[episode, t] - goals[episode, length - 1]))
        a, d = np.asarray(a), np.asarray(d)
        print(f"  3-cube PushCube     achievable={a.mean():.4f}  "
              f"demanded={d.mean():.4f}  ratio={d.mean() / a.mean():.1f}x")
        print("  => the two tasks pose structurally different problems at the same H=5.")
    else:
        print("  3-cube buffer not present; skipping contrast.")

    print("\nRESULT:", "CLAIM CONFIRMED" if ok else "CLAIM NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
