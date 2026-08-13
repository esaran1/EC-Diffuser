#!/usr/bin/env python3
"""Create a deterministic, non-copying episode split for audited PushCube data."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def build(health_report, seed=2606):
    episodes = health_report["inventory"]["episodes"]
    excluded = health_report["integrity"]["possible_clipped_or_invalid_state_episode_indices"]
    usable = sorted(set(range(episodes)).difference(excluded))
    random.Random(seed).shuffle(usable)
    validation_count = round(len(usable) * 0.10)
    test_count = round(len(usable) * 0.10)
    splits = {
        "train": sorted(usable[: len(usable) - validation_count - test_count]),
        "validation": sorted(usable[len(usable) - validation_count - test_count : -test_count]),
        "test": sorted(usable[-test_count:]),
    }
    membership = json.dumps(splits, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema_version": "ecdiffuser-episode-split-v1",
        "source_sha256": health_report["source"]["sha256"],
        "seed": seed,
        "split_unit": "episode",
        "excluded_episode_indices": excluded,
        "exclusion_reason": "object state reached the simulator clipping boundary (+/-1.0 or beyond)",
        "counts": {name: len(indices) for name, indices in splits.items()},
        "split_membership_sha256": hashlib.sha256(membership).hexdigest(),
        "splits": splits,
        "notes": [
            "No source arrays are copied or overwritten.",
            "Fit normalization on train only; validation is for predefined checkpoint diagnostics.",
            "The offline test split is not a substitute for seeded simulator evaluation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("health_report", type=Path)
    parser.add_argument("--seed", type=int, default=2606)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    health = json.loads(args.health_report.read_text())
    payload = json.dumps(build(health, args.seed), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
