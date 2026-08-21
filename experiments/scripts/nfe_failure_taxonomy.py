"""Failure-mode breakdown per NFE arm, pooled across evaluation replicates.

Categories follow the taxonomy used in experiments/isaacgym_flow_diagnosis.md so
the two reports stay comparable. Contact is a proxy: the end effector came within
a cube radius and the cube subsequently moved.
"""

import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

RESULTS = "experiments/isaacgym_control/nfe_study"
ORDER = ["flow_nfe1", "flow_nfe2", "flow_nfe4", "flow_nfe8", "flow_nfe16", "gaussian_nfe100"]


def classify(episode):
    """Assign one failure category. Order matters: earliest failure wins."""
    if episode["min_ee_to_cube"] > 0.08:
        return "1. never approaches"
    if episode["n_contacted"] == 0:
        return "2. approaches, no contact"
    if episode["cubes_farther"] > 0:
        return "3. contacts, pushes wrong direction"
    return "4. pushes correctly but insufficiently"


def main():
    by_arm = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS, "r*_*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        by_arm[payload["summary"]["label"]].extend(payload["episodes"])

    print("=== FAILURE TAXONOMY, pooled over replicates ===")
    header = f"{'arm':18s} {'n':>6s} {'fails':>6s}" + "".join(
        f"{c:>10s}" for c in ("never", "no-cont", "wrong-dir", "insuff")
    )
    print(header)

    rows = {}
    for label in ORDER:
        episodes = by_arm.get(label)
        if not episodes:
            continue
        failures = [e for e in episodes if e["success"] == 0]
        counts = Counter(classify(e) for e in failures)
        rows[label] = {
            "episodes": len(episodes),
            "failures": len(failures),
            "never_approaches": counts["1. never approaches"],
            "no_contact": counts["2. approaches, no contact"],
            "wrong_direction": counts["3. contacts, pushes wrong direction"],
            "insufficient": counts["4. pushes correctly but insufficiently"],
            "contact_rate": float(np.mean([e["n_contacted"] > 0 for e in episodes])),
            "mean_cubes_placed_on_failure": (
                float(np.mean([e["cubes_placed"] for e in failures])) if failures else None
            ),
        }
        r = rows[label]
        print(f"{label:18s} {r['episodes']:6d} {r['failures']:6d}"
              f"{r['never_approaches']:10d}{r['no_contact']:10d}"
              f"{r['wrong_direction']:10d}{r['insufficient']:10d}")

    print("\nContact rate and partial credit on failed episodes:")
    print(f"{'arm':18s} {'contact rate':>13s} {'cubes placed when failed':>26s}")
    for label in ORDER:
        if label not in rows:
            continue
        r = rows[label]
        placed = r["mean_cubes_placed_on_failure"]
        print(f"{label:18s} {r['contact_rate']:13.4f} "
              f"{(f'{placed:.2f} of 3' if placed is not None else '-'):>26s}")

    out = os.path.join(RESULTS, "..", "nfe_failure_taxonomy.json")
    with open(os.path.normpath(out), "w") as handle:
        json.dump(rows, handle, indent=2)
    print("\nwrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
