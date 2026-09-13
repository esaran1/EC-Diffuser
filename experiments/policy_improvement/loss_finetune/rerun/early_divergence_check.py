"""Section-16 early divergence check, from the ACTUAL production runs.

Once an arm finishes, compare its step-500 checkpoint against every other
finished arm's step-500 checkpoint. Different runtime objectives must produce
different weights. If they are byte-identical, the objective is still not
reaching optimization -- stop before burning the remaining arms, as happened
once already.

Compares TRAINABLE PARAMETERS only. `loss_weight_matrix` is a registered buffer
that differs across arms by construction, so including it would mask a genuine
failure by making the arms look divergent when their weights are not.

Exit 0 = ok (or nothing to compare yet), exit 1 = identical weights -> STOP.
"""
import argparse
import hashlib
import json
import os
import sys

import torch

ARMS = ("A", "B", "C", "D")
# Checkpoints to compare, cheapest/earliest first. `early10` comes from 10 real
# optimizer steps in the actual production run and is the mandatory in-run
# check: an arm that silently reverted to the canonical matrix is caught after
# ~10 steps rather than after 2,000. `500` is retained as a second, later
# confirmation. Arm A predates the early10 mechanism, so it only participates
# in the step-500 comparison.
STEP_LABELS = ("early10", "500")


def param_hash(path):
    data = torch.load(path, map_location="cpu")
    state = data["model"]
    digest = hashlib.sha256()
    for name in sorted(state):
        if name == "loss_weight_matrix":
            continue                      # buffer, differs by design
        tensor = state[name]
        if not torch.is_tensor(tensor):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().to(torch.float64).numpy().tobytes())
    return digest.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runroot", required=True)
    parser.add_argument("--sub", required=True)
    parser.add_argument("--arm", required=True)
    cli = parser.parse_args()

    report = {"per_label": {}, "verdict": "PASS"}
    any_collision = False
    compared_any = False

    for label in STEP_LABELS:
        found = {}
        for arm in ARMS:
            path = os.path.join(cli.runroot, "lossft_arm{}".format(arm), cli.sub,
                                "state_{}.pt".format(label))
            if os.path.exists(path):
                found[arm] = param_hash(path)

        print("[early_divergence] '{}' parameter hashes: {}".format(label, found))

        if len(found) < 2:
            print("[early_divergence]   fewer than two arms at '{}'; "
                  "nothing to compare yet".format(label))
            report["per_label"][label] = {"hashes": found, "compared": False}
            continue

        compared_any = True
        collisions = []
        arms = sorted(found)
        for i, x in enumerate(arms):
            for y in arms[i + 1:]:
                if found[x] == found[y]:
                    collisions.append((x, y, found[x]))

        for a, b, h in collisions:
            print("[early_divergence]   *** {} and {} IDENTICAL at '{}' ({}) ***".format(
                a, b, label, h))
        any_collision = any_collision or bool(collisions)
        report["per_label"][label] = {
            "hashes": found,
            "compared": True,
            "collisions": [{"a": a, "b": b, "sha16": h} for a, b, h in collisions],
            "verdict": "FAIL" if collisions else "PASS",
        }

    report["verdict"] = "FAIL" if any_collision else "PASS"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "EARLY_DIVERGENCE_{}.json".format(cli.arm))
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if any_collision:
        print("[early_divergence] VERDICT: FAIL -- objective not reaching optimization")
        return 1
    if not compared_any:
        print("[early_divergence] VERDICT: PASS (nothing comparable yet)")
        return 0
    print("[early_divergence] VERDICT: PASS -- all available arms distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
