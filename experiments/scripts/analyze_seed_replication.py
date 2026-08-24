"""Seed-42 vs seed-43 replication analysis at fixed H=100.

The primary endpoint is the equal-weight mean across 3/4/5 cubes of
per-object-success(Flow@4) - per-object-success(Flow@1).

Uncertainty uses a STRATIFIED PAIRED bootstrap: episodes are resampled
independently within each task (Flow@1 and Flow@4 share episodes, so pairs move
together), the per-task difference is computed, and the three task differences
are equal-weighted. Episodes are never pooled across tasks, and never across
training seeds.
"""

import glob
import json
import os

import numpy as np

NFE = "experiments/isaacgym_control/nfe_study"
PROBES = "experiments/isaacgym_control/fourcube"
S43 = "experiments/isaacgym_control/seed43"
TASKS = [("3cube", 3), ("4cube", 4), ("5cube", 5)]


def eps_of(paths):
    """Return {episode_id: cubes_placed} pooled over the given raw files."""
    out = {}
    for i, p in enumerate(sorted(paths)):
        with open(p) as fh:
            payload = json.load(fh)
        for e in payload["episodes"]:
            out[(i, e["episode"])] = e["cubes_placed"]
    return out


def seed42_files(task, nfe):
    if task == "3cube":                       # three replicates, pooled
        return sorted(glob.glob(f"{NFE}/r*_flow_nfe{nfe}.json"))
    return [f"{PROBES}/{task}_H100_flow_nfe{nfe}.json"]


def seed43_files(task, nfe):
    return [f"{S43}/seed43_{task}_H100_flow_nfe{nfe}.json"]


def paired_diffs(f4_paths, f1_paths, cubes):
    """Per-episode per-object-success difference, F4 minus F1, on shared episodes."""
    a, b = eps_of(f4_paths), eps_of(f1_paths)
    common = sorted(set(a) & set(b))
    return np.array([(a[k] - b[k]) / cubes for k in common], dtype=float)


def endpoint(loader, seed_label, rng_seed=0, n_boot=20000):
    per_task, arrays = {}, {}
    for task, cubes in TASKS:
        f4, f1 = loader(task, 4), loader(task, 1)
        if not all(os.path.exists(p) for p in f4 + f1):
            per_task[task] = None
            continue
        d = paired_diffs(f4, f1, cubes)
        arrays[task] = d
        per_task[task] = {"delta": float(d.mean()), "n": int(d.size)}

    if len(arrays) != len(TASKS):
        return {"seed": seed_label, "per_task": per_task, "primary": None}

    rng = np.random.RandomState(rng_seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        # Resample within each task independently, then equal-weight the tasks.
        boot[i] = np.mean([arrays[t][rng.randint(0, arrays[t].size, arrays[t].size)].mean()
                           for t, _ in TASKS])
    mean = float(np.mean([arrays[t].mean() for t, _ in TASKS]))
    return {
        "seed": seed_label,
        "per_task": per_task,
        "primary": mean,
        "primary_ci95": [float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))],
        "bootstrap_p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
    }


def main():
    s42 = endpoint(seed42_files, 42)
    s43 = endpoint(seed43_files, 43)

    print("=== PRIMARY ENDPOINT: equal-weight mean of per-object F4-F1 at H=100 ===")
    print(f"{'seed':>6s} {'3 cubes':>12s} {'4 cubes':>12s} {'5 cubes':>12s} {'mean':>12s} {'95% CI':>22s}")
    for r in (s42, s43):
        cells = []
        for t, _ in TASKS:
            v = r["per_task"].get(t)
            cells.append(f"{v['delta']:+12.4f}" if v else f"{'PENDING':>12s}")
        if r["primary"] is None:
            print(f"{r['seed']:>6d} " + " ".join(cells) + f" {'PENDING':>12s}")
        else:
            ci = f"[{r['primary_ci95'][0]:+.4f},{r['primary_ci95'][1]:+.4f}]"
            print(f"{r['seed']:>6d} " + " ".join(cells) + f" {r['primary']:+12.4f} {ci:>22s}")

    print("\n  n per task:")
    for r in (s42, s43):
        ns = {t: (r["per_task"][t]["n"] if r["per_task"].get(t) else None) for t, _ in TASKS}
        print(f"    seed {r['seed']}: {ns}")

    print("\n  Experimental unit: 2 independently trained Flow models.")
    print("  Bootstrap CIs are CHECKPOINT-LEVEL (uncertainty over episodes given a")
    print("  trained model). They are not algorithm-level evidence, and episodes")
    print("  are never pooled across training seeds.")

    out = {"seed42": s42, "seed43": s43,
           "note": ("Primary endpoint frozen before seed-43 training. Stratified "
                    "paired bootstrap: resample episodes within each task, "
                    "equal-weight the three task differences.")}
    os.makedirs("experiments/audit", exist_ok=True)
    with open("experiments/audit/seed_replication.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote experiments/audit/seed_replication.json")


if __name__ == "__main__":
    main()
