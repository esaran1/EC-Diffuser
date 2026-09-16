"""Baseline repeatability from EXISTING artifacts, before any new GPU work.

Uses only genuine same-checkpoint repeat groups from the registry: identical
checkpoint bytes, identical EMA convention, identical NFE, identical protocol,
all five sets VALID, no contention, non-degenerate success rate.

Reports signed and absolute macro repeat differences, per-E-set repeat spread,
and a pooled within-checkpoint SD. Explicitly distinguishes the maximum observed
shift from the standard deviation -- conflating those is the error this study
exists to correct.

Zero GPU.
"""
import itertools
import json
import math
import os
import statistics as st

REG = "experiments/policy_improvement/evaluator_resolution/evaluation_registry.json"
OUT = "experiments/policy_improvement/evaluator_resolution"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]


def main():
    reg = json.load(open(REG))
    by_tag = {r["tag"]: r for r in reg["records"] if r.get("evaluation_valid")}
    groups = reg["same_checkpoint_repeat_groups"]

    print("=" * 86)
    print("EXISTING SAME-CHECKPOINT REPEAT DATA")
    print("=" * 86)
    print("%-18s %-4s %-46s %s" % ("checkpoint", "R", "evaluations", "macros"))
    detail = {}
    for sha, tags in sorted(groups.items()):
        macros = [by_tag[t]["macro"] for t in tags if t in by_tag]
        if len(macros) < 2:
            continue
        detail[sha] = {"tags": tags, "macros": macros,
                       "per_set": {s: [by_tag[t]["per_set"][s] for t in tags
                                       if t in by_tag] for s in SETS}}
        print("%-18s %-4d %-46s %s" % (
            sha, len(macros), ",".join(tags)[:46],
            " ".join("%.4f" % m for m in macros)))

    # ---- pairwise macro repeat differences --------------------------------
    pair_signed, pair_abs = [], []
    for sha, d in detail.items():
        for a, b in itertools.combinations(d["macros"], 2):
            pair_signed.append(b - a)
            pair_abs.append(abs(b - a))

    print("\n" + "=" * 86)
    print("MACRO REPEAT DIFFERENCES (all within-checkpoint pairs)")
    print("=" * 86)
    print("  n pairs                       %d  (from %d checkpoints; pairs within"
          " a group are NOT independent)" % (len(pair_abs), len(detail)))
    print("  mean signed difference        %+.3f pp" % (100 * st.mean(pair_signed)))
    print("  mean ABSOLUTE difference      %.3f pp" % (100 * st.mean(pair_abs)))
    print("  median absolute difference    %.3f pp" % (100 * st.median(pair_abs)))
    print("  RMS repeat difference         %.3f pp" % (
        100 * math.sqrt(sum(x * x for x in pair_signed) / len(pair_signed))))
    print("  SD of signed differences      %.3f pp" % (100 * st.stdev(pair_signed)))
    print("  MAXIMUM absolute difference   %.3f pp   <- an extreme, NOT an SD"
          % (100 * max(pair_abs)))

    # ---- pooled within-checkpoint SD --------------------------------------
    # Pool residuals about each checkpoint's own mean: sum of squared deviations
    # over sum of (n_i - 1) degrees of freedom.
    ss, dof, per_ckpt = 0.0, 0, {}
    for sha, d in detail.items():
        m = d["macros"]
        if len(m) < 2:
            continue
        mu = st.mean(m)
        ss += sum((x - mu) ** 2 for x in m)
        dof += len(m) - 1
        per_ckpt[sha] = {"R": len(m), "mean": mu,
                         "sd": st.stdev(m) if len(m) > 1 else None}
    s_w = math.sqrt(ss / dof) if dof else None

    print("\n" + "=" * 86)
    print("POOLED WITHIN-CHECKPOINT SD (existing data)")
    print("=" * 86)
    print("  pooled s_w = %.3f pp   (dof = %d, across %d checkpoints)" % (
        100 * s_w, dof, len(per_ckpt)))
    print("\n  per-checkpoint:")
    for sha, v in sorted(per_ckpt.items()):
        print("    %-18s R=%d  mean %.4f  SD %s" % (
            sha, v["R"], v["mean"],
            "%.3f pp" % (100 * v["sd"]) if v["sd"] is not None else "n/a"))

    # ---- per-E-set --------------------------------------------------------
    print("\n" + "=" * 86)
    print("PER-E-SET REPEAT SPREAD (existing data)")
    print("=" * 86)
    print("  %-6s %10s %12s %12s %10s" % (
        "set", "pooled SD", "mean |diff|", "max |diff|", "n pairs"))
    per_set_sd = {}
    for s in SETS:
        ss_s, dof_s, diffs = 0.0, 0, []
        for sha, d in detail.items():
            vals = d["per_set"][s]
            if len(vals) < 2:
                continue
            mu = st.mean(vals)
            ss_s += sum((x - mu) ** 2 for x in vals)
            dof_s += len(vals) - 1
            diffs += [abs(b - a) for a, b in itertools.combinations(vals, 2)]
        sd = math.sqrt(ss_s / dof_s) if dof_s else None
        per_set_sd[s] = sd
        print("  %-6s %9.3f pp %11.3f pp %11.3f pp %10d" % (
            s, 100 * sd, 100 * st.mean(diffs), 100 * max(diffs), len(diffs)))

    out = {
        "n_repeat_groups": len(detail),
        "groups": {k: {"tags": v["tags"], "macros": v["macros"]}
                   for k, v in detail.items()},
        "macro_pairwise": {
            "n_pairs": len(pair_abs),
            "mean_signed": st.mean(pair_signed),
            "mean_abs": st.mean(pair_abs),
            "median_abs": st.median(pair_abs),
            "rms": math.sqrt(sum(x * x for x in pair_signed) / len(pair_signed)),
            "sd_signed": st.stdev(pair_signed),
            "max_abs": max(pair_abs),
        },
        "pooled_within_checkpoint_sd": s_w,
        "pooled_dof": dof,
        "per_checkpoint": per_ckpt,
        "per_set_pooled_sd": per_set_sd,
        "note": ("Maximum observed shift is an extreme of a small sample, not a "
                 "standard deviation. Pairs within a repeat group share "
                 "observations and are not independent."),
    }
    with open(os.path.join(OUT, "EXISTING_REPEATABILITY.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\nwritten %s/EXISTING_REPEATABILITY.json" % OUT)


if __name__ == "__main__":
    main()
