"""Re-express the completed loss-study effects against measured evaluator
resolution (directive section 34).

For each previously reported effect, report:
  - its magnitude relative to single-realization repeatability
  - the approximate R per policy that would be needed to resolve an effect of
    that size under the measured variance model

This does NOT resurrect any loss result as positive, and imposes no retroactive
significance threshold. The output is descriptive: "this effect is below/above
the current R=1 resolution", and "resolving it would need approximately R=X".

Zero GPU.
"""
import json
import math
import os

HERE = "experiments/policy_improvement/evaluator_resolution"

# Effects as reported in the completed loss study (percentage points).
EFFECTS = [
    ("terminal mask, legacy, R=1, 4-seed mean", 2.24,
     "original headline; later shown substantially single-realization"),
    ("terminal mask, legacy, R=2 physics-averaged", 0.39,
     "same policies, one extra realization averaged in"),
    ("terminal mask, clean first-step, 4-seed mean", 1.46,
     "corrected trainer; SD 2.98 pp across seeds, one seed reversed"),
    ("denominator control B'-A, legacy", 1.82,
     "masked numerator with A's denominator"),
    ("B' - B, legacy", -0.42, "denominator-only contrast"),
    ("lambda_action=2 vs masked lambda=1 (C-B)", -1.88, "action weighting"),
    ("lambda_action=5 vs masked lambda=1 (D-B)", -1.67, "action weighting"),
]


def main():
    ap = os.path.join(HERE, "ANCHOR_ANALYSIS.json")
    if not os.path.exists(ap):
        print("ANCHOR_ANALYSIS.json not available yet")
        return
    A = json.load(open(ap))
    s_w = A["s_w"]["conservative"]          # fraction
    s_w_pp = 100 * s_w
    rc_pp = 100 * A["repeatability_coefficient"]["conservative"]

    print("=" * 92)
    print("PRIOR LOSS EFFECTS vs MEASURED EVALUATOR RESOLUTION")
    print("=" * 92)
    print("  conservative within-checkpoint s_w      : %.3f pp" % s_w_pp)
    print("  single-evaluation repeatability coeff RC: +/- %.3f pp" % rc_pp)
    print("  SE of a B-A difference at R=1           : %.3f pp" % (
        s_w_pp * math.sqrt(2)))
    print()
    print("  %-46s %9s %10s %12s" % (
        "effect", "size pp", "|d|/SE_R1", "R needed"))
    rows = []
    se_r1 = s_w_pp * math.sqrt(2)
    for name, size, note in EFFECTS:
        ratio = abs(size) / se_r1 if se_r1 else float("nan")
        # R to reach 80% power at alpha=0.05 for a true effect of this size
        R = 2 * ((1.96 + 0.8416) * s_w_pp / abs(size)) ** 2 if size else float("inf")
        rows.append({"effect": name, "size_pp": size, "ratio_to_se_r1": ratio,
                     "R_needed_power80": math.ceil(R) if math.isfinite(R) else None,
                     "note": note})
        print("  %-46s %+9.2f %10.2f %12s" % (
            name, size, ratio,
            str(math.ceil(R)) if math.isfinite(R) else "inf"))

    print()
    print("  Reading: |d|/SE_R1 is how large the reported effect is relative to")
    print("  the standard error of a single-realization A/B difference. A value")
    print("  near or below 1 means a single evaluation of each policy could not")
    print("  distinguish that effect from evaluator variation.")
    print()
    print("  No significance threshold is applied retroactively, and no effect is")
    print("  reclassified as positive by this table.")

    out = {"conservative_s_w_pp": s_w_pp, "RC_pp": rc_pp, "se_r1_pp": se_r1,
           "effects": rows,
           "caveat": ("Descriptive re-expression only. Does not revive any loss "
                      "result and imposes no retroactive significance test.")}
    with open(os.path.join(HERE, "LOSS_REINTERPRETATION.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\nwritten %s/LOSS_REINTERPRETATION.json" % HERE)


if __name__ == "__main__":
    main()
