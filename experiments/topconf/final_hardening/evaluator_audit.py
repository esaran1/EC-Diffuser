"""Evaluator resolution audit — executable CPU utility.

Consumes repeated-outcome tensors Y[R, N] (R realizations x N fixed scenarios)
and reports the fixed-benchmark estimand, within-scenario variance, resolution
vs R, the R needed for a target effect, and a sign-reliability diagnostic.

Estimand (fixed benchmark):  J = (1/N) sum_i E_r[Y_ir]
Contrast variance:           Var[Delta] ~ 2 * mean_i p_i(1-p_i) / (N*R)

NOTE: adaptive allocation is deliberately NOT provided -- it was measured to be
worse than uniform replication (see RESOLUTION_AUDIT_FEASIBILITY.md).
"""
import numpy as np


def audit(Y):
    """Y: (R, N) binary outcomes for ONE policy arm on N fixed scenarios."""
    Y = np.asarray(Y, float)
    R, N = Y.shape
    p = Y.mean(0)
    within = float(np.mean(p * (1 - p)))
    if R > 1:                      # small-R bias correction
        within *= R / (R - 1)
    between = float(p.var(ddof=1))
    icc = between / (between + within) if (between + within) > 0 else float("nan")
    return {"R": R, "N": N, "J": float(p.mean()),
            "within_scenario_var": within, "between_scenario_var": between,
            "icc": icc,
            "n_robust_success": int((p == 1).sum()), "n_robust_fail": int((p == 0).sum()),
            "n_sensitive": int(((p > 0) & (p < 1)).sum())}


def resolution(within, N, R):
    """95% half-width on a two-arm fixed-benchmark contrast, in pp."""
    return 100 * 1.96 * np.sqrt(2 * within / (N * R))


def required_R(within, N, target_pp):
    """Smallest R whose 95% half-width on Delta is <= target_pp."""
    for R in range(1, 1001):
        if resolution(within, N, R) <= target_pp:
            return R
    return None


def sign_reliability(delta_pp, within, N, R):
    """Predicted P(correct sign) for a true effect delta_pp, given the evaluator."""
    from math import erf, sqrt
    se = 100 * np.sqrt(2 * within / (N * R))
    z = abs(delta_pp) / se
    return 0.5 * (1 + erf(z / sqrt(2)))


def report(Y, targets=(3.0, 5.0, 10.0)):
    a = audit(Y); w, N = a["within_scenario_var"], a["N"]
    lines = [f"fixed-benchmark J = {a['J']:.4f}   (R={a['R']}, N={N})",
             f"within-scenario var {w:.5f} | between {a['between_scenario_var']:.5f} | ICC {a['icc']:.3f}",
             f"scenarios: robust-success {a['n_robust_success']}, robust-fail {a['n_robust_fail']}, "
             f"physics-sensitive {a['n_sensitive']}",
             "", "resolution (95% half-width on a two-arm contrast):"]
    for R in (1, 2, 3, 5, 8):
        lines.append(f"   R={R}: {resolution(w, N, R):5.2f} pp")
    lines.append("")
    lines.append("required R for a target effect:")
    for t in targets:
        R = required_R(w, N, t)
        lines.append(f"   {t:4.1f} pp -> R = {R}")
    lines.append("")
    lines.append("predicted P(correct sign) at R=1 / R=3:")
    for d in targets:
        lines.append(f"   true effect {d:4.1f} pp: {100*sign_reliability(d,w,N,1):5.1f}% / "
                     f"{100*sign_reliability(d,w,N,3):5.1f}%")
    return "\n".join(lines), a


if __name__ == "__main__":
    import json, sys, glob
    files = sorted(glob.glob("experiments/evaluation_noise/results/r0_s42_nfe4_crnrep*.json"))
    S = [json.load(open(f)) for f in files]
    eps = sorted({e["episode"] for e in S[0]["episodes"]})
    Y = np.array([[{x["episode"]: float(x["success"]) for x in d["episodes"]}[e]
                   for e in eps] for d in S])
    txt, _ = report(Y)
    print(txt)
