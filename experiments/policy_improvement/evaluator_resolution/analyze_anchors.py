"""Primary statistical analysis of the evaluator-resolution anchor study.

Covers directive sections 19-29: per-anchor repeatability, within-checkpoint SD,
repeatability coefficient, same-policy null difference distribution, the
pseudo-treatment 3-vs-3 experiment, uncertainty vs R, two-policy difference
uncertainty, power/precision planning, GPU cost, and per-E-set resolution.

Statistical care taken throughout:
  - pairwise differences within a repeat group SHARE observations and are not
    independent; they are used descriptively only, with realization-level
    bootstrap for uncertainty
  - a maximum observed shift is never reported as a standard deviation
  - anchors are tested for heteroscedasticity before pooling; the conservative
    (noisier) anchor drives planning recommendations
  - power/precision numbers are planning approximations, not guarantees

Zero GPU.
"""
import itertools
import json
import math
import os
import random
import statistics as st

HERE = "experiments/policy_improvement/evaluator_resolution"
P2 = "experiments/policy_improvement/phase2"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
ANCHORS = ["canonical", "cleanA42"]
R_MAX = 6
EVAL_COST_GPU_H = 0.2225        # refreshed from measured runtime below if available


def load_anchor(anchor):
    out = []
    for r in range(1, R_MAX + 1):
        p = os.path.join(P2, "evalres_{}_R{}_diagnostics.json".format(anchor, r))
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ctrl = d.get("control", {})
        ok = (d.get("control_mean") is not None
              and d.get("n_valid_sets") == len(SETS)
              and all(ctrl.get(s, {}).get("status") == "VALID" for s in SETS)
              and not any(ctrl.get(s, {}).get("midset_max_foreign_mib") or 0
                          for s in SETS))
        if not ok:
            continue
        out.append({"R": r, "macro": d["control_mean"],
                    "per_set": {s: ctrl[s]["success"] for s in SETS},
                    "iso": ctrl.get("E0", {}).get("pre_snapshot", {}).get("iso")})
    return out


def describe(vals):
    n = len(vals)
    if n == 0:
        return None
    q = sorted(vals)
    out = {"n": n, "mean": st.mean(vals), "min": min(vals), "max": max(vals),
           "range": max(vals) - min(vals), "median": st.median(vals)}
    if n > 1:
        out["sd"] = st.stdev(vals)
        out["var"] = st.variance(vals)
        lo = q[int(0.25 * (n - 1))]
        hi = q[int(math.ceil(0.75 * (n - 1)))]
        out["iqr"] = hi - lo
        out["cv"] = out["sd"] / out["mean"] if out["mean"] else None
    return out


def pp(x):
    return "n/a" if x is None else "{:.3f}".format(100 * x)


def main():
    data = {a: load_anchor(a) for a in ANCHORS}
    have = {a: len(v) for a, v in data.items()}
    print("=" * 92)
    print("ANCHOR REALIZATIONS AVAILABLE: " + ", ".join(
        "{}={}".format(a, have[a]) for a in ANCHORS))
    print("=" * 92)
    if min(have.values()) < 2:
        print("insufficient repeats for analysis yet")
        return

    results = {"anchors": {}, "n_realizations": have}

    # ---- 19: full matrix, never hidden behind summaries --------------------
    for a in ANCHORS:
        rows = data[a]
        print("\n{}  ({} realizations)".format(a, len(rows)))
        print("  %-4s %10s %9s %9s %9s %9s %9s  %s" % (
            "R", "macro", *SETS, "timestamp"))
        for r in rows:
            print("  %-4d %10.4f %9.4f %9.4f %9.4f %9.4f %9.4f  %s" % (
                r["R"], r["macro"], *[r["per_set"][s] for s in SETS],
                r["iso"] or ""))
        macros = [r["macro"] for r in rows]
        dm = describe(macros)
        print("  macro: mean %.4f  SD %s pp  range %s pp  median %.4f" % (
            dm["mean"], pp(dm.get("sd")), pp(dm["range"]), dm["median"]))
        per_set_desc = {s: describe([r["per_set"][s] for r in rows]) for s in SETS}
        print("  per-set SD (pp): " + "  ".join(
            "{} {}".format(s, pp(per_set_desc[s].get("sd"))) for s in SETS))
        results["anchors"][a] = {"realizations": rows, "macro": dm,
                                 "per_set": per_set_desc}

    # ---- 20: within-checkpoint SD, heteroscedasticity check ---------------
    print("\n" + "=" * 92)
    print("WITHIN-CHECKPOINT REPEATABILITY SD")
    print("=" * 92)
    sds = {a: results["anchors"][a]["macro"].get("sd") for a in ANCHORS}
    for a in ANCHORS:
        print("  %-10s s_w = %s pp   (R=%d)" % (a, pp(sds[a]), have[a]))
    ss = dof = 0.0
    for a in ANCHORS:
        m = [r["macro"] for r in data[a]]
        mu = st.mean(m)
        ss += sum((x - mu) ** 2 for x in m)
        dof += len(m) - 1
    s_pooled = math.sqrt(ss / dof)
    ratio = max(sds.values()) / min(sds.values()) if min(sds.values()) else None
    print("  pooled s_w = %s pp  (dof %d)" % (pp(s_pooled), int(dof)))
    print("  variance ratio (larger/smaller) = %s" % (
        "n/a" if ratio is None else "{:.2f}".format(ratio ** 2)))
    hetero = ratio is not None and ratio ** 2 > 3.0
    print("  heteroscedastic (var ratio > 3)? %s" % hetero)
    s_conservative = max(v for v in sds.values() if v is not None)
    print("  CONSERVATIVE s_w for planning = %s pp (%s)" % (
        pp(s_conservative), "noisier anchor" if hetero else "noisier anchor, "
        "used regardless for safety"))
    results["s_w"] = {"per_anchor": sds, "pooled": s_pooled,
                      "conservative": s_conservative, "heteroscedastic": hetero}

    # ---- 21: repeatability coefficient ------------------------------------
    rc = 1.96 * math.sqrt(2) * s_conservative
    rc_pooled = 1.96 * math.sqrt(2) * s_pooled
    print("\n  Repeatability coefficient RC = 1.96*sqrt(2)*s_w")
    print("    using conservative s_w : +/- %s pp" % pp(rc))
    print("    using pooled s_w       : +/- %s pp" % pp(rc_pooled))
    print("    (approximate: assumes roughly normal, independent realizations)")
    results["repeatability_coefficient"] = {"conservative": rc, "pooled": rc_pooled}

    # ---- 22: empirical same-policy null difference ------------------------
    print("\n" + "=" * 92)
    print("SAME-POLICY NULL DIFFERENCE DISTRIBUTION (descriptive)")
    print("=" * 92)
    null = {}
    for a in ANCHORS:
        m = [r["macro"] for r in data[a]]
        diffs = [abs(x - y) for x, y in itertools.combinations(m, 2)]
        null[a] = diffs
        print("  %-10s %d pairs (NOT independent)  mean |d| %s pp  max %s pp" % (
            a, len(diffs), pp(st.mean(diffs)), pp(max(diffs))))
    alldiffs = [d for a in ANCHORS for d in null[a]]
    print("\n  descriptive P(|difference| >= threshold), pooled over both anchors:")
    for thr in (0.01, 0.02, 0.03, 0.04):
        frac = sum(1 for d in alldiffs if d >= thr) / len(alldiffs)
        print("    >= %.0f pp : %.3f  (%d/%d pairs)" % (
            100 * thr, frac, sum(1 for d in alldiffs if d >= thr), len(alldiffs)))
    # realization-level bootstrap for uncertainty on P(|d| >= 2pp)
    rng = random.Random(20260916)
    boots = []
    for _ in range(4000):
        vals = []
        for a in ANCHORS:
            m = [r["macro"] for r in data[a]]
            res = [rng.choice(m) for _ in m]
            vals += [abs(x - y) for x, y in itertools.combinations(res, 2)]
        boots.append(sum(1 for d in vals if d >= 0.02) / len(vals))
    boots.sort()
    print("    bootstrap 95%% interval for P(|d| >= 2 pp): [%.3f, %.3f]" % (
        boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]))
    results["null_differences"] = {a: null[a] for a in ANCHORS}

    # ---- 23: pseudo-treatment 3v3 -----------------------------------------
    print("\n" + "=" * 92)
    print("PSEUDO-TREATMENT EXPERIMENT (3 vs 3, ZERO real policy difference)")
    print("=" * 92)
    pseudo = {}
    for a in ANCHORS:
        m = [r["macro"] for r in data[a]]
        if len(m) < 6:
            print("  %-10s needs 6 realizations, have %d -- skipped" % (a, len(m)))
            continue
        effects, seen = [], set()
        for combo in itertools.combinations(range(len(m)), 3):
            comp = tuple(sorted(set(range(len(m))) - set(combo)))
            key = tuple(sorted([combo, comp]))
            if key in seen:            # A|B and B|A are the same split
                continue
            seen.add(key)
            eff = st.mean([m[i] for i in comp]) - st.mean([m[i] for i in combo])
            effects.append(eff)
            effects.append(-eff)       # both labelings are equally arbitrary
        pseudo[a] = effects
        ab = [abs(e) for e in effects]
        print("  %-10s %d unique splits  SD %s pp  max|effect| %s pp" % (
            a, len(seen), pp(st.stdev(effects)), pp(max(ab))))
        for thr in (0.01, 0.02, 0.03, 0.04):
            print("      fraction |pseudo-effect| >= %.0f pp : %.3f" % (
                100 * thr, sum(1 for e in ab if e >= thr) / len(ab)))
    results["pseudo_treatment"] = pseudo

    # ---- 24/25: uncertainty vs R ------------------------------------------
    print("\n" + "=" * 92)
    print("UNCERTAINTY VS NUMBER OF REALIZATIONS (conservative s_w = %s pp)" % pp(
        s_conservative))
    print("=" * 92)
    print("  %-4s %14s %16s %20s %14s" % (
        "R", "SE(mean) pp", "SE(B-A) pp", "95% half-width pp", "GPU-h per A/B"))
    vsR = {}
    for R in range(1, R_MAX + 1):
        se1 = s_conservative / math.sqrt(R)
        sed = s_conservative * math.sqrt(2.0 / R)
        half = 1.96 * sed
        cost = 2 * R * EVAL_COST_GPU_H
        vsR[R] = {"se_single": se1, "se_delta": sed, "half_width": half,
                  "gpu_h_per_ab": cost}
        print("  %-4d %13.3f %15.3f %19.3f %13.3f" % (
            R, 100 * se1, 100 * sed, 100 * half, cost))
    results["uncertainty_vs_R"] = vsR

    # ---- 26: power planning ----------------------------------------------
    print("\n" + "=" * 92)
    print("PLANNING: R per policy to detect a TRUE difference (alpha=0.05, power=0.80)")
    print("=" * 92)
    print("  approximation R >= 2 * ((1.96 + 0.84) * s_w / delta)^2")
    power = {}
    for delta_pp in (1, 2, 3, 4):
        delta = delta_pp / 100.0
        R = 2 * ((1.96 + 0.8416) * s_conservative / delta) ** 2
        power[delta_pp] = math.ceil(R)
        print("    delta = %d pp -> R >= %d per policy   (%.1f GPU-h per A/B pair)"
              % (delta_pp, math.ceil(R), 2 * math.ceil(R) * EVAL_COST_GPU_H))
    results["power_planning"] = power

    # ---- 27: precision planning ------------------------------------------
    print("\n" + "=" * 92)
    print("PLANNING: R per policy for a target 95%% CI half-width on B-A")
    print("=" * 92)
    prec = {}
    for hw_pp in (0.5, 1.0, 1.5, 2.0):
        hw = hw_pp / 100.0
        R = 2 * (1.96 * s_conservative / hw) ** 2
        prec[hw_pp] = math.ceil(R)
        print("    half-width <= %.1f pp -> R >= %d per policy   (%.1f GPU-h per A/B)"
              % (hw_pp, math.ceil(R), 2 * math.ceil(R) * EVAL_COST_GPU_H))
    results["precision_planning"] = prec

    # ---- 29: per-E-set ----------------------------------------------------
    print("\n" + "=" * 92)
    print("PER-E-SET RESOLUTION (diagnostic only -- no set is dropped)")
    print("=" * 92)
    print("  %-6s %12s %12s %12s %12s" % (
        "set", "canonical SD", "cleanA42 SD", "pooled SD", "max |diff|"))
    per_set_pooled = {}
    for s in SETS:
        sds_s, alld = {}, []
        ss_s = dof_s = 0.0
        for a in ANCHORS:
            v = [r["per_set"][s] for r in data[a]]
            sds_s[a] = st.stdev(v) if len(v) > 1 else None
            mu = st.mean(v)
            ss_s += sum((x - mu) ** 2 for x in v)
            dof_s += len(v) - 1
            alld += [abs(x - y) for x, y in itertools.combinations(v, 2)]
        pooled = math.sqrt(ss_s / dof_s) if dof_s else None
        per_set_pooled[s] = pooled
        print("  %-6s %11s %11s %11s %11s" % (
            s, pp(sds_s["canonical"]), pp(sds_s["cleanA42"]), pp(pooled),
            pp(max(alld)) if alld else "n/a"))
    results["per_set_pooled_sd"] = per_set_pooled

    # ---- 32: drift check --------------------------------------------------
    print("\n" + "=" * 92)
    print("TEMPORAL DRIFT CHECK (descriptive)")
    print("=" * 92)
    for a in ANCHORS:
        m = [r["macro"] for r in data[a]]
        idx = list(range(1, len(m) + 1))
        if len(m) > 2:
            mi, mm = st.mean(idx), st.mean(m)
            num = sum((i - mi) * (x - mm) for i, x in zip(idx, m))
            den = math.sqrt(sum((i - mi) ** 2 for i in idx)
                            * sum((x - mm) ** 2 for x in m))
            r_corr = num / den if den else 0.0
            print("  %-10s corr(macro, realization index) = %+.3f" % (a, r_corr))
        results["anchors"][a]["drift_corr"] = (
            r_corr if len(m) > 2 else None)
    print("  (a small |corr| with n=6 is not evidence of drift either way)")

    with open(os.path.join(HERE, "ANCHOR_ANALYSIS.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print("\nwritten %s/ANCHOR_ANALYSIS.json" % HERE)


if __name__ == "__main__":
    main()
