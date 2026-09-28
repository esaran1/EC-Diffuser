"""Closed-loop comparison, raw numbers first.

usage: analyze_cl.py BASE CAND [CAND...]   (names as used in tf_<name>_R<r> tags)
Resolution reference: conservative within-checkpoint SD 1.552 pp (evaluator-
resolution program), so SE(B-A) at R=3 = 1.552*sqrt(2/3) = 1.267 pp and the
95% half-width is 2.48 pp. Effects below ~3 pp are UNRESOLVED at R=3.
"""
import json, math, os, sys
import numpy as np
P2 = "experiments/policy_improvement/phase2"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
SW = 1.552


def load(name, R=3):
    out = {}
    for r in range(1, R + 1):
        p = os.path.join(P2, "tf_%s_R%d_diagnostics.json" % (name, r))
        if not os.path.exists(p):
            continue
        c = json.load(open(p)).get("control") or {}
        for s in SETS:
            v = c.get(s) or {}
            if v.get("status") == "VALID":
                out[(r, s)] = v
    return out


def table(name, d):
    print("\n%s" % name)
    print("   R  " + "".join("%8s" % s for s in SETS) + "   macro")
    macros = []
    for r in (1, 2, 3):
        row = [d.get((r, s), {}).get("success") for s in SETS]
        if all(x is not None for x in row):
            macros.append(np.mean(row))
        print("  R%d  " % r + "".join("%8.4f" % x if x is not None else "     n/a" for x in row)
              + ("   %.4f" % np.mean(row) if all(x is not None for x in row) else ""))
    return macros


def main():
    names = sys.argv[1:]
    D = {n: load(n) for n in names}
    M = {n: table(n, D[n]) for n in names}
    # trace consistency: trace final_success must equal evaluator success
    bad = [(n, k) for n in names for k, v in D[n].items()
           if v.get("trace") and abs(v["trace"]["final_success"] - v["success"]) > 1e-9]
    print("\ntrace/evaluator final-success consistency:", "OK" if not bad else "MISMATCH %s" % bad[:5])
    base = names[0]
    for c in names[1:]:
        print("\n=== %s  minus  %s ===" % (c, base))
        if len(M[base]) < 3 or len(M[c]) < 3:
            print("  incomplete: %d / %d valid realizations" % (len(M[base]), len(M[c]))); continue
        per_set = {s: 100 * (np.mean([D[c][(r, s)]["success"] for r in (1, 2, 3)])
                             - np.mean([D[base][(r, s)]["success"] for r in (1, 2, 3)])) for s in SETS}
        per_r = [100 * (M[c][i] - M[base][i]) for i in range(3)]
        delta = 100 * (np.mean(M[c]) - np.mean(M[base]))
        se = SW * math.sqrt(2 / 3)
        print("  macro R=3: %s %.2f%%  vs  %s %.2f%%   delta %+.2f pp" % (
            c, 100 * np.mean(M[c]), base, 100 * np.mean(M[base]), delta))
        print("  per set (pp): " + "  ".join("%s %+.2f" % (s, per_set[s]) for s in SETS))
        print("  per realization pairing R1..R3 (pp): " + " ".join("%+.2f" % x for x in per_r))
        print("  sets improved: %d/5   realizations improved: %d/3" % (
            sum(v > 0 for v in per_set.values()), sum(x > 0 for x in per_r)))
        print("  SE(delta) %.2f pp, 95%% half-width %.2f pp (conservative s_w)  |delta|/SE %.2f -> %s" % (
            se, 1.96 * se, abs(delta) / se,
            "UNRESOLVED (<~3 pp)" if abs(delta) < 3 else "beyond routine resolution"))
    print("\nstopping / near-goal trace (mean over valid sets x realizations):")
    keys = ["final_success", "reached_any", "lost_after_reach", "steps_to_first_reach",
            "pre_reach_action_mag", "post_reach_action_mag", "post_reach_max_d_mean", "final_d_mean"]
    print("  %-22s" % "" + "".join("%14s" % n[:13] for n in names))
    for k in keys:
        vals = []
        for n in names:
            xs = [v["trace"][k] for v in D[n].values() if v.get("trace") and v["trace"].get(k) is not None]
            vals.append(np.mean(xs) if xs else float("nan"))
        print("  %-22s" % k + "".join("%14.4f" % x for x in vals))


if __name__ == "__main__":
    main()
