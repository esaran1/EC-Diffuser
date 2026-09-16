"""Analyse the 2D field diagnostics (sections 24, 30).

The central question this answers: did the model ACTUALLY ADAPT to off-diagonal
times, or does off-diagonal error still track distance from the diagonal (the
confound that invalidated the closed warm-start branch)?

Zero GPU.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as fh:
        return json.load(fh)


def matrix(cells, grid, key):
    return np.array([[cells["{:.2f}_{:.2f}".format(ts, ta)][key]
                      for ta in grid] for ts in grid])


def diag_distance_corr(cells, grid, key):
    """Correlation between |t_s - t_a| and the metric, OFF the diagonal.

    The closed warm-start branch found corr approx -1.0 for cosine (i.e. error
    rose almost perfectly with distance from the trained diagonal). If a model
    has genuinely adapted to off-diagonal training, this dependence should be
    substantially weaker.
    """
    d, v = [], []
    for ts in grid:
        for ta in grid:
            if ts == ta:
                continue
            d.append(abs(ts - ta))
            v.append(cells["{:.2f}_{:.2f}".format(ts, ta)][key])
    d, v = np.array(d), np.array(v)
    if d.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(d, v)[0, 1])


def summarize(tag, path):
    p = load(path)
    grid, cells = p["grid"], p["cells"]
    out = {"tag": tag, "p_diag": p.get("p_diag"),
           "checkpoint_step": p.get("checkpoint_step")}
    print("\n" + "=" * 74)
    print("{}   p_diag={}   step={}".format(tag, p.get("p_diag"),
                                            p.get("checkpoint_step")))
    print("=" * 74)
    for key, label in (("a_err", "action velocity error"),
                       ("s_err", "state velocity error")):
        M = matrix(cells, grid, key)
        dia = np.array([cells["{:.2f}_{:.2f}".format(t, t)][key] for t in grid])
        off = M[~np.eye(len(grid), dtype=bool)]
        print("\n  {} (rows t_state, cols t_action)".format(label))
        print("          " + "".join("{:>9.2f}".format(t) for t in grid))
        for i, ts in enumerate(grid):
            print("    {:.2f}  ".format(ts)
                  + "".join("{:>9.4f}".format(M[i, j]) for j in range(len(grid))))
        corr = diag_distance_corr(cells, grid, key)
        print("    diagonal mean {:.4f} | off-diagonal mean {:.4f} | ratio {:.3f}"
              .format(dia.mean(), off.mean(), off.mean() / max(dia.mean(), 1e-9)))
        print("    corr(|t_s - t_a|, {}) off-diagonal = {:+.4f}".format(key, corr))
        out[key] = {"diag_mean": float(dia.mean()), "off_mean": float(off.mean()),
                    "off_over_diag": float(off.mean() / max(dia.mean(), 1e-9)),
                    "corr_dist": corr,
                    "matrix": M.tolist()}
    # semantic corners, in OUR convention (t=1 clean)
    c = lambda ts, ta, k: cells["{:.2f}_{:.2f}".format(ts, ta)][k]
    out["corners"] = {
        "state_clean_action_noisy_a_err": c(1.0, 0.0, "a_err"),
        "action_clean_state_noisy_s_err": c(0.0, 1.0, "s_err"),
        "both_noisy_a_err": c(0.0, 0.0, "a_err"),
        "both_clean_a_err": c(1.0, 1.0, "a_err"),
    }
    print("\n  semantic corners (our convention: t=1 CLEAN)")
    print("    state clean, action noisy -> action err {:.4f}  [inverse-model regime]"
          .format(out["corners"]["state_clean_action_noisy_a_err"]))
    print("    action clean, state noisy -> state  err {:.4f}  [forward-dynamics regime]"
          .format(out["corners"]["action_clean_state_noisy_s_err"]))
    print("    both noisy -> action err {:.4f}   both clean -> action err {:.4f}"
          .format(out["corners"]["both_noisy_a_err"],
                  out["corners"]["both_clean_a_err"]))
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = [os.path.join(HERE, "field_{}.json".format(a))
                 for a in ("uwm", "ctl")]
    results = []
    for p in paths:
        if not os.path.exists(p):
            print("missing: {}".format(p))
            continue
        results.append(summarize(os.path.basename(p).replace(".json", ""), p))

    if len(results) == 2:
        u = next((r for r in results if r.get("p_diag") == 0.0), None)
        c = next((r for r in results if r.get("p_diag") == 1.0), None)
        if u and c:
            print("\n" + "=" * 74)
            print("ADAPTATION VERDICT (section 23)")
            print("=" * 74)
            for k in ("a_err", "s_err"):
                print("\n  {}".format(k))
                print("    off/diag ratio   UWM {:.3f}  vs  CTL {:.3f}".format(
                    u[k]["off_over_diag"], c[k]["off_over_diag"]))
                print("    corr(dist,err)   UWM {:+.4f}  vs  CTL {:+.4f}".format(
                    u[k]["corr_dist"], c[k]["corr_dist"]))
            adapted = (u["a_err"]["off_over_diag"] < c["a_err"]["off_over_diag"])
            print("\n  UWM off-diagonal penalty LOWER than control: {}".format(adapted))
            print("  (Adaptation means the UWM arm's off-diagonal cells are")
            print("   relatively better than the diagonal-only control's, which by")
            print("   construction never saw off-diagonal times.)")
            results.append({"tag": "verdict", "uwm_adapted_offdiag": bool(adapted)})

    with open(os.path.join(HERE, "FIELD_ANALYSIS.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("\nwrote FIELD_ANALYSIS.json")


if __name__ == "__main__":
    main()
