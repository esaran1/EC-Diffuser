"""Compare the registered inference paths across arms (sections 26-28, 30).

Answers, offline:
  - does any non-diagonal path beat the diagonal on the SAME model?
  - does the UWM arm beat its matched diagonal control?
  - are paths compared at equal model calls (section 27)?

Applies the measured evaluator resolution as the interpretation rule: offline
metrics narrow the candidate set, but only closed-loop control at R=3 can
support a performance claim (section 48).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["P0_diagonal", "P1_state_leads", "P2_action_leads",
         "P3_state_first", "P4_action_first"]


def load(tag):
    p = os.path.join(HERE, "replay_{}.json".format(tag))
    return json.load(open(p)) if os.path.exists(p) else None


def table(d, which="ema"):
    rows = []
    for p in PATHS:
        if p in d.get(which, {}):
            r = d[which][p]
            rows.append((p, r["action_err_all"], r["action_err_step0"],
                         r["state_err"], r["model_calls"], r["finite"]))
    return rows


def main():
    arms = {t: load(t) for t in ("uwm", "ctl")}
    out = {}
    for tag, d in arms.items():
        if d is None:
            print("missing replay_{}.json".format(tag))
            continue
        print("\n" + "=" * 78)
        print("ARM {}   p_diag={}   step={}   drift={:.4f}".format(
            tag, d.get("p_diag"), d.get("checkpoint_step"),
            d.get("param_drift_l2", float("nan"))))
        print("=" * 78)
        for which in ("ema", "model"):
            if which not in d:
                continue
            print("\n  [{}]".format(which.upper()))
            print("  {:<20} {:>9} {:>11} {:>10} {:>6}".format(
                "path", "act_err", "act_err@0", "state_err", "calls"))
            rows = table(d, which)
            base = dict((r[0], r) for r in rows).get("P0_diagonal")
            for p, ae, a0, se, mc, fin in rows:
                mark = ""
                if base and p != "P0_diagonal":
                    d0 = (a0 - base[2]) / max(abs(base[2]), 1e-9) * 100
                    mark = "  ({:+.1f}% vs diag @0)".format(d0)
                print("  {:<20} {:>9.4f} {:>11.4f} {:>10.4f} {:>6d}{}".format(
                    p, ae, a0, se, mc, mark))
            if rows:
                calls = set(r[4] for r in rows)
                print("  equal model calls across paths: {} ({})".format(
                    len(calls) == 1, sorted(calls)))
        out[tag] = {"paths": {p: d["ema"][p] for p in PATHS if p in d.get("ema", {})},
                    "param_drift_l2": d.get("param_drift_l2"),
                    "p_diag": d.get("p_diag")}

    if len(out) == 2 and "uwm" in out and "ctl" in out:
        print("\n" + "=" * 78)
        print("UWM vs MATCHED DIAGONAL CONTROL (offline, EMA)")
        print("=" * 78)
        print("  {:<20} {:>12} {:>12} {:>12}".format(
            "path", "UWM act@0", "CTL act@0", "UWM - CTL"))
        for p in PATHS:
            u = out["uwm"]["paths"].get(p)
            c = out["ctl"]["paths"].get(p)
            if not u or not c:
                continue
            print("  {:<20} {:>12.4f} {:>12.4f} {:>+12.4f}".format(
                p, u["action_err_step0"], c["action_err_step0"],
                u["action_err_step0"] - c["action_err_step0"]))
        print("\n  Lower is better. A negative UWM-CTL means the independent-time")
        print("  model predicts the executed action more accurately than the")
        print("  matched diagonal control at the same path and model-call count.")
        print("\n  NOTE (section 48): this is an OFFLINE metric. It narrows the")
        print("  candidate set for closed-loop evaluation; it does not by itself")
        print("  support any control claim.")

    with open(os.path.join(HERE, "REPLAY_ANALYSIS.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\nwrote REPLAY_ANALYSIS.json")


if __name__ == "__main__":
    main()
