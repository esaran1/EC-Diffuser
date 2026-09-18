"""Did the section-23 extension to 10k change the verdict? Zero GPU.

Compares the registered adaptation metrics at 5k (the original pilot) and 10k
(the extension) for BOTH arms, so 'the UWM arm improved' cannot be confused
with 'both arms improved because they trained longer'.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def metrics(path, key="a_err", exclude_degenerate=False):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    c = d["cells"]
    dias, offs, dd, vv = [], [], [], []
    for ts in GRID:
        for ta in GRID:
            if exclude_degenerate and (ta == 1.0 if key == "a_err" else ts == 1.0):
                continue
            v = c["{:.2f}_{:.2f}".format(ts, ta)][key]
            if ts == ta:
                dias.append(v)
            else:
                offs.append(v)
                dd.append(abs(ts - ta))
                vv.append(v)
    return {
        "diag": float(np.mean(dias)), "off": float(np.mean(offs)),
        "ratio": float(np.mean(offs) / max(np.mean(dias), 1e-9)),
        "corr": float(np.corrcoef(dd, vv)[0, 1]),
        "corner_1_0": float(c["1.00_0.00"][key]),
        "p_diag": d.get("p_diag"),
    }


def main():
    rows = []
    for label, fn in (("uwm 5k", "field_uwm.json"), ("uwm 10k", "field_uwm_ext.json"),
                      ("ctl 5k", "field_ctl.json"), ("ctl 10k", "field_ctl_ext.json")):
        m = metrics(os.path.join(HERE, fn))
        me = metrics(os.path.join(HERE, fn), exclude_degenerate=True)
        if m:
            rows.append((label, m, me))

    print("=" * 84)
    print("ACTION BLOCK — full grid (the REGISTERED criterion)")
    print("=" * 84)
    print("  {:<10} {:>9} {:>9} {:>9} {:>10} {:>12}".format(
        "arm", "diag", "off", "off/diag", "corr", "corner(1,0)"))
    for label, m, _ in rows:
        print("  {:<10} {:>9.4f} {:>9.4f} {:>9.3f} {:>+10.4f} {:>12.4f}".format(
            label, m["diag"], m["off"], m["ratio"], m["corr"], m["corner_1_0"]))

    print("\n" + "=" * 84)
    print("ACTION BLOCK — excluding the degenerate t_action=1.0 column")
    print("=" * 84)
    print("  {:<10} {:>9} {:>9} {:>9} {:>10}".format(
        "arm", "diag", "off", "off/diag", "corr"))
    for label, _, me in rows:
        print("  {:<10} {:>9.4f} {:>9.4f} {:>9.3f} {:>+10.4f}".format(
            label, me["diag"], me["off"], me["ratio"], me["corr"]))

    d = {label: {"full": m, "excl_degenerate": me} for label, m, me in rows}

    print("\n" + "=" * 84)
    print("DID THE EXTENSION CHANGE THE VERDICT?")
    print("=" * 84)
    if "uwm 10k" in d and "ctl 10k" in d:
        for tag, sub in (("full grid", "full"),
                         ("excl. degenerate", "excl_degenerate")):
            u5, u10 = d["uwm 5k"][sub], d["uwm 10k"][sub]
            c5, c10 = d["ctl 5k"][sub], d["ctl 10k"][sub]
            print("\n  [{}]".format(tag))
            print("    UWM off/diag  {:.3f} -> {:.3f}   ({:+.1f}%)".format(
                u5["ratio"], u10["ratio"],
                100 * (u10["ratio"] - u5["ratio"]) / u5["ratio"]))
            print("    CTL off/diag  {:.3f} -> {:.3f}   ({:+.1f}%)".format(
                c5["ratio"], c10["ratio"],
                100 * (c10["ratio"] - c5["ratio"]) / c5["ratio"]))
            print("    gap (CTL/UWM) {:.2f}x -> {:.2f}x".format(
                c5["ratio"] / u5["ratio"], c10["ratio"] / u10["ratio"]))
            print("    UWM corr      {:+.4f} -> {:+.4f}".format(u5["corr"], u10["corr"]))
            print("    CTL corr      {:+.4f} -> {:+.4f}".format(c5["corr"], c10["corr"]))
        print("\n  The control is the guard: if BOTH arms improve similarly, the")
        print("  change is 'trained longer', not 'trained off-diagonal'.")
    else:
        print("  extension field data not yet available")

    with open(os.path.join(HERE, "COMPARE_5K_10K.json"), "w") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
    print("\nwrote COMPARE_5K_10K.json")


if __name__ == "__main__":
    main()
