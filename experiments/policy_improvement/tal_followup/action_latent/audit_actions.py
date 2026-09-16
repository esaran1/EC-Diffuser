"""Stage H action-dataset audit (directive section 40). CPU only, zero GPU.

Runs BEFORE any action-VAE training, because section 41 makes the bottleneck
decision conditional on a PREDECLARED dataset criterion:

    if the top 2 PCA dimensions explain >= 99% of nonterminal action variance,
    authorize latent dim = 2; otherwise do NOT invent a lossy 2-D bottleneck.

Separates REAL action-bearing rows from the CONSTRUCTED terminal zero-action
row for representation diagnostics (the terminal row is NOT deleted from policy
training -- this separation is for analysis only).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "diffuser")
sys.path.insert(0, ".")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "ACTION_DATASET_AUDIT.json")


def main():
    import diffuser.utils as utils
    sp = ("data/panda_push/uwm_ctl/3C_dlp_adalnpint_randcolor_H5_T4_seed42")
    if not os.path.exists(os.path.join(sp, "dataset_config.pkl")):
        sp = "data/panda_push/lossft_armAclean/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
    from diffuser.utils.serialization import load_config
    dcfg = load_config(sp, "dataset_config.pkl")
    dcfg._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
    ds = dcfg(seed=42)
    print("dataset: {} episodes/windows".format(len(ds)))

    n = min(4000, len(ds))
    rng = np.random.RandomState(20260916)
    idx = rng.choice(len(ds), size=n, replace=False)
    H = None
    rows = []
    for i in idx:
        tr = np.asarray(ds[int(i)].trajectories)
        H = tr.shape[0]
        rows.append(tr[:, :3])            # action coordinates
    A = np.stack(rows)                    # [n, H, 3]
    print("sampled {} windows, horizon {}".format(n, H))

    terminal = A[:, H - 1, :].reshape(-1, 3)
    nonterm = A[:, :H - 1, :].reshape(-1, 3)

    res = {"n_windows": int(n), "horizon": int(H)}

    print("\n=== terminal row (CONSTRUCTED zero-action target) ===")
    res["terminal"] = {
        "mean": terminal.mean(0).tolist(), "sd": terminal.std(0).tolist(),
        "min": terminal.min(0).tolist(), "max": terminal.max(0).tolist(),
        "n": int(terminal.shape[0]),
        "all_identical": bool(np.allclose(terminal, terminal[0])),
    }
    print("  mean {}  sd {}".format(np.round(terminal.mean(0), 5),
                                    np.round(terminal.std(0), 5)))
    print("  identical across all windows: {}".format(res["terminal"]["all_identical"]))

    print("\n=== nonterminal rows (REAL executed actions) ===")
    res["nonterminal"] = {
        "mean": nonterm.mean(0).tolist(), "sd": nonterm.std(0).tolist(),
        "min": nonterm.min(0).tolist(), "max": nonterm.max(0).tolist(),
        "n": int(nonterm.shape[0]),
    }
    print("  n = {}".format(nonterm.shape[0]))
    for d in range(3):
        print("  dim {}: mean {:+.4f} sd {:.4f} range [{:+.3f}, {:+.3f}]".format(
            d, nonterm[:, d].mean(), nonterm[:, d].std(),
            nonterm[:, d].min(), nonterm[:, d].max()))

    print("\n=== correlation between action dimensions ===")
    C = np.corrcoef(nonterm.T)
    res["correlation"] = C.tolist()
    for r in np.round(C, 4):
        print("  {}".format(r))

    print("\n=== PCA spectrum (the PREDECLARED bottleneck criterion) ===")
    X = nonterm - nonterm.mean(0)
    cov = np.cov(X.T)
    w, v = np.linalg.eigh(cov)
    w = w[::-1]
    frac = w / w.sum()
    cum = np.cumsum(frac)
    res["pca_eigenvalues"] = w.tolist()
    res["pca_frac"] = frac.tolist()
    res["pca_cumulative"] = cum.tolist()
    for i, (f, c) in enumerate(zip(frac, cum)):
        print("  PC{}: {:.4f} of variance   cumulative {:.4f}".format(i + 1, f, c))

    top2 = float(cum[1])
    authorize = top2 >= 0.99
    res["top2_cumulative"] = top2
    res["bottleneck_authorized"] = bool(authorize)
    print("\n  PREDECLARED CRITERION: top-2 PCA >= 99% of nonterminal variance?")
    print("  top-2 cumulative = {:.4f}  ->  latent dim 2 {}".format(
        top2, "AUTHORIZED" if authorize else "NOT AUTHORIZED"))
    if not authorize:
        print("  => do NOT invent a lossy 2-D bottleneck. Record that no")
        print("     meaningful lower-dimensional bottleneck exists (section 41).")

    print("\n=== action magnitude / multimodality ===")
    mag = np.linalg.norm(nonterm, axis=1)
    res["magnitude"] = {"mean": float(mag.mean()), "sd": float(mag.std()),
                        "p50": float(np.percentile(mag, 50)),
                        "p95": float(np.percentile(mag, 95)),
                        "frac_near_zero": float((mag < 1e-6).mean())}
    print("  |a| mean {:.4f} sd {:.4f} p50 {:.4f} p95 {:.4f}".format(
        mag.mean(), mag.std(), np.percentile(mag, 50), np.percentile(mag, 95)))
    print("  fraction of near-zero actions: {:.4f}".format(res["magnitude"]["frac_near_zero"]))

    print("\n=== temporal structure (lag-1 autocorrelation within a window) ===")
    if H > 2:
        a0 = A[:, :H - 2, :].reshape(-1, 3)
        a1 = A[:, 1:H - 1, :].reshape(-1, 3)
        ac = [float(np.corrcoef(a0[:, d], a1[:, d])[0, 1]) for d in range(3)]
        res["lag1_autocorr"] = ac
        print("  per-dim lag-1 autocorr: {}".format(np.round(ac, 4)))

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print("\nwrote {}".format(OUT))


if __name__ == "__main__":
    main()
