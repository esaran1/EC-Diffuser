"""Tables from raw units: results/units.csv (one row per unit) and paired
comparisons. Pairing key = (scenario set, realization, episode); every config
of a checkpoint sees the same scenario, colors and policy noise for that key.

  analyze.py csv
  analyze.py curve CKPT NE CFG1 CFG2 ...        success per config, paired vs first
  analyze.py pair CKPT NE CFG_A CFG_B [R]       A - B
"""
import csv, glob, json, os, sys
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.join(HERE, "results/units")
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]


def load_units():
    out = []
    for f in sorted(glob.glob(os.path.join(U, "*.json"))):
        j = json.load(open(f))
        if j.get("status") == "VALID" and j.get("layout", "canonical_16") == "canonical_16":
            out.append(j)
    return out


def write_csv():
    rows = load_units()
    keys = ["unit", "ckpt", "train_seed", "epoch", "internal_step", "ckpt_sha256", "num_entity", "batch",
            "layout", "set", "set_sha256", "realization", "nfe", "solver", "calls_per_plan", "precision",
            "n", "successes", "success", "lat_ms_mean", "lat_ms_p50", "wall_s", "status", "noise", "colors"]
    with open(os.path.join(HERE, "results/units.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("wrote results/units.csv ({} valid canonical units)".format(len(rows)))


def vec(ckpt, ne, cfg, Rs=("1", "2"), sets=SETS, k=""):
    nfe, sol, prec = cfg.split(":")
    v = []
    for s in sets:
        for R in Rs:
            p = os.path.join(U, "{}_c{}_b16_{}{}_{}_R{}_{}{}".format(ckpt, ne, sol, nfe, prec, R, s, k))
            if not os.path.exists(p + ".json") or json.load(open(p + ".json"))["status"] != "VALID":
                return None
            v.append(np.load(p + ".npz")["success"].astype(bool))
    return np.concatenate(v)


def mcnemar(a, b):
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum()); n = b10 + b01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b10, b01) + 1)) / 2 ** n) if n else 1.0
    return b10, b01, p


def boot_ci(a, b, B=20000, seed=0, q=(5, 95)):
    x = a.astype(float) - b.astype(float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, q)


def pair(a, b):
    lo, hi = boot_ci(a, b); b10, b01, p = mcnemar(a, b)
    return dict(sA=float(a.mean()), sB=float(b.mean()), d_pp=100 * float(a.mean() - b.mean()),
                ci90=[100 * float(lo), 100 * float(hi)], disc=[b10, b01], p=p, n=int(len(a)))


def fmt(r):
    return "{:.1f} vs {:.1f}  d={:+.1f}pp  90%CI[{:+.1f},{:+.1f}]  disc {}/{}  p={:.3f}  n={}".format(
        100 * r["sA"], 100 * r["sB"], r["d_pp"], r["ci90"][0], r["ci90"][1], r["disc"][0], r["disc"][1], r["p"], r["n"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "csv":
        write_csv()
    elif cmd == "curve":
        ckpt, ne, cfgs = sys.argv[2], sys.argv[3], sys.argv[4:]
        ref = vec(ckpt, ne, cfgs[0])
        for c in cfgs:
            v = vec(ckpt, ne, c)
            if v is None:
                print("{:16s} missing".format(c)); continue
            print("{:16s} {:.1f}%  (n={})   vs {}: {}".format(c, 100 * v.mean(), len(v), cfgs[0], fmt(pair(v, ref)) if c != cfgs[0] else "-"))
    elif cmd == "pair":
        ckpt, ne, A, B = sys.argv[2:6]; Rs = tuple(sys.argv[6].split(",")) if len(sys.argv) > 6 else ("1", "2")
        print(fmt(pair(vec(ckpt, ne, A, Rs), vec(ckpt, ne, B, Rs))))


def pooled(cells, A, B):
    """cells: [(ckpt, ne)]; per-cell paired diffs + pooled paired (episode-level
    concatenation within cells) and seed-level mean/min/max of cell effects."""
    a = [vec(c, n, A) for c, n in cells]; b = [vec(c, n, B) for c, n in cells]
    per = [100 * (x.mean() - y.mean()) for x, y in zip(a, b)]
    r = pair(np.concatenate(a), np.concatenate(b))
    return per, r


if __name__ == "__main__" and sys.argv[1] == "pooled":
    ne_list = sys.argv[2].split(","); A, B = sys.argv[3], sys.argv[4]
    seeds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["s42_400k", "s43_400k", "s44_400k"]
    for ne in ne_list + ([",".join(ne_list)] if len(ne_list) > 1 else []):
        cells = [(s, n) for s in seeds for n in ne.split(",")]
        per, r = pooled(cells, A, B)
        print("{} - {} | cubes {} | per-cell {} | pooled {}".format(A, B, ne, " ".join("{:+.1f}".format(p) for p in per), fmt(r)))


def write_episodes_csv():
    """Per-episode outcomes for every valid canonical unit (pairing key + success)."""
    import gzip
    with gzip.open(os.path.join(HERE, "results/episodes.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "ckpt", "train_seed", "num_entity", "set", "realization", "nfe", "solver",
                    "precision", "episode", "success", "final_d"])
        for j in load_units():
            z = np.load(os.path.join(U, j["unit"] + ".npz"))
            for e, (s, d) in enumerate(zip(z["success"], z["final_d"])):
                w.writerow([j["unit"], j["ckpt"], j["train_seed"], j["num_entity"], j["set"], j["realization"],
                            j["nfe"], j["solver"], j["precision"], e, int(s), "{:.5f}".format(float(d))])


if __name__ == "__main__" and sys.argv[1] == "episodes":
    write_episodes_csv(); print("wrote results/episodes.csv.gz")
